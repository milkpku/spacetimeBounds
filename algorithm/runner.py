import numpy as np
import torch
from .sample_pool import SamplePool
from .select_pool import SelectPool

class RunnerGAE:
  """ Given environment, actor and critic, sample given number of samples

  """
  def __init__(self, env, s_norm, actor, critic, sample_size, gamma, lam, exp_rate=1.0,
          use_importance_sampling=True, num_segments=10, sample_gamma=0.99, sample_base=0.2, sample_tmp=1, ckpt_sample_prob=None,
          use_state_evolution=True, num_selected_elite=200, select_delay=400, ckpt_select_set=None,
          use_gpu_model=False):
    """
      Inputs:
        env       gym.env_vec, vectorized environment, need to have following funcs:
                    env.num_envs
                    env.observation_space
                    env.set_sample_prob(prob)
                    obs = env.reset()
                    obs, rwds, news, infos = env.step(acs)

        s_norm    torch.nn, normalizer for input states, need to have following funcs:
                    obst_normed = s_norm(obst)
                    s_norm.record(obt)
                    s_norm.update()

        actor     torch.nn, actor model, need to have following funcs:
                    m = actor.act_distribution(obst_norm), where m is a Gaussian distribution

        critic    torch.nn, critic model, need to have following funcs:
                    vpreds = critic(obst_norm)

        sample_size   int, number of samples per run()

        gamma     float, discount factor of reinforcement learning

        lam       float, lambda for GAE algorithm

        exp_rate  float, probability to take stochastic action
    """
    self.env = env
    self.s_norm = s_norm
    self.actor = actor
    self.critic = critic
    self.nenv = nenv = env.num_envs
    self.obs = np.zeros((nenv, env.state_size), dtype=np.float32)
    self.obs[:] = env.reset()
    self._use_gpu = use_gpu_model
    if use_gpu_model:
      self.toTorch = lambda x: torch.cuda.FloatTensor(x)
    else:
      self.toTorch = lambda x: torch.FloatTensor(x)
    self.sample_size = sample_size
    self.news = [True for _ in range(nenv)]

    # lambda used in GAE
    self.lam = lam

    # discount rate
    self.gamma = gamma
    self.v_min = 0
    self.v_max = 1.0/(1.0-gamma)
    self.estimate_mode = 0

    # exploration rate
    self.exp_rate = exp_rate

    # importance sampling
    self.iter = 0
    self.use_importance_sampling = use_importance_sampling
    if use_importance_sampling:
      if ckpt_sample_prob is not None:
        self._div = len(ckpt_sample_prob)
        self.importance_pool = SamplePool(num_segments, sample_gamma, sample_base, sample_tmp)
        self.importance_pool._prob = ckpt_sample_prob
      else:
        self._div = num_segments
        self.importance_pool = SamplePool(num_segments, sample_gamma, sample_base, sample_tmp)

    # select set
    self.use_state_evolution = use_state_evolution
    if use_state_evolution:
      if ckpt_select_set is not None:
        self.select_pool = SelectPool(self._div, num_selected_elite)
        self._select_delay = -1
        self.select_pool._state_val = ckpt_select_set["select_val"]
        self.select_pool._state = ckpt_select_set["select_set"]
      else:
        self.select_pool = SelectPool(num_segments, num_selected_elite)
        self._select_delay = select_delay

  def set_exp_rate(self, exp_rate):
    self.exp_rate = exp_rate

  def _prev_run(self):
    """ Update sampling pool and initialization poses & velocities before sampling
    """
    # initializing sampling strategy
    self.iter += 1

    if self.use_importance_sampling:
      self.importance_pool.update_prob()
      val = self.importance_pool._val_pool
      prob = self.importance_pool._prob
      print("sampler estimated val: ", val)
      print("sampling prob: ", prob)
    else:
      prob = np.ones(1)

    # use select set
    if self.use_state_evolution and self.iter > self._select_delay:
      select_set = self.select_pool.get_select_set()
      noise_level = self.select_pool.get_noise_level()
    else:
      select_set = None
      noise_level = None

    self.env.set_sample_data(prob, select_set, noise_level)

    self.fail_phase = []
    self.start_phase = []
    self.end_rwds = []
    self.bound_channels = []

  def _post_run(self, data):
    """
      record data to do importance sampling and selecting
    """

    # record for importance sampling
    if self.use_importance_sampling:
      val = data["vtargs"]
      phase = data["obs"][:, 0]
      start = data["news"]

      start_val = val[start]
      start_phase = phase[start]
      self.importance_pool.record(start_phase, start_val)

    # record for select set
    if self.use_state_evolution:
      val = data["vtargs"]
      phase = data["obs"][:, 0]
      states = data["poses"]
      self.select_pool.record(val, phase, states)
      self.select_pool.update()
      np.set_printoptions(precision=3)
      val_mean = np.array(self.select_pool._val_mean)
      val_var = np.array(self.select_pool._val_var)
      noise_level = np.array(self.select_pool.get_noise_level())
      print("select set avg val: ", val_mean)
      print("select val var: ", val_var)
      print("select noise level: ", noise_level)

  def run(self):
    """ run policy and get data for PPO training

      ==========================================================
      steps     0       1       2       ...     t       t+1
      -----------------------------------------------------------
      new       True    False   True    ...     False   True
      ob        s0      s1      s2      ...     s_t     s_{t+1}
      pose      p0      p1      p2      ...     p_t     p_{t+1}
      exp       True    True    False   ...     True    True
      ac        a0      a1      a2      ...     a_t     a_{t+1}
      --------------------------------------------------------- >> where yeild happens
      rwd       r0      r1      r2      ...     r_t     r_{t+1}
      fail      False   False   False   ...     True    False
      vpred     v(s0)   v(s1)   v(s2)   ...     v(s_t)  v(s_{t+1})
      alogp     lp(a0)  lp(a1)  lp(a2)  ...     lp(a_t) lp(a_{t+1})
                        |                       |
                        v                       v
      end_obs   -       ep0_end -       ...     ep1_end -
      end_vpreds-       ep0_val -       ...     ep1_val -

      for vectorized env, all data are stored in mb_${var}s

      besides new, ob, exp, ac, rwd, fail, there is information of networks:
        vpred   predicted value of current state s_t
        alogp   log_prob of choosing current action a_t
    """

    # clear record of fail phase
    self._prev_run()

    # estimated number of steps env_vec need to take
    #nsteps = int(1.02 * self.sample_size / (self.nenv * self.exp_rate))
    nsteps = int(self.sample_size / self.nenv) + 1

    mb_news = []
    mb_obs  = []
    mb_poses= []
    mb_exps = []
    mb_acs  = []
    mb_rwds = []
    mb_vpreds = []
    mb_alogps = []
    mb_ends   = []  # if the episode ends here
    mb_fails  = []  # if the episode ends here cause of failure
    mb_bounds = []  # if the episode ends here cause of failure of bounds
    mb_wraps  = []  # if the episode ends here cause of succeed to the end
    mb_trusts = []  # if the state's value is trustable
    mb_ob_ends= []
    mb_vends  = []

    # update normalizer, then freeze until finish training
    self.s_norm.update()
    if self._use_gpu:
      self.s_norm.cpu()
      self.actor.cpu()
    for _ in range(nsteps):
      # normalize input state before feed to actor & critic
      obst = torch.FloatTensor(self.obs)
      self.s_norm.record(obst)
      obst_norm = self.s_norm(obst)

      with torch.no_grad():
        # with probability exp_rate to act stochastically
        m = self.actor.act_distribution(obst_norm)
        exp = np.random.rand() < self.exp_rate
        acs = m.sample() if exp else m.mean
        alogps = torch.sum(m.log_prob(acs), dim=1).cpu().numpy()
        acs = acs.cpu().numpy()
        #vpreds = self.critic(obst_norm).view(-1).numpy()

      mb_news.append(self.news.copy())
      mb_obs.append(self.obs.copy())
      mb_exps.append([exp]*self.nenv)
      mb_acs.append(acs)
      #mb_vpreds.append(vpreds)
      mb_alogps.append(alogps)

      self.obs[:], rwds, self.news, infos = self.env.step(acs)

      mb_rwds.append(rwds)

      # gather reset pose infomation
      poses = [infos[i]["pose"] for i in range(self.nenv)]
      mb_poses.append(poses)

      # classify those stop by timer to be success, and those stoped by contacting
      # ground or torque exceeding limit as fail
      fails = [infos[i]["terminate"] or not infos[i]["valid_episode"] for i in range(self.nenv)]
      bound = [infos[i]["bound"] for i in range(self.nenv)]
      wraps = [infos[i]["wrap_end"] for i in range(self.nenv)]
      trusts= [infos[i]["trust_rwd"] for i in range(self.nenv)]
      mb_fails.append(fails)
      mb_bounds.append(bound)
      mb_wraps.append(wraps)
      mb_trusts.append(trusts)

      # if is done, calculate vend pred using critic
      ends = np.zeros(self.nenv)
      for i, done in enumerate(self.news):
        if done:
          ob_end = infos[i]["end_ob"]
          mb_ob_ends.append(ob_end)
          ends[i] = 1

      mb_ends.append(ends)

      # record the last rwd for start_phase, then update start phase
      for i, done in enumerate(self.news):
        if done:
          self.start_phase.append(infos[i]["start_phase"])
          self.end_rwds.append(rwds[i])

      # record the fail phases, not the end phases
      for i, done in enumerate(self.news):
        if done and infos[i]["terminate"]:
          self.fail_phase.append(infos[i]["end_ob"][0])

      # record the active bound channel
      for i, done in enumerate(self.news):
        if done and infos[i]["bound"]:
          self.bound_channels.extend(infos[i]["bound_active"])

    if self._use_gpu:
      self.s_norm.cuda()
      self.actor.cuda()

    mb_start_phases = np.array(self.start_phase)
    mb_end_rwds     = np.array(self.end_rwds)
    mb_fail_phases  = np.array(self.fail_phase)
    mb_bound_channels=np.array(self.bound_channels)

    mb_news = np.asarray(mb_news,   dtype=np.bool)
    mb_obs  = np.asarray(mb_obs,    dtype=np.float32)
    mb_poses= np.asarray(mb_poses,  dtype=np.float32)
    mb_exps = np.asarray(mb_exps,   dtype=np.bool)
    mb_acs  = np.asarray(mb_acs,    dtype=np.float32)
    mb_rwds = np.asarray(mb_rwds,   dtype=np.float32)
    #mb_vpreds=np.asarray(mb_vpreds, dtype=np.float32)
    mb_alogps=np.asarray(mb_alogps, dtype=np.float32)
    mb_fails= np.asarray(mb_fails,  dtype=np.bool)
    mb_bounds=np.asarray(mb_bounds,  dtype=np.bool)
    mb_wraps= np.asarray(mb_wraps,  dtype=np.bool)
    mb_trusts=np.asarray(mb_trusts, dtype=np.bool)
    mb_ends = np.asarray(mb_ends,  dtype=np.bool)
    mb_ob_ends= np.asarray(mb_ob_ends, dtype=np.float32)
    #mb_vends= np.asarray(mb_vends,  dtype=np.float32)

    # edit rwds according to trusts
    mean_rwd = mb_rwds[mb_trusts].mean()
    mb_rwds[np.logical_not(mb_trusts)] = mean_rwd

    # evaluate vpred and vends separately
    with torch.no_grad():
      obst = torch.Tensor(mb_obs)
      obst_norm = self.s_norm(obst)
      mb_vpreds = self.critic(obst_norm)
      dim0, dim1, dim2 = mb_vpreds.shape
      mb_vpreds = mb_vpreds.reshape(dim0, dim1)
      mb_vpreds = mb_vpreds.cpu().data.numpy()

      mb_vends = np.zeros(mb_ends.shape)
      if len(mb_ob_ends) > 0:
        obst = torch.Tensor(mb_ob_ends)
        obst_norm = self.s_norm(obst)
        vends = self.critic(obst_norm)
        mb_vends[mb_ends] = vends.cpu().view(-1)

    # GAE(\lam) algorithm to estimate vtarg and adv
    # if end is succ_end, then estimate by assuming following reward be the same
    # as current reward, so estimated value shoud be (self.v_max * rwd)
    with torch.no_grad():
      obst = self.toTorch(self.obs)
      obst_norm = self.s_norm(obst)
      last_vpreds = self.critic(obst_norm).cpu().view(-1).numpy()

      fail_end = np.logical_and(self.news, mb_fails[-1])
      succ_end = np.logical_and(self.news, np.logical_not(mb_fails[-1]))
      wrap_end = np.logical_and(self.news, mb_wraps[-1])
      last_vpreds[fail_end] = self.v_min
      if self.estimate_mode == 0:
        last_vpreds[succ_end] = self.v_max * mb_rwds[-1][succ_end]
      elif self.estimate_mode == 1:
        last_vpreds[succ_end] = mb_vends[-1][succ_end]
      elif self.estimate_mode == 2:
        last_vpreds[succ_end] = self.v_max
      else:
        assert(False and "not supported mode %d" % self.estimate_mode)
      last_vpreds[wrap_end] = self.v_max * mb_rwds[-1][wrap_end]

    mb_vtargs= np.zeros_like(mb_rwds)
    mb_advs  = np.zeros_like(mb_rwds)

    mb_nextvalues = mb_advs
    mb_nextvalues[:-1] = mb_vpreds[1:]
    fail_end = np.logical_and(mb_news[1:], mb_fails[:-1])
    succ_end = np.logical_and(mb_news[1:], np.logical_not(mb_fails[:-1]))
    wrap_end = np.logical_and(mb_news[1:], mb_wraps[:-1])
    mb_nextvalues[:-1][fail_end] = self.v_min
    if self.estimate_mode == 0:
      mb_nextvalues[:-1][succ_end] = self.v_max * mb_rwds[:-1][succ_end]
    elif self.estimate_mode == 1:
      mb_nextvalues[:-1][succ_end] = mb_vends[:-1][succ_end]
    elif self.estimate_mode == 2:
      mb_nextvalues[:-1][succ_end] = self.v_max
    else:
      assert(False and "not supported mode %d" % self.estimate_mode)
    mb_nextvalues[:-1][wrap_end] = self.v_max * mb_rwds[:-1][wrap_end]

    mb_nextvalues[-1] = last_vpreds

    mb_delta = mb_advs
    mb_delta = mb_rwds + self.gamma * mb_nextvalues - mb_vpreds

    lastgaelam = 0
    for t in reversed(range(nsteps)):
      if t == nsteps - 1:
         nextnonterminal = 1.0 - self.news
      else:
         nextnonterminal = 1.0 - mb_news[t+1]
      mb_advs[t] = lastgaelam = mb_delta[t] + self.gamma * self.lam * nextnonterminal * lastgaelam

    mb_vtargs = mb_advs + mb_vpreds

    keys = ["news", "obs", "poses", "exps", "acs", "rwds", "fails", "bounds", "advs", "vtargs", "a_logps"]
    contents = map(sf01, (mb_news, mb_obs, mb_poses, mb_exps, mb_acs, mb_rwds, mb_fails, mb_bounds, mb_advs, mb_vtargs, mb_alogps))

    data = {}
    for key, cont in zip(keys, contents):
      data[key] = cont

    data["samples"] = data["news"].size
    data["explores"] = data["exps"].sum()
    if mb_start_phases.size > 0:
      data["start_phases"] = mb_start_phases
      data["end_rwds"] = mb_end_rwds
    else:
      data["start_phases"] = np.array([0]) # phases not allowed to be None
      data["end_rwds"] = np.array([-1])

    if mb_fail_phases.size > 0:
      data["fail_phases"] = mb_fail_phases
    else:
      data["fail_phases"] = np.array([-1]) # phases not allowed to be None

    if mb_bound_channels.size > 0:
      data["bound_channels"] = mb_bound_channels
    else:
      data["bound_channels"] = np.array([-1])

    # record data for pooling and initialization states
    self._post_run(data)

    return data

  def test(self):
    """ Test current policy with unlimited timer

      Outputs:
        avg_step
        avg_rwd
    """
    alive = np.array([True for _ in range(self.nenv)])
    any_alive = True
    acc_rwd = np.zeros(self.nenv)
    acc_step = np.zeros(self.nenv)

    # prepare environment, set mode to TEST
    self.env.set_mode(1)
    self.obs = self.env.reset()
    self.news = [True for _ in range(self.nenv)]
    if self._use_gpu:
      self.s_norm.cpu()
      self.actor.cpu()
    while any_alive:
      # normalize input state before feed to actor & critic
      obst = torch.FloatTensor(self.obs)
      self.s_norm.record(obst)
      obst_norm = self.s_norm(obst)

      with torch.no_grad():
        # with probability exp_rate to act stochastically
        acs = self.actor.act_deterministic(obst_norm)
        acs = acs.cpu().numpy()

      self.obs[:], rwds, self.news, infos = self.env.step(acs)

      # decide which are alive, since timer is set to max, so using self.news as fails
      alive = np.logical_and(alive, np.logical_not(self.news))

      # record the rwd and step for alive agents
      acc_rwd += rwds * alive
      acc_step += alive

      # decide if any are alive
      any_alive = np.any(alive)

    if self._use_gpu:
      self.s_norm.cuda()
      self.actor.cuda()

    avg_step = np.mean(acc_step)
    avg_rwd  = np.mean(acc_rwd)

    # turn mode back to train IMPORTANT
    self.env.set_mode(0)
    self.obs = self.env.reset()

    return avg_step, avg_rwd

  def set_estimate(self, mode):
    """ Set end state estimate mode
      mode 0: use last reward as following rewards
      mode 1: use vpred as following end state value
      mode 2: use r_max and r_min as following rewards
    """
    assert(mode == 0 or mode == 1 or mode == 2 and "mode %d not supported" % mode)
    self.estimate_mode = mode

  def save_info(self, data):
    if self.use_importance_sampling:
      data["importance"] = self.importance_pool._prob
    if self.use_state_evolution and self.iter > self._select_delay:
      data["select_set"] = self.select_pool.get_select_set()
      data["select_val"] = self.select_pool._state_val
      data["val_var"] = self.select_pool.get_noise_level()
    return data

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def test_gaelam():
  nsample = 100
  nenv = 10
  mb_values= np.random.rand(nsample, nenv)
  mb_rwds  = np.random.rand(nsample, nenv)
  mb_news  = np.random.rand(nsample, nenv) > 0
  mb_fails = np.random.rand(nsample, nenv) > 0
  news = np.random.rand(nenv) > 0
  last_vpreds = np.random.rand(nenv)
  #fail_end = np.logical_and(news, mb_fails[-1])
  #succ_end = np.logical_and(news, np.logical_not(mb_fails[-1]))
  last_vpreds[news] = 0

  gamma = 0.95
  lam   = 0.99

  # like's implementation
  mb_vtarg = np.zeros_like(mb_rwds)
  mb_advs  = np.zeros_like(mb_rwds)

  mb_nextvalues = mb_advs
  mb_nextvalues[:-1] = mb_values[1:]
  mb_nextvalues[:-1][mb_news[1:]] = 0
  #fail_end = np.logical_and(mb_news[1:], mb_fails[:-1])
  #succ_end = np.logical_and(mb_news[1:], np.logical_not(mb_fails[:-1]))
  #mb_nextvalues[:-1][fail_end] = self.v_min
  #mb_nextvalues[:-1][succ_end] = self.v_max
  mb_nextvalues[-1] = last_vpreds

  mb_delta = mb_advs
  mb_delta = mb_rwds + gamma * mb_nextvalues - mb_values

  lastgaelam = 0
  for t in reversed(range(nsample)):
    if t == nsample - 1:
       nextnonterminal = 1.0 - news
    else:
       nextnonterminal = 1.0 - mb_news[t+1]
    mb_advs[t] = lastgaelam = mb_delta[t] + gamma * lam * nextnonterminal * lastgaelam

  mb_vtarg = mb_advs + mb_values
  mb_advs_like = mb_advs

  # baselines' implementation
  mb_rewards = mb_rwds
  last_values = last_vpreds
  mb_returns = np.zeros_like(mb_rewards)
  mb_advs = np.zeros_like(mb_rewards)
  lastgaelam = 0
  for t in reversed(range(nsample)):
    if t == nsample - 1:
      nextnonterminal = 1.0 - news
      nextvalues = last_values
    else:
      nextnonterminal = 1.0 - mb_news[t+1]
      nextvalues = mb_values[t+1]
    delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_values[t]
    mb_advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
  mb_returns = mb_advs + mb_values

  vtarg_err = abs(mb_vtarg - mb_returns).max()
  adv_err = abs(mb_advs_like - mb_advs).max()
  print("vtarg error: %f, adv error: %f" % (vtarg_err, adv_err))

if __name__=="__main__":
  # test gae algorithm
  test_gaelam()
