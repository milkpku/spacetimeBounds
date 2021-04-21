import numpy as np
from multiprocessing import Process, Pipe
from env import VecEnv, CloudpickleWrapper

RESERVE_RATE = 0.2

def worker(remote, parent_remote, env_fn_wrapper):
  parent_remote.close()
  env = env_fn_wrapper.x()
  prob = None
  select_set = None
  noise_level = None

  def sample_selection():
    if prob is None:
      phase = env._rand.rand()
    else:
      div = env._rand.choice(len(prob), p=prob)
      phase = (div + env._rand.rand())/len(prob)

    if select_set is None or np.random.rand() < RESERVE_RATE:
      return phase, None, None
    else:
      # select randomly
      choice = int(np.random.rand() * (len(select_set[div])-1))
      state = select_set[div][choice]
      noise = noise_level[div]
      return None, state, noise

  try:
    while True:
      cmd, data = remote.recv()
      if cmd == 'step':
        ob, reward, done, info = env.step(data)
        if done:
          info["end_ob"] = ob
          phase, state, noise = sample_selection()
          if noise is not None:
            env.set_reset_noise(noise)
          ob = env.reset(phase, state)
        remote.send((ob, reward, done, info))
      elif cmd == 'get_state_size':
        remote.send(env.get_state_size())
      elif cmd == 'get_reset_data':
        state = env.get_reset_data(data)
        remote.send(state)
      elif cmd == 'reset':
        phase, state, noise = sample_selection()
        if noise is not None:
          env.set_reset_noise(noise)
        ob = env.reset(phase, state)
        remote.send(ob)
      elif cmd == 'reset_from_data':
        ob = env.reset(None, data)
        remote.send(ob)
      elif cmd == 'close':
        remote.close()
        break
      elif cmd == 'set_task_t':
        env.set_task_t(data)
        remote.send(True)
      elif cmd == 'set_sample_data':
        prob = data[0]
        select_set = data[1]
        noise_level = data[2]
        remote.send(True)
      elif cmd == 'set_mode':
        env.set_mode(data)
        remote.send(True)
      elif cmd == 'calc_sub_rewards':
        err_vec = env.calc_sub_rewards()
        remote.send(err_vec)
      else:
        raise NotImplementedError
  #except KeyboardInterrupt:
  #  print('SubprocVecEnv worker: got KeyboardInterrupt')
  finally:
    env.close()

class SubprocVecEnv(VecEnv):
  """
  VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
  Recommended to use when num_envs > 1 and step() can be a bottleneck.
  """
  def __init__(self, env_fns):
    """
    Arguments:

    env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
    """
    self.waiting = False
    self.closed = False
    self.num_envs = len(env_fns)
    self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
    self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
           for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
    for p in self.ps:
      p.daemon = True  # if the main process crashes, we should not cause things to hang
      p.start()
    for remote in self.work_remotes:
      remote.close()

    self.set_task_t(0.5)
    self.viewer = None
    #self.specs = [f().spec for f in env_fns]
    #VecEnv.__init__(self, len(env_fns), observation_space, action_space)
    VecEnv.__init__(self, len(env_fns))


    self.remotes[0].send(('get_state_size', None))
    self.state_size = self.remotes[0].recv()
    self.annealing_sample = 0
    self.mode = 0
    self.prob = None

  def step_async(self, actions):
    self._assert_not_closed()
    for remote, action in zip(self.remotes, actions):
      remote.send(('step', action))
    self.waiting = True

  def step_wait(self):
    self._assert_not_closed()
    results = [remote.recv() for remote in self.remotes]
    self.waiting = False
    obs, rews, dones, infos = zip(*results)
    return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

  def reset(self):
    self._assert_not_closed()
    for remote in self.remotes:
      remote.send(('reset', None))
    return _flatten_obs([remote.recv() for remote in self.remotes])

  def reset_from_data(self, states):
    self._assert_not_closed()
    for remote, state in zip(self.remotes, states):
      remote.send(('reset_from_data', state))
    return _flatten_obs([remote.recv() for remote in self.remotes])

  def close_extras(self):
    self.closed = True
    if self.waiting:
      for remote in self.remotes:
        remote.recv()
    for remote in self.remotes:
      remote.send(('close', None))
    for p in self.ps:
      p.join()

  def get_reset_data(self, align_origin=True):
    self._assert_not_closed()
    for remote in self.remotes:
      remote.send(('get_reset_data', align_origin))
    return _flatten_obs([remote.recv() for remote in self.remotes])

  def _assert_not_closed(self):
    assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

  def set_task_t(self, t):
    self.task_t = t
    self._assert_not_closed()
    for remote in self.remotes:
      remote.send(('set_task_t', t))
    for remote in self.remotes:
      remote.recv()

  def set_mode(self, mode):
    self.mode = mode
    self._assert_not_closed()
    for remote in self.remotes:
      remote.send(("set_mode", mode))
    for remote in self.remotes:
      remote.recv()

  def set_sample_data(self, prob, select_set, noise):
    self._assert_not_closed()
    for remote in self.remotes:
      remote.send(('set_sample_data', (prob, select_set, noise)))
    for remote in self.remotes:
      remote.recv()

  def calc_sub_rewards(self):
    self._assert_not_closed()
    for remote in self.remotes:
      remote.send(('calc_sub_rewards', None))
    self.waiting = True
    results = [remote.recv() for remote in self.remotes]
    self.waiting = False
    return _flatten_obs(results)

def _flatten_obs(obs):
  assert isinstance(obs, list) or isinstance(obs, tuple)
  assert len(obs) > 0

  if isinstance(obs[0], dict):
    import collections
    assert isinstance(obs, collections.OrderedDict)
    keys = obs[0].keys()
    return {k: np.stack([o[k] for o in obs]) for k in keys}
  else:
    return np.stack(obs)

