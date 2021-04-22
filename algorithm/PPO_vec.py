from .runner import RunnerGAE
from .data_tools import PPO_Dataset

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
from tensorboardX import SummaryWriter

### curriculum scheduling from DeepMimic ###
CURRIC_SAMPLES  = 3.2e7
MAX_T = 20
MIN_T = 0.5
############################

def calc_grad_norm(model):
  total_norm = 0
  for p in model.parameters():
    if p.requires_grad == True:
      param_norm = p.grad.data.norm(2)
      total_norm += param_norm.item() ** 2
  total_norm = total_norm ** (1./2)
  return total_norm

# from DeepMimic
def map_samples_to_task_t(samples):
  if samples > CURRIC_SAMPLES:
    return MAX_T
  t = MIN_T + (MAX_T - MIN_T) * np.power(samples/CURRIC_SAMPLES, 4)
  return t

def record_channels(tag, channel, glob_step, writer):
  print(channel)
  bins = np.arange(channel.min() - 0.5, channel.max() + 1.5)
  counts, limits = np.histogram(channel, bins)
  buck_lim = bins[1:] - 0.5
  writer.add_histogram_raw(tag, min=channel.min(), max=channel.max(), num=channel.size, sum=0,
    sum_squares=0, bucket_limits=buck_lim, bucket_counts=counts, global_step=glob_step)

def PPO_vec(actor, critic, s_norm, vec_env, exp_id,
        save_dir="./",
        ### training settings
        iter_num=70000,
        sample_size=4096,
        epoch_size=1,
        batch_size=256,
        checkpoint_batch=100,
        test_batch=10,

        ### PPO parameter
        gamma=0.95,
        lam=0.95,
        clip_threshold=0.2,

        ### learning rate
        actor_lr=2.5e-6,
        actor_wdecay=5e-4,
        actor_momentum=0.9,
        critic_lr=1e-2,
        critic_wdecay=5e-4,
        critic_momentum=0.9,
        max_grad_norm=100,          # prevent explosion

        ### importance sampling
        use_importance_sampling=True,
        num_segments = 10,
        ckpt_sample_prob=None,

        ### initial state evolution
        use_state_evolution=True,
        num_selected_elite=200,
        ckpt_select_set=None,

        ### other settings
        use_end_state_estimation=True,
        use_deepmimic_scheduling=False,
        use_gpu_model=False):
  """
  PPO algorithm of reinforcement learning
  Inputs:
    actor       policy network, with following methods:
                  m = actor.distributions(ob)
                  ac = actor.act_deterministic()

    critic

    s_norm

    vec_env     vectorized environment, with following methods:
                  ob = env.reset()
                  ob, rew, done, _ = env.step(ac)
    exp_id      string, experiment id, checkpoints and training monitor data
                will be saved to ./${exp_id}/
  """

  actor_optim = optim.SGD(actor.parameters(), actor_lr, momentum=actor_momentum, weight_decay=actor_wdecay)
  critic_optim = optim.SGD(critic.parameters(), critic_lr, momentum=critic_momentum, weight_decay=critic_wdecay)
  exp_rate = 1.0

  # set up environment and data generator
  runner_args = {
      "sample_size": sample_size,
      "gamma": gamma,
      "lam": lam,
      "exp_rate": exp_rate,
      "use_importance_sampling": use_importance_sampling,
      "num_segments": num_segments,
      "ckpt_sample_prob": ckpt_sample_prob,
      "use_state_evolution": use_state_evolution,
      "num_selected_elite": num_selected_elite,
      "ckpt_select_set": ckpt_select_set,
      "use_gpu_model": use_gpu_model,
  }
  runner = RunnerGAE(vec_env, s_norm, actor, critic, **runner_args)

  # use estimated value from critic for end state in GAE calculation
  # when the episode/trajectory is interrupted rather than terminated
  if use_end_state_estimation:
    runner.set_estimate(1)
  else:
    runner.set_estimate(2)

  if use_gpu_model:
    T = lambda x: torch.cuda.FloatTensor(x)
  else:
    T = lambda x: torch.FloatTensor(x)

  writer = SummaryWriter("%s/%s" % (save_dir, exp_id))
  total_sample = 0
  train_sample = 0
  anneal_sample = 0 #vec_env.annealing_sample
  for it in range(iter_num + 1):
    # sample data with gae estimated adv and vtarg
    anneal_rate = np.clip(anneal_sample / CURRIC_SAMPLES, 0, 1)
    if use_deepmimic_scheduling:
      t = map_samples_to_task_t(anneal_sample)
      vec_env.set_task_t(t)
    else:
      vec_env.set_task_t(MAX_T)

    # can update exp_rate by runner.set_exp_rate()
    data = runner.run()
    dataset = PPO_Dataset(data)

    atarg = dataset.advantage
    atarg = (atarg - atarg.mean()) / (atarg.std() + 1e-5) # trick: standardized advantage function

    adv_clip_rate = np.mean(np.abs(atarg) > 4)
    adv_max = np.max(atarg)
    adv_min = np.min(atarg)
    val_min = 0;
    val_max = 1 / (1-gamma);
    vtarg = dataset.vtarget
    vtarg_clip_rate = np.mean(np.logical_or(vtarg < val_min, vtarg > val_max))
    vtd_max = np.max(vtarg)
    vtd_min = np.min(vtarg)

    atarg = np.clip(atarg, -4, 4)
    vtarg = np.clip(vtarg, val_min, val_max)

    dataset.advantage = atarg
    dataset.vtarget = vtarg

    # logging interested variables
    N = np.clip(data["news"].sum(), a_min=1, a_max=None) # prevent divding 0
    avg_rwd = data["rwds"].sum()/N
    avg_step = data["samples"]/N
    rwd_per_step = avg_rwd / avg_step
    # examine end_rwd, fail and phases
    rwd_end = data["end_rwds"]
    N_end = np.clip(len(rwd_end), a_min=1, a_max=None) # prevent dividing 0
    avg_rwd_end = sum(rwd_end) / N_end
    fail_rate = sum(data["fails"])/N_end
    bound_rate = sum(data["bounds"])/N_end
    start_phases = data["start_phases"]
    fail_phases = data["fail_phases"]
    bound_channels = data["bound_channels"]
    total_sample += data["samples"]
    train_sample += data["explores"]
    anneal_sample += data["explores"]
    writer.add_scalar("sample/avg_rwd",     avg_rwd,        it)
    writer.add_scalar("sample/avg_step",    avg_step,       it)
    writer.add_scalar("sample/rwd_per_step",rwd_per_step,   it)
    writer.add_scalar("sample/end_rwd",     avg_rwd_end,    it)
    writer.add_scalar("anneal/anneal_rate", anneal_rate,    it)
    writer.add_scalar("anneal/exp_rate",    exp_rate, it)
    writer.add_scalar("anneal/total_samples", total_sample, it)
    writer.add_scalar("anneal/train_samples", train_sample, it)
    writer.add_scalar("anneal/anealing_samples", anneal_sample, it)
    writer.add_histogram("sample/start_phase",start_phases, it)
    writer.add_histogram("sample/fail_phase",  fail_phases, it)
    # special handling of bound channels
    record_channels("sample/bound_channels", bound_channels, it, writer)

    if (it % test_batch == 0):
      test_step, test_rwd = runner.test()
      writer.add_scalar("sample/test_step", test_step,      it)
      writer.add_scalar("sample/test_rwd",  test_rwd,       it)

    print("\n===== iter %d ====="% it)
    print("avg_rwd       = %f" % avg_rwd)
    print("avg_step      = %f" % avg_step)
    print("rwd_per_step  = %f" % rwd_per_step)
    print("test_rwd      = %f" % test_rwd)
    print("test_step     = %f" % test_step)
    print("fail_rate     = %f" % fail_rate)
    print("bound_rate    = %f" % bound_rate)
    print("end_rwd       = %f" % avg_rwd_end)
    print("anneal_rate   = %f" % anneal_rate)
    print("exp_rate      = %f" % exp_rate)
    print("total_samples = %d" % total_sample)
    print("adv_clip_rate = %f, (%f, %f)" % (adv_clip_rate, adv_min, adv_max))
    print("vtd_clip_rate = %f, (%f, %f)" % (vtarg_clip_rate, vtd_min, vtd_max))

    writer.add_scalar("debug/fail_rate",    fail_rate,          it)
    writer.add_scalar("debug/bound_rate",   bound_rate,         it)
    writer.add_scalar("debug/adv_clip",     adv_clip_rate,      it)
    writer.add_scalar("debug/vtarg_clip",   vtarg_clip_rate,    it)

    # start training
    pol_loss_avg    = 0
    pol_surr_avg    = 0
    pol_sym_avg     = 0
    pol_abound_avg  = 0
    vf_loss_avg     = 0
    clip_rate_avg   = 0

    actor_grad_avg  = 0
    critic_grad_avg = 0

    for epoch in range(epoch_size):
      #print("iter %d, epoch %d" % (it, epoch))

      for bit, batch in enumerate(dataset.batch_sample(batch_size)):
        # prepare batch data
        ob, ac, atarg, tdlamret, log_p_old = batch
        ob = T(ob)
        ac = T(ac)
        atarg = T(atarg)
        tdlamret = T(tdlamret).view(-1, 1)
        log_p_old = T(log_p_old)

        # clean optimizer cache
        actor_optim.zero_grad()
        critic_optim.zero_grad()

        # calculate new log_pact
        ob_normed = s_norm(ob)
        m = actor.act_distribution(ob_normed)
        vpred = critic(ob_normed)
        log_pact = m.log_prob(ac)
        if log_pact.dim() == 2:
          log_pact = log_pact.sum(dim=1)

        # PPO object, clip advantage object
        ratio = torch.exp(log_pact - log_p_old)
        surr1 = ratio * atarg
        surr2 = torch.clamp(ratio, 1.0 - clip_threshold, 1.0 + clip_threshold) * atarg
        pol_surr = -torch.mean(torch.min(surr1, surr2))

        if (surr2.mean() > 12):
          assert(False)
          from IPython import embed; embed()

        # action bound penalty, normalized
        violation_min = torch.clamp(ac - actor.a_min, max=0) / actor.a_std;
        violation_max = torch.clamp(ac - actor.a_max, min=0) / actor.a_std;
        violation = torch.sum(torch.pow(violation_min, 2) + torch.pow(violation_max, 2), dim=1)
        pol_abound = 0.5 * torch.mean(violation)

        # trick: add penalty for violation of bound
        pol_loss = pol_surr + pol_abound
        pol_surr_avg += pol_surr.item()
        pol_abound_avg += pol_abound.item()
        pol_loss_avg += pol_loss.item()


        # critic vpred loss
        vf_criteria = nn.MSELoss()
        vf_loss = vf_criteria(vpred, tdlamret) / (critic.v_std**2) # trick: normalize v loss

        vf_loss_avg += vf_loss.item()

        if (not np.isfinite(pol_loss.item())):
          print("pol_loss infinite")
          assert(False)
          from IPython import embed; embed()

        if (not np.isfinite(vf_loss.item())):
          print("vf_loss infinite")
          assert(False)
          from IPython import embed; embed()

        pol_loss.backward(retain_graph=True)
        vf_loss.backward(retain_graph=True)

        nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)

        actor_grad_avg += calc_grad_norm(actor)
        critic_grad_avg+= calc_grad_norm(critic)

        # for debug use
        clip_rate = (torch.abs(ratio - 1.0) > clip_threshold).float()
        clip_rate = torch.mean(clip_rate)
        clip_rate_avg += clip_rate.item()

        actor_optim.step()
        critic_optim.step()


    batch_num = epoch_size * (sample_size // batch_size)
    pol_loss_avg    /= batch_num
    pol_surr_avg    /= batch_num
    pol_sym_avg     /= batch_num
    pol_abound_avg  /= batch_num
    vf_loss_avg     /= batch_num
    clip_rate_avg   /= batch_num

    writer.add_scalar("debug/clip_rate", clip_rate_avg,     it)
    writer.add_scalar("train/pol_loss",  pol_loss_avg,      it)
    writer.add_scalar("train/pol_surr",  pol_surr_avg,      it)
    writer.add_scalar("train/ab_loss",   pol_abound_avg,    it)
    writer.add_scalar("train/vf_loss",   vf_loss_avg,       it)

    print("pol_loss      = %f" % pol_loss_avg)
    print("pol_surr      = %f" % pol_surr_avg)
    print("ab_loss       = %f" % pol_abound_avg)
    print("sym_loss      = %f" % pol_sym_avg)
    print("vf_loss       = %f" % vf_loss_avg)
    print("clip_rate     = %f" % clip_rate_avg)

    actor_grad_avg /= batch_num
    critic_grad_avg/= batch_num
    writer.add_scalar("train/actor_grad", actor_grad_avg,   it)
    writer.add_scalar("train/critic_grad", critic_grad_avg, it)
    print("actor_grad    = %f" % actor_grad_avg)
    print("critic_grad   = %f" % critic_grad_avg)

    # save checkpoint
    if (it % checkpoint_batch == 0):
      print("save check point ...")
      actor.cpu()
      critic.cpu()
      s_norm.cpu()
      data = {"actor": actor.state_dict(),
              "critic": critic.state_dict(),
              "s_norm": s_norm.state_dict(),
              "samples": anneal_sample}
      if use_gpu_model:
        actor.cuda()
        critic.cuda()
        s_norm.cuda()

      data = runner.save_info(data)

      torch.save(data, "%s/%s/checkpoint_%d.tar" % (save_dir, exp_id, it))

if __name__=="__main__":

  import time
  from env import DeepMimicSimEnv, SubprocVecEnv
  import gym.spaces
  from model import Normalizer, Actor, Critic

  import argparse
  import multiprocessing
  num_cpu = multiprocessing.cpu_count()
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", type=str, default="run", help="task to be performed")
  parser.add_argument("--id", type=str, default="PPO_vec_test", help="experiment id")
  parser.add_argument("--workers", type=int, default=num_cpu, help="number of workers to sample data")
  parser.add_argument("--ckpt", type=str, default=None, help="load trained data as start")
  parser.add_argument("--samples", type=int, default=0, help="annealing sample numbers")
  args = parser.parse_args()

  env = DeepMimicSimEnv(args.task)
  s_dim = env.get_state_size()
  a_dim = env.get_action_size()
  a_bound = env.action_space
  env.close()

  s_norm = Normalizer(s_dim)
  s_norm.mean.data[0] = 0.5
  actor = Actor(s_dim, a_dim, a_bound)
  critic= Critic(s_dim, 0, 1/(1-gamma))

  def make_env(seed):
    return lambda: DeepMimicEnv(env_args, seed)
  vec_env = SubprocVecEnv([make_env(i + time.process_time_ns() % 1000) for i in range(args.workers)])

  if (args.ckpt is not None):
    try:
      checkpoint = torch.load(args.ckpt)
      actor.load_state_dict(checkpoint["actor"])
      critic.load_state_dict(checkpoint["critic"])
      s_norm.load_state_dict(checkpoint["s_norm"])

      if "samples" in checkpoint.keys():
        vec_env.set_annealing_samples(checkpoint["samples"])
      else:
        vec_env.set_annealing_samples(args.samples)

      print("load from %s" % args.ckpt)

    except:
      print("fail to load from %s" % args.ckpt)
      assert(False)

  stamp = time.strftime("%Y-%b-%d-%H%M%S", time.localtime())
  exp_id = "%s-%s" % (args.id, stamp)

  PPO_vec(actor, critic, s_norm, vec_env, exp_id)
