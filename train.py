import numpy as np
import torch
use_gpu = torch.cuda.is_available()

def init_model(env, model_args, ckpt=None):
  # get input/output size and range
  s_dim = env.get_state_size()
  a_dim = env.get_action_size()
  a_min = env.a_min
  a_max = env.a_max
  a_noise = model_args["noise"] * np.ones(a_dim)

  # get reference memory for FFC
  ref_mem = env._mocap.get_ref_mem()
  if not model_args["with_ffc"]:
    ref_mem.fill(0)
  ref_mem = ref_mem[:, 1:]  # no phase velocity

  # automatically use gpu
  if use_gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

  from model import Normalizer, Actor, Critic
  GAMMA = file_args["train_args"]["gamma"]
  non_norm = [0] #FMD0
  s_norm = Normalizer(s_dim, non_norm)
  actor = Actor(s_dim, a_dim, a_min, a_max, a_noise, ref_mem.shape[0])
  critic= Critic(s_dim, 0, 1/(1-GAMMA))

  actor.set_reference(ref_mem)
  actor.ref_mem.requires_grad = False

  if (args.ckpt is not None):
    try:
      checkpoint = torch.load(args.ckpt)
      actor.load_state_dict(checkpoint["actor"])
      critic.load_state_dict(checkpoint["critic"])
      s_norm.load_state_dict(checkpoint["s_norm"])
      print("load from %s" % args.ckpt)

    except:
      print("fail to load from %s" % args.ckpt)
      assert(False)

  return s_norm, actor, critic


if __name__=="__main__":
  import argparse
  import multiprocessing
  num_cpu = multiprocessing.cpu_count()
  parser = argparse.ArgumentParser()
  # training args
  parser.add_argument("arg_file", type=str, help="arg file for training")
  # training
  parser.add_argument("--id", type=str, default="spacetime_test", help="experiment id")
  parser.add_argument("--iters", type=int, default=None, help="number of iterations, use arg_file if not set")
  parser.add_argument("--workers", type=int, default=num_cpu, help="number of workers to sample data")
  # start from previous ckpt
  parser.add_argument("--ckpt", type=str, default=None, help="load trained data as start")
  args = parser.parse_args()

  import json
  file_args = json.load(open(args.arg_file))

  # environment args
  env_name = file_args["env_name"]
  env_args = file_args["env_args"]
  model_args = file_args["model_args"]
  train_args = file_args["train_args"]

  from env import make_env, make_vec_env
  vec_env = make_vec_env(env_name, env_args, args.workers)

  test_env = make_env(env_name, env_args)
  s_norm, actor, critic = init_model(test_env, model_args, args.ckpt)

  import time
  stamp = time.strftime("%Y-%b-%d-%H%M%S", time.localtime())
  exp_id = "%s-%s" % (args.id, stamp)

  train_args["exp_id"] = exp_id
  train_args["vec_env"] = vec_env
  if args.iters:
    train_args["iter_num"] = args.iters
  train_args["actor"] = actor
  train_args["critic"] = critic
  train_args["s_norm"] = s_norm
  train_args["use_gpu_model"] = use_gpu

  if (args.ckpt is not None):
    if "importance" in checkpoint.keys():
      train_args["ckpt_sample_prob"] = checkpoint["importance"]
    if "select_set" in checkpoint.keys():
      train_args["ckpt_select_set"] = checkpoint

  from algorithm import PPO_vec
  PPO_vec(**train_args)
