from env import SpacetimeBoundsEnv
import numpy as np
from Kinematic import KinematicCore

class ImitationEnv(SpacetimeBoundsEnv):
  def __init__(self, **kwargs):
    """
        Initialize FDM0E environment
    """
    super().__init__(**kwargs)

  def calc_reward(self):
    t = self.curr_phase * self._mocap._cycletime
    count, pose, vel = self._mocap.slerp(t)
    self._curr_kin_pose = pose
    self._curr_kin_vel = vel
    if self._use_global_root_ori:
      local_sim_pose = self._curr_sim_pose.copy()
      local_sim_pose[0] = 0
      local_sim_pose[2] = 0
      local_sim_vel = self._curr_sim_vel
      local_kin_pose = self._curr_kin_pose.copy()
      local_kin_pose[0] = 0
      local_kin_pose[2] = 0
      local_kin_vel = self._curr_kin_vel
      reward = self._skeleton.get_reward(local_sim_pose, local_sim_vel, local_kin_pose, local_kin_vel)
    else:
      local_sim_pose, local_sim_vel = self._skeleton.toLocalFrame(self._curr_sim_pose, self._curr_sim_vel)
      local_kin_pose, local_kin_vel = self._skeleton.toLocalFrame(self._curr_kin_pose, self._curr_kin_vel)
      reward = self._skeleton.get_reward(local_sim_pose, local_sim_vel, local_kin_pose, local_kin_vel)
    return reward

if __name__=="__main__":
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # base
  parser.add_argument("--task", type=str, default='run', help="task to perform")
  parser.add_argument("--engine", type=str, default='pybullet', help="simulation engine, select from pybullet and pyPhysX")
  parser.add_argument("--contact", type=str, default='walk', help="contact type, selected from walk, cartwheel, crawl, roll")
  parser.add_argument("--noself", dest='sc', action='store_const', const=False, default=True, help="no self-collision")
  parser.add_argument("--vis", dest='vis', action='store_const', const=True, default=False, help="visualize environment")
  #FDM0
  parser.add_argument("--env", type=str, help="select from FDM0, FDM0g, FDM0z, FDM0zg")
  parser.add_argument("--ckpt", type=str, help="checkpoint")
  args = parser.parse_args()

  if "z" in args.env:
    heading_vec = [0, 0, 1]
  else:
    heading_vec = [1, 0, 0]

  kwargs = {
          # base
          "task": args.task,
          "seed": 0,
          "engine": args.engine,
          "contact": args.contact,
          "self_collision": args.sc,
          "enable_draw": args.vis,
          # FDM0
          "use_global_root_ori": True,
          "heading_vec": heading_vec,
          "use_state_lim": True,
          "bound": "./data/bounds/default_new_bound.txt",
          "rel_root_pos": not "g" in args.env,
          "rel_root_ori": False,
          "rel_endeffector": True,
          }

  test_env = FDM0REnv(**kwargs)

  import torch
  from model import load_FDM
  model = load_FDM(args.ckpt)

  data = torch.load(args.ckpt)
  if "select_set" in data.keys():
    select_set = data["select_set"]
    from env import test_model_select
    test_model_select(test_env, model, select_set)
  else:
    from env import test_model
    test_model(test_env, model)

