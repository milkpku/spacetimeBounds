from env import SpacetimeBoundsEnv
import numpy as np
from Kinematic import KinematicCore

# 1 root(id 0) NONE
# 2 chest(id 1) SPHERICAL
# 3 neck(id 2) SPHERICAL
# 4 right_hip(id 3) SPHERICAL
# 5 right_knee(id 4) REVOLUTE
# 6 right_ankle(id 5) SPHERICAL
# 7 right_shoulder(id 6) SPHERICAL
# 8 right_elbow(id 7) REVOLUTE
# 9 right_wrist(id 8) FIXED
# 10 left_hip(id 9) SPHERICAL
# 11 left_knee(id 10) REVOLUTE
# 12 left_ankle(id 11) SPHERICAL
# 13 left_shoulder(id 12) SPHERICAL
# 14 left_elbow(id 13) REVOLUTE
# 15 left_wrist(id 14) FIXED

convex_nodes = {
  "default": list(range(15)),
  "upper": [0, 1, 2, 6, 7, 8, 12, 13, 14],
  "lower": [0, 3, 4, 5, 9, 10, 11],
  "endeffector": [0, 5, 8, 11, 14],
}

class StyleVolumeEnv(SpacetimeBoundsEnv):
  def __init__(self, nodes="default", scale=0.12, enlarge=True, **kwargs):
    """
        Initialize FDM0E environment
    """
    super().__init__(**kwargs)

    # joints' weight for penalty
    self._nodes = convex_nodes[nodes]
    self._scale = scale
    self._enlarge = enlarge

  def calc_volume(self):
    try:
      char = self._skeleton._kin_core.getCharacter()
      volume = KinematicCore.jointConvexHullVolume(char, self._nodes)
    except:
      from IPython import embed; embed()
    return volume

  def calc_reward(self):
    vol = self.calc_volume()

    r_diff = np.exp(-vol/self._scale)

    if self._enlarge:
      rwd = (1-r_diff) # encourage volume
    else:
      rwd = r_diff  # discourage volume

    return rwd


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
  #FDM0_volume
  parser.add_argument("--scheme", type=str, help="joint node scheme, select from %s" % str(convex_nodes.keys()))
  parser.add_argument("--scale", type=float, default=0.12, help="scale of volume energy")
  parser.add_argument("--enlarge", action='store_const', const=True, default=False)
  parser.add_argument("--ckpt", type=str, default="torch_policy/jump.tar", help="checkpoint")
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
          # FDM0_joints
          "joint_nodes": convex_nodes[args.scheme],
          "scale": args.scale,
          "enlarge": args.enlarge,
          }

  test_env = FDM0VolumeEnv(**kwargs)

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

