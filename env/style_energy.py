from env import SpacetimeBoundsEnv
import numpy as np
from Kinematic import KinematicCore

class StyleEnergyEnv(SpacetimeBoundsEnv):
  def __init__(self, begin_energy=20, end_energy=100, **kwargs):
    """
        Initialize FDM0E environment
    """
    super().__init__(**kwargs)

    # begin_energy can be larger than end_energy, then the environment will
    # discourage high energy
    self._low = begin_energy
    self._high = end_energy
    self._gap = end_energy - begin_energy

  def calc_reward(self):
    char = self._skeleton._kin_core.getCharacter()
    kin_energy = KinematicCore.kinematicEnergy(char, True)

    rwd = (kin_energy - self._low) / self._gap
    rwd = np.clip(rwd, 0, 1)
    return rwd

if __name__=="__main__":
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  #base
  parser.add_argument("--task", type=str, default='run', help="task to perform")
  parser.add_argument("--engine", type=str, default='pybullet', help="simulation engine, select from pybullet and pyPhysX")
  parser.add_argument("--contact", type=str, default='walk', help="contact type, selected from walk, cartwheel, crawl, roll")
  parser.add_argument("--noself", dest='sc', action='store_const', const=False, default=True, help="no self-collision")
  parser.add_argument("--vis", dest='vis', action='store_const', const=True, default=False, help="visualize environment")
  #FDM0
  parser.add_argument("--env", type=str, help="select from FDM0, FDM0g, FDM0z, FDM0zg")
  #FDM0_energy
  parser.add_argument("--range", type=float, nargs=2, default=[40, 60], help="range of linear reward")
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
          # FDM0_energy
          "begin_energy": args.range[0],
          "end_energy": args.range[1],
          }

  test_env = FDM0EnergyEnv(**kwargs)

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

