from env import BaseEnv
import numpy as np
from utils.motion_bound import build_motion_bound
import time

class SpacetimeBoundsEnv(BaseEnv):
  def __init__(self,
          use_global_root_ori=True,
          heading_vec=[1, 0, 0],
          use_state_lim=True,
          bound="data/bounds/default_new_bound.txt",
          rel_root_pos=True,
          rel_root_ori=True,
          rel_endeffector=True,
          **kwargs):
    """ Initialize FDM0 environment
    """
    super().__init__(**kwargs)

    # representation
    self._use_global_root_ori = use_global_root_ori

    # motion bound
    self._use_state_lim = use_state_lim
    self._bound = build_motion_bound(bound)
    self._rel_root_pos = rel_root_pos
    self._rel_root_ori = rel_root_ori
    self._rel_endeffector = rel_endeffector

    # heading vector
    self._skeleton.set_heading_vec(heading_vec)

  def get_state_size(self):
    single_state = self._skeleton.build_state();
    # 1 dim for phase, 1 sim state,
    state_size = 1 + single_state.size
    return state_size

  def get_action_size(self):
    return  self._skeleton.expdof

  def build_action_bound(self):
    a_min = np.array(self._skeleton.build_a_min())
    a_max = np.array(self._skeleton.build_a_max())

    return a_min, a_max

  def get_reset_data(self, align_origin=True):
    """ data used to reset env
    """
    phase = self.curr_phase % 1
    pose, vel = self._engine.get_pose()
    if align_origin:
      pose[0] = 0
      pose[2] = 0

    state = np.concatenate([[phase], pose, vel])
    return state

  def reset(self, phase=None, state=None):
    if state is not None:
      offset = 0
      start_phase = state[offset]
      offset += 1
      pose_dim = self._mocap.get_pose_dim()
      pose = state[offset : offset+pose_dim]
      offset += pose_dim
      vel_dim = self._mocap.get_vel_dim()
      vel = state[offset : offset+vel_dim]
    else:
      start_phase = self._rand.rand() if phase is None else phase
      count, pose, vel = self._mocap.slerp(start_phase * self._mocap._cycletime)

    # align with mocap data, to facilitate abs root compare
    cnt, kin_pose, kin_vel = self._mocap.slerp(start_phase*self._mocap._cycletime)
    pose[0] = kin_pose[0]
    pose[2] = kin_pose[2]

    # resolve ground contact
    # atlas model is not clean, so only humanoid3d model is supported
    if self._model=="humanoid3d":
      low_y = self._skeleton.lowest_height(pose.tolist())
      if low_y < 0:
        pose[1] += 0.001 - low_y

    self.start_phase = start_phase
    self.curr_phase = start_phase
    self.phase_vel = 1.0 / self._mocap._cycletime

    self._engine.set_pose(pose, vel)

    return self.record_state()

  def set_action(self, action):
    self.target_pose = self._skeleton.exp_to_targ_pose(action, True)
    self._engine.set_spd_target(self.target_pose)

  def record_info(self):
    info = {"terminate": self.check_terminate(),
            "valid_episode": self.check_valid_episode(),
            "start_phase": self.start_phase,
            "wrap_end": self.check_wrap_end(),
            "bound": self.check_state_diff(),
            "pose": self.get_reset_data(),
            "trust_rwd": True,
            }
    if info["bound"]:
      # record active channel of motion bound
      info["bound_active"] = self.check_state_diff_active_group()
    return info

  def post_update(self):
    self.curr_phase += self.phase_vel * self._sim_step

  def record_state(self):
    """
      state is made up with 2 items:
       - phase
       - sim_state
    """
    # build state using current reference root, and without global info
    pose, vel = self._engine.get_pose()
    self._curr_sim_pose = pose
    self._curr_sim_vel = vel
    self._skeleton.set_pose(pose)
    self._skeleton.set_vel(vel)
    ori_pos = pose[:3]
    ori_rot = pose[3:7]
    sim_state = self._skeleton.build_state(ori_pos, ori_rot, self._use_global_root_ori)

    phase = self.curr_phase % 1.0

    state = np.concatenate(([phase], sim_state))
    return state

  def calc_reward(self):
    return 1.0

  def check_terminate(self):
    if self._mode == 0:  # training mode
      return self.check_state_diff()
    elif self._mode == 1: # test mode
      return self._contact_ground
    else:
      assert(False and "not supported mode")

  def check_state_diff(self):
    if not self._use_state_lim or self._mode==1:
      return False

    curr_sim_pose = self._curr_sim_pose

    t = self.curr_phase * self._mocap._cycletime
    count, pose, vel = self._mocap.slerp(t)
    pose[:3] += count * self._mocap._cyc_offset
    curr_kin_pose = pose

    bound = self._bound.slerp(self.curr_phase)
    self._skeleton.set_state_diff_lim(bound)

    return self._skeleton.check_state_diff(curr_sim_pose, curr_kin_pose,
            self._rel_root_pos, self._rel_root_ori, self._rel_endeffector)

  def check_state_diff_active_group(self):
    """ return list of active joints which break motion bound
    """
    if not self._use_state_lim or self._mode==1:
      return False

    curr_sim_pose = self._curr_sim_pose.copy()

    t = self.curr_phase * self._mocap._cycletime
    count, pose, vel = self._mocap.slerp(t)
    pose[:3] += count * self._mocap._cyc_offset
    curr_kin_pose = pose

    bound = self._bound.slerp(self.curr_phase)
    self._skeleton.set_state_diff_lim(bound)

    status_vec = self._skeleton.check_state_diff_vec(curr_sim_pose, curr_kin_pose, self._rel_root_pos, self._rel_root_ori, self._rel_endeffector)

    active_group = list(filter(lambda x: status_vec[x], range(len(status_vec))))
    return active_group

if __name__=="__main__":
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # base
  parser.add_argument("--task", type=str, default='run', help="task to perform")
  parser.add_argument("--engine", type=str, default='pybullet', help="simulation engine, select from pybullet and pyPhysX")
  parser.add_argument("--contact", type=str, default='walk', help="contact type, selected from walk, cartwheel, crawl, roll")
  parser.add_argument("--bound", type=str, default='data/bounds/default_new_bound.txt', help="bound")
  parser.add_argument("--noself", dest='sc', action='store_const', const=False, default=True, help="no self-collision")
  parser.add_argument("--vis", dest='vis', action='store_const', const=True, default=False, help="visualize environment")
  #FDM0
  parser.add_argument("--env", type=str, help="select from FDM0, FDM0g, FDM0z, FDM0zg")
  parser.add_argument("--ckpt", type=str, default="torch_policy/jump.tar", help="checkpoint")
  #record
  parser.add_argument("--record", dest='record', action='store_const', const=True, default=False, help="record motion data")
  parser.add_argument("--random", dest='random', action='store_const', const=True, default=False, help="random initialize")
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
          "bound": args.bound,
          "rel_root_pos": not "g" in args.env,
          "rel_root_ori": False,
          "rel_endeffector": True,
          }

  test_env = FastDeepMimicEnv(**kwargs)

  import torch
  from model import load_FDM
  model = load_FDM(args.ckpt)

  data = torch.load(args.ckpt)
  from env import test_model
  if "select_set" in data.keys():
    select_set = data["select_set"]
    test_model(test_env, model, select_set, args.record, args.random)
  else:
    test_model(test_env, model, None, args.record, args.random)

