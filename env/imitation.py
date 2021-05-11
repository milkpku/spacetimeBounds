from env import SpacetimeBoundsEnv
import numpy as np
from Kinematic import KinematicCore

class ImitationEnv(SpacetimeBoundsEnv):
  def __init__(self, **kwargs):
    """
        Initialize imitation environment
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
