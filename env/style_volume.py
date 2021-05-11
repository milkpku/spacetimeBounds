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
        Initialize volume stylize environment
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
