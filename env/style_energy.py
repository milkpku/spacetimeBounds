from env import SpacetimeBoundsEnv
import numpy as np
from Kinematic import KinematicCore

class StyleEnergyEnv(SpacetimeBoundsEnv):
  def __init__(self, begin_energy=20, end_energy=100, **kwargs):
    """
        Initialize energy stylize environment
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
