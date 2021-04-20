from abc import ABC, abstractmethod

ACT_STEPTIME  = 1./30.
SUBSTEPS = 20
VIS_STEPTIME = 1.0/150
SIM_STEPTIME  = ACT_STEPTIME / SUBSTEPS

class SimEngine(ABC):
  """ simulation engine for character controll

  """

  @abstractmethod
  def __init__(self, skeleton, self_collision, sim_timestep, model):
    pass

  @abstractmethod
  def set_steptime(self, steptime, substeps):
    pass

  @abstractmethod
  def set_allowed_fall_contact(self, body_ids):
    pass

  @abstractmethod
  def set_kp(self, kp):
    pass

  @abstractmethod
  def set_kd(self, kd):
    pass

  @abstractmethod
  def set_force_limit(self, force_lim):
    pass

  @abstractmethod
  def get_pose(self):
    pass

  @abstractmethod
  def set_pose(self, pose, vel):
    pass

  @abstractmethod
  def set_spd_target(self, pose):
    pass

  @abstractmethod
  def step_sim(self):
    pass

  @abstractmethod
  def check_fall(self):
    """ Check if fobiddent parts make contact with ground
        True if there is contact, False if there is not
    """
    pass

  @abstractmethod
  def check_self_contact(self):
    """ Check if character has self contacts
        True if there is contact, False if there is not
    """
    pass

  @abstractmethod
  def close(self):
    pass
