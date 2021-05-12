from env import SpacetimeBoundsEnv
import numpy as np
from Kinematic import KinematicCore

class SpacetimeRegEnv(SpacetimeBoundsEnv):
  def __init__(self,
               e_scale=16, ca_scale=16, a_scale=40, aa_scale=400,
               abs_acc=False,
               **kwargs):
    """
        Initialize SpacetimeRegEnv environment
    """
    super().__init__(**kwargs)

    # regularization
    self._e_scale = e_scale
    self._ca_scale = ca_scale
    self._a_scale = a_scale
    self._aa_scale = aa_scale
    self._abs_acc = abs_acc

    self.init_params()

    self._prev_cv = None
    self._curr_cv = None

    self._prev_vs = None
    self._curr_vs = None

    self._prev_omgs = None
    self._curr_omgs = None

  def init_params(self):
    mocap = self._mocap

    get_bound = lambda p: min(p['max'], p['upper_fence'])

    e_param = mocap.get_energy_statistics()
    self._e_penalty = get_bound(e_param) * self._e_scale

    com_param, body_params = mocap.get_acceleration_statistics(self._abs_acc)
    self._ca_penalty = get_bound(com_param) * self._ca_scale
    self._a_penalties = map(get_bound, body_params)
    self._a_penalties = np.array(list(self._a_penalties)) * self._a_scale

    joint_params = mocap.get_angular_acceleration_statistics(False)
    self._aa_penalties = map(get_bound, joint_params)
    self._aa_penalties = (np.array(list(self._aa_penalties)) + 0.1) * self._aa_scale

  def get_CoM_velocity(self):
    vel = self._skeleton.get_com_vel()
    return vel

  def get_kin_energy(self):
    char = self._skeleton._kin_core.getCharacter()
    kin_energy = KinematicCore.kinematicEnergy(char, True)
    return kin_energy

  def get_body_velocities(self):
    vels = list(map(self._skeleton.get_body_vel, range(self._skeleton.num_joints)))
    vels = np.array(vels)
    return vels

  def get_joint_local_omgs(self):
    omgs = list(map(self._skeleton.get_joint_local_omg, range(self._skeleton.num_joints)))
    omgs = np.array(omgs)
    return omgs

  def record_state(self):
    """
      state is made up with 2 items:
       - phase
       - sim_state
    """
    pose, vel = self._engine.get_pose()

    # build state using current reference root, and without global info
    self._skeleton.set_pose(pose)
    self._skeleton.set_vel(vel)
    self._curr_sim_pose = pose
    self._curr_sim_vel = vel
    ori_pos = pose[:3]
    ori_rot = pose[3:7]
    sim_state = self._skeleton.build_state(ori_pos, ori_rot, self._use_global_root_ori)

    phase = self.curr_phase % 1.0

    state = np.concatenate(([phase], sim_state))

    # get com velocity and body parts velocity
    vel = self.get_CoM_velocity()
    self._prev_cv = self._curr_cv
    self._curr_cv = vel.copy()

    vels = self.get_body_velocities()
    self._prev_vs = self._curr_vs
    self._curr_vs = vels.copy()

    # get joint local omg
    omgs = self.get_joint_local_omgs()
    self._prev_omgs = self._curr_omgs
    self._curr_omgs = omgs.copy()

    return state

  def calc_reg(self):
    # kinematic energy
    kin_energy = self.get_kin_energy()
    reg_energy = kin_energy / self._e_penalty

    # linear acceleration
    com_acc = (self._curr_cv - self._prev_cv) * 30
    body_accs = (self._curr_vs - self._prev_vs) * 30
    if not self._abs_acc:
      body_accs = body_accs - com_acc

    err_ca = abs(com_acc[1])
    err_as = np.sqrt((body_accs**2).sum(axis=1))

    reg_com = err_ca / self._ca_penalty
    reg_as = err_as / self._a_penalties

    # angular acceleration
    angular_acc = (self._curr_omgs - self._prev_omgs) * 30

    err_aa = np.sqrt((angular_acc**2).sum(axis=1))

    reg_aa = err_aa / self._aa_penalties

    # sum up all regularizations
    regs = [reg_energy, reg_com, reg_as.mean(), reg_aa.mean()]
    regs = np.array(regs)
    r_reg = np.exp(-regs.mean())

    return r_reg

  def calc_reward(self):

    r_reg = self.calc_reg()
    return r_reg
