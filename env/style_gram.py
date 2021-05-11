from env import SpacetimeBoundsEnv
from utils.humanoid_kin import HumanoidSkeleton
from utils.humanoid_mocap import HumanoidMocap
from Kinematic import KinematicCore
from model import MotionEncoder
from model.motion_encoder import init_motion_encoder, process_gram_matrix
from utils.export_feature import process_trajectory
import torch
import numpy as np


class StyleGramEnv(SpacetimeBoundsEnv):
  def __init__(self, style, window=None, scale=0.06,
               e_scale=4, ca_scale=20, a_scale=20,
               at_scale=4, ay_scale=20,
               z_scale=2, cv_scale=4, cv_ratio=1,
               abs_acc=False, avoid_self=False,
               **kwargs):
    """
        Initialize Gram matrix style transfer environment
    """
    super().__init__(**kwargs)
    if window:
      self._window = window
    else:
      if self._mocap._is_wrap:
        self._window = int(self._mocap._cycletime * 60)
      else:
        assert(False and "style reference is not cyclic")
    self._min_window = 7
    self._scale = scale

    self._history = []
    char_file = "data/characters/humanoid3d_cmu.txt"
    ctrl_file = "data/controllers/humanoid3d_cmu_ctrl.txt"
    self._style_skeleton = HumanoidSkeleton(char_file, ctrl_file)
    style_file = "data/motions/humanoid3d_%s.txt" % style
    self._style_G = self.style_gram_matrix(style_file)

    # regularization
    self._e_scale = e_scale
    self._ca_scale = ca_scale
    self._a_scale = a_scale
    self._at_scale = at_scale
    self._ay_scale = ay_scale
    self._z_off_scale = z_scale
    self._cv_scale = cv_scale
    self._cv_ratio = cv_ratio
    self._abs_acc = abs_acc
    self._avoid_self = avoid_self

    self.init_params(style_file)

    self._prev_cv = None
    self._curr_cv = None

    self._prev_vs = None
    self._curr_vs = None

    self._prev_omgs = None
    self._curr_omgs = None

  def init_params(self, motion_file):
    mocap = HumanoidMocap(self._skeleton, motion_file)

    get_bound = lambda p: min(p['max'], p['upper_fence'])

    e_param = mocap.get_energy_statistics()
    self._e_penalty = get_bound(e_param) * self._e_scale

    com_param, body_params = mocap.get_acceleration_statistics(self._abs_acc)
    self._ca_penalty = get_bound(com_param) * self._ca_scale
    self._a_penalties = map(get_bound, body_params)
    self._a_penalties = np.array(list(self._a_penalties)) * self._a_scale

    com_vel = mocap._com_vel
    com_vel_l = np.sqrt((com_vel**2).sum(axis=1))
    self._cv_goal = np.array([com_vel_l.mean(), 0, 0])
    self._cv_penalty = max(com_vel_l.std(), 0.2 * com_vel_l.mean()) * self._cv_scale

    self._cv_goal *= self._cv_ratio
    self._cv_penalty *= self._cv_ratio

    twist_params, yaw_params = mocap.get_angular_acceleration_statistics()
    self._at_penalties = map(get_bound, twist_params)
    self._at_penalties = np.array(list(self._at_penalties)) * self._at_scale + 0.1
    self._ay_penalties = map(get_bound, yaw_params)
    self._ay_penalties = np.array(list(self._ay_penalties)) * self._ay_scale + 0.1

  def style_gram_matrix(self, motion_file):
    # load style motion
    style_mocap = HumanoidMocap(self._style_skeleton, motion_file, extend_none_wrap=False)

    # transfer to features
    if style_mocap._is_wrap:
      clip_time = 25 * style_mocap._cycletime + 12/60
    else:
      clip_time = style_mocap._cycletime

    length = int(clip_time * 60)
    traj = []
    for i in range(length):
      cnt, pose, vel = style_mocap.slerp(i/60)
      pose[:3] += style_mocap._cyc_offset * cnt
      pose[:3] *= 19
      self._style_skeleton.set_pose(pose)
      jointposes = self._style_skeleton.get_pos_for_gram()
      traj.append(jointposes)

    traj = np.array(traj)
    feature = process_trajectory(traj).transpose()
    dim, l = feature.shape
    feature = feature.reshape(1, dim, l)

    # to gram matrix
    G = process_gram_matrix(feature)

    return G

  def get_state_size(self):
    single_state = self._skeleton.build_state()
    # 1 dim for phase, 2 for additional x, z, 1 for sim state (which contains y)
    state_size = 1 + single_state.size
    return state_size

  def style_distance(self):
    if (len(self._history) < self._min_window):
      return 0

    # turn history records to feature
    traj = np.array(self._history)
    feature = process_trajectory(traj)
    feature = feature.transpose()

    content_G = process_gram_matrix(feature)
    err = np.square(content_G - self._style_G).mean()
    err = np.sqrt(err)

    return err

  def get_foot_forces(self):
    """ get foot supporting forces
    Output:

      FL   vector, contact force on left foot
      FR   vector, contact force on right foot
    """
    left_forces = []
    right_forces = []
    for sim_step in self._contact_forces:
      l_f = np.array([0., 0., 0.])
      r_f = np.array([0., 0., 0.])
      for part, pos, force in sim_step:
        if part == 5:
          r_f += force
        if part == 11:
          l_f += force
      left_forces.append(l_f)
      right_forces.append(r_f)

    left_forces = np.array(left_forces)
    right_forces = np.array(right_forces)

    FL = left_forces.mean(axis=0)
    FR = right_forces.mean(axis=0)

    return FL, FR

  def get_CoMz_offset(self):
    pos = self._skeleton.get_com_pos()
    return pos[2]

  def get_CoM_velocity(self):
    vel = self._skeleton.get_com_vel()
    return vel

  def get_body_velocities(self):
    vels = list(map(self._skeleton.get_body_vel, range(self._skeleton.num_joints)))
    vels = np.array(vels)
    return vels

  def get_joint_local_omgs(self):
    omgs = list(map(self._skeleton.get_joint_local_omg, range(self._skeleton.num_joints)))
    omgs = np.array(omgs)
    return omgs

  def get_kin_energy(self):
    char = self._skeleton._kin_core.getCharacter()
    kin_energy = KinematicCore.kinematicEnergy(char, True)
    return kin_energy

  def reset(self, phase=None, state=None):
    self._history = []
    return super().reset(phase, state)

  def record_state(self):
    """
      state is made up with 2 items:
       - phase
       - sim_state
    """
    pose, vel = self._engine.get_pose()

    # add to history and mantain features for style comparing
    style_pose = pose.copy()
    style_pose[:3] *= 19
    self._style_skeleton.set_pose(style_pose)
    pos_gram = self._style_skeleton.get_pos_for_gram()
    if len(self._history) > 1:
      pos_gram_mid = (pos_gram + self._history[-1])/2
      self._history.append(pos_gram_mid)
      self._history.append(pos_gram)
    else:
      # when start from [], first position
      self._history.append(pos_gram)

    if len(self._history) > self._window:
      # pop out two frames, since the record freq is 60Hz
      self._history.pop(0)
      self._history.pop(0)

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

  def calc_reward(self):

    # penalizing self contact
    if self._avoid_self and self._engine.check_self_contact():
      return 0

    # style reward
    err_s = self.style_distance()
    r_s = np.exp(-err_s/self._scale)

    # z offset, c velocity
    z_off = self.get_CoMz_offset()
    err_z = abs(z_off)
    vel = self.get_CoM_velocity()
    err_cv = np.linalg.norm(vel - self._cv_goal)
    reg_off = np.array([err_z/self._z_off_scale, err_cv/self._cv_penalty])

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
    twist_acc = angular_acc[:, 1]
    yaw_acc = angular_acc[:, [0, 2]]

    twist_err = abs(twist_acc)
    yaw_err = np.sqrt((yaw_acc**2).sum(axis=1))

    reg_aa = np.concatenate([twist_err/self._at_penalties, yaw_err/self._ay_penalties])

    # sum up all regularizations
    regs = [reg_off.mean(), reg_energy, reg_com, reg_as.mean(), reg_aa.mean()]
    regs = np.array(regs)
    r_reg = np.exp(-regs.mean())

    rwd = r_s * r_reg

    #return np.array([r_s, r_reg])
    return rwd

  def record_info(self):
    info = super().record_info()
    info["trust_rwd"] = len(self._history) > self._min_window
    return info
