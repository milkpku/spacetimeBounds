import sys
sys.path.append("../")
import numpy as np
import json
from Kinematic import KinematicCore
from utils.quaternion import Quaternion

class HumanoidMocap(object):
  """ MOCAP interpolator

      get_phase() get phase at time t

      slerp(t) get information at time t


      pose representation:
        - root pos (x, y, z)
        - root rot (w, x, y, z) in world frame
        - joint rots theta or (w, z, y, z) in local frame, VelRel

      vel w/ padding representation:
        - root vel (vx, vy, vz)
        - root angular vel (wx, wy, wz) in world frame
        - joint angular vel w or (wx, wy, wz, 0) in local frame, VelRel

      vel w/o padding representation
        - root vel (vx, vy, vz)
        - root angular vel (wx, wy, wz) in world frame
        - joint angular vel w or (wx, wy, wz) in local frame, VelRel
  """
  def __init__(self, skeleton, mocap_file, extend_none_wrap=True):

    self._skeleton = skeleton   # HumanoidSkeleton

    self._durations = None
    self._frames    = None

    self._cycletime = None   # total time of mocap
    self._spf = None         # seconds per frame
    self._cyc_offset = None  # cycle offset of base
    self._is_wrap = None

    with open(mocap_file, 'r') as f:
      data = json.load(f)

    frame_data = np.array(data["Frames"])
    loop_type = data["Loop"]
    self._is_wrap = (loop_type == "wrap")

    # assert frame duration should be uniform
    self._durations = frame_data[:, 0]
    tmp = self._durations[:-1]
    assert(abs(tmp - tmp.mean()).max() < 1e-10), tmp
    self._spf = self._durations[0]

    # if is non-wrap motion, extend final frame 0.5 seconds
    if not self._is_wrap and extend_none_wrap:
      n_ext = int(0.5/self._spf)
      frame_data[-1, 0] = self._spf
      repeated_frames = np.repeat([frame_data[-1]], n_ext, axis=0)
      frame_data = np.concatenate([frame_data, repeated_frames])
      frame_data[-1, 0] = 0
      self._durations = frame_data[:, 0]

    # calculate cycle time and other information
    self._cycletime = np.sum(self._durations)

    self._frames = frame_data[:, 1:]
    self.num_frames = frame_data.shape[0]
    self._cyc_offset = self._frames[-1][:3] - self._frames[0][:3]
    self._cyc_offset[1] = 0
    assert(self._frames.shape[1] == self._skeleton.dof)

    # normalize quaternions
    for offset, dof in zip(self._skeleton.pos_start, self._skeleton.joint_dof):
      if dof == 4:
        for i in range(self.num_frames):
          quat = Quaternion.fromWXYZ(self._frames[i, offset:offset+4])
          self._frames[i, offset:offset+4] = quat.wxyz()

    # precompute velocity
    self._vels_pad = np.zeros((self.num_frames, self._skeleton.dof))

    for i in range(self._vels_pad.shape[0]-1):
      curr_frame = self._frames[i]
      next_frame = self._frames[i+1]
      dt = self._durations[i]
      self._vels_pad[i] = self._skeleton.computeVel(curr_frame, next_frame, dt, True)
    self._vels_pad[-1] = self._vels_pad[-2]

    self._vels_pad_noise = self._vels_pad.copy()

    self._vels_pad = self._butterworth_filter(self._vels_pad)

    self._vels_comp_noise = self._vels_pad_noise[:, self._skeleton.comp2pad]
    self._vels_comp = self._vels_pad[:, self._skeleton.comp2pad]
    assert(np.isfinite(self._vels_pad.sum())), print("infinite vels in mocap file")

    # precompute com pose and smooth
    self._com = []
    self._com_vel = []
    for p, v in zip(self._frames, self._vels_pad):
      self._skeleton.set_pose(p)
      self._skeleton.set_vel(v)
      self._com.append(self._skeleton.get_com_pos())
      self._com_vel.append(self._skeleton.get_com_vel())
    self._com = np.array(self._com)
    self._com_vel = np.array(self._com_vel)

    self._com_smooth = self._com.copy()
    self._com_smooth = self._butterworth_filter(self._com_smooth)

  def get_pose_dim(self):
    return self._frames[0].size

  def get_vel_dim(self, padding=True):
    return self._vels_pad[0].size if padding else self._vels_comp[0].size

  def get_phase(self, t):
    """ Get phase from t

      Inputs:
        t           float, current time

      Outputs:
        phase       float, current phase
    """
    phase = t / self._cycletime
    phase -= np.floor(phase)
    if phase < 0:
      phase += 1.0

    return phase

  def _butterworth_filter(self, vels):
    # TODO butterworth filter
    for i in range(vels.shape[1]):
        vels[:, i] = KinematicCore.butterworthFilter(self._spf, 6, vels[:, i])
    return vels

  def slerp(self, t, padding=True):
    """ Interpolate between two frames, indexed by time t

      Inputs:
        t           float, current time
        padding     decide whether return vel_pad or vel_cmp

      Outputs:
        cycle_count int, number of cycles it finishs
        pose        np.array of float, character pose
        vel         np.array of float, velcity during frame, w/ or w/o padding
    """
    # calculate cycle count
    cycle_count = np.floor(t / self._cycletime)
    t -= cycle_count * self._cycletime
    if (t < 0):
      cycle_count -= 1
      t += self._cycletime

    if not self._is_wrap and cycle_count > 0:
      count = 0
      pose = self._frames[-1].copy()
      if padding:
        vel = np.zeros_like(self._vels_pad[-1])
      else:
        vel = np.zeros_like(self._vels_comp[-1])

      return count, pose, vel

    # index frames
    t = t / self._spf
    idx = int(t)
    idx_next = idx + 1
    t -= idx
    assert(idx >= 0)
    assert(idx_next < self._frames.shape[0])

    # calculate pos
    curr_frame = self._frames[idx]
    next_frame = self._frames[idx_next]
    pose = self._skeleton.slerp(curr_frame, next_frame, t)

    if padding:
      vel = self._vels_pad[idx] * (1-t) + self._vels_pad[idx_next] * t
    else:
      vel = self._vels_comp[idx] * (1-t) + self._vels_comp[idx_next] * t

    return cycle_count, pose, vel

  def get_com_pos(self, t, smooth=False):
    """ Get com pos at t

      Inputs:
        t   float, global time starting from mocap's beginning

      Outputs:
        count  int, number of cycles that has past
        com    vec3, current com pos
        smooth bool, whether use smooth or not
    """
    cycle_count = np.floor(t / self._cycletime)
    t -= cycle_count * self._cycletime
    if (t < 0):
      cycle_count -= 1
      t += self._cycletime

    if not self._is_wrap and cycle_count > 0:
      count = 0
      if smooth:
        com = self._com_smooth[-1]
      else:
        com = self._com[-1]

      return count, com

    # index frames
    t = t / self._spf
    idx = int(t)
    idx_next = idx + 1
    t -= idx
    assert(idx >= 0)
    assert(idx_next < self._com.shape[0])

    # calculate pos
    if smooth:
      curr_com = self._com_smooth[idx]
      next_com = self._com_smooth[idx_next]
    else:
      curr_com = self._com[idx]
      next_com = self._com[idx_next]
    com = curr_com + (next_com - curr_com) * t

    return cycle_count, com

  def get_ref_mem(self):
    """ Construct reference memory, action represented in exponential map, and
    a phase velocity is given

    """
    a_dim = self._skeleton.expdof + 1
    ref_mem = np.zeros((self.num_frames, a_dim))
    for i in range(self.num_frames):
      ref_mem[i, 1:] = self._skeleton.targ_pose_to_exp(self._frames[i])
    ref_mem[:, 0] = 1.0 / self._cycletime
    return ref_mem

  def resample(self, timestep=0.0333):
    """ Resample mocap, make it timestep as given

      Inputs:
        timestep  expected timestep for resampling

      Outputs:
        mocap     np.array data of mocap frames

    """
    frames = np.ceil(self._cycletime / timestep)
    frames = int(frames)
    dt = self._cycletime / frames
    mocap = np.zeros((frames+1, self._frames.shape[1] + 1))
    for i in range(frames):
      count, pose, vel = self.slerp(i * dt)
      mocap[i, 0] = dt
      mocap[i, 1:] = pose

    # final duration is 0
    mocap[-1, 0] = 0
    mocap[-1, 1:] = self._frames[-1]

    return mocap

  def adjust_y_pos(self, x, y):
      # blend y
      from scipy.interpolate import interp1d
      f = interp1d(x, y, 'slinear')
      fx = np.arange(0, 347)
      fy = f(fx)
      self._frames[:, 1] += fy
      import matplotlib.pyplot as plt
      plt.plot(fy)
      plt.scatter(x, y)
      plt.show()

  def show_com(self):
    import matplotlib.pyplot as plt
    plt.subplot(311)
    plt.plot(self._frames[:, :3])
    plt.subplot(312)
    plt.plot(self._com)
    plt.subplot(313)
    plt.plot(self._com_smooth)
    plt.show()

  def get_acceleration_statistics(self, abs_acc=False):
    # get position of body parts and com
    l = max(len(self._frames), 120)
    spf = self._cycletime / (l+1)
    nj = self._skeleton.num_joints
    body_pos = np.zeros((l, nj, 3))
    com_pos = np.zeros((l, 3))
    for i in range(l):
      t = i * spf
      cnt, pose, vel = self.slerp(t)
      self._skeleton.set_pose(pose)

      for b in range(nj):
        pos = self._skeleton.get_body_pos(b)
        body_pos[i, b] = pos

      pos = self._skeleton.get_com_pos()
      com_pos[i] = pos

    # calculate velocity and acceleration of body parts and com
    body_vel = (body_pos[1:] - body_pos[:-1]) / spf
    body_vel_smooth = np.zeros_like(body_vel)
    for i in range(3):
      body_vel_smooth[:, :, i] = self._butterworth_filter(body_vel[:, :, i])
    body_acc = (body_vel_smooth[1:] - body_vel_smooth[:-1]) / spf

    com_vel = (com_pos[1:] - com_pos[:-1]) / spf
    com_vel_smooth = self._butterworth_filter(com_vel)
    com_acc = (com_vel_smooth[1:] - com_vel_smooth[:-1]) / spf

    if abs_acc:
      body_acc_rel = body_acc
    else:
      body_acc_rel = body_acc - com_acc.reshape(len(com_acc), 1, 3)

    # calculate acceleration magnitude and get parameters
    com_acc_l = np.sqrt((com_acc**2).sum(axis=1))
    body_acc_rel_l = np.sqrt((body_acc_rel**2).sum(axis=2))
    body_acc_rel_l = body_acc_rel_l.transpose()

    com_param = box_param(com_acc_l)
    body_params = list(map(box_param, body_acc_rel_l))

    return com_param, body_params

  def get_angular_acceleration_statistics(self, seperate=True):
    # get position of body parts and com
    l = max(int(self._cycletime*30), 120)
    spf = 1/30
    nj = self._skeleton.num_joints
    joint_local_omg = np.zeros((l, nj, 3))
    for i in range(l):
      t = i * spf
      cnt, pose, vel = self.slerp(t)
      self._skeleton.set_pose(pose)
      self._skeleton.set_vel(vel)

      for b in range(nj):
        omg = self._skeleton.get_joint_local_omg(b)
        joint_local_omg[i, b] = omg

    # calculate velocity and acceleration of body parts and com
    joint_local_omg_smooth = np.zeros_like(joint_local_omg)
    for i in range(3):
      joint_local_omg_smooth[:, :, i] = self._butterworth_filter(joint_local_omg[:, :, i])
    joint_local_angular_acc = (joint_local_omg_smooth[1:] - joint_local_omg_smooth[:-1]) / spf

    if not seperate:
        tmp = joint_local_angular_acc
        joint_local_acc = np.sqrt(tmp[:, :, 0]**2 + tmp[:, :, 1]**2 + tmp[:, :, 2]**2)
        joint_local_acc = joint_local_acc.transpose()
        joint_acc_params = list(map(box_param, joint_local_acc))
        return joint_acc_params

    # calculate acceleration magnitude and get parameters
    joint_local_twist_acc = abs(joint_local_angular_acc[:, :, 1])
    joint_local_yaw_acc = np.sqrt(joint_local_angular_acc[:, :, 0]**2 + joint_local_angular_acc[:, :, 2]**2)

    joint_local_twist_acc = joint_local_twist_acc.transpose()
    joint_local_yaw_acc = joint_local_yaw_acc.transpose()

    twist_params = list(map(box_param, joint_local_twist_acc))
    yaw_params = list(map(box_param, joint_local_yaw_acc))

    return twist_params, yaw_params

  def get_energy_statistics(self):
    # get position of body parts and com
    l = max(len(self._frames), 120)
    spf = self._cycletime / (l+1)
    kin_energies = []
    for i in range(l):
      t = i * spf
      cnt, pose, vel = self.slerp(t)
      self._skeleton.set_pose(pose)
      self._skeleton.set_vel(vel)

      char = self._skeleton._kin_core.getCharacter()
      e_rel = KinematicCore.kinematicEnergy(char, True)

      kin_energies.append(e_rel)

    return box_param(kin_energies)

def box_param(x):
  """ get box parameter of x: median, q1, q3, max, min, upper_fence, lower_fence

    Outputs:
      params   dict of median, q1, q3, max, min, upper_fence, lower_fence
  """
  x_sorted = np.sort(x)

  l = len(x_sorted)

  params = {
      'median': x_sorted[l//2],
      'q1': x_sorted[l//4],
      'q3': x_sorted[3*l // 4],
      'max': x_sorted[-1],
      'min': x_sorted[0],
  }

  params['upper_fence'] = params['q3'] + 1.5 * (params['q3'] - params['q1'])
  params['lower_fence'] = params['q1'] - 1.5 * (params['q3'] - params['q1'])

  return params

def test_mocap(task, ckpt):
  from IPython import embed
  from humanoid_kin import HumanoidSkeleton
  char_file = "/home/likem/Work/motionBound/data/characters/humanoid3d.txt"
  ctrl_file = "/home/likem/Work/motionBound/data/controllers/humanoid3d_ctrl.txt"
  skeleton = HumanoidSkeleton(char_file, ctrl_file)

  mocap_file = "/home/likem/Work/motionBound/data/motions/humanoid3d_%s.txt" % task
  mocap = HumanoidMocap(skeleton, mocap_file)

  import torch
  data = torch.load(ckpt)

  for t, pose, vel in zip(data["t"], data["pose"], data["vel"]):
    count, pose_calc, vel_calc = mocap.slerp(t)

    pose_diff = pose - pose_calc
    vel_diff = vel - vel_calc
    assert(np.abs(pose_diff).max() < 1e-6), embed()
    assert(np.abs(vel_diff).max() < 1e-5), embed()

  ref_mem = mocap.get_ref_mem()
  assert(ref_mem)


def change_dir(frames):
  # change direction to x+ axis, starting position at origin
  ori_pos = frames[0, 1:4].copy()
  ori_pos[1] = 0
  vel_dir = frames[-1, 1:4] - frames[0, 1:4]
  heading_theta = np.arctan2(-vel_dir[2], vel_dir[0])
  inv_rot = Quaternion.fromAngleAxis(-heading_theta, np.array([0, 1, 0]))
  for i in range(frames.shape[0]):
    frame = frames[i]
    frames[i, 1:4] = inv_rot.rotate(frame[1:4] - ori_pos)
    root_rot = Quaternion.fromWXYZ(frame[4:8])
    new_rot = inv_rot.mul(root_rot)
    frames[i, 4:8] = new_rot.pos_wxyz()

  return frames

def save_mocap(frames, outfile):
  frames = frames.tolist()

  with open(outfile, 'w') as wh:
    wh.write("{\n")
    wh.write('"Loop":"wrap",\n')
    wh.write('"Frames":\n')
    wh.write('[\n')
    contents = ",\n".join(map(str, frames))
    wh.write(contents + "\n")
    wh.write(']\n')
    wh.write("}")

  print("write to %s" % outfile)

def resample_mocap(mocap_file, outfile, resample_frequency):
  from humanoid_kin import HumanoidSkeleton
  char_file = "/home/likem/Work/motionBound/data/characters/humanoid3d.txt"
  ctrl_file = "/home/likem/Work/motionBound/data/controllers/humanoid3d_ctrl.txt"
  skeleton = HumanoidSkeleton(char_file, ctrl_file)

  mocap = HumanoidMocap(skeleton, mocap_file)

  timestep = 1.0 / resample_frequency
  frames = mocap.resample(timestep)

  return frames


if __name__=="__main__":

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("src", type=str, help="input mocap data")
  parser.add_argument("-o","--out", type=str, help="output file name")
  parser.add_argument("--rate", type=float, default=30, help="resample frequency")

  args = parser.parse_args()
  frames = resample_mocap(args.src, args.out, args.rate)
  frames = change_dir(frames)
  save_mocap(frames, args.out)
