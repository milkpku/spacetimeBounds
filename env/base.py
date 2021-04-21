from abc import ABC, abstractmethod
import numpy as np
import time
import argparse

from utils.humanoid_kin import HumanoidSkeleton
from utils.humanoid_mocap import HumanoidMocap
from utils.humanoid_vis import HumanoidVis
from sim import engine_builder

# timesteps
ACT_STEPTIME  = 1./30.
SUBSTEPS = 20
VIS_STEPTIME = 1.0/150
SIM_STEPTIME  = ACT_STEPTIME / SUBSTEPS

# task and contact
allowed_contacts = {
        "walk":     "walk.txt",
        "cartwheel":"cartwheel.txt",
        "crawl":    "crawl.txt",
        "roll":     "roll.txt",
        "knee":     "knee.txt",
        }

class BaseEnv(ABC):
  """
  environment abstraction

  __init__
  - build_action_bound (abstract)

  get_state_size (abstract)
  get_action_size (abstract)

  get_reset_data (abstract)

  reset (abstract)

  step
  - set_action (abstract)
  - check_valid_episode
  - record_state (abstract)
  - record_info  (abstract)
  - calc_reward  (abstract)
  - is_episode_end

  check_terminate (abstract)
  check_fall
  check_wrap_end
  check_time_end

  set_mode
  set_task_t

  """
  def __init__(self, task, seed=0, model="humanoid3d", engine="pybullet", contact="walk",
               self_collision=True, enable_draw=False,
               record_contact=False, record_torques=False):
    self._task = task
    self._rand = np.random.RandomState(seed)
    self._model= model

    self._ctrl_step = ACT_STEPTIME
    self._nsubsteps = SUBSTEPS
    self._sim_step = self._ctrl_step / self._nsubsteps

    self._enable_draw = enable_draw
    self._vis_step = VIS_STEPTIME
    self._follow_character = True

    self._record_contact = record_contact

    # initialize kinematic parts
    char_file = "data/characters/%s.txt" % self._model
    ctrl_file = "data/controllers/%s_ctrl.txt" % self._model
    motion_file = "data/motions/%s_%s.txt" % (self._model, self._task)
    self._skeleton = HumanoidSkeleton(char_file, ctrl_file)
    self._mocap = HumanoidMocap(self._skeleton, motion_file)
    self.curr_phase = None

    # initialize visual parts, need to be the first when using pybullet environment
    if self._enable_draw:
      # draw timer
      self._prev_clock = time.perf_counter()
      self._left_time = 0

      # visual part
      self._visual = HumanoidVis(self._skeleton, self._model)
      cnt, pose, vel = self._mocap.slerp(0)
      self._sim_char = self._visual.add_character("sim", [227/255, 170/255, 14/255, 1])

      #self._kin_char = self._visual.add_character("kin", [1, 1, 1, 0.4])
      self._kin_char = self._visual.add_character("kin", [44/255, 160/255, 44/255, 1])

      self._visual.camera_follow(self._sim_char, 2, 180, 0)

    # initialize simulation parts
    self._engine = engine_builder(engine, self._skeleton, self_collision, self._sim_step, self._model)

    contact_parser = argparse.ArgumentParser()
    contact_parser.add_argument("N", type=int, nargs="+", help="allowed contact body ids")
    contact_file = "data/contacts/%s_%s" % (self._model, allowed_contacts[contact])
    with open(contact_file) as fh:
      s = fh.read().split()
      args = contact_parser.parse_args(s)
      allowed_body_ids = args.N
    self._engine.set_allowed_fall_contact(allowed_body_ids)
    self._contact_ground = False

    # joint torque monitor # debug use
    self._monitor_joints = record_torques
    if self._monitor_joints:
      self._monitored_joints = list(range(self._skeleton.num_joints))
      self._engine.set_monitored_joints(self._monitored_joints)

    # initialize reinfrocement learning action bound
    self.a_min, self.a_max = self.build_action_bound()

    self._max_t = max(20, self._mocap._cycletime)
    self._task_t = 0.5
    self._mode = 0   # 0 for max_t and 1 for task_t

  @abstractmethod
  def get_state_size(self):
    assert(False and "not implemented")

  @abstractmethod
  def get_action_size(self):
    assert(False and "not implemented")

  @abstractmethod
  def build_action_bound(self):
    assert(False and "not implemented")

  @abstractmethod
  def get_reset_data(self):
    assert(False and "not implemented")

  @abstractmethod
  def reset(self, phase=None, state=None):
    """ Reset environment

      Inputs:
        phase
        state

      Outputs:
        obs
    """
    assert(False and "not implemented")

  def step(self, action):
    """ Step environment

      Inputs:
        action

      Outputs:
        ob
        r
        done
        info
    """
    self.set_action(action)

    self._contact_forces = []

    self._joint_torques = []

    # if not terminated during simulation
    self._contact_ground = False
    for i in range(self._nsubsteps):
      self.update()
      if self.check_fall():
        self._contact_ground = True

    is_fail = self.check_terminate()
    ob = self.record_state()
    r = 0 if is_fail else self.calc_reward()
    done = is_fail or self.is_episode_end()
    info = self.record_info()

    return ob, r, done, info

  @abstractmethod
  def set_action(self, action):
    assert(False and "not implemented")

  def update(self):
    # update draw
    if self._enable_draw and self.time_to_draw(self._sim_step):
      self.update_draw()

    self._engine.step_sim(self._sim_step)
    if self._record_contact:
      self.update_contact_forces()

    if self._monitor_joints:
      self.update_joint_torques()

    self.post_update()

  def time_to_draw(self, timestep):
    self._left_time -= timestep
    if self._left_time < 0:
      self._left_time += self._vis_step
      return True
    else:
      return False

  def update_draw(self):
    # synchronize sim pose and kin pose
    sim_pose, sim_vel = self._engine.get_pose()
    kin_t = self.curr_phase * self._mocap._cycletime
    cnt, kin_pose, kin_vel = self._mocap.slerp(kin_t)

    # offset kinematic pose
    kin_pose[:3] += cnt * self._mocap._cyc_offset
    kin_pose[0] += 1.5

    # wait until time
    time_remain = self._vis_step - (time.perf_counter() - self._prev_clock)
    if time_remain > 0:
      time.sleep(time_remain)
    self._prev_clock = time.perf_counter()

    # draw on window
    self._visual.set_pose(self._sim_char, sim_pose, sim_vel)
    self._visual.set_pose(self._kin_char, kin_pose, kin_vel)

    # adjust cameral pose
    if self._follow_character:
      self._visual.camera_follow(self._sim_char)

  def update_contact_forces(self):
    # draw contact forces
    contact_forces = self._engine.report_contact_force()
    self._contact_forces.append(contact_forces)

  def draw_contact_force(self):
    contact_forces = self._engine.report_contact_force()
    for part, pos, force in contact_forces:
      self._visual.visual_force(pos, force)

  def update_joint_torques(self):
    jt = self._engine.get_monitored_joint_torques()
    self._joint_torques.append(jt)

  def draw_joint_torques(self):
    jt = self._engine.get_monitored_joint_torques()
    pose, vel = self._engine.get_pose()
    self._skeleton.set_pose(pose)
    for i, torque in zip(self._monitored_joints, jt):
      pos = self._skeleton.get_joint_pos(i)
      self._visual.visual_force(pos, np.array(torque))

  @abstractmethod
  def record_info(self):
    assert(False and "not implemented")

  @abstractmethod
  def post_update(self):
    assert(False and "not implemented")

  @abstractmethod
  def record_state(self):
    assert(False and "not implemented")

  @abstractmethod
  def calc_reward(self):
    assert(False and "not implemented")

  def is_episode_end(self):
    """ Check if episode is end
      episode will normally end if time excced max time
    """
    return self.check_wrap_end() or self.check_time_end()

  @abstractmethod
  def check_terminate(self):
    """ Check if simulation or task failed
    """
    assert(False and "not implemented")

  def check_fall(self):
    """ Check if any not allowed body part contact ground
    """
    # check fail
    return self._engine.check_fall()

  def check_valid_episode(self):
    # TODO check valid
    #return self._core.CheckValidEpisode()
    return True

  def check_time_end(self):
    if self._mode == 0:
      return self._engine.t >= self._task_t
    elif self._mode == 1:
      return self._engine.t >= self._max_t
    else:
      assert(False and "no supported mode")

  def check_wrap_end(self):
    """ Check if time is up for non-wrap motions
        True if motion is non-wrap and reaches phase 1.0
        False if motion is wrap or hasn't reaches phase 1.0
    """
    loop_term = not self._mocap._is_wrap and self.curr_phase > 1.0
    return loop_term

  def set_mode(self, mode):
    # 0 for train, 1 for test using max time
    assert(mode >=0 and mode < 2)
    self._mode = mode

  def set_task_t(self, t):
    """ Set the max t an episode can have under training mode
    """
    self._task_t = min(t, self._max_t)

  def set_max_t(self, t):
    """ Set the max t an episode can have under test mode
    """
    self._max_t = t

  def close(self):
    self._engine.close()
