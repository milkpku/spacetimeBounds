import numpy as np
import math
from utils import bullet_client
from utils.humanoid_kin import HumanoidSkeleton
from utils.humanoid_mocap import HumanoidMocap
from utils.humanoid_vis import HumanoidVis
import pybullet as p1

#from IPython import embed
import time

###
ACT_STEPTIME  = 1./30.
SUBSTEPS = 20
VIS_STEP = 1.0/150
SIM_STEPTIME  = ACT_STEPTIME / SUBSTEPS
###

class VisMocapEnv():
    def __init__(self, mocap_file, pybullet_client=None, model="humanoid3d"):
      self._isInitialized = False
      self.rand_state = np.random.RandomState()
      self._motion_file = mocap_file
      self.enable_draw = True
      self.follow_character = True
      self._model = model
      self.init()

    def init(self):
      """
        initialize environment with bullet backend
      """
      # mocap data
      if self._model == "humanoid3d":
        char_file = "data/characters/humanoid3d.txt"
        ctrl_file = "data/controllers/humanoid3d_ctrl.txt"
        self._skeleton = HumanoidSkeleton(char_file, ctrl_file)
        self._mocap = HumanoidMocap(self._skeleton, self._motion_file)
        print("mocap duration: %f" % self._mocap._cycletime)
      elif self._model == "atlas":
        char_file = "data/characters/atlas.txt"
        ctrl_file = "data/controllers/atlas_ctrl.txt"
        self._skeleton = HumanoidSkeleton(char_file, ctrl_file)
        self._mocap = HumanoidMocap(self._skeleton, self._motion_file)
        print("mocap duration: %f" % self._mocap._cycletime)
      elif self._model == "atlas_jason":
        char_file = "data/characters/atlas_jason.txt"
        ctrl_file = "data/controllers/atlas_jason_ctrl.txt"
        self._skeleton = HumanoidSkeleton(char_file, ctrl_file)
        self._mocap = HumanoidMocap(self._skeleton, self._motion_file)
        print("mocap duration: %f" % self._mocap._cycletime)
      else:
        assert(False)

      self._visual = HumanoidVis(self._skeleton, self._model)
      #color = [227/255, 170/255, 14/255, 1] # sim
      color = [44/255, 160/255, 44/255, 1] # ref
      self._char = self._visual.add_character("mocap", color)
      self._visual.camera_follow(self._char, 2, 0, 0)

      self._pybullet_client = self._visual._pybullet_client
      self._play_speed = self._pybullet_client.addUserDebugParameter("play_speed", 0, 2, 1.0)
      self._phase_ctrl = self._pybullet_client.addUserDebugParameter("frame", 0, 1, 0)

    def reset(self, phase=None):
      startTime = self.rand_state.rand() if phase is None else phase
      startTime *= self._mocap._cycletime
      self.t = startTime
      self.start_phase = self._mocap.get_phase(self.t)
      count, pose, vel = self._mocap.slerp(startTime)

      if self.enable_draw:
        self.synchronize_sim_char()
        self._prev_clock = time.time()
        self._visual.camera_follow(self._char)

    def step(self):
      speed = self._pybullet_client.readUserDebugParameter(self._play_speed)
      phase = self._pybullet_client.readUserDebugParameter(self._phase_ctrl)
      #phase /= self._mocap.num_frames
      if (self.start_phase != phase):
        self.reset(phase)
      self.update(VIS_STEP, speed)

    def update(self, timeStep, speed):
      self.wait_till_timestep(VIS_STEP)
      self.synchronize_sim_char()
      if self.follow_character:
        self._visual.camera_follow(self._char)
        #cnt, pos = self._mocap.get_com_pos(self.t, True)
        #pos += cnt * self._mocap._cyc_offset
        #pos[1] = 1
        #self._visual.camera_follow(self._char, None, None, None, pos)

      self.t += timeStep * speed

    def synchronize_sim_char(self):
      count, pose, vel = self._mocap.slerp(self.t)
      pose[:3] += count * self._mocap._cyc_offset
      self._visual.set_pose(self._char, pose, vel)

    def getKeyboardEvents(self):
      return self._pybullet_client.getKeyboardEvents()

    def isKeyTriggered(self, keys, key):
      o = ord(key)
      #print("ord=",o)
      if o in keys:
        return keys[ord(key)] & self._pybullet_client.KEY_WAS_TRIGGERED
      return False

    def wait_till_timestep(self, timeStep):
      time_remain = timeStep - (time.time() - self._prev_clock)
      if time_remain > 0:
        time.sleep(time_remain)
      self._prev_clock = time.time()

def show_mocap(mocap_file, model):
  env = VisMocapEnv(mocap_file, None, model)
  #env._mocap.show_com()
  env.reset()
  while True:
    env.step()

if __name__=="__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--mocap", type=str, default='data/motions/humanoid3d_jump.txt', help="task to perform")
  parser.add_argument("--model", type=str, default='humanoid3d', help="model")
  args = parser.parse_args()

  show_mocap(args.mocap, args.model)
