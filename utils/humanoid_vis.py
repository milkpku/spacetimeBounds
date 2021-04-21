import math
import numpy as np
from utils import bullet_client
from utils.humanoid_kin import JointType
import pybullet as p1

URDF_DIR = "data/urdf"

class HumanoidVis(object):

  def __init__(self, skeleton, model="humanoid3d"):
    self._skeleton = skeleton
    self.characters = dict()
    # init pybullet client
    self._init_physics()
    self._model = model # humanoid3d or atlas

  def _init_physics(self):
    self._pybullet_client =  bullet_client.BulletClient(connection_mode=p1.GUI)
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI,0)
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_Y_AXIS_UP,1)

    # load ground plane
    self._pybullet_client.setAdditionalSearchPath(URDF_DIR)
    z2y = self._pybullet_client.getQuaternionFromEuler([-math.pi*0.5,0,0])
    self._planeId = self._pybullet_client.loadURDF("plane_implicit.urdf",[0,0,0],z2y, useMaximalCoordinates=True)

    # set simulation environment parameters
    self._pybullet_client.setGravity(0,-9.8,0)

    self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=10)
    self._pybullet_client.changeDynamics(self._planeId, linkIndex=-1, lateralFriction=0.9)

    self._pybullet_client.setTimeStep(1.0/60.0)
    self._pybullet_client.setPhysicsEngineParameter(numSubSteps=1)

  def add_character(self, name, color):
    assert(name not in self.characters.keys())
    char_id = self._new_character(color)
    self.characters[name] = char_id
    return name

  def _new_character(self, color):

    if self._model == "humanoid3d":
      kin_model = self._pybullet_client.loadURDF(
        "humanoid/humanoid.urdf", [0,0.889540259,0],
        useFixedBase=True, flags=self._pybullet_client.URDF_MAINTAIN_LINK_ORDER)
    elif self._model == "atlas":
      kin_model = self._pybullet_client.loadURDF(
        "atlas/atlas.urdf", [0,0,0],
        useFixedBase=True, flags=self._pybullet_client.URDF_MAINTAIN_LINK_ORDER)
    else:
      raise NotImplementedError

    self._pybullet_client.changeDynamics(kin_model, -1, linearDamping=0, angularDamping=0)

    # set kinematic character dymanic, collision and vision property
    kin_act_state = (self._pybullet_client.ACTIVATION_STATE_SLEEP
                    + self._pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING
                    + self._pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)
    self._pybullet_client.setCollisionFilterGroupMask(kin_model,-1,collisionFilterGroup=0,collisionFilterMask=0)
    self._pybullet_client.changeDynamics(kin_model,-1,activationState=kin_act_state)
    self._pybullet_client.changeVisualShape(kin_model,-1, rgbaColor=color, textureUniqueId=-1)
    for j in range (self._pybullet_client.getNumJoints(kin_model)):
      self._pybullet_client.setCollisionFilterGroupMask(kin_model,j,collisionFilterGroup=0,collisionFilterMask=0)
      self._pybullet_client.changeDynamics(kin_model,j,activationState=kin_act_state)
      self._pybullet_client.changeVisualShape(kin_model,j, rgbaColor=color, textureUniqueId=-1)

    return kin_model

  def set_pose(self, char_name, pose, vel):
    """ Set character state in physics engine
      Inputs:
        pose   np.array of float, self._skeleton.pos_dim, position of base and
               orintation of joints, represented in local frame
        vel    np.array of float, self._skeleton.vel_dim, velocity of base and
               angular velocity of joints, represented in local frame

        phys_model  pybullet model unique Id, self._sim_model or self._kin_model
    """
    assert(char_name in self.characters.keys())
    phys_model = self.characters[char_name]
    s = self._skeleton
    pos = pose[:3]
    orn_wxyz = pose[3:7]
    orn = [orn_wxyz[1], orn_wxyz[2], orn_wxyz[3], orn_wxyz[0]]
    v   = vel[:3]
    omg = vel[3:6]
    self._pybullet_client.resetBasePositionAndOrientation(phys_model, pos, orn)
    self._pybullet_client.resetBaseVelocity(phys_model, v, omg)

    for i in range(s.num_joints):
      jtype = s.joint_types[i]
      p_off = s.pos_start[i]
      if jtype is JointType.BASE:
        pass
      elif jtype is JointType.FIXED:
        pass
      elif jtype is JointType.REVOLUTE:
        orn = [pose[p_off]]
        omg = [vel[p_off]]
        self._pybullet_client.resetJointStateMultiDof(phys_model, i, orn, omg)
      elif jtype is JointType.SPHERE:
        orn_wxyz = pose[p_off : p_off+4]
        orn = [orn_wxyz[1], orn_wxyz[2], orn_wxyz[3], orn_wxyz[0]]
        omg = vel[p_off : p_off+3]
        self._pybullet_client.resetJointStateMultiDof(phys_model, i, orn, omg)

  def camera_follow(self, char_name, dis=None, yaw=None, pitch=None, pos=None):
    assert(char_name in self.characters.keys())
    phys_model = self.characters[char_name]

    _pos = np.array(self._get_base_pose(phys_model))
    _pos[1] = 1
    _pos[0] += 0.75

    cam_info = self._pybullet_client.getDebugVisualizerCamera()
    _yaw, _pitch, _dis = cam_info[8], cam_info[9], cam_info[10]
    dis = _dis if dis is None else dis
    yaw = _yaw if yaw is None else yaw
    pitch = _pitch if pitch is None else pitch
    pos = _pos if pos is None else pos
    self._pybullet_client.resetDebugVisualizerCamera(dis, yaw, pitch, pos)

  def _get_base_pose(self, phys_model):
    pos, orn = self._pybullet_client.getBasePositionAndOrientation(phys_model)
    return pos

  def _get_joint_pose(self, phys_model, jid):
    info = self._pybullet_client.getLinkState(phys_model, jid)
    pos = info[4]
    orn = info[5]
    return pos, orn
