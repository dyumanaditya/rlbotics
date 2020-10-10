import os
import gym
import time
import numpy as np
import pybullet as p
import pybullet_data
from gym.utils import seeding


class PandaDrillerEnv(gym.Env):
	metadata = {'render.modes': ['human', 'rgb_array'],
				'video.frames_per_second': 50}

	def __init__(self, render):
		self.path = os.path.abspath(os.path.dirname(__file__))
		self.isRender = render
		self.seed()
		self.reset()

		# Initialise joint info
		self.joints_min, self.joints_max = np.array(),
		self.num_arm_joints = p.getNumJoints(self.arm_id)
		for j in range(self.num_arm_joints):
			self.joints_min.append(p.getJointInfo(self.arm_id, j)[8])
			self.joints_max.append(p.getJointInfo(self.arm_id, j)[9])

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		# Connect to physics client
		if self.isRender:
			p.connect(p.GUI)
		else:
			p.connect(p.DIRECT)

		p.resetSimulation()
		p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

		# Load Robots
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		tableOrientation = p.getQuaternionFromEuler([0, 0, np.pi/2])
		drillOrientation = p.getQuaternionFromEuler([0, -np.pi/2, np.pi])

		self.plane_id = p.loadURDF('plane.urdf')
		self.arm_id = p.loadURDF('franka_panda/panda.urdf', [-0.6, 0, 0.93], useFixedBase=True)
		self.table_id = p.loadURDF('table/table.urdf', [0, 0, 0], tableOrientation, globalScaling=1.5, useFixedBase=True)
		self.drill_id = p.loadURDF(os.path.join(self.path, 'drill.urdf'), [-0.265, 0, 1.73], drillOrientation, globalScaling=0.013)

		p.setRealTimeSimulation(1)
		self._grab_drill()
		self._generate_plane()
		self._get_camera_img()
		p.setRealTimeSimulation(0)

	def _grab_drill(self):
		time.sleep(0.5)
		p.setJointMotorControl2(self.arm_id, 5, p.POSITION_CONTROL, targetPosition=1)
		p.setJointMotorControl2(self.arm_id, 3, p.POSITION_CONTROL, targetPosition=-1)
		p.setJointMotorControl2(self.arm_id, 6, p.POSITION_CONTROL, targetPosition=0.8)
		p.setJointMotorControl2(self.arm_id, 9, p.POSITION_CONTROL, targetPosition=0.04)
		p.setJointMotorControl2(self.arm_id, 10, p.POSITION_CONTROL, targetPosition=0.04)

		time.sleep(0.5)
		p.setJointMotorControl2(self.arm_id, 9, p.POSITION_CONTROL, targetPosition=0)
		p.setJointMotorControl2(self.arm_id, 10, p.POSITION_CONTROL, targetPosition=0)

		time.sleep(0.5)
		p.setGravity(0, 0, -9.8)
		p.setJointMotorControl2(self.arm_id, 3, p.POSITION_CONTROL, targetPosition=1)
		time.sleep(0.5)

	def _generate_plane(self):
		# Min Max constraints for drilling on plane
		# min = [-0.2, 0.2, 0]
		# max = [0.2, -0.2, 0]

		hole_position = [self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(-0.2, 0.2), 0]
		plane_orientation = p.getQuaternionFromEuler(list(self.np_random.uniform(low=0, high=2*np.pi, size=3)))
		# plane_orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])

		plane_visual = p.createVisualShape(
			p.GEOM_MESH,
			fileName=os.path.join(self.path, 'plane', 'plane.obj')
		)

		plane = p.createMultiBody(
			basePosition=[0, 0, 1.3],
			baseVisualShapeIndex=plane_visual,
			baseOrientation=plane_orientation,
			baseInertialFramePosition=[0.6, 0, 1.3]
		)

		# Wood texture for plane
		texture = p.loadTexture(os.path.join(self.path, 'plane', 'texture.jpeg'))
		p.changeVisualShape(plane, -1, textureUniqueId=texture)

		hole_visual = p.createVisualShape(
			p.GEOM_MESH,
			rgbaColor=[25, 0, 0, 1],
			visualFramePosition=hole_position,
			fileName=os.path.join(self.path, 'plane', 'targetHole0_02.obj')
		)

		hole = p.createMultiBody(
			basePosition=[0, 0, 1.3],
			baseVisualShapeIndex=hole_visual,
			baseOrientation=plane_orientation
		)

	def _get_camera_img(self):
		view_matrix1 = p.computeViewMatrix(
			cameraEyePosition=[0, 0, 2.5],
			cameraTargetPosition=[0, 0, 0],
			cameraUpVector=[1, 0, 0]
		)

		view_matrix2 = p.computeViewMatrix(
			cameraEyePosition=[-0.2, 1.5, 1.3],
			cameraTargetPosition=[-0.2, 0, 1.4],
			cameraUpVector=[0, 1, 0]
		)

		projection_matrix1 = p.computeProjectionMatrixFOV(
			fov=30,
			aspect=1.0,
			nearVal=0.01,
			farVal=2
		)

		projection_matrix2 = p.computeProjectionMatrixFOV(
			fov=40,
			aspect=1.0,
			nearVal=0.01,
			farVal=2
		)

		width1, height1, rgb_img1, depth_img1, seg_img1 = p.getCameraImage(
			width=200,
			height=200,
			viewMatrix=view_matrix1,
			projectionMatrix=projection_matrix1
		)

		width2, height2, rgb_img2, depth_img2, seg_img2 = p.getCameraImage(
			width=200,
			height=200,
			viewMatrix=view_matrix2,
			projectionMatrix=projection_matrix2
		)

	def render(self, mode='human'):
		if mode == 'human':
			self.isRender = True
		elif mode == 'rgb_array':
			pass

	def step(self, action):
		action = action[0]
		action = np.clip(action, -1, 1).astype(np.float32)

		# Map to appropriate range according to joint limits
		for i in range(self.num_arm_joints):
			action[i] = self._map_linear(action[i], self.joints_min[i], self.joints_max[i])

		joint_ind = list(range(self.num_arm_joints))
		p.setJointMotorControlArray(self.arm_id, joint_ind, p.POSITION_CONTROL, targetPositions=action)

	def _map_linear(self, val, minimum, maximum):
		old_range = 1 - (-1)
		new_range = maximum - minimum
		new_value = (((val - (-1)) * new_range) / old_range) + minimum
		return new_value




env = PandaDrillerEnv(render=True)
while 1:
	time.sleep(0.1)
	# env._get_camera_img()

