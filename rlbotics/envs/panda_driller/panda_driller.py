import os
import gym
import time
import math
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from gym.utils import seeding


class PandaDrillerEnv(gym.Env):
	metadata = {'render.modes': ['human', 'rgb'],
				'video.frames_per_second': 50}

	def __init__(self, render=False, obs_mode='rgb'):
		self.path = os.path.abspath(os.path.dirname(__file__))
		self.is_render = render
		self.obs_mode = obs_mode
		self.max_timesteps = 1000
		self.timesteps = 0
		self.done = False

		# Connect to physics client
		p.connect(p.GUI) if self.is_render else p.connect(p.DIRECT)

		# Load Robot and other objects
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		arm_base_pos = [-0.6, 0, 0.93]
		self.drill_base_pos = [-0.24, 0, 1.69]
		self.drill_orientation = p.getQuaternionFromEuler([0, -np.pi/2, np.pi])
		table_orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])

		p.loadURDF('plane.urdf')
		self.hole = -1
		self.plane = -1
		self.arm_id = p.loadURDF('franka_panda/panda.urdf', arm_base_pos, useFixedBase=True)
		self.table_id = p.loadURDF('table/table.urdf', [0, 0, 0], table_orientation, globalScaling=1.5, useFixedBase=True)
		self.drill_id = p.loadURDF(os.path.join(self.path, 'drill', 'drill.urdf'), self.drill_base_pos, self.drill_orientation, globalScaling=0.013)

		# Initialise joint info
		self.joint_states, self.joint_limits, self.velocity_limits = [], [], []
		self.num_arm_joints = p.getNumJoints(self.arm_id)
		for j in range(self.num_arm_joints):
			self.joint_states.append(p.getJointState(self.arm_id, j)[0])
			self.joint_limits.append(p.getJointInfo(self.arm_id, j)[8:10])
			v = p.getJointInfo(self.arm_id, j)[11]
			self.velocity_limits.append(np.inf if v == 0 else v)

		# Initialise environment spaces
		self.action_space = spaces.Box(-1, 1, (self.num_arm_joints,), dtype=np.float32)
		if self.obs_mode == 'rgb':
			self.observation_space = spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8)
		elif self.obs_mode == 'rgbd':
			self.observation_space = spaces.Box(0.01, 1000, shape=(224, 224, 4), dtype=np.uint16)
		elif self.obs_mode == 'rgbds':
			self.observation_space = spaces.Box(0.01, 1000, shape=(224, 224, 5), dtype=np.uint16)

		# Initialise env
		self.seed()
		self.reset()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		self.done = False
		p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

		# Reset joint states
		for j in range(self.num_arm_joints):
			p.resetJointState(self.arm_id, j, 0)
		p.resetBasePositionAndOrientation(self.drill_id, self.drill_base_pos, self.drill_orientation)
		p.removeBody(self.hole)
		p.removeBody(self.plane)

		# Temp!!!
		self._generate_plane()
		print(p.getMatrixFromQuaternion(self.drill_orientation))
		print(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0,0,0])))
		print()
		time.sleep(1000)

		p.setRealTimeSimulation(1)
		self._grab_drill()
		self._generate_plane()
		p.setRealTimeSimulation(0)
		p.setGravity(0, 0, -9.8)

	def _grab_drill(self):
		time.sleep(0.1)
		p.setJointMotorControl2(self.arm_id, 5, p.POSITION_CONTROL, targetPosition=1)
		p.setJointMotorControl2(self.arm_id, 3, p.POSITION_CONTROL, targetPosition=-1)
		p.setJointMotorControl2(self.arm_id, 6, p.POSITION_CONTROL, targetPosition=0.8)
		p.setJointMotorControl2(self.arm_id, 9, p.POSITION_CONTROL, targetPosition=0.04)
		p.setJointMotorControl2(self.arm_id, 10, p.POSITION_CONTROL, targetPosition=0.04)

		time.sleep(0.1)
		p.setJointMotorControl2(self.arm_id, 9, p.POSITION_CONTROL, targetPosition=0)
		p.setJointMotorControl2(self.arm_id, 10, p.POSITION_CONTROL, targetPosition=0)

		time.sleep(0.1)
		p.setJointMotorControl2(self.arm_id, 3, p.POSITION_CONTROL, targetPosition=1)
		time.sleep(0.1)

	def _generate_plane(self):
		# Min Max constraints for drilling on plane
		# min = [-0.2, 0.2, 0]
		# max = [0.2, -0.2, 0]

		plane_orientation = [0, 0, 0]
		#plane_orientation[0] = self.np_random.uniform(0, np.pi/4)
		#plane_orientation[1] = self.np_random.uniform(3*np.pi/4, np.pi)
		#plane_orientation[2] = self.np_random.uniform(0, np.pi/2)
		#plane_orientation = p.getQuaternionFromEuler(plane_orientation)
		plane_scale = [self.np_random.uniform(1, 1.4), self.np_random.uniform(1, 1.4), 1]
		hole_position = [self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(-0.2, 0.2), 0]

		plane_visual = p.createVisualShape(
			p.GEOM_MESH,
			meshScale=plane_scale,
			fileName=os.path.join(self.path, 'plane', 'plane.obj')
		)

		plane_collision = p.createCollisionShape(
			p.GEOM_MESH,
			meshScale=plane_scale,
			fileName=os.path.join(self.path, 'plane', 'plane.obj')
		)

		self.plane = p.createMultiBody(
			basePosition=[0, 0, 1.4],
			baseVisualShapeIndex=plane_visual,
			baseCollisionShapeIndex=plane_collision,
			baseOrientation=plane_orientation,
			baseInertialFramePosition=[0.6, 0, 1.3]
		)

		# Random texture for plane
		tex = self.np_random.randint(0, 20)
		texture = p.loadTexture(os.path.join(self.path, 'plane', 'textures', str(tex)+'.jpg'))
		p.changeVisualShape(self.plane, -1, textureUniqueId=texture)

		hole_visual = p.createVisualShape(
			p.GEOM_MESH,
			rgbaColor=[25, 0, 0, 1],
			visualFramePosition=hole_position,
			fileName=os.path.join(self.path, 'plane', 'targetHole.obj')
		)

		self.hole = p.createMultiBody(
			basePosition=[0, 0, 1.4],
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
			width=224,
			height=224,
			viewMatrix=view_matrix1,
			projectionMatrix=projection_matrix1
		)

		width2, height2, rgb_img2, depth_img2, seg_img2 = p.getCameraImage(
			width=224,
			height=224,
			viewMatrix=view_matrix2,
			projectionMatrix=projection_matrix2
		)
		print(np.shape(seg_img1))
		time.sleep(10)

	def render(self, mode='human'):
		if mode == 'human':
			self.is_render = True
		elif mode == 'rgb_array':
			pass

	def step(self, action):
		self.timesteps += 1
		action = np.clip(action, -1, 1).astype(np.float32)

		# Map to appropriate range according to joint joint_limits
		# And get relative action
		action = self._map_linear(action)
		rel_action = (action.T - self.joint_states).T

		# Compute max time needed to complete motion
		TIME = np.max(rel_action.T / self.velocity_limits)
		TIME = math.ceil(TIME / (1/240))

		sub_action = rel_action / TIME
		joint_ind = list(range(self.num_arm_joints))

		for step in range(TIME):
			target_pos = np.squeeze(self.joint_states + sub_action.T * step)
			p.setJointMotorControlArray(self.arm_id, joint_ind, p.POSITION_CONTROL, targetPositions=target_pos)
			p.stepSimulation()
			time.sleep(1/240)

		# Update joint states
		for j in range(self.num_arm_joints):
			self.joint_states[j] = p.getJointState(self.arm_id, j)[0]

		# reward = self._compute_reward(rel_action)

	def _map_linear(self, action):
		for i in range(self.num_arm_joints):
			val = action[i]
			minimum = self.joint_limits[i][0]
			maximum = self.joint_limits[i][1]
			action[i] = (((val - (-1)) * (maximum - minimum)) / (1 - (-1))) + minimum
		return action

	def _compute_reward(self, rel_action, sparse=False):
		rew = 0

		if sparse:
			pass
		else:
			# Check if drill has dropped
			if len(p.getContactPoints(self.arm_id, self.drill_id)) == 0:
				self.done = True
				rew -= 300
			# Check if drill is touching the table
			elif len(p.getContactPoints(self.drill_id, self.table_id)) != 0 and not self.done:
				rew -= 200
			# Check if Panda is touching the table
			if len(p.getContactPoints(self.arm_id, self.table_id)) != 0:
				rew -= 200
			# Check if Panda is touching the plane
			if len(p.getContactPoints(self.arm_id, self.plane)) != 0:
				rew -= 200

			# Compute electricity cost
			scale_factor = 10
			electricity_cost = np.sum(np.abs(rel_action) * scale_factor)

			#

			if self.timesteps == self.max_timesteps:
				self.done = True

	#def _get_transformation_matrix(self, rotation_matrix, translation):


	def close(self):
		p.disconnect()








env = PandaDrillerEnv(render=True)
#time.sleep(10)
while 1:
	#time.sleep(0.1)

	act = np.random.uniform(-1, 1, (12,1))
	# env.step(act)
	env._get_camera_img()

