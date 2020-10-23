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
	metadata = {'render.modes': ['human', 'rgb', 'rgbd', 'rgbds'],
				'video.frames_per_second': 50}

	def __init__(self, render=False, obs_mode='rgb'):
		self.path = os.path.abspath(os.path.dirname(__file__))
		self.obs_mode = obs_mode
		self.max_timesteps = 1000
		self.timestep = 0
		self.done = False

		# Connect to physics client
		p.connect(p.GUI) if render else p.connect(p.DIRECT)

		# Load Robot and other objects
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		arm_base_pos = [-0.6, 0, 0.93]
		self.drill_base_pos = [-0.15, 0, 1.6]
		# self.drill_base_pos = [0.2, 0.2, 1.501]
		self.drill_orientation = p.getQuaternionFromEuler([0, -np.pi/2, np.pi])
		self.drill_bit_vector = np.array([0, 0, -1])
		table_orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])

		p.loadURDF('plane.urdf')
		self.hole = -1
		self.plane = -1
		self.arm_id = p.loadURDF('franka_panda/panda.urdf', arm_base_pos, useFixedBase=True)
		self.table_id = p.loadURDF('table/table.urdf', [0, 0, 0], table_orientation, globalScaling=1.5, useFixedBase=True)
		self.drill_id = p.loadURDF(os.path.join(self.path, 'drill', 'drill.urdf'), self.drill_base_pos, self.drill_orientation, globalScaling=0.013)

		# Initialise joint info
		self.joint_states, self.joint_limits, self.velocity_limits = [], [], []
		self.num_joints = p.getNumJoints(self.arm_id)
		for j in range(self.num_joints):
			self.joint_states.append(p.getJointState(self.arm_id, j)[0])
			self.joint_limits.append(p.getJointInfo(self.arm_id, j)[8:10])
			v = p.getJointInfo(self.arm_id, j)[11]
			self.velocity_limits.append(np.inf if v == 0 else v)

		# Initialise environment spaces
		self.action_space = spaces.Box(-1, 1, (self.num_joints,), dtype=np.float32)
		if self.obs_mode == 'rgb':
			self.observation_space = spaces.Box(0, 255, shape=(2, 224, 224, 3), dtype=np.uint8)
		elif self.obs_mode == 'rgbd':
			self.observation_space = spaces.Box(0.01, 1000, shape=(2, 224, 224, 4), dtype=np.uint16)
		elif self.obs_mode == 'rgbds':
			self.observation_space = spaces.Box(0.01, 1000, shape=(2, 224, 224, 5), dtype=np.uint16)

		# Initialise env
		self.seed()
		self.reset()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		self.done = False
		p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
		p.setRealTimeSimulation(1)
		p.setGravity(0, 0, 0)

		# Reset joint states
		p.setJointMotorControlArray(self.arm_id, list(range(self.num_joints)), controlMode=p.POSITION_CONTROL, targetPositions=[0]*self.num_joints)
		p.resetBasePositionAndOrientation(self.drill_id, self.drill_base_pos, self.drill_orientation)
		time.sleep(0.1)
		p.removeBody(self.hole)
		p.removeBody(self.plane)

		self._grab_drill()
		self._generate_plane()
		#p.setRealTimeSimulation(0)
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

	def _update_drill_vector(self):
		current_drill_orientation = np.array(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.drill_id)[1]))
		current_drill_orientation = current_drill_orientation - np.array(p.getEulerFromQuaternion(self.drill_orientation))
		rotation_matrix = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(current_drill_orientation))).reshape(3, 3)
		self.drill_bit_vector = np.dot(rotation_matrix, np.array([0, 0, -1]))

	def _generate_plane(self):
		# Min Max constraints for drilling on plane
		# min = [-0.2, 0.2, 0]
		# max = [0.2, -0.2, 0]

		plane_orientation = [0, 0, 0]
		plane_orientation[0] = self.np_random.uniform(0, np.pi/4)
		plane_orientation[1] = self.np_random.uniform(3*np.pi/4, np.pi)
		plane_orientation[2] = self.np_random.uniform(0, np.pi/2)
		plane_orientation = p.getQuaternionFromEuler(plane_orientation)
		plane_scale = [self.np_random.uniform(1, 1.4), self.np_random.uniform(1, 1.4), 1]
		hole_relative_pos = [self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(-0.2, 0.2), 0]
		# hole_relative_pos = [0.2, 0.2, 0]
		self.hole_pos = np.array(p.multiplyTransforms([0, 0, 1.4], plane_orientation, hole_relative_pos, plane_orientation)[0])

		# Compute plane normal
		rotation_matrix = np.array(p.getMatrixFromQuaternion(plane_orientation)).reshape(3, 3)
		self.plane_normal = np.dot(rotation_matrix, np.array([0, 0, -1]))

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
			baseOrientation=plane_orientation
		)

		# Random texture for plane
		texture_id = self.np_random.randint(0, 20)
		texture = p.loadTexture(os.path.join(self.path, 'plane', 'textures', str(texture_id) + '.jpg'))
		p.changeVisualShape(self.plane, -1, textureUniqueId=texture)

		hole_visual = p.createVisualShape(
			p.GEOM_MESH,
			rgbaColor=[25, 0, 0, 1],
			visualFramePosition=hole_relative_pos,
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

		_, _, rgb_img1, depth_img1, seg_img1 = p.getCameraImage(
			width=224,
			height=224,
			viewMatrix=view_matrix1,
			projectionMatrix=projection_matrix1
		)

		_, _, rgb_img2, depth_img2, seg_img2 = p.getCameraImage(
			width=224,
			height=224,
			viewMatrix=view_matrix2,
			projectionMatrix=projection_matrix2
		)
		return [(rgb_img1, rgb_img2), (depth_img1, depth_img2), (seg_img1, seg_img2)]

	def render(self, mode='human'):
		if mode == 'human':
			pass

		elif mode == 'rgb':
			img = self._get_camera_img()
			img1, img2 = img[0][0], img[0][1]
			return img1[:, :, :3], img2[:, :, :3]

		elif mode == 'rgbd':
			img = self._get_camera_img()
			img1, img2 = img[0][0], img[0][1]
			dep1, dep2 = img[1][0], img[1][1]
			img1, img2 = img1[:, :, :3], img2[:, :, :3]
			return np.dstack((img1, dep1)), np.dstack((img2, dep2))

		elif mode == 'rgbds':
			img = self._get_camera_img()
			img1, img2 = img[0][0], img[0][1]
			dep1, dep2 = img[1][0], img[1][1]
			seg1, seg2 = img[2][0], img[2][1]
			img1, img2 = img1[:, :, :3], img2[:, :, :3]
			return np.dstack((img1, dep1, seg1)), np.dstack((img2, dep2, seg2))

	def step(self, action):
		self.timestep += 1
		action = np.clip(action, -1, 1).astype(np.float32)

		# Map to appropriate range according to joint joint_limits
		# And get relative action
		joint_angles = self._map_linear(action)
		rel_joint_angles = (joint_angles.T - self.joint_states).T

		# Compute max time needed to complete motion
		max_time = np.max(rel_joint_angles.T / self.velocity_limits)
		max_time = math.ceil(max_time / (1 / 240))

		delta_joint_angles = rel_joint_angles / max_time
		joint_idx = list(range(self.num_joints))

		reward = 0
		for t in range(max_time):
			if self.done:
				break
			target_pos = np.squeeze(self.joint_states + delta_joint_angles.T * t)
			p.setJointMotorControlArray(self.arm_id, joint_idx, p.POSITION_CONTROL, targetPositions=target_pos)
			p.stepSimulation()
			reward += self._compute_reward(delta_joint_angles)
			time.sleep(1/240)

		# Update joint states
		for j in range(self.num_joints):
			self.joint_states[j] = p.getJointState(self.arm_id, j)[0]

		new_obs = self.render(mode=self.obs_mode)
		return new_obs, reward, self.done, {}

	def _map_linear(self, joint_angles):
		for i in range(self.num_joints):
			val = joint_angles[i]
			minimum = self.joint_limits[i][0]
			maximum = self.joint_limits[i][1]
			joint_angles[i] = (((val - (-1)) * (maximum - minimum)) / (1 - (-1))) + minimum
		return joint_angles

	def _compute_reward(self, delta_joint_angles, sparse=False):
		reward = 0

		if sparse:
			# Check if drilling task is complete
			self._update_drill_vector()
			cos_theta = np.dot(self.drill_bit_vector, self.plane_normal)/(np.linalg.norm(self.drill_bit_vector)*np.linalg.norm(self.plane_normal))
			theta = math.degrees(math.acos(cos_theta))

			drill_pos = np.array(p.getBasePositionAndOrientation(self.drill_id)[0])
			targ_dist = drill_pos - self.hole_pos
			if theta < 1 and np.linalg.norm(targ_dist) <= 0.101:
				reward = 1
				self.done = True
		else:
			# Check if drill has dropped
			if len(p.getContactPoints(self.arm_id, self.drill_id, linkIndexA=9)) == 0:
				#self.done = True
				reward -= 200
			# Check if drill is touching the table
			elif len(p.getContactPoints(self.drill_id, self.table_id)) != 0 and not self.done:
				reward -= 200
			# Check if Panda is touching the table
			if len(p.getContactPoints(self.arm_id, self.table_id)) != 0:
				reward -= 200
			# Check if Panda is touching the plane
			if len(p.getContactPoints(self.arm_id, self.plane)) != 0:
				reward -= 200

			# Check angle between drill bit and plane normal
			angle_scale = 2
			self._update_drill_vector()
			cos_theta = np.dot(self.drill_bit_vector, self.plane_normal)/(np.linalg.norm(self.drill_bit_vector)*np.linalg.norm(self.plane_normal))
			theta = math.degrees(math.acos(cos_theta))
			reward -= theta * angle_scale

			# Check if drilling task is complete
			dist_scale = 10
			drill_pos = np.array(p.getBasePositionAndOrientation(self.drill_id)[0])
			targ_dist = np.linalg.norm(drill_pos - self.hole_pos)
			reward -= targ_dist * dist_scale
			if theta < 1 and targ_dist <= 0.102:
				reward += 500
				self.done = True

			# Compute electricity cost
			electricity_scale = 1
			electricity_cost = np.sum(np.abs(delta_joint_angles) * electricity_scale)

			# Compute final reward
			reward = reward - electricity_cost

			if self.timestep == self.max_timesteps:
				reward = -1 if sparse and not self.done else reward
				self.timestep = 0
				self.done = True

		return reward

	def close(self):
		p.disconnect()



env = PandaDrillerEnv(render=True)
while 1:
	# time.sleep(0.1)
	#
	#act = env.action_space.sample()
	act = np.zeros((12,1))

	#new_obs, rew, done, info = env.step(act)
	#print(rew, done)
	# if done:
	# 	env.reset()
	# env._get_camera_img()

