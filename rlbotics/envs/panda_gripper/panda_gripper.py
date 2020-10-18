import os
import gym
import time
import math
import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
from gym import spaces
from gym.utils import seeding


class PandaGripperEnv(gym.Env):
	metadata = {'render.modes': ['human', 'rgb_array'],
				'video.frames_per_second': 60}

	def __init__(self, render, first_person_view=True, num_of_parts=5, obs_mode='rgb'):
		self.path = os.path.abspath(os.path.dirname(__file__))
		self.num_of_parts = num_of_parts
		self.first_person_view = first_person_view
		self.movement_penalty_constant = 10
		self.steps_taken = 0
		self.obs_mode = obs_mode
		self.max_timesteps = 5
		self.done = False

		# Connect to physics client
		p.connect(p.GUI) if render else p.connect(p.DIRECT)

		p.setAdditionalSearchPath(pybullet_data.getDataPath())

		# load urdfs
		table_orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])

		self.plane_id = p.loadURDF('plane.urdf')
		self.arm_id = p.loadURDF('franka_panda/panda.urdf', [0, 0, 0.94], useFixedBase=True)
		self.table_id = p.loadURDF('table/table.urdf', [0.5, 0, 0], table_orientation, globalScaling=1.5, useFixedBase=True)

		self.from_tray_pos = [0.5, 0.4, 0.94]
		self.to_tray_pos = [0.5, -0.4, 0.94]

		self.from_tray_id = p.loadURDF('tray/traybox.urdf', self.from_tray_pos)
		self.to_tray_id = p.loadURDF('tray/traybox.urdf', self.to_tray_pos)

		# bounding box of from tray and to tray
		self.from_tray_min_AABB, self.from_tray_max_AABB = p.getAABB(self.from_tray_id)
		self.to_tray_min_AABB, self.to_tray_max_AABB = p.getAABB(self.to_tray_id)

		self.parts_id = []
		self.other_id = [self.table_id, self.arm_id, self.from_tray_id, self.to_tray_id]

		# load data for parts (mass, scale)
		self.parts_data = pd.read_csv(os.path.join(self.path, 'parts', 'parts_data.csv'))

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
		self.steps_taken = 0
		self.done = False

		p.setGravity(0, 0, -9.8)
		p.setRealTimeSimulation(True)

		p.setJointMotorControlArray(self.arm_id, list(range(self.num_arm_joints)), controlMode=p.POSITION_CONTROL, targetPositions=[0]*self.num_arm_joints)

		# time.sleep(20)
		for i in range(self.num_of_parts):
			self._add_random_object()
			time.sleep(1)

		# for i in range(self.num_of_parts):
		# 	cubeOrientation = p.getQuaternionFromEuler([np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)])
		# 	self.parts_id.append(p.loadURDF('cube_small.urdf', [np.random.uniform(self.from_tray_pos[0] - 0.2, self.from_tray_pos[0] + 0.2), np.random.uniform(self.from_tray_pos[1] - 0.2, self.from_tray_pos[1] + 0.2), 2], cubeOrientation))

		time.sleep(1)
		p.setRealTimeSimulation(False)

		img = self._get_camera_img()

		return img

	def _add_random_object(self):
		object_num = np.random.randint(8)
		# object_num = 0

		object_scale = self.parts_data.loc[object_num, 'scale']
		object_mass = self.parts_data.loc[object_num, 'mass']
		object_orientation = p.getQuaternionFromEuler([np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)])

		object_visual = p.createVisualShape(
			p.GEOM_MESH,
			meshScale=[object_scale] * 3,
			fileName=os.path.join(self.path, 'parts', str(object_num) + '.obj'),
			rgbaColor=np.hstack((np.random.rand(3), 1))
		)

		object_collision = p.createCollisionShape(
			p.GEOM_MESH,
			meshScale=[object_scale] * 3,
			fileName=os.path.join(self.path, 'parts', str(object_num) + '.obj')
		)

		self.parts_id.append(p.createMultiBody(
			basePosition=[np.random.uniform(self.from_tray_pos[0] - 0.1, self.from_tray_pos[0] + 0.1), np.random.uniform(self.from_tray_pos[1] - 0.1, self.from_tray_pos[1] + 0.1), 1.5],
			baseVisualShapeIndex=object_visual,
			baseCollisionShapeIndex=object_collision,
			baseOrientation=object_orientation,
			baseMass=object_mass
		))

	def _get_camera_img(self):
		if self.first_person_view:
			# Center of mass position and orientation (of link-7)
			pos, ori, _, _, _, _ = p.getLinkState(self.arm_id, 11, computeForwardKinematics=True)

			rot_matrix = p.getMatrixFromQuaternion(ori)
			rot_matrix = np.array(rot_matrix).reshape(3, 3)

			# Initial vectors: z-axis, y-axis
			init_camera_vector = (0, 0, 1)
			init_camera_up_vector = (0, 1, 0)

			# Rotated vectors
			camera_vector = rot_matrix.dot(init_camera_vector)
			camera_up_vector = rot_matrix.dot(init_camera_up_vector)

			projection_matrix = p.computeProjectionMatrixFOV(
				fov=60,
				aspect=1.0,
				nearVal=0.01,
				farVal=100
			)

			view_matrix = p.computeViewMatrix(
				cameraEyePosition=pos,
				cameraTargetPosition=pos + 0.1 * camera_vector,
				cameraUpVector=camera_up_vector
			)

			_, _, rgba_img, depth_img, seg_img = p.getCameraImage(
				width=224,
				height=224,
				viewMatrix=view_matrix,
				projectionMatrix=projection_matrix
			)

		else:
			view_matrix = p.computeViewMatrix(
				cameraEyePosition=[0.5, 0, 2.5],
				cameraTargetPosition=[0.5, 0, 0.94],
				cameraUpVector=[1, 0, 0]
			)

			projection_matrix = p.computeProjectionMatrixFOV(
				fov=60,
				aspect=1.0,
				nearVal=0.01,
				farVal=100
			)

			_, _, rgba_img, depth_img, seg_img = p.getCameraImage(
				width=224,
				height=224,
				viewMatrix=view_matrix,
				projectionMatrix=projection_matrix
			)

		# Remove alpha channel
		rgb_img = rgba_img[:,:,:3]

		if self.obs_mode == 'rgb':
			obs = rgb_img
		elif self.obs_mod == 'rgbd':
			obs = np.dstack((rgb_img, depth_img))
		elif self.obs_mod == 'rgbds':
			obs = np.dstack((rgb_img, depth_img, seg_img))
		else:
			print("bad input")
			obs = rgb_img

		return obs

	def render(self, mode='human'):
		if mode == 'human':
			self.is_render = True
		elif mode == 'rgb_array':
			pass

	def step(self, action, camera_mode='rgb'):
		self.steps_taken += 1

		action = np.clip(action, -1, 1).astype(np.float32)
		action = self._map_linear(action)

		rew, self.done = self._compute_reward(action)

		# execute action in the next few lines
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

		obs = self._get_camera_img()

		return obs, rew, self.done

	def _compute_reward(self, action):
		done = 0

		# ids of overlapping objects with from tray and to tray
		overlapping_objects_with_from_tray = np.array(p.getOverlappingObjects(self.from_tray_min_AABB, self.from_tray_max_AABB))[:, 0]
		overlapping_objects_with_to_tray = np.array(p.getOverlappingObjects(self.to_tray_min_AABB, self.to_tray_max_AABB))[:, 0]

		# ids of overlapping cubes with from tray and to tray
		overlapping_cubes_with_from_tray = [id for id in overlapping_objects_with_from_tray if id not in self.other_id]
		overlapping_cubes_with_to_tray = [id for id in overlapping_objects_with_to_tray if id not in self.other_id]

		# print("num of cubes in from tray is ", len(overlapping_cubes_with_from_tray))
		# print("num of cubes in to tray is ", len(overlapping_cubes_with_to_tray))

		# reward = number of cubes outside from tray + number of cubes in to tray
		reward = self.num_of_parts - len(overlapping_cubes_with_from_tray) + len(overlapping_cubes_with_to_tray)

		if reward == 20 or self.steps_taken >= self.max_timesteps:
			done = 1
			# TODO: deduct large reard if max steps exceeded

		# movement penalty
		current_joint_pos = np.array(p.getJointStates(self.arm_id, list(range(self.num_arm_joints))))[:, 0]

		# penalty = sum(abs(current - target)) * some contstant
		movement_penalty = np.absolute(action - current_joint_pos).sum(axis=0) * self.movement_penalty_constant

		reward -= movement_penalty

		return reward, done


	def _map_linear(self, action):
		for i in range(self.num_arm_joints):
			val = action[i]
			minimum = self.joint_limits[i][0]
			maximum = self.joint_limits[i][1]
			action[i] = (((val - (-1)) * (maximum - minimum)) / (1 - (-1))) + minimum
		return action

env = PandaGripperEnv(render=True)

# while 1:
# 	time.sleep(0.1)

# for i in range(12):
# 	positionMin = [0] * 12
# 	positionMin[i] = env.limits[i][0]
# 	img, rew, done = env.step(positionMin)
# 	time.sleep(1)
# 	print(rew, done)
#
# 	positionMax = [0] * 12
# 	positionMax[i] = env.limits[i][1]
# 	img, rew, done = env.step(positionMax)
# 	time.sleep(1)
# 	print(rew, done)
#
# 	positionZero = [0] * 12
# 	img, rew, done = env.step(positionZero)
# 	time.sleep(1)
# 	print(rew, done)

for i in range(100):
	act = np.random.uniform(-1, 1, 12)
	obs, rew, done = env.step(act)
	print(obs.shape, rew, done)
