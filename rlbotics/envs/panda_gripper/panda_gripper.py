import os
import gym
import time
import math
import cv2 as cv
import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
from gym import spaces
from gym.utils import seeding

from rlbotics.envs.common.domain_randomizer import DomainRandomizer


class PandaGripperEnv(gym.Env):
	metadata = {'render.modes': ['human', 'rgb', 'rgbd', 'rgbds'],
				'video.frames_per_second': 60}

	def __init__(self, render=False, num_of_parts=5, obs_mode='rgb'):
		self.path = os.path.abspath(os.path.dirname(__file__))
		self.obs_mode = obs_mode
		self.max_timesteps = 1000
		self.timestep = 0
		self.done = False

		self.num_of_parts = num_of_parts
		self.movement_penalty_constant = 10

		# Connect to physics client
		p.connect(p.GUI) if render else p.connect(p.DIRECT)

		p.setAdditionalSearchPath(pybullet_data.getDataPath())

		# load urdfs
		table_orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])

		self.tray_1_id = None
		self.tray_2_id = None

		self.plane_id = p.loadURDF('plane.urdf')
		self.arm_id = p.loadURDF('franka_panda/panda.urdf', [0, 0, 0.94], useFixedBase=True)
		self.table_id = p.loadURDF('table/table.urdf', [0.5, 0, 0], table_orientation, globalScaling=1.5, useFixedBase=True)

		self.parts_id = []
		self.other_id = [self.table_id, self.arm_id]

		# load data for parts (mass, scale)
		self.parts_data = pd.read_csv(os.path.join(self.path, 'parts', 'parts_data.csv'))

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
			self.observation_space = spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8)
		elif self.obs_mode == 'rgbd':
			self.observation_space = spaces.Box(0.01, 1000, shape=(224, 224, 4), dtype=np.uint16)
		elif self.obs_mode == 'rgbds':
			self.observation_space = spaces.Box(0.01, 1000, shape=(224, 224, 5), dtype=np.uint16)

		# Initialise env
		self.seed()
		self.domain_randomizer = DomainRandomizer(self.np_random)
		#self.reset()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		self.timestep = 0
		self.done = False

		# remove old parts
		if self.tray_1_id is not None:
			p.removeBody(self.tray_1_id)
		if self.tray_2_id is not None:
			p.removeBody(self.tray_2_id)

		# self.tray_1_pos = [0.5, 0.4, 0.94]
		# self.tray_2_pos = [0.5, -0.4, 0.94]

		self.tray_1_pos = [np.random.uniform(0.1, 0.9), np.random.uniform(0.3, 0.7), 0.94]
		self.tray_2_pos = [np.random.uniform(0.1, 0.9), np.random.uniform(-0.3, -0.7), 0.94]

		self.tray_1_id = p.loadURDF('tray/traybox.urdf', self.tray_1_pos)
		self.tray_2_id = p.loadURDF('tray/traybox.urdf', self.tray_2_pos)

		p.changeVisualShape(self.tray_1_id, -1, rgbaColor=np.hstack((np.random.rand(3), 1)))
		p.changeVisualShape(self.tray_2_id, -1, rgbaColor=np.hstack((np.random.rand(3), 1)))

		# bounding box of from tray and to tray
		self.tray_1_min_AABB, self.from_tray_max_AABB = p.getAABB(self.tray_1_id)
		self.tray_2_min_AABB, self.to_tray_max_AABB = p.getAABB(self.tray_2_id)

		for part in self.parts_id:
			p.removeBody(part)

		self.parts_id = []
		self.other_id.append(self.tray_1_id)
		self.other_id.append(self.tray_2_id)

		p.setGravity(0, 0, -9.8)
		p.setRealTimeSimulation(True)

		p.setJointMotorControlArray(self.arm_id, list(range(self.num_joints)), controlMode=p.POSITION_CONTROL, targetPositions=[0]*self.num_joints)

		# add the random objects in tray 1
		for i in range(self.num_of_parts):
			self._add_random_object()
			time.sleep(0.1)

		# add cubes in tray 1
		# for i in range(self.num_of_parts):
		# 	cubeOrientation = p.getQuaternionFromEuler([np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)])
		# 	self.parts_id.append(p.loadURDF('cube_small.urdf', [np.random.uniform(self.tray_1_pos[0] - 0.2, self.tray_1_pos[0] + 0.2), np.random.uniform(self.tray_1_pos[1] - 0.2, self.tray_1_pos[1] + 0.2), 2], cubeOrientation))

		p.setRealTimeSimulation(False)

		new_obs = self.render(mode=self.obs_mode)

		return new_obs

	def step(self, action, camera_mode='rgb'):
		self.timestep += 1
		action = np.clip(action, -1, 1).astype(np.float32)

		# Map to appropriate range according to joint joint_limits
		# And get relative action
		joint_angles = self._map_linear(action)

		rew = self._compute_reward(joint_angles)

		# execute action in the next few lines
		rel_joint_angles = (joint_angles.T - self.joint_states).T

		# Compute max time needed to complete motion
		max_time = np.max(rel_joint_angles.T / self.velocity_limits)
		max_time = math.ceil(max_time / (1 / 240))

		delta_joint_angles = rel_joint_angles / max_time
		joint_idx = list(range(self.num_joints))

		for t in range(max_time):
			target_pos = np.squeeze(self.joint_states + delta_joint_angles.T * t)
			p.setJointMotorControlArray(self.arm_id, joint_idx, p.POSITION_CONTROL, targetPositions=target_pos)
			p.stepSimulation()
			time.sleep(1/240)

		# Update joint states
		for j in range(self.num_joints):
			self.joint_states[j] = p.getJointState(self.arm_id, j)[0]

		new_obs = self.render(mode=self.obs_mode)

		return new_obs, rew, self.done

	def render(self, mode='human'):
		rgb_img, depth_img, seg_img = self._get_camera_img()

		# remove alpha channel from rgba image

		if mode == 'human':
			pass

		elif mode == 'rgb':
			obs = rgb_img

		elif mode == 'rgbd':
			obs = np.dstack((rgb_img, depth_img))

		elif mode == 'rgbds':
			obs = np.dstack((rgb_img, depth_img, seg_img))

		else:
			print("bad input")
			obs = rgb_img

		return obs

	def close(self):
		p.disconnect()

	def _map_linear(self, joint_angles):
		for i in range(self.num_joints):
			val = joint_angles[i]
			minimum = self.joint_limits[i][0]
			maximum = self.joint_limits[i][1]
			joint_angles[i] = (((val - (-1)) * (maximum - minimum)) / (1 - (-1))) + minimum
		return joint_angles

	def _add_random_object(self):
		object_num = np.random.randint(7)
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
			basePosition=[np.random.uniform(self.tray_1_pos[0] - 0.1, self.tray_1_pos[0] + 0.1), np.random.uniform(self.tray_1_pos[1] - 0.1, self.tray_1_pos[1] + 0.1), 1.5],
			baseVisualShapeIndex=object_visual,
			baseCollisionShapeIndex=object_collision,
			baseOrientation=object_orientation,
			baseMass=object_mass
		))

	def _get_camera_img(self, domain_randomization=True):
		view_matrix = p.computeViewMatrix(
			cameraEyePosition=[0.5, 0, 2.5],
			cameraTargetPosition=[0.5, 0, 0.94],
			cameraUpVector=[1, 0, 0]
		)

		projection_matrix = p.computeProjectionMatrixFOV(
			fov=70,
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

		rgb_img = rgba_img[:,:,:3]

		if domain_randomization:
			rgb_img = self.domain_randomizer.randomize_lighting(rgb_img)

		# cv.imshow("img", rgb_img)
		# cv.waitKey(0)

		return rgb_img, depth_img, seg_img

	def _compute_reward(self, action):
		# idxs of all overlapping objects with from tray and to tray including arm etc
		all_overlapping_objects_with_tray_1 = np.array(p.getOverlappingObjects(self.tray_1_min_AABB, self.from_tray_max_AABB))[:, 0]
		all_overlapping_objects_with_tray_2 = np.array(p.getOverlappingObjects(self.tray_2_min_AABB, self.to_tray_max_AABB))[:, 0]

		# idxs of overlapping object with from tray and to tray
		overlapping_objects_with_tray_1 = [id for id in all_overlapping_objects_with_tray_1 if id not in self.other_id]
		overlapping_objects_with_tray_2 = [id for id in all_overlapping_objects_with_tray_2 if id not in self.other_id]

		# print("num of cubes in from tray is ", len(overlapping_objects_with_tray_1))
		# print("num of cubes in to tray is ", len(overlapping_objects_with_tray_2))

		# reward = number of cubes outside from tray + number of cubes in to tray
		reward = self.num_of_parts - len(overlapping_objects_with_tray_1) + len(overlapping_objects_with_tray_2)

		if reward == self.num_of_parts * 2 or self.timestep >= self.max_timesteps:
			self.done = True

			# deduct big amount if max_timesteps exceeded
			if self.timestep >= self.max_timesteps:
				reward -= 200

		# movement penalty
		current_joint_pos = np.array(p.getJointStates(self.arm_id, list(range(self.num_joints))))[:, 0]

		# penalty = sum(abs(current - target)) * some contstant
		movement_penalty = np.absolute(action - current_joint_pos).sum(axis=0) * self.movement_penalty_constant

		reward -= movement_penalty

		return reward


env = PandaGripperEnv(render=True)

for epoch in range(10):
	print("epoch : ", epoch)
	print("resetting")
	obs = env.reset()

	for i in range(100):
		# act = np.random.uniform(-1, 1, 12)
		act = env.action_space.sample()
		obs, rew, done = env.step(act)

		print(obs[0].shape, rew, done)
		if done == True:
			print("done")
			break

env.close()
