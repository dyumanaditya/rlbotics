import gym
import math
import numpy as np
import pybullet as p
from gym import spaces
from gym.utils import seeding

from rlbotics.envs.common.domain_randomizer import DomainRandomizer
from rlbotics.envs.worlds.driller_world.driller_world import DrillerWorld


class DrillerGym(DrillerWorld, gym.Env):
	metadata = {'render.modes': ['rgb', 'rgbd', 'rgbds'],
				'video.frames_per_second': 50}

	def __init__(self, robot, render=False, obs_mode='rgb', domain_randomization=True):
		super().__init__(robot, render)
		self.domain_randomization = domain_randomization
		self.obs_mode = obs_mode
		self.max_timesteps = 1000
		self.timestep = 0
		self.done = False

		# Initialise environment spaces
		self.action_space = spaces.Box(-1, 1, (self.arm.num_joints,), dtype=np.float32)
		if self.obs_mode == 'rgb':
			self.observation_space = spaces.Box(0, 255, shape=(2, 224, 224, 3), dtype=np.uint8)
		elif self.obs_mode == 'rgbd':
			self.observation_space = spaces.Box(0.01, 1000, shape=(2, 224, 224, 4), dtype=np.uint16)
		elif self.obs_mode == 'rgbds':
			self.observation_space = spaces.Box(0.01, 1000, shape=(2, 224, 224, 5), dtype=np.uint16)

		# Initialise env
		self.seed()
		self.domain_randomizer = DomainRandomizer(self.np_random)
		self.reset()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		self.done = False
		self.timestep = 0
		p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

		# TODO: Reset drill position

		self.arm.reset()
		p.removeBody(self.hole)
		p.removeBody(self.plane)

		# Randomize physics constraints and drill color
		self.domain_randomizer.randomize_physics_constraints(self.drill_id)
		self.domain_randomizer.randomize_color(self.drill_id)

		self.grab_drill()
		self.generate_plane()

		obs = self.render(mode=self.obs_mode)
		return obs

	def step(self, action):
		self.timestep += 1
		action = np.clip(action, -1, 1).astype(np.float32)

		# Map to appropriate range according to joint joint_limits
		joint_angles = self._map_linear(action)

		# TODO: Complete motion using robot base class

	def render(self, mode='rgb'):
		img = self.get_camera_img()
		img1, img2, dep1, dep2, seg1, seg2 = img[0][0], img[0][1], img[1][0], img[1][1], img[2][0], img[2][1]

		if mode == 'rgb':
			return img1, img2

		elif mode == 'rgbd':
			return np.dstack((img1, dep1)), np.dstack((img2, dep2))

		elif mode == 'rgbds':
			return np.dstack((img1, dep1, seg1)), np.dstack((img2, dep2, seg2))

	def close(self):
		p.disconnect()

	def get_camera_img(self):
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

		# Remove alpha channel
		rgb_img1, rgb_img2 = rgb_img1[:, :, :3], rgb_img2[:, :, :3]
		if self.domain_randomization:
			rgb_img1, rgb_img2 = self.domain_randomizer.randomize_lighting(rgb_img1, rgb_img2)

		return [(rgb_img1, rgb_img2), (depth_img1, depth_img2), (seg_img1, seg_img2)]

	def _map_linear(self, joint_angles):
		for i in range(self.arm.num_joints):
			val = joint_angles[i]
			minimum = self.arm.joint_lower_limits[0]
			maximum = self.arm.joint_upper_limits[1]
			joint_angles[i] = (((val - (-1)) * (maximum - minimum)) / (1 - (-1))) + minimum
		return joint_angles

	def _compute_reward(self, delta_joint_angles, sparse=False):
		# TODO: FIX THIS!!
		reward = 0

		if sparse:
			# Check if drill has dropped
			c1 = len(p.getContactPoints(self.arm_id, self.drill_id, linkIndexA=9))
			c2 = len(p.getContactPoints(self.arm_id, self.drill_id, linkIndexA=10))
			if c1 + c2 == 0:
				self.done = True
				reward = -1

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
			c1 = len(p.getContactPoints(self.arm_id, self.drill_id, linkIndexA=9))
			c2 = len(p.getContactPoints(self.arm_id, self.drill_id, linkIndexA=10))
			if c1 + c2 == 0:
				self.done = True
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




