import gym
import time
import pybullet as p
import pybullet_data
from gym.utils import seeding
import numpy as np

class PandaGripperEnv(gym.Env):
	metadata = {'render.modes': ['human', 'rgb_array'],
				'video.frames_per_second': 60}

	def __init__(self, render, first_person_view=False, num_of_cubes=10):
		self.is_render = render
		self.num_of_cubes = num_of_cubes
		self.first_person_view = first_person_view
		self.movement_penalty_constant = 10
		self.seed()
		self.reset()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		# Connect to physics client
		if self.is_render:
			p.connect(p.GUI)
		else:
			p.connect(p.DIRECT)

		p.resetSimulation()
		p.setGravity(0, 0, -9.8)
		p.setRealTimeSimulation(True)

		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		table_orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])


		self.plane_id = p.loadURDF('plane.urdf')
		self.arm_id = p.loadURDF('franka_panda/panda.urdf', [0, 0, 0.94], useFixedBase=True)
		self.table_id = p.loadURDF('table/table.urdf', [0.5, 0, 0], table_orientation, globalScaling=1.5, useFixedBase=True)

		self.from_tray_id = p.loadURDF('tray/traybox.urdf', [0.5, 0.4, 0.94])
		self.to_tray_id = p.loadURDF('tray/traybox.urdf', [0.5, -0.4, 0.94])

		# bounding box of from tray and to tray
		self.from_tray_min_AABB, self.from_tray_max_AABB = p.getAABB(self.from_tray_id)
		self.to_tray_min_AABB, self.to_tray_max_AABB = p.getAABB(self.to_tray_id)

		self.cubes_id = []
		self.other_id = [self.table_id, self.arm_id, self.from_tray_id, self.to_tray_id]

		self.num_of_joints = p.getNumJoints(self.arm_id)
		self.limits = self.get_limits()
		p.setJointMotorControlArray(self.arm_id, list(range(self.num_of_joints)), controlMode=p.POSITION_CONTROL, targetPositions=[0]*self.num_of_joints)

		# time.sleep(20)

		for i in range(self.num_of_cubes):
			cubeOrientation = p.getQuaternionFromEuler([np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)])
			self.cubes_id.append(p.loadURDF('cube_small.urdf', [np.random.uniform(0.3, 0.7), np.random.uniform(0.2, 0.6), 2], cubeOrientation))

		time.sleep(1)

		# for i in range(self.num_of_joints):
		# 	print(p.getJointInfo(self.arm_id, i))

		self.init_up_vector = (1, 0, 0)
		self.init_camera_vector = (0, 0, 1)

		fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
		self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)

		if self.first_person_view:
			pass

		else:
			self.view_matrix = p.computeViewMatrix((0.5,0,2.5),(0.5,0,0.94), self.init_up_vector)
			img = p.getCameraImage(1000, 1000, self.view_matrix, self.projection_matrix)

		# p.getCameraImage(640, 480)

		return img

	def get_limits(self):
		limits = []

		for i in range(self.num_of_joints):
			#print(p.getJointInfo(arm_id, i)[8:10])
			limits.append(p.getJointInfo(self.arm_id, i)[8:10])

		return limits

	def render(self, mode='human'):
		if mode == 'human':
			self.is_render = True
		elif mode == 'rgb_array':
			pass

	def step(self, action):
		action = np.clip(action, -1, 1).astype(np.float32)

		for i in range(self.num_of_joints):
			action[i] = self._linear_map(action[i], -1, 1, self.limits[i][0], self.limits[i][1])

		rew, done = self._compute_reward(action)

		# execute action
		p.setJointMotorControlArray(self.arm_id, list(range(self.num_of_joints)), controlMode=p.POSITION_CONTROL, targetPositions=action)
		# time.sleep(1)

		if self.first_person_view:
			pass
		else:
			img = p.getCameraImage(100, 100, self.view_matrix, self.projection_matrix)

		return img, rew, done

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
		reward = self.num_of_cubes - len(overlapping_cubes_with_from_tray) + len(overlapping_cubes_with_to_tray)

		if reward == 20:
			done = 1

		current_joint_pos = np.array(p.getJointStates(self.arm_id, list(range(self.num_of_joints))))[:, 0]

		movement_penalty = np.absolute(action - current_joint_pos).sum(axis=0) * self.movement_penalty_constant

		reward -= movement_penalty

		return reward, done


	def _linear_map(self, x, in_min, in_max, out_min, out_max):
		return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


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
	img, rew, done = env.step(act)
	print(rew, done)
