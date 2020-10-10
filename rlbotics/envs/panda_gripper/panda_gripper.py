import gym
import time
import pybullet as p
import pybullet_data
from gym.utils import seeding
import numpy as np

class PandaGripperEnv(gym.Env):
	metadata = {'render.modes': ['human', 'rgb_array'],
				'video.frames_per_second': 60}

	def __init__(self, render, firstPersonView=False, numOfCubes=10):
		self.isRender = render
		self.numOfCubes = numOfCubes
		self.firstPersonView = firstPersonView
		self.seed()
		self.reset()

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
		p.setGravity(0, 0, -9.8)
		p.setRealTimeSimulation(True)

		# Load Robots
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		tableOrientation = p.getQuaternionFromEuler([0, 0, np.pi/2])


		self.planeId = p.loadURDF('plane.urdf')
		self.armId = p.loadURDF('franka_panda/panda.urdf', [0, 0, 0.94], useFixedBase=True)
		self.tableId = p.loadURDF('table/table.urdf', [0.5, 0, 0], tableOrientation, globalScaling=1.5, useFixedBase=True)

		self.fromTrayId = p.loadURDF('tray/traybox.urdf', [0.5, 0.4, 0.94])
		self.toTrayId = p.loadURDF('tray/traybox.urdf', [0.5, -0.4, 0.94])

		self.cubesId = []

		for i in range(self.numOfCubes):
			cubeOrientation = p.getQuaternionFromEuler([np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)])
			self.cubesId.append(p.loadURDF('cube_small.urdf', [np.random.uniform(0.3, 0.7), np.random.uniform(0.2, 0.6), 2], cubeOrientation))

		time.sleep(1)

		self.numOfJoints = p.getNumJoints(self.armId)

		for i in range(self.numOfJoints):
			print(p.getJointInfo(self.armId, i))

		self.init_up_vector = (1, 0, 0)
		self.init_camera_vector = (0, 0, 1)

		fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
		self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)

		if self.firstPersonView:
			pass

		else:
			self.view_matrix = p.computeViewMatrix((0.5,0,2.5),(0.5,0,0.94), self.init_up_vector)
			img = p.getCameraImage(1000, 1000, self.view_matrix, self.projection_matrix)

		# p.getCameraImage(640, 480)

		return img

	def render(self, mode='human'):
		if mode == 'human':
			self.isRender = True
		elif mode == 'rgb_array':
			pass

	def step(self, action):
		reward = 0
		done = 0

		for joint in range(self.numOfJoints):
			p.setJointMotorControl2(self.armId, joint, controlMode=p.POSITION_CONTROL, targetPosition=action[joint])

		if self.firstPersonView:
			pass

		else:
			img = p.getCameraImage(1000, 1000, self.view_matrix, self.projection_matrix)


		return img, reward, done


env = PandaGripperEnv(render=True)
# while 1:
# 	time.sleep(0.1)


# for i in range(100):
# 	act = np.random.rand(1, 12).squeeze(0)
# 	time.sleep(0.1)
# 	env.step(act)
