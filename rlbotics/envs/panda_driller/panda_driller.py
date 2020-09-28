import gym
import math
import pybullet as p
import pybullet_data
from gym.utils import seeding


class PandaDrillerEnv(gym.Env):
	metadata = {'render.modes': ['human', 'rgb_array'],
				'video.frames_per_second': 60}

	def __init__(self, render):
		self.isRender = render
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
		p.setGravity(0, 0, 9.8)

		# Load Robots
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		tableOrientation = p.getQuaternionFromEuler([0, 0, math.pi/2])

		self.planeId = p.loadURDF('plane.urdf')
		self.armId = p.loadURDF('franka_panda/panda.urdf', [0, 0, 0.93], useFixedBase=True)
		self.tableId = p.loadURDF('table/table.urdf', [0.5, 0, 0], tableOrientation, globalScaling=1.5, useFixedBase=True)

		#p.setAdditionalSearchPath('/home/Documents/rlbotics/envs/panda_driller/drill')
		self.drillVisualShapeId = p.createVisualShape(p.GEOM_MESH, fileName='drill/drill.obj')
		self.drillCollisionShapeId = p.createCollisionShape(p.GEOM_MESH, fileName='drill.obj')
		self.drillId = p.createMultiBody(baseVisualShapeIndex=self.drillVisualShapeId,
										baseCollisionShapeIndex=self.drillCollisionShapeId)


	def render(self, mode='human'):
		if mode == 'human':
			self.isRender = True
		elif mode == 'rgb_array':
			pass

	def step(self, action):
		pass






		# # Get joint info
		# self.num_joints = p.getNumJoints(self.armId)

	# def step(self, action):
	# 	for joint in range(self.num_joints):
	# 		p.setJointMotorControl2(self.armId, joint, controlMode=p.POSITION_CONTROL, targetPosition=action[joint])
	# 		p.stepSimulation()


import time
env = PandaDrillerEnv(render=True)
time.sleep(1000)
# act = [12, 20] * 6
# for i in range(100):
# 	time.sleep(0.1)
# 	env.step(act)
