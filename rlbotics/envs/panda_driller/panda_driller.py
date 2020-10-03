import gym
import math
import time
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
		# p.setGravity(0, 0, -9.8)

		# Load Robots
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		tableOrientation = p.getQuaternionFromEuler([0, 0, math.pi/2])
		drillOrientation = p.getQuaternionFromEuler([0, -math.pi/2, math.pi])

		self.planeId = p.loadURDF('plane.urdf')
		self.armId = p.loadURDF('franka_panda/panda.urdf', [0, 0, 0.93], useFixedBase=True)
		self.drillId = p.loadURDF('drill/drill.urdf', [0.335, 0, 1.73], drillOrientation, globalScaling=0.013)
		self.tableId = p.loadURDF('table/table.urdf', [0.5, 0, 0], tableOrientation, globalScaling=1.5, useFixedBase=True)

		time.sleep(5)
		p.setRealTimeSimulation(1)
		self._grab_drill()
		p.setRealTimeSimulation(0)

	def _grab_drill(self):
		time.sleep(0.5)
		p.setJointMotorControl2(self.armId, 5, p.POSITION_CONTROL, targetPosition=1)
		p.setJointMotorControl2(self.armId, 3, p.POSITION_CONTROL, targetPosition=-1)
		p.setJointMotorControl2(self.armId, 6, p.POSITION_CONTROL, targetPosition=0.8)
		p.setJointMotorControl2(self.armId, 9, p.POSITION_CONTROL, targetPosition=0.5)
		p.setJointMotorControl2(self.armId, 10, p.POSITION_CONTROL, targetPosition=0.5)

		time.sleep(0.5)
		p.setJointMotorControl2(self.armId, 9, p.POSITION_CONTROL, targetPosition=-0.5)
		p.setJointMotorControl2(self.armId, 10, p.POSITION_CONTROL, targetPosition=-0.5)

		time.sleep(0.5)
		p.setGravity(0, 0, -9.8)
		p.setJointMotorControl2(self.armId, 3, p.POSITION_CONTROL, targetPosition=1)
		time.sleep(0.5)


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



env = PandaDrillerEnv(render=True)
while 1:
	time.sleep(0.1)

# act = [12, 20] * 6
# for i in range(100):
# 	time.sleep(0.1)
# 	env.step(act)
