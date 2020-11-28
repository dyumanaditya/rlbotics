import os
import numpy as np
import pybullet as p
import pybullet_data

from rlbotics.envs.robots.panda import Panda


class DrillerEnv:
	def __init__(self, render, robot):
		self.physics_client = p.connect(p.GUI) if render else p.connect(p.DIRECT)
		self.path = os.path.abspath(os.path.dirname(__file__))

		# Load Robot and other objects
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		arm_base_pos = [-0.6, 0, 0.93]
		arm_base_orn = p.getQuaternionFromEuler([0, 0, 0])
		self.drill_base_pos = [-0.12, 0, 1.601]
		self.drill_orientation = p.getQuaternionFromEuler([0, -np.pi/2, np.pi])
		self.drill_bit_vector = np.array([0, 0, -1])
		table_orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])

		p.loadURDF('plane.urdf')
		if robot == 'panda':
			self.arm = Panda(self.physics_client, arm_base_pos, arm_base_orn)

		self.table_id = p.loadURDF('table/table.urdf', [0, 0, 0], table_orientation, globalScaling=1.5, useFixedBase=True)
		self.drill_id = p.loadURDF(os.path.join(self.path, 'drill', 'drill.urdf'), self.drill_base_pos, self.drill_orientation, globalScaling=0.013)

		self.hole = -1
		self.plane = -1

	def reset(self):
		self.arm.reset()
		p.removeBody(self.hole)
		p.removeBody(self.plane)
		self.grab_drill()
		self.generate_plane()
		# TODO: add drill resetting

	def grab_drill(self):
		pass

	def update_drill_vector(self):
		pass

	def generate_plane(self):
		pass

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

		return [(rgb_img1, rgb_img2), (depth_img1, depth_img2), (seg_img1, seg_img2)]




