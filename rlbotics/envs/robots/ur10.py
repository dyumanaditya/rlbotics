import os
import time
import math
import numpy as np
import pybullet as p

from rlbotics.envs.common.utils import draw_frame


class UR10:
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None):
		self.physics_client = physics_client
		self.robot_path = os.path.abspath(os.path.join('..', 'models', 'robots', 'ur10', 'ur10.urdf'))
		self.gripper_path = os.path.abspath(os.path.join('..', 'models', 'gripper', 'robotiq_2f_85', 'robotiq_2f_85.urdf'))
		flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
		self.robot_id = p.loadURDF(self.robot_path, base_pos, base_orn, useFixedBase=True, flags=flags,
								   physicsClientId=self.physics_client)




ps = p.connect(p.GUI)
arm = UR10(ps, [0,0,0], [1,0,0,0])

