import pybullet as p


class PandaGripper:
	"""
	Gripper specs for Panda robot ONLY. None signifies 'get info from urdf'
	"""
	def __init__(self):
		self.gripper_info = {}
		dof = 2
		ee_idx = 11
		joint_indices = [9, 10]
		joint_lower_limits = None
		joint_upper_limits = None
		joint_ranges = None
		joint_velocity_limits = None
		initial_joint_positions = [0.0, 0.0]

		if ee_idx is None:
			self.create_ee_idx()

		self.gripper_info['dof'] = dof
		self.gripper_info['ee_idx'] = ee_idx
		self.gripper_info['joint_indices'] = joint_indices
		self.gripper_info['joint_lower_limits'] = joint_lower_limits
		self.gripper_info['joint_upper_limits'] = joint_upper_limits
		self.gripper_info['joint_ranges'] = joint_ranges
		self.gripper_info['joint_velocity_limits'] = joint_velocity_limits
		self.gripper_info['initial_pose'] = initial_joint_positions

	def create_ee_idx(self):
		# Create a constraint by adding a fixed joint
		pass

	def open_gripper(self, gripper_id, joint_indices, joint_velocity_limits, physics_client, width=0.08):
		width = min(width, 0.08)
		target = [width/2, width/2]
		for i in range(len(target)):
			p.setJointMotorControl2(gripper_id, joint_indices[i], p.POSITION_CONTROL, target[i],
									maxVelocity=joint_velocity_limits[i], physicsClientId=physics_client)

	def close_gripper(self, gripper_id, joint_indices, joint_velocity_limits, physics_client, width=0.0):
		width = max(width, 0.0)
		target = [width/2, width/2]
		for i in range(len(target)):
			p.setJointMotorControl2(gripper_id, joint_indices[i], p.POSITION_CONTROL, target[i],
									maxVelocity=joint_velocity_limits[i], physicsClientId=physics_client)
