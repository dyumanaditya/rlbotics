import os
import time
import math
import numpy as np
import pybullet as p
import pybullet_data

from rlbotics.envs.common.utils import draw_frame


class UR10:
	def __init__(self, physics_client, base_pos, base_orn, initial_arm_joint_positions=None, initial_gripper_joint_positions=None):
		p.setRealTimeSimulation(1)      # SEE ABOUT THIS LATER. This is needed to complete motion		
		self.physics_client = physics_client

		self.path = os.path.abspath(os.path.dirname(__file__))

		self.robot_path = os.path.join(os.path.dirname(self.path), 'models', 'robots', 'universal_robots', 'ur10', 'ur10.urdf')
		self.gripper_path = os.path.join(os.path.dirname(self.path), 'models', 'grippers', 'robotiq', 'robotiq_2f_85', 'robotiq_2f_85.urdf')


		flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
		self.robot_id = p.loadURDF(self.robot_path, base_pos, base_orn, useFixedBase=True, flags=flags,
									physicsClientId=self.physics_client)
		self.gripper_id = p.loadURDF(self.gripper_path, [0, 1, 0], flags=flags,
									physicsClientId=self.physics_client)

		self.arm_num_joints = p.getNumJoints(self.robot_id)
		self.gripper_num_joints = p.getNumJoints(self.gripper_id)

		self.arm_revolute_joint_indices, self.arm_prismatic_joint_indices, self.arm_fixed_joint_indices = [], [], []
		self.gripper_revolute_joint_indices, self.gripper_prismatic_joint_indices, self.gripper_fixed_joint_indices = [], [], []

		self.arm_joint_lower_limits, self.arm_joint_upper_limits, self.arm_joint_ranges = [], [], []
		self.gripper_joint_lower_limits, self.gripper_joint_upper_limits, self.gripper_joint_ranges = [], [], []

		self.arm_velocity_limits = []
		self.gripper_velocity_limits = []

		print("gathering arm joint info: ")

		for joint_idx in range(self.arm_num_joints):
			joint_info = p.getJointInfo(self.robot_id, joint_idx, self.physics_client)
			joint_type = joint_info[2]

			print(joint_idx, joint_type)

			if joint_type == p.JOINT_REVOLUTE:
				self.arm_revolute_joint_indices.append(joint_idx)
			elif joint_type == p.JOINT_PRISMATIC:
				self.arm_prismatic_joint_indices.append(joint_idx)
			elif joint_type == p.JOINT_FIXED:
				self.arm_fixed_joint_indices.append(joint_idx)

			if joint_type != p.JOINT_FIXED:
				self.arm_joint_lower_limits.append(joint_info[8])
				self.arm_joint_upper_limits.append(joint_info[9])
				self.arm_joint_ranges.append(joint_info[9] - joint_info[8])

				self.arm_velocity_limits.append(joint_info[11])

		print("gathering gripper joint info: ")

		for joint_idx in range(self.gripper_num_joints):
			joint_info = p.getJointInfo(self.gripper_id, joint_idx, self.physics_client)
			joint_type = joint_info[2]

			print(joint_idx, joint_type)

			if joint_type == p.JOINT_REVOLUTE:
				self.gripper_revolute_joint_indices.append(joint_idx)
			elif joint_type == p.JOINT_PRISMATIC:
				self.gripper_prismatic_joint_indices.append(joint_idx)
			elif joint_type == p.JOINT_FIXED:
				self.gripper_fixed_joint_indices.append(joint_idx)

			if joint_type != p.JOINT_FIXED:
				self.gripper_joint_lower_limits.append(joint_info[8])
				self.gripper_joint_upper_limits.append(joint_info[9])
				self.gripper_joint_ranges.append(joint_info[9] - joint_info[8])

				self.gripper_velocity_limits.append(joint_info[11])

		self.arm_num_dof = 6
		self.gripper_num_dof = 6
		self.end_effector_idx = 8

		arm_link_orn = p.getLinkState(self.robot_id, self.end_effector_idx)[5]

		cid = p.createConstraint(self.robot_id, self.end_effector_idx, self.gripper_id, -1, 
								p.JOINT_FIXED, [0,0,0], [0,0,0.03], [0,0,0], arm_link_orn, arm_link_orn)

			# Initial pose
		if initial_arm_joint_positions is not None:
			self.initial_arm_joint_positions = initial_arm_joint_positions
		else:
			self.initial_arm_joint_positions = [0, -np.pi/2, np.pi/2, 0, np.pi/2, 0, 0, 0, 0, 0]

		if initial_gripper_joint_positions is not None:
			self.initial_gripper_joint_positions = initial_gripper_joint_positions
		else:
			self.initial_gripper_joint_positions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

		# Add debugging frame on the end effector
		pos, orn = self.get_cartesian_pose('quaternion')
		self.ee_ids = draw_frame(pos, orn)

	def reset(self):
		arm_joint_indices = self.arm_revolute_joint_indices + self.arm_prismatic_joint_indices
		arm_joint_indices.sort()
		for pos_idx, joint_idx in enumerate(arm_joint_indices):
			p.resetJointState(self.robot_id, joint_idx, self.initial_arm_joint_positions[pos_idx], physicsClientId=self.physics_client)

		gripper_joint_indices = self.gripper_revolute_joint_indices + self.gripper_prismatic_joint_indices
		gripper_joint_indices.sort()
		for pos_idx, joint_idx in enumerate(gripper_joint_indices):
			p.resetJointState(self.gripper_id, joint_idx, self.initial_gripper_joint_positions[pos_idx], physicsClientId=self.physics_client)

		pos, orn = self.get_cartesian_pose('quaternion')
		self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)
	
	def get_joint_limits(self, mode='arm'):
		if mode == 'arm':
			return self.arm_joint_lower_limits, self.arm_joint_upper_limits
		elif mode == 'gripper':
			return self.gripper_joint_lower_limits, self.gripper_joint_upper_limits

	def get_joint_positions(self, mode='arm'):
		if mode == 'arm':
			arm_joint_indices = self.arm_revolute_joint_indices + self.arm_prismatic_joint_indices
			arm_joint_indices.sort()
			return np.array([j[0] for j in p.getJointStates(self.robot_id, arm_joint_indices, physicsClientId=self.physics_client)])
		elif mode == 'gripper':
			gripper_joint_indices = self.gripper_revolute_joint_indices + self.gripper_prismatic_joint_indices
			gripper_joint_indices.sort()
			return np.array([j[0] for j in p.getJointStates(self.gripper_id, gripper_joint_indices, physicsClientId=self.physics_client)])

	def get_cartesian_pose(self, orientation_format='euler'):
		state = p.getLinkState(self.robot_id, self.end_effector_idx, computeForwardKinematics=True,
								physicsClientId=self.physics_client)
		pos = state[0]
		orn = state[1]

		T_arm_gripper_pos = [0, 0, 0.15]
		T_arm_gripper_orn = [0, 0, 0, 1]

		T_world_gripper = p.multiplyTransforms(pos, orn, T_arm_gripper_pos, T_arm_gripper_orn)

		new_pos = T_world_gripper[0]

		if orientation_format == 'euler':
			new_orn = p.getEulerFromQuaternion(T_world_gripper[1], physicsClientId=self.physics_client)
		else:
			new_orn = T_world_gripper[1]
		return new_pos, new_orn

	def get_image(self, view_dist=0.5, width=224, height=224):
		# Get end-effector pose
		_, _, _, _, w_pos, w_orn = p.getLinkState(self.robot_id, self.end_effector_idx,
												computeForwardKinematics=True,
												physicsClientId=self.physics_client)
		# Camera frame w.r.t end-effector
		cam_orn = p.getQuaternionFromEuler([0.0, 0.0, -np.pi/2], physicsClientId=self.physics_client)
		cam_pos = [-0.02, 0.0, 0.0]

		# Compute camera frame from end effector frame
		pos, orn = p.multiplyTransforms(w_pos, w_orn, cam_pos, cam_orn, physicsClientId=self.physics_client)

		# Get camera frame rotation matrix from quaternion
		rot_mat = p.getMatrixFromQuaternion(orn, physicsClientId=self.physics_client)
		rot_mat = np.array(rot_mat).reshape(3, 3)

		# Initial camera view direction and up direction
		init_view_vec = [0, 0, 1]
		init_up_vec = [0, 1, 0]

		# Transform vectors based on the camera frame
		view_vec = rot_mat.dot(init_view_vec)
		up_vec = rot_mat.dot(init_up_vec)

		view_matrix = p.computeViewMatrix(
			cameraEyePosition=pos,
			cameraTargetPosition=pos + view_dist * view_vec,
			cameraUpVector=up_vec,
			physicsClientId=self.physics_client
		)

		# Camera parameters and projection matrix
		fov, aspect, near_plane, far_plane = 70, 1.0, 0.01, 100
		projection_matrix = p.computeProjectionMatrixFOV(
			fov=fov,
			aspect=aspect,
			nearVal=near_plane,
			farVal=far_plane,
			physicsClientId=self.physics_client
		)

		# Extract camera image
		w, h, rgba_img, depth_img, seg_img = p.getCameraImage(
			width=width,
			height=height,
			viewMatrix=view_matrix,
			projectionMatrix=projection_matrix,
			physicsClientId=self.physics_client
		)
		rgb_img = rgba_img[:, :, :3]
		return rgb_img, depth_img, seg_img

	def set_cartesian_pose(self, pose):
		pos = pose[:3]
		roll, pitch, yaw = list(map(lambda x: x % (2*np.pi), pose[3:]))

		# Map RPY : -pi < RPY <= pi
		eul_orn = [-(2*np.pi - roll) if roll > np.pi else roll,
					-(2*np.pi - pitch) if pitch > np.pi else pitch,
					-(2*np.pi - yaw) if yaw > np.pi else yaw]

		orn = p.getQuaternionFromEuler(eul_orn, physicsClientId=self.physics_client)

		T_gripper_arm_pos = [0, 0, -0.15]
		T_gripper_arm_orn = [0, 0, 0, 1]

		T_world_arm = p.multiplyTransforms(pos, orn, T_gripper_arm_pos, T_gripper_arm_orn)

		joint_positions = p.calculateInverseKinematics(self.robot_id, self.end_effector_idx, T_world_arm[0], orn,
														self.arm_joint_lower_limits, self.arm_joint_upper_limits,
														self.arm_joint_ranges, self.initial_arm_joint_positions,
														maxNumIterations=100, physicsClientId=self.physics_client)
		joint_idx = 0
		target_joint_positions = []
		for i in range(self.arm_num_joints):
			if i in self.arm_fixed_joint_indices:
				continue
			else:
				target_joint_positions.append(joint_positions[joint_idx])
				joint_idx += 1
		target_joint_positions = np.array(target_joint_positions)
		self.set_joint_positions(target_joint_positions)

	def set_joint_positions(self, target_joint_positions, control_freq=1./240.):
		arm_joint_indices = self.arm_revolute_joint_indices + self.arm_prismatic_joint_indices
		arm_joint_indices.sort()

		current_joint_positions = self.get_joint_positions(mode='arm')
		joint_positions_diff = target_joint_positions - current_joint_positions

		# Compute time to complete motion
		max_total_time = np.max(joint_positions_diff / self.arm_velocity_limits)
		num_timesteps = math.ceil(max_total_time / control_freq)
		delta_joint_positions = joint_positions_diff / num_timesteps

		for t in range(1, num_timesteps+1):
			joint_positions = current_joint_positions + delta_joint_positions * t
			p.setJointMotorControlArray(self.robot_id, arm_joint_indices, p.POSITION_CONTROL,
										targetPositions=joint_positions, physicsClientId=self.physics_client)

		p.stepSimulation(self.physics_client)
		time.sleep(control_freq)

		# Update end effector frame display
		pos, orn = self.get_cartesian_pose('quaternion')
		self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)

	def open_gripper(self):
		pass

	def close_gripper(self):
		pass

def main():
	physics_client = p.connect(p.GUI)
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	p.setGravity(0, 0, -0.98)

	# Create kukass
	ur_10 = UR10(physics_client, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1])

	ur_10.get_image()

	# Target pose
	target_cart_pose = [0.3, 0.0, 0.08, 0.0, np.pi, 0.0]
	time.sleep(2)
	ur_10.set_cartesian_pose(target_cart_pose)
	time.sleep(2)
	print(ur_10.get_cartesian_pose())

	# Open gripper
	ur_10.open_gripper()
	#time.sleep(3)
	ur_10.close_gripper()

	# Get final image
	rgb, _, _ = ur_10.get_image()

	# dummy to keep window open
	while(1):
		time.sleep(0.01)

	physics_client.disconnect()


if __name__ == '__main__':
    main()

