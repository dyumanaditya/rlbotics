import os
import yaml
import math
import time
import numpy as np
import pybullet as p
import pybullet_data

from rlbotics.envs.common.utils import draw_frame
from rlbotics.envs.common.robot_gripper_class_dict import gripper_class_dict


class Manipulator:
	def __init__(self, physics_client, robot_info, base_pos, base_orn, gripper_name):
		p.setRealTimeSimulation(1)      # SEE ABOUT THIS LATER. This is needed to complete motion
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		self.physics_client = physics_client

		# Load YAML info
		yaml_file = os.path.join('..', 'common', 'load_info.yaml')
		with open(yaml_file, 'r') as stream:
			load_info = yaml.safe_load(stream)

		# Expand robot_info into attributes
		self.robot_dof = robot_info['dof']
		self.robot_name = robot_info['robot_name']
		self.in_built_gripper = robot_info['in_built_gripper']
		self.robot_joint_indices = robot_info['joint_indices']
		self.robot_joint_lower_limits = robot_info['joint_lower_limits']
		self.robot_joint_upper_limits = robot_info['joint_upper_limits']
		self.robot_joint_ranges = robot_info['joint_ranges']
		self.robot_joint_velocity_limits = robot_info['joint_velocity_limits']
		self.robot_initial_joint_positions = robot_info['initial_pose']

		# Create gripper object and expand into attributes
		gripper = gripper_class_dict[gripper_name]()
		self.gripper_dof = gripper.gripper_info['dof']
		self.gripper_name = gripper_name
		self.ee_idx = gripper.gripper_info['ee_idx']
		self.gripper_joint_indices = gripper.gripper_info['joint_indices']
		self.gripper_joint_lower_limits = gripper.gripper_info['joint_lower_limits']
		self.gripper_joint_upper_limits = gripper.gripper_info['joint_upper_limits']
		self.gripper_joint_ranges = gripper.gripper_info['joint_ranges']
		self.gripper_joint_velocity_limits = gripper.gripper_info['joint_velocity_limits']
		self.gripper_initial_joint_positions = gripper.gripper_info['initial_pose']

		self.initial_joint_positions = self.robot_initial_joint_positions + self.gripper_initial_joint_positions

		# Combine gripper and robot if gripper is not in built
		if self.in_built_gripper:
			if load_info[self.robot_name]['location'] == 'local':
				self.robot_path = os.path.join('..', 'models', load_info[self.robot_name]['relative_path'])
			elif load_info[self.robot_name]['location'] == 'pybullet_data':
				self.robot_path = load_info[self.robot_name]['relative_path']

		else:
			# TODO: Combine URDFs
			if load_info[self.gripper_name]['location'] == 'local':
				self.gripper_path = os.path.join('..', 'models', load_info[self.gripper_name]['relative_path'])
			elif load_info[self.gripper_name]['location'] == 'pybullet_data':
				self.gripper_path = load_info[self.gripper_name]['relative_path']
			# Combine urdfs and return combined urdfs path: robot_path
			pass

		# Load robot
		self.robot_id = p.loadURDF(self.robot_path, base_pos, base_orn, useFixedBase=True, physicsClientId=self.physics_client)

		# Check attributes to fill in any missed out information
		self._check_attributes()

		# Add debugging frame on the end effector
		pos, orn = self.get_cartesian_pose('quaternion')
		self.ee_ids = draw_frame(pos, orn)

	def _check_attributes(self):
		temp_client = p.connect(p.DIRECT)

		# Sync gripper info with robot info: gripper indices come after robot indices
		if not self.in_built_gripper:
			robot_id = p.loadURDF(self.robot_path, physicsClientId=temp_client)
			gripper_idx_offset = p.getNumJoints(robot_id)

			# Verify robot joint specs
			if self.robot_joint_indices is None:
				self.robot_joint_indices = self._get_joint_indices(robot_id, temp_client)
			if self.robot_dof is None:
				self.robot_dof = len(self.robot_joint_indices)
			if self.robot_joint_lower_limits is None:
				self.robot_joint_lower_limits = self._get_joint_limits_from_urdf(robot_id, temp_client)[0]
			if self.robot_joint_upper_limits is None:
				self.robot_joint_upper_limits = self._get_joint_limits_from_urdf(robot_id, temp_client)[1]
			if self.robot_joint_ranges is None:
				self.robot_joint_ranges = self._get_joint_ranges_from_urdf(robot_id, temp_client)
			if self.robot_joint_velocity_limits is None:
				self.robot_joint_velocity_limits = self._get_joint_velocity_limits_from_urdf(robot_id, temp_client)

			p.removeBody(robot_id, temp_client)
			gripper_id = p.loadURDF(self.gripper_path, physicsClientId=temp_client)

			# Verify gripper joint specs
			if self.gripper_joint_indices is None:
				self.gripper_joint_indices = self._get_joint_indices(gripper_id, temp_client)
			self.gripper_joint_indices = list(map(lambda x: x+gripper_idx_offset, self.gripper_joint_indices))
			if self.gripper_dof is None:
				self.gripper_dof = len(self.gripper_joint_indices)
			if self.gripper_joint_lower_limits is None:
				self.gripper_joint_lower_limits = self._get_joint_limits_from_urdf(gripper_id, temp_client)[0]
			if self.gripper_joint_upper_limits is None:
				self.gripper_joint_upper_limits = self._get_joint_limits_from_urdf(gripper_id, temp_client)[1]
			if self.gripper_joint_ranges is None:
				self.gripper_joint_ranges = self._get_joint_ranges_from_urdf(gripper_id, temp_client)
			if self.gripper_joint_velocity_limits is None:
				self.gripper_joint_velocity_limits = self._get_joint_velocity_limits_from_urdf(gripper_id, temp_client)

			p.removeBody(gripper_id, temp_client)

		else:
			robot_id = p.loadURDF(self.robot_path, physicsClientId=temp_client)

			# Verify robot joint specs
			if self.robot_joint_indices is None:
				self.robot_joint_indices = self._get_joint_indices(robot_id, temp_client)[:self.robot_dof]
			if self.robot_joint_lower_limits is None:
				self.robot_joint_lower_limits = self._get_joint_limits_from_urdf(robot_id, temp_client)[0][:self.robot_dof]
			if self.robot_joint_upper_limits is None:
				self.robot_joint_upper_limits = self._get_joint_limits_from_urdf(robot_id, temp_client)[1][:self.robot_dof]
			if self.robot_joint_ranges is None:
				self.robot_joint_ranges = self._get_joint_ranges_from_urdf(robot_id, temp_client)[:self.robot_dof]
			if self.robot_joint_velocity_limits is None:
				self.robot_joint_velocity_limits = self._get_joint_velocity_limits_from_urdf(robot_id, temp_client)[:self.robot_dof]

			# Verify gripper joint specs
			if self.gripper_joint_indices is None:
				self.gripper_joint_indices = self._get_joint_indices(robot_id, temp_client)[self.robot_dof:]
			if self.gripper_joint_lower_limits is None:
				self.gripper_joint_lower_limits = self._get_joint_limits_from_urdf(robot_id, temp_client)[0][self.robot_dof:]
			if self.gripper_joint_upper_limits is None:
				self.gripper_joint_upper_limits = self._get_joint_limits_from_urdf(robot_id, temp_client)[1][self.robot_dof:]
			if self.gripper_joint_ranges is None:
				self.gripper_joint_ranges = self._get_joint_ranges_from_urdf(robot_id, temp_client)[self.robot_dof:]
			if self.gripper_joint_velocity_limits is None:
				self.gripper_joint_velocity_limits = self._get_joint_velocity_limits_from_urdf(robot_id, temp_client)[self.robot_dof:]

			p.removeBody(robot_id, temp_client)
		p.disconnect(temp_client)

	def _get_joint_indices(self, id, physics_client):
		indices = []
		for idx in range(p.getNumJoints(id, physics_client)):
			joint_type = p.getJointInfo(id, idx, physics_client)[2]
			if joint_type == p.JOINT_FIXED:
				continue
			else:
				indices.append(idx)
		return indices

	def _get_joint_limits_from_urdf(self, id, physics_client):
		lower_limits, upper_limits = [], []
		for idx in range(p.getNumJoints(id, physics_client)):
			joint_info = p.getJointInfo(id, idx, physics_client)
			joint_type = joint_info[2]
			if joint_type != p.JOINT_FIXED:
				lower_limits.append(joint_info[8])
				upper_limits.append(joint_info[9])
		return lower_limits, upper_limits

	def _get_joint_ranges_from_urdf(self, id, physics_client):
		lower_limits, upper_limits = self._get_joint_limits_from_urdf(id, physics_client)
		joint_ranges = []
		for i, j in zip(lower_limits, upper_limits):
			joint_ranges.append(j-i)
		return joint_ranges

	def _get_joint_velocity_limits_from_urdf(self, id, physics_client):
		velocity_limits = []
		for idx in range(p.getNumJoints(id, physics_client)):
			joint_info = p.getJointInfo(id, idx, physics_client)
			joint_type = joint_info[2]
			if joint_type != p.JOINT_FIXED:
				velocity_limits.append(joint_info[11])
		return velocity_limits

	def reset(self):
		joint_indices = self.robot_joint_indices + self.gripper_joint_indices
		for pos_idx, joint_idx in enumerate(joint_indices):
			p.resetJointState(self.robot_id, joint_idx, self.initial_joint_positions[pos_idx], physicsClientId=self.physics_client)

		pos, orn = self.get_cartesian_pose('quaternion')
		self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)

	def get_joint_limits(self):
		lower_limits = self.robot_joint_lower_limits + self.gripper_joint_lower_limits
		upper_limits = self.robot_joint_upper_limits + self.gripper_joint_upper_limits
		return lower_limits, upper_limits

	def get_joint_ranges(self):
		ranges = self.robot_joint_ranges + self.gripper_joint_ranges
		return ranges

	def get_joint_positions(self):
		joint_indices = self.robot_joint_indices + self.gripper_joint_indices
		return np.array([j[0] for j in p.getJointStates(self.robot_id, joint_indices, physicsClientId=self.physics_client)])

	def get_cartesian_pose(self, orientation_format='euler'):
		state = p.getLinkState(self.robot_id, self.ee_idx, computeForwardKinematics=True, physicsClientId=self.physics_client)
		pos = list(state[0])
		if orientation_format == 'euler':
			orn = list(p.getEulerFromQuaternion(state[1], physicsClientId=self.physics_client))
		else:
			orn = list(state[1])
		return pos, orn

	def get_image(self, view_dist=0.5, width=224, height=224):
		# Get end-effector pose
		_, _, _, _, w_pos, w_orn = p.getLinkState(self.robot_id, self.ee_idx,
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

		joint_lower_limits, joint_upper_limits = self.get_joint_limits()
		joint_ranges = self.get_joint_ranges()
		joint_positions = p.calculateInverseKinematics(self.robot_id, self.ee_idx, pos, orn,
													   joint_lower_limits, joint_upper_limits,
													   joint_ranges, self.initial_joint_positions,
													   maxNumIterations=100, physicsClientId=self.physics_client)
		# Remove gripper positions
		joint_positions = joint_positions[:self.robot_dof]
		target_joint_positions = np.array(joint_positions)
		self.set_joint_positions(target_joint_positions)

	def set_joint_positions(self, target_joint_positions, control_freq=1./240.):
		joint_indices = self.robot_joint_indices

		current_joint_positions = self.get_joint_positions()[:self.robot_dof]
		joint_positions_diff = target_joint_positions - current_joint_positions

		# Compute time to complete motion
		max_total_time = np.max(joint_positions_diff / self.robot_joint_velocity_limits)
		num_timesteps = math.ceil(max_total_time / control_freq)
		delta_joint_positions = joint_positions_diff / num_timesteps

		for t in range(1, num_timesteps+1):
			joint_positions = current_joint_positions + delta_joint_positions * t
			p.setJointMotorControlArray(self.robot_id, joint_indices, p.POSITION_CONTROL,
										targetPositions=joint_positions, physicsClientId=self.physics_client)

			p.stepSimulation(self.physics_client)
			time.sleep(control_freq)

			# Update end effector frame display
			pos, orn = self.get_cartesian_pose('quaternion')
			self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)
