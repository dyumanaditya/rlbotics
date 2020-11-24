import time
import math
import numpy as np
import pybullet as p
import pybullet_data

from rlbotics.envs.common.utils import draw_frame


class Panda:
    def __init__(self, physics_client, base_pos, base_orn):
        self.physics_client = physics_client
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        self.robot_id = p.loadURDF('franka_panda/panda.urdf', base_pos, base_orn, useFixedBase=True, flags=flags,
                                   physicsClientId=self.physics_client)

        # 7 Revolute, 2 Prismatic, 3 Fixed
        self.num_dofs = 7
        self.end_effector_idx = 11
        self.num_joints = p.getNumJoints(self.robot_id)

        # Initialize position and velocity limits
        self.joint_lower_limits, self.joint_upper_limits, self.joint_range, self.velocity_limits = [], [], [], []
        for joint_idx in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_idx, physicsClientId=self.physics_client)
            self.joint_lower_limits.append(joint_info[8])
            self.joint_upper_limits.append(joint_info[9])
            self.joint_range.append(joint_info[9] - joint_info[8])
            max_velocity = joint_info[11]
            self.velocity_limits.append(np.inf if max_velocity == 0 else max_velocity)

        # Initial pose
        self.initial_joint_positions = [0.0, 0.54, 0.0, -1.6, 0.0, 2.0, 0.0, 0.02, 0.02]

        # Add debugging frame on the end effector
        pos, orn = p.getLinkState(self.robot_id, self.end_effector_idx, computeForwardKinematics=True,
                                  physicsClientId=self.physics_client)[4:6]
        self.ee_ids = draw_frame(pos, orn)

    def reset(self):
        joint_idx = 0
        for i in range(self.num_joints):
            joint_type = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)[2]
            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                p.resetJointState(self.robot_id, i, self.initial_joint_positions[joint_idx],
                                  physicsClientId=self.physics_client)
                joint_idx += 1

    def get_joint_limits(self):
        return self.joint_lower_limits, self.joint_upper_limits

    def get_joint_positions(self):
        return np.array([j[0] for j in p.getJointStates(self.robot_id, range(self.num_joints),
                                                        physicsClientId=self.physics_client)])

    def get_cartesian_pose(self, orientation_format="euler"):
        state = p.getLinkState(self.robot_id, self.end_effector_idx, physicsClientId=self.physics_client)
        pos = list(state[0])
        if orientation_format == "euler":
            orn = list(p.getEulerFromQuaternion(state[1], physicsClientId=self.physics_client))
        else:
            orn = list(state[1])
        pose = pos + orn
        return pose

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
        roll, pitch, yaw = pose[3:]

        eul_orn = [min(np.pi, max(-np.pi, roll)),
                   min(np.pi, max(-np.pi, pitch)),
                   min(np.pi, max(-np.pi, yaw))]

        orn = p.getQuaternionFromEuler(eul_orn, physicsClientId=self.physics_client)

        joint_positions = p.calculateInverseKinematics(self.robot_id, self.end_effector_idx, pos, orn,
                                                       self.joint_lower_limits, self.joint_upper_limits,
                                                       self.joint_range, self.initial_joint_positions,
                                                       maxNumIterations=100, physicsClientId=self.physics_client)
        target_joint_positions = np.zeros(self.num_joints)
        joint_idx = 0
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            joint_type = info[2]
            if joint_type == p.JOINT_REVOLUTE:
                target_joint_positions[i] = joint_positions[joint_idx]
                joint_idx += 1
        self.set_joint_positions(target_joint_positions)

    def set_joint_positions(self, target_joint_positions, control_freq=1./240):
        joint_indices = list(range(self.num_joints))
        current_joint_positions = self.get_joint_positions()
        joint_positions_diff = target_joint_positions - current_joint_positions

        # Compute time to complete motion
        max_total_time = np.max(joint_positions_diff / self.velocity_limits)
        num_timesteps = math.ceil(max_total_time / control_freq)
        delta_joint_positions = joint_positions_diff / num_timesteps

        for t in range(num_timesteps):
            joint_positions = current_joint_positions + delta_joint_positions * t
            p.setJointMotorControlArray(self.robot_id, joint_indices, p.POSITION_CONTROL,
                                        targetPositions=joint_positions, physicsClientId=self.physics_client)

            # Update end effector frame display
            pos, orn = p.getLinkState(self.robot_id, self.end_effector_idx, computeForwardKinematics=True,
                                      physicsClientId=self.physics_client)[4:6]
            self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)

            p.stepSimulation(self.physics_client)
            time.sleep(control_freq)

    def open_gripper(self, width=0.8):
        width = min(width, 0.8)
        for i in [9, 10]:
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, width/2, force=10,
                                    physicsClientId=self.physics_client)

            # Update end effector frame display
            pos, orn = p.getLinkState(self.robot_id, self.end_effector_idx, computeForwardKinematics=True,
                                      physicsClientId=self.physics_client)[4:6]
            self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)

    def close_gripper(self, width=0.0):
        width = max(width, 0.0)
        for i in [9, 10]:
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, width/2, force=10,
                                    physicsClientId=self.physics_client)

            # Update end effector frame display
            pos, orn = p.getLinkState(self.robot_id, self.end_effector_idx, computeForwardKinematics=True,
                                      physicsClientId=self.physics_client)[4:6]
            self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)

