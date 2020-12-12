import time
import pybullet as p
from math import radians

from rlbotics.envs.common.utils import draw_frame
from rlbotics.envs.robots.manipulator import Manipulator


class Panda(Manipulator):
    """
    Class for Franka Emika Panda. Datasheet specs taken from:
    https://s3-eu-central-1.amazonaws.com/franka-de-uploads/uploads/Datasheet-EN.pdf
    Angle Units: degrees
    """
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
        robot_info = {}
        dof = 7
        robot_name = 'panda'
        in_built_gripper = True
        joint_indices = [0, 1, 2, 3, 4, 5, 6]
        joint_lower_limits = [-166, -101, -166, -176, -166, -1, -166]
        joint_upper_limits = [166, 101, 166, -4, 166, 215, 166]
        joint_ranges = [ul - ll for ll, ul in zip(joint_lower_limits, joint_upper_limits)]
        joint_velocity_limits = [150, 150, 150, 150, 180, 180, 180]

        # Convert all degree specs to radians
        joint_lower_limits = list(map(radians, joint_lower_limits))
        joint_upper_limits = list(map(radians, joint_upper_limits))
        joint_ranges = list(map(radians, joint_ranges))
        joint_velocity_limits = list(map(radians, joint_velocity_limits))

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0.0, 0.3, 0.0, -1.2, 0.0, 2.0, 0.0]

        robot_info['dof'] = dof
        robot_info['robot_name'] = robot_name
        robot_info['in_built_gripper'] = in_built_gripper
        robot_info['joint_indices'] = joint_indices
        robot_info['joint_lower_limits'] = joint_lower_limits
        robot_info['joint_upper_limits'] = joint_upper_limits
        robot_info['joint_ranges'] = joint_ranges
        robot_info['joint_velocity_limits'] = joint_velocity_limits
        robot_info['initial_pose'] = initial_joint_positions

        super().__init__(physics_client, robot_info=robot_info, base_pos=base_pos, base_orn=base_orn, gripper_name='panda_gripper')

    def open_gripper(self, width=0.08):
        width = min(width, 0.08)
        target = [width/2, width/2]
        for i in range(self.gripper_dof):
            p.setJointMotorControl2(self.robot_id, self.gripper_joint_indices[i], p.POSITION_CONTROL, target[i],
                                    maxVelocity=self.gripper_joint_velocity_limits[i], physicsClientId=self.physics_client)
        time.sleep(1)

        # Update end effector frame display
        pos, orn = self.get_cartesian_pose('quaternion')
        self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)

    def close_gripper(self, width=0.0):
        width = max(width, 0.0)
        target = [width/2, width/2]
        for i in range(self.gripper_dof):
            p.setJointMotorControl2(self.robot_id, self.gripper_joint_indices[i], p.POSITION_CONTROL, target[i],
                                    maxVelocity=self.gripper_joint_velocity_limits[i], physicsClientId=self.physics_client)
        time.sleep(1)

        # Update end effector frame display
        pos, orn = self.get_cartesian_pose('quaternion')
        self.ee_ids = draw_frame(pos, orn, replacement_ids=self.ee_ids)
