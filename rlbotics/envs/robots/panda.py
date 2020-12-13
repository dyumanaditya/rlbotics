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
