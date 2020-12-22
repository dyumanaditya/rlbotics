from rlbotics.envs.robots.manipulator import Manipulator
import numpy as np


class UR10(Manipulator):
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
        robot_name = 'ur10'

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0, -1.4, 1.8, 0, np.pi/2, 0]

        initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
                        'initial_joint_positions': initial_joint_positions}

        super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name)


class UR5(Manipulator):
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
        robot_name = 'ur5'

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0, -1.4, 1.8, 0, np.pi/2, 0]

        initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
                        'initial_joint_positions': initial_joint_positions}

        super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name)


class UR3(Manipulator):
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
        robot_name = 'ur3'

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0, -1.4, 1.8, 0, np.pi/2, 0]

        initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
                        'initial_joint_positions': initial_joint_positions}

        super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name=gripper_name)
