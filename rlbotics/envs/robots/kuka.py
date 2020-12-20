from rlbotics.envs.robots.manipulator import Manipulator


class Iiwa(Manipulator):
    def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
        robot_name = 'Iiwa'

        # Initial pose
        if initial_joint_positions is None:
            initial_joint_positions = [0.0, 0.3, 0.0, -1.2, 0.0, 2.0, 0.0]

        initial_pose = {'base_pos': base_pos, 'base_orn': base_orn,
                        'initial_joint_positions': initial_joint_positions}

        super().__init__(physics_client, robot_name=robot_name, initial_pose=initial_pose, gripper_name='panda_gripper')
