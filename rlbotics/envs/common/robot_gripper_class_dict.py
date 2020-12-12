# IMPORT ALL ROBOT AND GRIPPER CLASSES:
# Robot classes:
from rlbotics.envs.robots.panda import Panda
from rlbotics.envs.robots.kuka_iiwa import KukaIiwa
from rlbotics.envs.robots.ur10 import UR10

# Gripper classes:
from rlbotics.envs.grippers.panda_gripper import PandaGripper


robot_class_dict = {
	'panda': Panda,
	'kuka_iiwa': KukaIiwa,
	'ur10': UR10
}

gripper_class_dict = {
	'panda_gripper': PandaGripper
}
