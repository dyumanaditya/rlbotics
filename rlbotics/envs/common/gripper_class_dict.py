# IMPORT ALL GRIPPER CLASSES
from rlbotics.envs.grippers.panda_gripper import PandaGripper
from rlbotics.envs.grippers.robotiq_2f_85 import Robotiq2f85

gripper_class_dict = {
	'panda_gripper': PandaGripper,
	'robotiq_2f_85': Robotiq2f85
}
