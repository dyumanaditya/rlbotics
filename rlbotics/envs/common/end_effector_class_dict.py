# IMPORT ALL GRIPPER CLASSES
from rlbotics.envs.robots.end_effectors.robotiq import Robotiq2f85
from rlbotics.envs.robots.end_effectors.panda_gripper import PandaGripper
from rlbotics.envs.robots.end_effectors.welding_torches import WeldingTorch


end_effector_class_dict = {
	'panda_gripper': PandaGripper,
	'robotiq_2f_85': Robotiq2f85,
	'welding_torch': WeldingTorch
}
