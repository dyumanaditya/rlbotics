# IMPORT ALL ROBOT CLASSES:
from rlbotics.envs.robots.panda import Panda
from rlbotics.envs.robots.kuka_iiwa import KukaIiwa
from rlbotics.envs.robots.ur10 import UR10


robot_class_dict = {
	'panda': Panda,
	'kuka_iiwa': KukaIiwa,
	'ur10': UR10
}
