# IMPORT ALL ROBOT CLASSES:
from rlbotics.envs.robots.panda import Panda
from rlbotics.envs.robots.kuka import KukaIiwa
from rlbotics.envs.robots.universal_robots import UR10, UR5, UR3
from rlbotics.envs.robots.fanuc import Cr7ia, Cr7ial, Cr35ia, Lrmate200i, Lrmate200ib, Lrmate200ib3l, Lrmate200ic, \
	Lrmate200ic5f, Lrmate200ic5h, Lrmate200ic5hs, Lrmate200ic5l, M6ib, M6ib6s, M10ia, M10ia7l, M16ib, M20ia, M20ia10l, \
	M20ib, M430ia2f, M430ia2p, M710ic45m, M710ic50, M900ia, M900ib, R1000ia


robot_class_dict = {
	'panda': Panda,
	'kuka_iiwa': KukaIiwa,
	'ur10': UR10,
	'ur5': UR5,
	'ur3': UR3,
	'fanuc_cr7ia': Cr7ia,
	'fanuc_cr7ial': Cr7ial,
	'fanuc_cr35ia': Cr35ia,
	'fanuc_lrmate200i': Lrmate200i,
	'fanuc_lrmate200ib': Lrmate200ib,
	'fanuc_lrmate200ib3l': Lrmate200ib3l,
	'fanuc_lrmate200ic': Lrmate200ic,
	'fanuc_lrmate200ic5f': Lrmate200ic5f,
	'fanuc_lrmate200ic5h': Lrmate200ic5h,
	'fanuc_lrmate200ic5hs': Lrmate200ic5hs,
	'fanuc_lrmate200ic5l': Lrmate200ic5l,
	'fanuc_m6ib': M6ib,
	'fanuc_m6ib6s': M6ib6s,
	'fanuc_m10ia': M10ia,
	'fanuc_m10ia7l': M10ia7l,
	'fanuc_m16ib': M16ib,
	'fanuc_m20ia': M20ia,
	'fanuc_m20ia10l': M20ia10l,
	'fanuc_m20ib': M20ib,
	'fanuc_m430ia2f': M430ia2f,
	'fanuc_m430ia2p': M430ia2p,
	'fanuc_m710ic45m': M710ic45m,
	'fanuc_m710ic50': M710ic50,
	'fanuc_m900ia': M900ia,
	'fanuc_m900ib': M900ib,
	'fanuc_r1000ia': R1000ia
}
