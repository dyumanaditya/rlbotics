from rlbotics.envs.robots.universal_robots import UR10, UR5, UR3
from rlbotics.envs.robots.panda import Panda

import pybullet as p
import numpy as np
import time

physics_client = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)

robot1 = UR10(physics_client, [0,0,0], [0,0,0,1], gripper_name='robotiq_2f_85')
# robot2 = UR5(physics_client, [0,1,0], [0,0,0,1], gripper_name='robotiq_2f_85')
# # robot3 = KukaIiwa(physics_client, [0,0.5,0], [0,0,0,1], gripper_name='robotiq_2f_85')
# robot4 = Panda(physics_client, [0,0,0], [0,0,0,1])

robot1.reset()
# robot2.reset()
# # robot3.reset()
# robot4.reset()
time.sleep(1)


while True:
    time.sleep(1)
    # robot1.set_cartesian_pose([1, 0, 0.2, 0, 0, 0])
    # robot2.set_cartesian_pose([0.5, 1, 0.2, 0, np.pi, 0])
    # robot4.set_cartesian_pose([0.5, 0, 0.2, 0, np.pi, 0])
