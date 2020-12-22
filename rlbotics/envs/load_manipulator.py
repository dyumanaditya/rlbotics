from rlbotics.envs.robots.universal_robots import UR10
from rlbotics.envs.robots.panda import Panda
from rlbotics.envs.robots.kuka import Iiwa

import pybullet as p
import time

physics_client = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)

robot = UR10(physics_client, [0,0,0], [0,0,0,1], gripper_name='robotiq_2f_85')
# robot = Iiwa(physics_client, [0,0,0], [0,0,0,1], gripper_name='robotiq_2f_85')
# robot = Panda(physics_client, [0,0,0], [0,0,0,1])



while True:
    robot.reset()