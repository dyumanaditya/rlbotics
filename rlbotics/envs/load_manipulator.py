from rlbotics.envs.robots.panda import Panda
import pybullet as p
import time

physics_client = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)

robot = Panda(physics_client, [0,0,0], [0,0,0,1])

while True:
    time.sleep(1)