from rlbotics.envs.robots.panda import Panda
from rlbotics.envs.robots.abb import Irb5400, Irb6600

import pybullet as p
import pybullet_data
import numpy as np
import time
import os

class BodyInWhiteWorld:
    def __init__(self, robot, gripper, render, use_ee_cam=False):
        self.use_ee_cam = use_ee_cam
        self.gripper = gripper
        self.physics_client = p.connect(p.GUI) if render else p.connect(p.DIRECT)
        self.path = os.path.abspath(os.path.dirname(__file__))
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane_id = p.loadURDF('plane.urdf', physicsClientId=self.physics_client)


    def reset_world(self):
        rail_orientation = p.getQuaternionFromEuler([np.pi/2,0,0])
        body_in_white_orientation = p.getQuaternionFromEuler([0,0,np.pi/2])

        rail_visual = p.createVisualShape(
            p.GEOM_MESH,
            meshScale=[0.001] * 3,
            fileName=os.path.join(os.path.dirname(self.path), 'models', 'misc', 'body_in_white', 'rail.obj'),
            rgbaColor=[0.5, 0.5, 0.5]
        )

        rail_collision = p.createCollisionShape(
            p.GEOM_MESH,
            meshScale=[0.001] * 3,
            fileName=os.path.join(os.path.dirname(self.path), 'models', 'misc', 'body_in_white', 'rail.obj'),
        )

        body_in_white_visual = p.createVisualShape(
            p.GEOM_MESH,
            meshScale=[0.001] * 3,
            fileName=os.path.join(os.path.dirname(self.path), 'models', 'misc', 'body_in_white', 'car1.obj'),
            rgbaColor=[0.5, 0.5, 0.5]
        )

        body_in_white_collision = p.createCollisionShape(
            p.GEOM_MESH,
            meshScale=[0.001] * 3,
            fileName=os.path.join(os.path.dirname(self.path), 'models', 'misc', 'body_in_white', 'car1.obj'),
        )

        self.rail_1 = p.createMultiBody(
            basePosition=[0.5,0,0],
            baseVisualShapeIndex=rail_visual,
            baseCollisionShapeIndex=rail_collision,
            baseOrientation=rail_orientation,
            baseMass=1000
        )

        self.rail_2 = p.createMultiBody(
            basePosition=[-0.5,0,0],
            baseVisualShapeIndex=rail_visual,
            baseCollisionShapeIndex=rail_collision,
            baseOrientation=rail_orientation,
            baseMass=1000
        )

        self.body_in_white = p.createMultiBody(
            basePosition=[0,0,0.19],
            baseVisualShapeIndex=body_in_white_visual,
            baseCollisionShapeIndex=body_in_white_collision,
            baseOrientation=body_in_white_orientation,
            baseMass=1000,
            baseInertialFramePosition=[1, 0, 0]
        )

        self.robot1 = Irb5400(self.physics_client, [-2.5,-0,0], [0,0,0,1], gripper_name=self.gripper)
        self.robot2 = Irb5400(self.physics_client, [2.5,-0,0], [0,0,1,0], gripper_name=self.gripper)
        # self.robot3 = Irb5400(self.physics_client, [-2.5,1.5,0], [0,0,0,1], gripper_name=self.gripper)
        # self.robot4 = Irb5400(self.physics_client, [2.5,1.5,0], [0,0,1,0], gripper_name=self.gripper)
        self.robot5 = Irb6600(self.physics_client, [-2.5,3,0], [0,0,0,1], gripper_name=self.gripper)
        self.robot6 = Irb6600(self.physics_client, [2.5,3,0], [0,0,1,0], gripper_name=self.gripper)

        # self.robot1.reset()
        # self.robot2.reset()
        # # self.robot3.reset()
        # # self.robot4.reset()
        # self.robot5.reset()
        # self.robot6.reset()



world = BodyInWhiteWorld('UR10', 'robotiq_2f_85', render=True)

world.reset_world()

while True:
    time.sleep(1)

