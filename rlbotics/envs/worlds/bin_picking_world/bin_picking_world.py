import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import pandas as pd

from rlbotics.envs.robots.panda import Panda
from rlbotics.envs.robots.kuka import Kuka

class BinPickingWorld:
    def __init__(self, robot, render, num_of_parts=5):
        self.physics_client = p.connect(p.GUI) if render else p.connect(p.DIRECT)
        self.path = os.path.abspath(os.path.dirname(__file__))
    

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(True)
        p.setGravity(0, 0, -0.98)

        self.num_of_parts = num_of_parts

        # Load Robot and other objects
        arm_base_pos = [0, 0, 0.94]
        arm_base_orn = p.getQuaternionFromEuler([0, 0, 0])

        table_orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])

        if robot == 'panda':
            self.arm = Panda(self.physics_client, arm_base_pos, arm_base_orn)

        elif robot == 'kuka':
            self.arm = Kuka(self.physics_client, arm_base_pos, arm_base_orn)

        self.tray_1_id = None
        self.tray_2_id = None

        self.plane_id = p.loadURDF('plane.urdf')
        self.table_id = p.loadURDF('table/table.urdf', [0.5, 0, 0], table_orientation, globalScaling=1.5, useFixedBase=True)

        self.parts_id = []
        self.other_id = [self.table_id, self.arm]

        self.parts_data = pd.read_csv(os.path.join(self.path, 'parts', 'parts_data.csv'))


    def reset(self):
        if self.tray_1_id is not None:
            p.removeBody(self.tray_1_id)
        if self.tray_2_id is not None:
            p.removeBody(self.tray_2_id)

        self.tray_1_pos = [np.random.uniform(0.4, 0.9), np.random.uniform(0.3, 0.7), 0.94]
        self.tray_2_pos = [np.random.uniform(0.4, 0.9), np.random.uniform(-0.3, -0.7), 0.94]

        self.tray_1_id = p.loadURDF('tray/traybox.urdf', self.tray_1_pos)
        self.tray_2_id = p.loadURDF('tray/traybox.urdf', self.tray_2_pos)

        # bounding box of from tray and to tray
        self.tray_1_min_AABB, self.from_tray_max_AABB = p.getAABB(self.tray_1_id)
        self.tray_2_min_AABB, self.to_tray_max_AABB = p.getAABB(self.tray_2_id)

        for part in self.parts_id:
            p.removeBody(part)

        self.parts_id = []
        self.other_id.append(self.tray_1_id)
        self.other_id.append(self.tray_2_id)

        self.arm.reset()

        # add the random objects in tray 1
        for _ in range(self.num_of_parts):
            self.add_random_object()
            time.sleep(0.1)

    def add_random_object(self):
        object_num = np.random.randint(7)
        # object_num = 0

        object_scale = self.parts_data.loc[object_num, 'scale']
        object_mass = self.parts_data.loc[object_num, 'mass']
        object_orientation = p.getQuaternionFromEuler([np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)])

        object_visual = p.createVisualShape(
            p.GEOM_MESH,
            meshScale=[object_scale] * 3,
            fileName=os.path.join(self.path, 'parts', str(object_num) + '.obj'),
            rgbaColor=np.hstack((np.random.rand(3), 1))
        )

        object_collision = p.createCollisionShape(
            p.GEOM_MESH,
            meshScale=[object_scale] * 3,
            fileName=os.path.join(self.path, 'parts', str(object_num) + '.obj')
        )

        self.parts_id.append(p.createMultiBody(
            basePosition=[np.random.uniform(self.tray_1_pos[0] - 0.1, self.tray_1_pos[0] + 0.1), np.random.uniform(self.tray_1_pos[1] - 0.1, self.tray_1_pos[1] + 0.1), 1.5],
            baseVisualShapeIndex=object_visual,
            baseCollisionShapeIndex=object_collision,
            baseOrientation=object_orientation,
            baseMass=object_mass
        ))

    def get_camera_img(self):
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.5, 0, 2.5],
            cameraTargetPosition=[0.5, 0, 0.94],
            cameraUpVector=[1, 0, 0]
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=70,
            aspect=1.0,
            nearVal=0.01,
            farVal=100
        )

        _, _, rgba_img, depth_img, seg_img = p.getCameraImage(
            width=224,
            height=224,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )

        rgb_img = rgba_img[:,:,:3]

        return rgb_img, depth_img, seg_img
