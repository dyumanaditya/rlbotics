import pybullet as p
import pybullet_data
import time
import math
import numpy as np

p.connect(p.GUI)
flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
robot_id = p.loadURDF('combined.urdf', useFixedBase=False, flags=flags)
while(1):
    time.sleep(0.1)