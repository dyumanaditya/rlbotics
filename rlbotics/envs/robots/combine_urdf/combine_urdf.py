import pybullet as p
from pybullet_utils import bullet_client as bc
from pybullet_utils import urdfEditor as ed


def combine_urdf(arm_info, gripper_info):
	arm_name = arm_info['arm_name']
	arm_path = arm_info['arm_path']
	gripper_name = gripper_info['gripper_name']
	gripper_path = gripper_info['gripper_path']
	jointPivotXYZInParent = gripper_info['jointPivotXYZInParent']
	jointPivotRPYInParent = gripper_info['jointPivotRPYInParent']
	jointPivotXYZInChild = gripper_info['jointPivotXYZInChild']
	jointPivotRPYInChild = gripper_info['jointPivotRPYInChild']

	p0 = bc.BulletClient(connection_mode=p.DIRECT)
	p1 = bc.BulletClient(connection_mode=p.DIRECT)

	arm_id = p1.loadURDF(arm_path, flags=p0.URDF_USE_IMPLICIT_CYLINDER)
	gripper_id = p0.loadURDF(gripper_path)

	arm_link_idx = p.getNumJoints(arm_id) - 1

	ed0 = ed.UrdfEditor()
	ed0.initializeFromBulletBody(arm_id, p1._client)
	ed1 = ed.UrdfEditor()
	ed1.initializeFromBulletBody(gripper_id, p0._client)

	parentLinkIndex = arm_link_idx

	new_joint = ed0.joinUrdf(ed1, parentLinkIndex, jointPivotXYZInParent, jointPivotRPYInParent,
							jointPivotXYZInChild, jointPivotRPYInChild, p0._client, p1._client)
	new_joint.joint_type = p0.JOINT_FIXED

	robot_path = f'rlbotics/envs/models/combined/{arm_name}_{gripper_name}.urdf'
	ed0.saveUrdf(robot_path)
	return robot_path
