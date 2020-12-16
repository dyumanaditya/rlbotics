from rlbotics.envs.robots.manipulator import Manipulator


# TODO: Add initial joint positions
class Cr7ia(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_cr7ia'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class Cr7ial(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_cr7ial'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class Cr35ia(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_cr35ia'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class Lrmate200i(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_lrmate200i'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class Lrmate200ib(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_lrmate200ib'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class Lrmate200ib3l(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_lrmate200ib3l'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class Lrmate200ic(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_lrmate200ic'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class Lrmate200ic5f(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_lrmate200ic5f'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class Lrmate200ic5h(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_lrmate200ic5h'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class Lrmate200ic5hs(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_lrmate200ic5hs'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class Lrmate200ic5l(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_lrmate200ic5l'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class M6ib(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_m6ib'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class M6ib6s(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_m6ib6s'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class M10ia(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_m10ia'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class M10ia7l(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_m10ia7l'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class M16ib(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_m16ib20'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class M20ia(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_m20ia'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class M20ia10l(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_m20ia10l'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class M20ib(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_m20ib25'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class M430ia2f(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_m430ia2f'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class M430ia2p(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_m430ia2p'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class M710ic45m(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_m710ic45m'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class M710ic50(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_m710ic50'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class M900ia(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_m900ia260l'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class M900ib(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_m900ib700'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)


class R1000ia(Manipulator):
	def __init__(self, physics_client, base_pos, base_orn, initial_joint_positions=None, gripper_name=None):
		robot_name = 'fanuc_r1000ia80f'

		# Initial pose
		if initial_joint_positions is None:
			initial_joint_positions = []

		initial_position_info = {'base_pos': base_pos, 'base_orn': base_orn,
								 'initial_joint_positions': initial_joint_positions}

		super().__init__(physics_client, robot_name=robot_name, initial_position_info=initial_position_info,
						 gripper_name=gripper_name)
