from gym.envs.registration import register

# TODO: Fill in None with appropriate values
register(
	id='PandaDriller-v0',
	entry_point='rlbotics.envs:PandaDrillerEnv',
	max_episode_steps=None,
	reward_threshold=None
)

register(
	id='PandaGripper-v0',
	entry_point='rlbotics.envs:PandaGripperEnv',
	max_episode_steps=None,
	reward_threshold=None
)
