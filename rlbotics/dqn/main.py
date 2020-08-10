import gym
from rlbotics.dqn.dqn import DQN
import rlbotics.dqn.hyperparameters as h
from rlbotics.common.logger import Logger


def main():
	# Build environment
	env = gym.make(h.env_name)
	agent = DQN(env)
	obs = env.reset()

	# Episode related information
	ep_counter = 0
	ep_rew = 0

	for iteration in range(h.max_iterations):
		if h.render:
			env.render()

		# Take action
		act = agent.get_action(obs)
		new_obs, rew, done, _ = env.step(act)

		# Store experience
		agent.store_transition(obs, act, rew, new_obs, done)

		obs = new_obs
		ep_rew += rew
		agent.logger.log(epsilon=agent.epsilon)

		# Episode done
		if done:
			obs = env.reset()

			# Display results
			print("episode: {}, total reward: {}".format(ep_counter, ep_rew))

			# Logging
			ep_counter += 1
			ep_rew = 0
			continue

		# Update Policy
		agent.update_policy()

		# Update target policy
		if iteration % h.update_target_freq == 0:
			agent.update_target_policy()

	# End
	env.close()
	agent.logger.writer.close()


if __name__ == '__main__':
	main()
