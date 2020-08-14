import gym
import torch
from rlbotics.dqn.dqn import DQN
import rlbotics.dqn.hyperparameters as h


def main():
	# Build environment
	env = gym.make(h.env_name)
	agent = DQN(env)
	obs = env.reset()

	# Episode related information
	ep_counter = 0
	ep_rew = 0

	# Set device
	gpu = 0
	device = torch.device("cuda:{gpu}" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
		torch.cuda.set_device(device)

	for iteration in range(h.max_iterations):
		if h.render:
			env.render()

		# Take action
		act = agent.get_action(obs)
		new_obs, rew, done, _ = env.step(act)

		# Store experience
		agent.store_transition(obs, act, rew, new_obs, done)
		ep_rew += rew
		obs = new_obs

		# Episode done
		if done:
			obs = env.reset()

			# Display results
			print("episode: {}, total reward: {}".format(ep_counter, ep_rew))
			agent.logger.writer.add_scalar("return/episode", ep_rew, ep_counter)

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


if __name__ == '__main__':
	main()
