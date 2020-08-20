import gym
import torch
import argparse

from rlbotics.dqn.dqn import DQN
import rlbotics.dqn.hyperparameters as h
from rlbotics.common.visualization import plot


def argparser():
	"""
    Input argument parser.
    Loads default hyperparameters from hyperparameters.py
    :return: Parsed arguments
    """
	parser = argparse.ArgumentParser()
	# General Parameters
	parser.add_argument('--seed', type=int, default=h.seed)
	parser.add_argument('--env_name', type=str, default=h.env_name)
	parser.add_argument('--gamma', type=float, default=h.gamma)
	parser.add_argument('--lr', type=float, default=h.lr)
	parser.add_argument('--max_iterations', type=int, default=h.max_iterations)
	parser.add_argument('--render', type=bool, default=h.render)

	# DQN Specific Parameters
	parser.add_argument('--batch_size', type=int, default=h.batch_size)
	parser.add_argument('--buffer_size', type=int, default=h.buffer_size)
	parser.add_argument('--epsilon', type=float, default=h.epsilon)
	parser.add_argument('--min_epsilon', type=float, default=h.min_epsilon)
	parser.add_argument('--exp_decay', type=float, default=h.exp_decay)
	parser.add_argument('-linear_decay', type=float, default=h.linear_decay)

	# Policy/Target Network
	parser.add_argument('--hidden_sizes', type=int, default=h.hidden_sizes)
	parser.add_argument('--activations', type=str, default=h.activations)
	parser.add_argument('--optimizer', type=str, default=h.optimizer)
	parser.add_argument('--loss_type', type=str, default=h.loss_type)
	parser.add_argument('--update_target_freq', type=int, default=h.update_target_freq)
	parser.add_argument('--use_grad_clip', type=bool, default=h.use_grad_clip)

	return parser.parse_args()


def main():
	args = argparser()
	# Build environment
	env = gym.make(args.env_name)
	agent = DQN(args, env)
	obs = env.reset()

	# Episode related information
	ep_counter = 0
	ep_rew = 0

	# Set device
	gpu = 0
	device = torch.device(f"cuda:{gpu}"if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
		torch.cuda.set_device(device)

	for iteration in range(args.max_iterations):
		if args.render:
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
			print("episode: {}, total reward: {}, epsilon: {}".format(ep_counter, ep_rew, agent.epsilon))

			# Logging
			ep_counter += 1
			ep_rew = 0
			continue

		# Update Policy
		agent.update_policy()

		# Update target policy
		if ep_counter % args.update_target_freq == 0:
			agent.update_target_policy()

	# End
	env.close()
	plot('DQN', args.env_name, args.seed, 'episodes', 'rewards', True)


if __name__ == '__main__':
	main()
