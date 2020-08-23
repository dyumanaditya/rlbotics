import gym
import torch
import argparse

from rlbotics.ddpg.ddpg import DDPG
import rlbotics.ddpg.hyperparameters as h
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
	parser.add_argument('--max_iterations', type=int, default=h.max_iterations)
	parser.add_argument('--render', type=bool, default=h.render)
	parser.add_argument('--use_grad_clip', type=bool, default=h.use_grad_clip)

	# DDPG Specific
	parser.add_argument('--batch_size', type=int, default=h.batch_size)
	parser.add_argument('--buffer_size', type=int, default=h.buffer_size)
	parser.add_argument('--polyak', type=float, default=h.polyak)
	parser.add_argument('--act_noise', type=float, default=h.act_noise)
	parser.add_argument('--noise_type', type=str, default=h.noise_type)
	parser.add_argument('--random_steps', type=int, default=h.random_steps)
	parser.add_argument('--update_after', type=int, default=h.update_after)

	# Policy and Q Network specific
	parser.add_argument('--save_freq', type=int, default=h.save_freq)
	parser.add_argument('--pi_lr', type=float, default=h.pi_lr)
	parser.add_argument('--q_lr', type=float, default=h.q_lr)
	parser.add_argument('--pi_hidden_sizes', nargs='+', type=int, default=h.pi_hidden_sizes)
	parser.add_argument('--q_hidden_sizes', nargs='+', type=int, default=h.q_hidden_sizes)
	parser.add_argument('--pi_activations', nargs='+', type=str, default=h.pi_activations)
	parser.add_argument('--q_activations', nargs='+', type=str, default=h.q_activations)
	parser.add_argument('--pi_optimizer', type=str, default=h.pi_optimizer)
	parser.add_argument('--q_optimizer', type=str, default=h.q_optimizer)
	parser.add_argument('--q_loss_type', type=str, default=h.q_loss_type)
	parser.add_argument('--weight_decay', type=float, default=h.weight_decay)

	return parser.parse_args()


def main():
	args = argparser()
	# Build environment
	env = gym.make(args.env_name)
	agent = DDPG(args, env)
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

		# Take action Random in the beginning for exploration
		act = agent.get_action(obs) if iteration > args.random_steps else env.action_space.sample()
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

			# Logging
			ep_counter += 1
			ep_rew = 0

		# Update Actor Critic
		agent.update_actor_critic()

	# End
	env.close()
	plot('DDPG', args.env_name, args.seed, 'episodes', 'rewards', True)


if __name__ == '__main__':
	main()
