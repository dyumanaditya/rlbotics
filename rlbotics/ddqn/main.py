import gym
import argparse

from rlbotics.ddqn.ddqn import DDQN
import rlbotics.ddqn.hyperparameters as h
from rlbotics.common.plotter import Plotter


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
	parser.add_argument('--render', type=int, default=h.render)
	parser.add_argument('--use_grad_clip', type=int, default=h.use_grad_clip)
	parser.add_argument('--save_freq', type=int, default=h.save_freq)
	parser.add_argument('--resume', type=int, default=h.resume)

	# DDQN Specific Parameters
	parser.add_argument('--batch_size', type=int, default=h.batch_size)
	parser.add_argument('--buffer_size', type=int, default=h.buffer_size)
	parser.add_argument('--epsilon', type=float, default=h.epsilon)
	parser.add_argument('--min_epsilon', type=float, default=h.min_epsilon)
	parser.add_argument('--exp_decay', type=float, default=h.exp_decay)
	parser.add_argument('-linear_decay', type=float, default=h.linear_decay)

	# Policy/Target Network
	parser.add_argument('--hidden_sizes', nargs='+', type=int, default=h.hidden_sizes)
	parser.add_argument('--activations', nargs='+', type=str, default=h.activations)
	parser.add_argument('--optimizer', type=str, default=h.optimizer)
	parser.add_argument('--loss_type', type=str, default=h.loss_type)
	parser.add_argument('--update_target_freq', type=int, default=h.update_target_freq)

	return parser.parse_args()


def main():
	args = argparser()
	# Build environment
	env = gym.make(args.env_name)
	env.seed(args.seed)
	agent = DDQN(args, env)
	obs = env.reset()

	# Episode related information (resume if necessary)
	ep_counter = agent.logger.tensorboard_updated
	ep_rew = 0

	# If resuming training, x is where we left off
	x = agent.steps_done
	for iteration in range(x, args.max_iterations):
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
			print(f"episode: {ep_counter}, total reward: {ep_rew}, epsilon: {agent.epsilon}, iter: {iteration}")

			# Increment ep_counter after policy updates start
			ep_rew = 0
			if iteration >= agent.batch_size:
				ep_counter += 1

		# Update Policy
		agent.update_policy()

		# Update target policy
		if ep_counter % args.update_target_freq == 0:
			agent.update_target_policy()

	# End
	env.close()
	p = Plotter()
	p.plot_individual('Episode/Reward', 'episodes', 'rewards', 'DDQN', args.env_name, args.seed, True)


if __name__ == '__main__':
	main()
