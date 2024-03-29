import os
import gym
import torch
import argparse
import numpy as np
from gym import wrappers


def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--algo', type=str, required=True)
	parser.add_argument('--env', type=str, required=True)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--verbose', type=bool, default=True)
	parser.add_argument('--video', type=bool, default=False)
	return parser.parse_args()


def main():
	"""
	Testing trained agents in environments.
	NOTE: This HAS to be run through terminal
	"""
	args = argparser()
	env = gym.make(args.env)

	continuous = False if len(env.action_space.shape) == 0 else True
	argmax = True if not continuous else False
	model_name = 'model.pth' if args.algo.lower() in ['dqn', 'ddqn'] else 'pimodel.pth'
	model_path = os.path.join('experiments', 'models', args.algo.upper() + '_' + args.env + '_' + str(args.seed), model_name)

	model = torch.load(model_path)
	obs_dim = env.observation_space.shape[0]

	if continuous:
		act_limit = env.action_space.high[0]

	# Save video of test
	if args.video:
		video_file = os.path.join('tests', 'videos', args.algo.upper() + '_' + args.env + '_' + str(args.seed))
		env = wrappers.Monitor(env, video_file, video_callable=lambda episode_id: True, force=True)

	# Run tests
	max_episodes = 30
	ep_count = 1
	ep_rew = 0
	obs = env.reset()
	while ep_count <= max_episodes:
		env.render()

		obs = torch.from_numpy(obs).float()
		obs = obs.view(-1, obs_dim)
		with torch.no_grad():
			act = model(obs).numpy()
			if argmax:
				act = act.argmax().item()
			if continuous:
				act = np.clip(act * act_limit, -act_limit, act_limit)[0]

		new_obs, rew, done, info = env.step(act)
		obs = new_obs
		ep_rew += rew

		if done:
			if args.verbose:
				print(f'Episode: {ep_count}, Total Reward: {ep_rew}')

			ep_rew = 0
			ep_count += 1
			obs = env.reset()

	env.close()


if __name__ == '__main__':
	main()
