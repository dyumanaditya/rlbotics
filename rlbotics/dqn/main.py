import gym
import matplotlib.pyplot as plt
from rlbotics.dqn.dqn import DQN
import rlbotics.dqn.hyperparameters as h


def main():
	# Build environment
	env = gym.make(h.env_name)
	agent = DQN(env)

	# Episode related information
	max_rew = []
	time_step = 1
	ep_counter = 0
	max_iterations = 100000

	for e in range(h.num_episodes):
		done = False
		ep_rew = 0
		obs = env.reset()

		while not done:
			if time_step >= max_iterations:
				break
			if h.render:
				env.render()

			# Take action
			act = agent.get_action(obs)
			new_obs, rew, done, info = env.step(act)

			# Decay epsilon
			if agent.epsilon > h.min_epsilon:
				agent.epsilon *= h.epsilon_decay

			# Store experience
			agent.store_transition(obs, act, rew, new_obs, done)

			# Update target model
			if time_step % h.update_target_freq == 0:
				agent.update_target_policy()

			# Learn:
			agent.update_policy()

			# Logging
			obs = new_obs
			time_step += 1
			ep_rew += rew

		# Episode done
		max_rew.append(ep_rew)
		ep_counter += 1

		# Display results
		print("episode: {}, total reward: {}".format(ep_counter, ep_rew))

	plt.xlabel('episodes')
	plt.ylabel('max-rewards')
	plt.plot(max_rew, 'b-')
	plt.show()


if __name__ == '__main__':
	main()