import gym
import matplotlib.pyplot as plt
from rlbotics.dqn.dqn import DQN
import rlbotics.dqn.hyperparameters as h


def train(agent):
	max_rew = []
	time_step = 1

	for e in range(h.num_episodes):
		done = False
		ep_rew = 0
		state = agent.env.reset()

		while not done:
			if h.render:
				agent.env.render()
			# Take action
			act = agent.policy.get_action(state, agent.epsilon)
			new_obs, rew, done, info = agent.env.step(act)

			# Decay epsilon
			if agent.epsilon > h.min_epsilon:
				agent.epsilon *= h.epsilon_decay

			# Store experience
			agent.memory.store_sample(state, act, rew, new_obs, done)

			# Update target model
			if time_step % h.update_target_net == 0:
				agent.update_target()

			# Learn:
			agent.update_policy()
			time_step += 1
			ep_rew += rew

		# Episode done
		max_rew.append(ep_rew)

		if e % 20 == 0:
			print("episode:", e, "  reward:", ep_rew)

	plt.xlabel('episodes')
	plt.ylabel('max-rewards')
	plt.plot(max_rew, 'b-')
	plt.show()


def main():
	env = gym.make(h.env_name)
	agent = DQN(0, env)
	train(agent)


if __name__ == '__main__':
	main()