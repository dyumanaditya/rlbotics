import torch
import numpy as np
from rlbotics.common.loss import loss
import rlbotics.dqn.hyperparameters as h
from rlbotics.dqn.replay_buffer import ReplayBuffer
from rlbotics.common.policies import MLPEpsilonGreedy
from rlbotics.common.approximators import MLP


class DQN:
	def __init__(self, args, env):
		"""
		:param args: ArgsParser that include hyperparameters
		:param env: Gym environment
		"""
		# General Setup
		self.env = env
		self.obs_dim = self.env.observation_space.shape[0]
		self.act_dim = self.env.action_space.n
		self.epsilon = h.epsilon
		self.criterion = loss(h.loss)

		# Random Replay Memory
		self.memory = ReplayBuffer(h.memory_limit, h.batch_size)

		# Build Policy
		self._build_policy()
		self.update_target()

	def _build_policy(self):
		layer_sizes = [self.obs_dim] + h.hidden_sizes + [self.act_dim]
		self.policy = MLPEpsilonGreedy(layer_sizes, h.activations, h.optimizer, h.lr)
		self.target_policy = MLP(layer_sizes, h.activations)
		print(self.policy.summary())

	def update_target(self):
		self.target_policy.model.load_state_dict(self.policy.model.state_dict())

<<<<<<< Updated upstream
	def update_policy(self):
		if len(self.memory) < h.start_learning:
=======
	def train(self):
		scores, episodes = [], []

		for episode in range(self.num_episodes):
			done = False
			score = 0
			time_step = 0
			state = self.env.reset()

			while not done:
				if self.render:
					self.env.render()

				# Take action
				action = self.policy.get_action(state, self.epsilon)
				next_state, reward, done, info = self.env.step(action)

				# Decay epsilon
				if self.epsilon > self.min_epsilon:
					self.epsilon *= self.epsilon_decay

				# Punish if episode ends for cartpole
				# if h.env_name == 'CartPole-v1':
				# 	reward = reward if not done or score == 499 else -100

				# Store experience
				self.memory.store_sample(state, action, reward, next_state, done)

				# Update target model
				if time_step % self.update_target_net == 0:
					self.update_target()

				# Learn:
				self.learn()
				time_step += 1
				score += reward

			# Episode done
			# if h.env_name == 'CartPole-v1':
			# 	score = score if score==500 else score +100
			scores.append(score)
			episodes.append(episode)
			print("episode:", episode, "  score:", score, "  memory length:",
				  len(self.memory), "  epsilon:", self.epsilon)
		plt.xlabel('episodes')
		plt.ylabel('score')
		plt.plot(episodes, scores, 'b-')
		plt.show()

	def learn(self):
		if len(self.memory) < self.start_learning:
>>>>>>> Stashed changes
			return

		mini_batch = self.memory.sample()
		mini_batch = np.array(mini_batch).T
		states = np.vstack(mini_batch[0])		# States
		actions = list(mini_batch[1])			# Actions
		rewards = list(mini_batch[2])			# Rewards
		next_states = np.vstack(mini_batch[3])	# Next States
		done = list(mini_batch[4])				# Done
		print(states.shape)


		target = self.policy.predict(states)
		q_val = self.policy.predict(states)
<<<<<<< Updated upstream
		q_val_targetNet = self.target_policy.predict(next_states)
		for i in range(h.batch_size):
=======
		q_val_targetNet = self.target_policy.predict(next_states).detach().max(1)[0]
		for i in range(self.batch_size):
>>>>>>> Stashed changes
			if done[i]:
				target[i][actions[i]] = rewards[i]
			else:
				target[i][actions[i]] = rewards[i] + h.gamma * torch.max(q_val_targetNet[i]).item()

		self.loss = self.criterion(q_val, target)
		self.policy.train(self.loss)
