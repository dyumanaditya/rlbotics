import numpy as np
import torch.nn as nn
import rlbotics.dqn.hyperparameters as h
from rlbotics.dqn.replay_buffer import ReplayBuffer
from rlbotics.common.policies import MLPEpsilonGreedy
from rlbotics.common.approximators import MLP
import matplotlib.pyplot as plt


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
		self.render = h.render

		# Hyperparameters
		self.lr = h.lr
		self.gamma = h.gamma
		self.epsilon = h.epsilon
		self.min_epsilon = h.min_epsilon
		self.epsilon_decay = h.epsilon_decay
		self.batch_size = h.batch_size
		self.memory_limit = h.memory_limit
		self.num_episodes = h.num_episodes

		# Policy Network Hyperparameters
		self.start_learning = h.start_learning
		self.hidden_sizes = h.hidden_sizes
		self.activations = h.activations
		self.layer_types = h.layer_types
		self.optimizer = h.optimizer
		self.loss = h.loss

		# Target Network Hyperparameters
		self.update_target_net = h.update_target_net

		# Random Replay Memory
		self.memory = ReplayBuffer(self.memory_limit, self.batch_size)

		# Build Policy
		self._build_policy()

	def _build_policy(self):
		IO_sizes = [self.obs_dim, self.act_dim]

		self.policy = MLPEpsilonGreedy(IO_sizes, self.hidden_sizes, self.activations,
									   self.layer_types, self.optimizer, self.lr)
		self.target_policy = MLP(IO_sizes, self.hidden_sizes, self.activations, self.layer_types)
		print(self.policy.summary())

	def update_target(self):
		self.target_policy.model.load_state_dict(self.policy.model.state_dict())

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
				if h.env_name == 'CartPole-v1':
					reward = reward if not done or score == 499 else -100

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
			if h.env_name == 'CartPole-v1':
				score = score if score==500 else score +100
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
			return

		mini_batch = self.memory.sample()
		mini_batch = np.array(mini_batch).T
		states = np.vstack(mini_batch[0])		# States
		actions = list(mini_batch[1])			# Actions
		rewards = list(mini_batch[2])			# Rewards
		next_states = np.vstack(mini_batch[3])	# Next States
		done = list(mini_batch[4])				# Done

		target = self.policy.predict(states)
		q_val = self.policy.predict(states)
		q_val_targetNet = self.target_policy.predict(next_states)
		for i in range(self.batch_size):
			if done[i]:
				target[i][actions[i]] = rewards[i]
			else:
				target[i][actions[i]] = rewards[i] + self.gamma * torch.max(q_val_targetNet[i]).item()

		loss = self.compute_loss(q_val, target)
		self.policy.train(loss)

	def compute_loss(self, x, y):
		loss = nn.MSELoss()(x, y)
		return loss

import torch
torch.autograd.set_detect_anomaly(True)
import gym
env = gym.make(h.env_name)
agent = DQN(0, env)
agent.train()


