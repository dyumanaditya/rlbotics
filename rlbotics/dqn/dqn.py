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

	def update_policy(self):
		if len(self.memory) < h.start_learning:
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
		for i in range(h.batch_size):
			if done[i]:
				target[i][actions[i]] = rewards[i]
			else:
				target[i][actions[i]] = rewards[i] + h.gamma * torch.max(q_val_targetNet[i]).item()

		self.loss = self.criterion(q_val, target)
		self.policy.train(self.loss)
