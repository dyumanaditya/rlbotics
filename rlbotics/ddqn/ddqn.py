import torch
import torch.nn as nn

from rlbotics.dqn.replay_buffer import ReplayBuffer
import rlbotics.dqn.hyperparameters as h
from rlbotics.common.policies import MLPEpsilonGreedy
from rlbotics.common.approximators import MLP
from rlbotics.common.logger import Logger


class DDQN:
	def __init__(self, env):
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.n

		# Replay buffer
		self.memory = ReplayBuffer(h.buffer_size)

		# Logger
		self.logger = Logger('DDQN')

		# Decaying epsilon (exp. and linear)
		self.epsilon = h.epsilon

		# Gradient clipping
		if h.grad_clip:
			self.grad_clip = (-1, 1)
		else:
			self.grad_clip = None

		# Steps
		self.steps_done = 0

		# Loss function
		self.criterion = nn.MSELoss()

		# Build policies
		self._build_policy()



	def _build_policy(self):
		layer_sizes = [self.obs_dim] + h.hidden_sizes + [self.act_dim]
		self.policy = MLPEpsilonGreedy(layer_sizes=layer_sizes,
									   activations=h.activations,
									   optimizer=h.optimizer,
									   lr=h.lr)

		self.target_policy = MLP(layer_sizes=layer_sizes, activations=h.activations)
		self.update_target_policy()

	def get_action(self, obs):
		action = self.policy.get_action(obs, self.epsilon)
		self.decay_epsilon(mode='exp')
		return action

	def decay_epsilon(self, mode):
		if mode == 'exp':
			# self.epsilon = max(h.min_epsilon, self.epsilon*h.epsilon_decay)
			self.epsilon = h.min_epsilon + (h.epsilon - h.min_epsilon) * math.exp(-1. * self.steps_done / h.epsilon_decay)
			self.steps_done += 1
		elif mode == 'linear':
			self.epsilon = max(h.min_epsilon, self.epsilon-h.linear_decay)

	def store_transition(self, obs, act, rew, new_obs, done):
		self.memory.add(obs, act, rew, new_obs, done)

		# Log Done, reward, epsilon data
		#self.logger.save_tabular(done=done, rewards=rew, epsilon=self.epsilon)

	def update_policy(self):
		if len(self.memory) < h.batch_size:
			return

		# Sample batch of transitions
		transition_batch = self.memory.sample(h.batch_size)

		# Extract batches and convert to tensors
		obs_batch = torch.as_tensor(transition_batch.obs, dtype=torch.float)
		act_batch = torch.as_tensor(transition_batch.act)
		rew_batch = torch.as_tensor(transition_batch.rew)
		new_obs_batch = torch.as_tensor(transition_batch.new_obs, dtype=torch.float)
		done_batch = torch.as_tensor(transition_batch.done)
		not_done_batch = torch.logical_not(done_batch)

		# Update
		out = self.policy.predict(obs_batch)

		q_values = self.policy.predict(obs_batch).gather(1, act_batch.unsqueeze(1))
		next_state_q_values = self.policy.predict(new_obs_batch[not_done_batch]).argmax(1)
		target_values = torch.zeros(h.batch_size, 1)
		target_values[not_done_batch] = self.target_policy.predict(new_obs_batch[not_done_batch]).gather(1, next_state_q_values.unsqueeze(1)).detach()

		expected_q_values = rew_batch.unsqueeze(1) + h.gamma * target_values

		loss = self.criterion(q_values, expected_q_values)
		self.policy.learn(loss, grad_clip=self.grad_clip)

	def update_target_policy(self):
		self.target_policy.load_state_dict(self.policy.state_dict())
