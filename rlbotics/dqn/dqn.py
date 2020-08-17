import torch
import math

from rlbotics.common.loss import losses
from rlbotics.common.logger import Logger
from rlbotics.common.approximators import MLP
from rlbotics.ddqn.replay_buffer import ReplayBuffer
from rlbotics.common.policies import MLPEpsilonGreedy


class DQN:
	def __init__(self, args, env):
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.n

		# General parameters
		self.lr = args.lr
		self.gamma = args.gamma
		self.seed = args.seed

		# DQN specific parameters
		self.epsilon = args.epsilon
		self.min_epsilon = args.min_epsilon
		self.exp_decay = args.exp_decay
		self.linear_decay = args.linear_decay
		self.batch_size = args.batch_size
		self.buffer_size = args.buffer_size

		# Policy network parameters
		self.loss_type = args.loss_type
		self.optimizer = args.optimizer
		self.use_grad_clip = args.use_grad_clip
		self.activations = args.activations
		self.hidden_sizes = args.hidden_sizes

		# Replay buffer
		self.memory = ReplayBuffer(self.buffer_size)

		# Logger
		self.logger = Logger('DQN', args.env_name, self.seed)

		# Gradient clipping
		if self.use_grad_clip:
			self.grad_clip = (-1, 1)
		else:
			self.grad_clip = None

		# Steps
		self.steps_done = 0

		# Loss function
		self.criterion = losses(self.loss_type)

		# Build policies
		self._build_policy()

		# Log parameter data
		total_params = sum(p.numel() for p in self.policy.parameters())
		trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
		self.logger.log(hyperparameters=vars(args), total_params=total_params, trainable_params=trainable_params)

	def _build_policy(self):
		layer_sizes = [self.obs_dim] + self.hidden_sizes + [self.act_dim]
		self.policy = MLPEpsilonGreedy(layer_sizes=layer_sizes,
									   activations=self.activations,
									   optimizer=self.optimizer,
									   lr=self.lr)

		self.target_policy = MLP(layer_sizes=layer_sizes, activations=self.activations)
		self.update_target_policy()

	def get_action(self, obs):
		action = self.policy.get_action(obs, self.epsilon)
		self.decay_epsilon(mode='exp')
		return action

	def decay_epsilon(self, mode):
		if mode == 'exp':
			self.epsilon = self.min_epsilon + (self.epsilon - self.min_epsilon) * math.exp(-1. * self.steps_done / self.exp_decay)
			self.steps_done += 1
		elif mode == 'linear':
			self.epsilon = max(self.min_epsilon, self.epsilon-self.linear_decay)

	def store_transition(self, obs, act, rew, new_obs, done):
		self.memory.add(obs, act, rew, new_obs, done)

		# Log Done, reward, epsilon data
		self.logger.log(name='transitions', done=done, rewards=rew, epsilon=self.epsilon)

	def update_policy(self):
		if len(self.memory) < self.batch_size:
			return

		# Sample batch of transitions
		transition_batch = self.memory.sample(self.batch_size)

		# Extract batches and convert to tensors
		obs_batch = torch.as_tensor(transition_batch.obs, dtype=torch.float)
		act_batch = torch.as_tensor(transition_batch.act)
		rew_batch = torch.as_tensor(transition_batch.rew)
		new_obs_batch = torch.as_tensor(transition_batch.new_obs, dtype=torch.float)
		done_batch = torch.as_tensor(transition_batch.done)
		not_done_batch = torch.logical_not(done_batch)

		# Update
		q_values = self.policy.predict(obs_batch).gather(1, act_batch.unsqueeze(1))
		target_values = torch.zeros(self.batch_size)
		target_values[not_done_batch] = self.target_policy.predict(new_obs_batch[not_done_batch]).max(1)[0].detach()

		expected_q_values = rew_batch + self.gamma * target_values
		expected_q_values = expected_q_values.unsqueeze(1)

		loss = self.criterion(q_values, expected_q_values)
		self.policy.learn(loss, grad_clip=self.grad_clip)

		# Log Model
		self.logger.log_model(self.policy)

	def update_target_policy(self):
		self.target_policy.load_state_dict(self.policy.state_dict())
