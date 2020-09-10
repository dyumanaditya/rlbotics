import os
import math
import torch
import pandas as pd
from copy import deepcopy

from rlbotics.common.loss import losses
from rlbotics.common.logger import Logger
from rlbotics.dqn.replay_buffer import ReplayBuffer
from rlbotics.common.policies import MLPEpsilonGreedy


class DQN:
	def __init__(self, args, env):
		self.env_name = args.env_name
		self.act_dim = env.action_space.n
		self.obs_dim = env.observation_space.shape[0]

		# General parameters
		self.lr = args.lr
		self.seed = args.seed
		self.gamma = args.gamma
		self.resume = args.resume
		self.save_freq = args.save_freq

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

		# Set device
		gpu = 0
		self.device = torch.device(f"cuda:{gpu}"if torch.cuda.is_available() else "cpu")

		# Replay buffer
		self.memory = ReplayBuffer(self.buffer_size, self.seed)

		# Logger
		self.logger = Logger('DQN', args.env_name, self.seed, resume=self.resume)

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
		if self.resume:
			self.resume_training()

		# Log parameter data
		total_params = sum(p.numel() for p in self.policy.parameters())
		trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
		self.logger.log(hyperparameters=vars(args), total_params=total_params, trainable_params=trainable_params)

	def _build_policy(self):
		layer_sizes = [self.obs_dim] + self.hidden_sizes + [self.act_dim]
		self.policy = MLPEpsilonGreedy(layer_sizes=layer_sizes,
									   activations=self.activations,
									   seed=self.seed).to(self.device)

		self.target_policy = deepcopy(self.policy).to(self.device)
		self.policy.summary()

		# Set Optimizer
		if self.optimizer == 'Adam':
			self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
		elif self.optimizer == 'RMSprop':
			self.q_optim = torch.optim.RMSprop(self.policy.parameters(), lr=self.lr)
		else:
			raise NameError(str(self.optimizer) + ' Optimizer not supported')

	def get_action(self, obs):
		action = self.policy.get_action(obs, self.epsilon)
		self.decay_epsilon(mode='exp')
		return action

	def decay_epsilon(self, mode):
		if mode == 'exp':
			self.epsilon = self.min_epsilon + (1.0 - self.min_epsilon) * math.exp(-self.exp_decay * self.steps_done)
			self.steps_done += 1
		elif mode == 'linear':
			self.epsilon = max(self.min_epsilon, self.epsilon-self.linear_decay)

	def store_transition(self, obs, act, rew, new_obs, done):
		self.memory.add(obs, act, rew, new_obs, done)

		# Log Done, reward, epsilon data only after policy updates start
		if len(self.memory) >= self.batch_size:
			self.logger.log(name='transitions', done=done, rewards=rew, epsilon=self.epsilon)

	def update_policy(self):
		if len(self.memory) < self.batch_size:
			return

		# Sample batch of transitions
		transition_batch = self.memory.sample(self.batch_size)

		# Extract batches and convert to tensors
		obs_batch = torch.as_tensor(transition_batch.obs, dtype=torch.float).to(self.device)
		act_batch = torch.as_tensor(transition_batch.act).to(self.device)
		rew_batch = torch.as_tensor(transition_batch.rew).to(self.device)
		new_obs_batch = torch.as_tensor(transition_batch.new_obs, dtype=torch.float).to(self.device)
		done_batch = torch.as_tensor(transition_batch.done).to(self.device)
		not_done_batch = torch.logical_not(done_batch).to(self.device)

		# Update
		q_values = self.policy(obs_batch).gather(1, act_batch.unsqueeze(1))
		target_values = torch.zeros(self.batch_size).to(self.device)
		target_values[not_done_batch] = self.target_policy(new_obs_batch[not_done_batch]).max(1)[0].detach()

		expected_q_values = rew_batch + self.gamma * target_values
		expected_q_values = expected_q_values.unsqueeze(1).to(self.device)

		loss = self.criterion(q_values, expected_q_values.float()).to(self.device)

		# Learn
		self.optim.zero_grad()
		loss.backward()
		if self.grad_clip is not None:
			for param in self.policy.parameters():
				param.grad.data.clamp_(self.grad_clip[0], self.grad_clip[1])
		self.optim.step()

		# Log Model and Loss
		if self.steps_done % self.save_freq == 0:
			self.logger.log_model(self.policy)
			self.logger.log_state_dict(self.optim.state_dict(), name='optim')
		self.logger.log(name='policy_updates', loss=loss.item())

	def update_target_policy(self):
		self.target_policy.load_state_dict(self.policy.state_dict())

	def resume_training(self):
		print('Resuming training...')
		print('Loading previously saved models...')

		# Load saved models
		model_file = os.path.join('experiments', 'models', 'DQN' + '_' + self.env_name + '_' + str(self.seed))
		self.policy = torch.load(os.path.join(model_file, 'model.pth'))
		self.taget_policy = deepcopy(self.policy)

		# Load optimizer state_dicts
		self.optim.load_state_dict(torch.load(os.path.join(model_file, 'optim')))

		# Start where we left off
		log_file = os.path.join('experiments', 'logs', 'DQN' + '_' + self.env_name + '_' + str(self.seed), 'transitions.csv')
		log = pd.read_csv(log_file)
		self.steps_done = len(log) + self.batch_size
