import torch
import numpy as np
from copy import deepcopy

from rlbotics.common.loss import losses
from rlbotics.common.logger import Logger
from rlbotics.common.approximators import MLP
from rlbotics.common.policies import MLPContinuous
from rlbotics.ddpg.replay_buffer import ReplayBuffer
from rlbotics.ddpg.utils import OUNoise, GaussianNoise


class DDPG:
	"""
	NOTE: For Continuous environments only
	"""
	def __init__(self, args, env):
		self.act_dim = env.action_space.shape[0]
		self.obs_dim = env.observation_space.shape[0]
		self.act_lim = env.action_space.high[0]

		# General parameters
		self.seed = args.seed
		self.gamma = args.gamma
		self.save_freq = args.save_freq
		self.use_grad_clip = args.use_grad_clip

		# DDPG Specific Parameters
		self.batch_size = args.batch_size
		self.buffer_size = args.buffer_size
		self.polyak = args.polyak
		self.act_noise = args.act_noise
		self.noise_type = args.noise_type
		self.random_steps = args.random_steps
		self.update_after = args.update_after

		# Policy Network Parameters
		self.pi_lr = args.pi_lr
		self.pi_hidden_sizes = args.pi_hidden_sizes
		self.pi_activations = args.pi_activations
		self.pi_optimizer = args.pi_optimizer

		# Q Network Parameters
		self.q_lr = args.q_lr
		self.q_hidden_sizes = args.q_hidden_sizes
		self.q_activations = args.q_activations
		self.q_optimizer = args.q_optimizer
		self.q_loss_type = args.q_loss_type
		self.weight_decay = args.weight_decay

		# Both networks
		self.weight_init = args.weight_init
		self.batch_norm = args.batch_norm

		# Set device
		gpu = 0
		self.device = torch.device(f"cuda:{gpu}"if torch.cuda.is_available() else "cpu")

		# Initialize action noise
		if self.noise_type == 'ou':
			self.noise = OUNoise(self.seed, mu=np.zeros(self.act_dim))
		elif self.noise_type == 'gaussian':
			self.noise = GaussianNoise(self.seed, self.act_noise, self.act_dim)

		# Replay buffer
		self.memory = ReplayBuffer(self.buffer_size, self.seed)

		# Logger
		self.logger = Logger('DDPG', args.env_name, self.seed)

		# Gradient clipping
		if self.use_grad_clip:
			self.grad_clip = (-1, 1)
		else:
			self.grad_clip = None

		# Steps
		self.steps_done = -1

		# Loss function
		self.q_criterion = losses(self.q_loss_type)

		# Build pi and q Networks
		self._build_policy()
		self._build_q_function()

		# Log parameter data
		total_params = sum(p.numel() for p in self.pi.parameters())
		trainable_params = sum(p.numel() for p in self.pi.parameters() if p.requires_grad)
		self.logger.log(hyperparameters=vars(args), total_params=total_params, trainable_params=trainable_params)

	def _build_policy(self):
		layer_sizes = [self.obs_dim] + self.pi_hidden_sizes + [self.act_dim]
		self.pi = MLPContinuous(seed=self.seed,
								lr=self.pi_lr,
								act_lim=self.act_lim,
								layer_sizes=layer_sizes,
								batch_norm=self.batch_norm,
								optimizer=self.pi_optimizer,
								weight_init=self.weight_init,
					  			activations=self.pi_activations).to(self.device)

		self.pi.summary()
		self.pi_target = deepcopy(self.pi).to(self.device)

		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.pi_target.parameters():
			p.requires_grad = False

	def _build_q_function(self):
		layer_sizes = [self.obs_dim + self.act_dim] + self.q_hidden_sizes + [1]
		self.q = MLP(seed=self.seed,
					 lr=self.pi_lr,
					 layer_sizes=layer_sizes,
					 batch_norm=self.batch_norm,
					 optimizer=self.pi_optimizer,
					 weight_init=self.weight_init,
					 activations=self.pi_activations).to(self.device)

		self.q.summary()
		self.q_target = deepcopy(self.q).to(self.device)

		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters():
			p.requires_grad = False

	def _compute_q_loss(self, batch):
		obs_batch = torch.as_tensor(batch.obs, dtype=torch.float).to(self.device)
		act_batch = torch.as_tensor(batch.act).to(self.device)
		rew_batch = torch.as_tensor(batch.rew).unsqueeze(1).to(self.device)
		new_obs_batch = torch.as_tensor(batch.new_obs, dtype=torch.float).to(self.device)
		done_batch = torch.as_tensor(batch.done).to(self.device)
		not_done_batch = torch.logical_not(done_batch).unsqueeze(1).to(self.device)

		pred_q = self.q(torch.cat([obs_batch, act_batch], dim=-1))

		# Bellman backup for Q function
		with torch.no_grad():
			targ_pi = self.pi.get_action(new_obs_batch)
			targ_q  = self.q_target(torch.cat([new_obs_batch, targ_pi], dim=-1))
			expected_q = rew_batch + self.gamma * (targ_q * not_done_batch)

		loss = self.q_criterion(pred_q, expected_q.float()).to(self.device)
		return loss

	def _compute_pi_loss(self, batch):
		obs_batch = torch.as_tensor(batch.obs, dtype=torch.float).to(self.device)
		pi = self.pi.get_action(obs_batch)
		q = self.q(torch.cat([obs_batch, pi], dim=-1))
		return -q.mean()

	def update_actor_critic(self):
		if self.steps_done < self.update_after:
			return

		# Sample batch of transitions
		transition_batch = self.memory.sample(self.batch_size)

		# Update q Network
		q_loss = self._compute_q_loss(transition_batch)
		self.q.learn(q_loss, grad_clip=self.grad_clip)

		# Freeze Q-network so you don't waste computational effort
		# computing gradients for it during the policy learning step.
		for p in self.q.parameters():
			p.requires_grad = False

		# Update pi Network
		pi_loss = self._compute_pi_loss(transition_batch)
		self.pi.learn(pi_loss, grad_clip=self.grad_clip)

		# Unfreeze Q-network so you can optimize it at next DDPG step.
		for p in self.q.parameters():
			p.requires_grad = True

		# Log Model and Loss
		if self.steps_done % self.save_freq == 0:
			self.logger.log_model(self.q, name='q')
			self.logger.log_model(self.pi, name='pi')
		self.logger.log(name='policy_updates', q_loss=q_loss.item(), pi_loss=pi_loss.item())

		# Update Target Networks
		self._update_target_actor_critic()

	def _update_target_actor_critic(self):
		# Polyak averaging
		with torch.no_grad():
			for p, p_targ in zip(self.q.parameters(), self.q_target.parameters()):
				p_targ.data.copy_(self.polyak*p.data + (1-self.polyak)*p_targ.data)
			for p, p_targ in zip(self.pi.parameters(), self.pi_target.parameters()):
				p_targ.data.copy_(self.polyak*p.data + (1-self.polyak)*p_targ.data)

	def get_action(self, obs):
		self.pi.eval()
		action = self.pi.get_action(obs).detach().numpy()
		action += self.noise()
		return np.clip(action, -self.act_lim, self.act_lim)[0]

	def store_transition(self, obs, act, rew, new_obs, done):
		self.memory.add(obs, act, rew, new_obs, done)
		self.steps_done += 1

		# Log Done, reward data
		self.logger.log(name='transitions', done=done, rewards=rew)
