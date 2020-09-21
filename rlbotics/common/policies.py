import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
import numpy as np
from rlbotics.common.approximators import MLP
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


class MLPSoftmaxPolicy(MLP):
	def __init__(self, layer_sizes, activations, seed, batch_norm=False, weight_init=None):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, batch_norm=batch_norm, weight_init=weight_init)
		torch.manual_seed(seed)

	def get_action(self, obs):
		with torch.no_grad():
			act_logits = self.forward(obs)

		act_dist = Categorical(logits=act_logits)
		return act_dist.sample().item()

	def get_log_prob(self, obs, act):
		act_logits = self.forward(obs)
		act_dist = Categorical(logits=act_logits)
		log_p = act_dist.log_prob(act)
		return log_p

	def get_distribution(self, obs):
		act_logits = self.forward(obs)
		act_dist = Categorical(logits=act_logits)
		return act_dist


class MLPGaussianPolicy(MLP):
	def __init__(self, act_lim, layer_sizes, activations, seed, batch_norm=False, weight_init=None):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, batch_norm=batch_norm, weight_init=weight_init)
		torch.manual_seed(seed)
		self.act_lim = act_lim

		log_std = -0.5 * np.ones(layer_sizes[-1], dtype=np.float32)
		self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

	def get_action(self, obs):
		with torch.no_grad():
			mu = self.forward(obs) * self.act_lim

		std = torch.exp(self.log_std)
		act_dist = Normal(mu, std)
		return act_dist.sample().numpy()[0]

	def get_log_prob(self, obs, act):
		mu = self.forward(obs)
		std = torch.exp(self.log_std)
		act_dist = Normal(mu, std)
		log_p = act_dist.log_prob(act).sum(axis=-1)
		return log_p

	def get_distribution(self, obs):
		mu = self.forward(obs)
		std = torch.exp(self.log_std)
		act_dist = Normal(mu, std)
		return act_dist


class MLPSquashedGaussianPolicy(MLP):
	def __init__(self, act_lim, layer_sizes, activations, seed, batch_norm=False, weight_init=None):
		super().__init__(layer_sizes=layer_sizes[:-1], activations=activations[:-1], seed=seed, batch_norm=batch_norm, weight_init=weight_init)
		self.mu_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])
		self.log_std_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])
		self.act_limit = act_lim

		self.LOG_STD_MAX = 2
		self.LOG_STD_MIN = -20

	def get_action(self, obs, deterministic=False, with_logprob=True):
		net_out = self.forward(obs)
		mu = self.mu_layer(net_out)
		log_std = self.log_std_layer(net_out)
		log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
		std = torch.exp(log_std)

		# Pre-squash distribution and sample
		pi_distribution = Normal(mu, std)
		if deterministic:
			# Only used for evaluating policy at test time.
			pi_action = mu
		else:
			pi_action = pi_distribution.rsample()

		if with_logprob:
			logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
			logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
		else:
			logp_pi = None

		pi_action = torch.tanh(pi_action)
		pi_action = self.act_limit * pi_action

		return pi_action, logp_pi

class MLPEpsilonGreedy(MLP):
	def __init__(self, layer_sizes, activations, seed, batch_norm=False, weight_init=None):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, batch_norm=batch_norm, weight_init=weight_init)
		random.seed(seed)
		self.action_size = layer_sizes[-1]

	def get_action(self, obs, epsilon):
		if random.random() < epsilon:
			action = random.randrange(self.action_size)
		else:
			with torch.no_grad():
				output = self.forward(obs)
				action = output.argmax().item()
		return action


class MLPActorContinuous(MLP):
	def __init__(self, act_lim, layer_sizes, activations, seed, batch_norm=False, weight_init=None):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, batch_norm=batch_norm, weight_init=weight_init)
		self.act_lim = act_lim

	def get_action(self, obs):
		return self.forward(obs) * self.act_lim			# Multiply to scale to action space


class MLPQFunctionContinuous(MLP):
	def __init__(self, layer_sizes, activations, seed, batch_norm=False, weight_init=None):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, batch_norm=batch_norm, weight_init=weight_init)

	def get_q_value(self, obs, act):
		return self.forward(torch.cat([obs, act], dim=-1))
