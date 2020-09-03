import torch
import random
import numpy as np
from rlbotics.common.approximators import MLP
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


class MLPSoftmaxPolicy(MLP):
	def __init__(self, layer_sizes, activations, seed, optimizer='Adam', lr=0.01, weight_decay=0, batch_norm=False, weight_init=None):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, optimizer=optimizer, lr=lr,
						 weight_decay=weight_decay, batch_norm=batch_norm, weight_init=weight_init)
		torch.manual_seed(seed)

	def get_action(self, obs):
		with torch.no_grad():
			act_logits = self.predict(obs)

		act_dist = Categorical(logits=act_logits)
		return act_dist.sample().item()

	def get_log_prob(self, obs, act):
		act_logits = self.predict(obs)
		act_dist = Categorical(logits=act_logits)
		log_p = act_dist.log_prob(act)
		return log_p

	def get_distribution(self, obs):
		act_logits = self.predict(obs)
		act_dist = Categorical(logits=act_logits)
		return act_dist


class MLPGaussianPolicy(MLP):
	def __init__(self, layer_sizes, activations, seed, optimizer='Adam', lr=0.01, weight_decay=0, batch_norm=False, weight_init=None):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, optimizer=optimizer, lr=lr,
						 weight_decay=weight_decay, batch_norm=batch_norm, weight_init=weight_init)
		torch.manual_seed(seed)

		log_std = -0.5 * np.ones(layer_sizes[-1], dtype=np.float32)
		self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

	def get_action(self, obs):
		with torch.no_grad():
			act_logits = self.predict(obs)

		std = torch.exp(self.log_std)
		act_dist = Normal(act_logits, std)
		return act_dist.sample().item()

	def get_log_prob(self, obs, act):
		act_logits = self.predict(obs)
		std = torch.exp(self.log_std)
		act_dist = Normal(act_logits, std)
		log_p = act_dist.log_prob(act).sum(axis=-1)
		return log_p

	def get_policy(self, obs):
		act_logits = self.predict(obs)
		std = torch.exp(self.log_std)
		act_dist = Normal(mu, std)
		return act_dist


class MLPEpsilonGreedy(MLP):
	def __init__(self, layer_sizes, activations, seed, optimizer='Adam', lr=0.01, weight_decay=0, batch_norm=False, weight_init=None):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, optimizer=optimizer, lr=lr,
						 weight_decay=weight_decay, batch_norm=batch_norm, weight_init=weight_init)
		random.seed(seed)
		self.action_size = layer_sizes[-1]

	def get_action(self, obs, epsilon):
		if random.random() < epsilon:
			action = random.randrange(self.action_size)
		else:
			with torch.no_grad():
				output = self.predict(obs)
				action = output.argmax().item()
		return action


class MLPContinuous(MLP):
	def __init__(self, act_lim, layer_sizes, activations, seed, optimizer='Adam', lr=0.01, weight_decay=0,
				 batch_norm=False, weight_init=None):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, optimizer=optimizer, lr=lr,
						 weight_decay=weight_decay, batch_norm=batch_norm, weight_init=weight_init)
		self.act_lim = act_lim

	def get_action(self, obs):
		return self.predict(obs) * self.act_lim			# Multiply to scale to action space
