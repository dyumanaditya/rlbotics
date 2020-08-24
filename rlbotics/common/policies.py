import torch
import random
import numpy as np
from rlbotics.common.approximators import MLP
from torch.distributions import Categorical
import torch.nn.functional as F


class MLPSoftmaxPolicy(MLP):
	def __init__(self, layer_sizes, activations, seed, optimizer='Adam', lr=0.01):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, optimizer=optimizer, lr=lr)
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


class MLPGaussian(MLP):
	def __init__(self, layer_sizes, activations, seed, optimizer='Adam', lr=0.01, weight_decay=0):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, optimizer=optimizer, lr=lr, weight_decay=weight_decay)
		torch.manual_seed(seed)

	def get_policy(self, obs):
		pass

	def get_action(self, obs):
		pass

	def get_log_prob(self, obs, act):
		pass


class MLPEpsilonGreedy(MLP):
	def __init__(self, layer_sizes, activations, seed, optimizer='Adam', lr=0.01, weight_decay=0):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, optimizer=optimizer, lr=lr, weight_decay=weight_decay)
		random.seed(seed)
		torch.manual_seed(seed)
		self.action_size = layer_sizes[-1]

	def get_action(self, obs, epsilon):
		if random.random() < epsilon:
			action = random.randrange(self.action_size)
		else:
			with torch.no_grad():
				output = self.predict(obs)
				action = output.argmax().item()
		return action
