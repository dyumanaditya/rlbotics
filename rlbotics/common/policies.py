import torch
import random
import numpy as np
from rlbotics.common.approximators import MLP
from torch.distributions import Categorical
import torch.nn.functional as F


class MLPSoftmaxPolicy(MLP):
	def __init__(self, layer_sizes, activations, seed, optimizer='Adam', lr=0.01, weight_decay=0, batch_norm=False):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, optimizer=optimizer, lr=lr,
						 weight_decay=weight_decay, batch_norm=batch_norm)
		torch.manual_seed(seed)

	def get_action(self, obs):
		if type(obs) == np.ndarray:
			obs = torch.from_numpy(obs).float().unsqueeze(0)

		with torch.no_grad():
			act_logits = self.mlp(obs)

		action_prob = F.softmax(act_logits, dim=-1)
		act_dist = Categorical(action_prob)
		act = act_dist.sample()
		return act

	def get_log_prob(self, obs, act):
		if type(obs) == np.ndarray:
			obs = torch.from_numpy(obs).float().unsqueeze(0)
		act_logits = self.mlp(obs)
		action_prob = F.softmax(act_logits, dim=-1)
		act_dist = Categorical(action_prob)
		log_p = act_dist.log_prob(act)
		return log_p

	def get_distribution(self, obs):
		if type(obs) == np.ndarray:
			obs = torch.from_numpy(obs).float().unsqueeze(0)
		act_logits = self.mlp(obs)
		action_prob = F.softmax(act_logits, dim=-1)
		act_dist = Categorical(action_prob)
		return act_dist


class MLPGaussian(MLP):
	def __init__(self, layer_sizes, activations, seed, optimizer='Adam', lr=0.01, weight_decay=0, batch_norm=False):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, optimizer=optimizer, lr=lr,
						 weight_decay=weight_decay, batch_norm=batch_norm)
		torch.manual_seed(seed)

	def get_policy(self, obs):
		pass

	def get_action(self, obs):
		pass

	def get_log_prob(self, obs, act):
		pass


class MLPEpsilonGreedy(MLP):
	def __init__(self, layer_sizes, activations, seed, optimizer='Adam', lr=0.01, weight_decay=0, batch_norm=False):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, optimizer=optimizer, lr=lr,
						 weight_decay=weight_decay, batch_norm=batch_norm)
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
