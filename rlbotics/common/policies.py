import torch
import random
import numpy as np
from rlbotics.common.approximators import MLP
from torch.distributions import Categorical
import torch.nn.functional as F


class MLPSoftmaxPolicy(MLP):
<<<<<<< HEAD
	def __init__(self, layer_sizes, activations, seed, optimizer='Adam', lr=0.01):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, optimizer=optimizer, lr=lr)
=======
	def __init__(self, layer_sizes, activations, seed, optimizer='Adam', lr=0.01, weight_decay=0, batch_norm=False):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, optimizer=optimizer, lr=lr,
						 weight_decay=weight_decay, batch_norm=batch_norm)
>>>>>>> 6217b6f2d9704876c70c2fbd96203e9ddc080790
		torch.manual_seed(seed)

	def get_action(self, obs):
		with torch.no_grad():
			act_logits = self.predict(obs)

<<<<<<< HEAD
		act_dist = Categorical(logits=act_logits)
		return act_dist.sample().item()

	def get_log_prob(self, obs, act):
		act_logits = self.predict(obs)
		act_dist = Categorical(logits=act_logits)
=======
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
>>>>>>> 6217b6f2d9704876c70c2fbd96203e9ddc080790
		log_p = act_dist.log_prob(act)
		return log_p

	def get_distribution(self, obs):
<<<<<<< HEAD
		act_logits = self.predict(obs)
		act_dist = Categorical(logits=act_logits)
=======
		if type(obs) == np.ndarray:
			obs = torch.from_numpy(obs).float().unsqueeze(0)
		act_logits = self.mlp(obs)
		action_prob = F.softmax(act_logits, dim=-1)
		act_dist = Categorical(action_prob)
>>>>>>> 6217b6f2d9704876c70c2fbd96203e9ddc080790
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
