import torch
import random
import itertools
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
	def __init__(self, layer_sizes, activations, seed, optimizer='Adam', lr=0.01, weight_decay=0, batch_norm=False, weight_init=None):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, optimizer=optimizer, lr=lr,
						 weight_decay=weight_decay, batch_norm=batch_norm, weight_init=weight_init)
		torch.manual_seed(seed)

		log_std = -0.5 * np.ones(layer_sizes[-1], dtype=np.float32)
		self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

	def get_action(self, obs):
		with torch.no_grad():
			mu = self.forward(obs)

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
				output = self.forward(obs)
				action = output.argmax().item()
		return action


class MLPActorContinuous(MLP):
	def __init__(self, act_lim, layer_sizes, activations, seed, optimizer='Adam', lr=0.01, weight_decay=0,
				 batch_norm=False, weight_init=None):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, optimizer=optimizer, lr=lr,
						 weight_decay=weight_decay, batch_norm=batch_norm, weight_init=weight_init)
		self.act_lim = act_lim

	def get_action(self, obs):
		return self.forward(obs) * self.act_lim			# Multiply to scale to action space


class MLPQFunctionContinuous(MLP):
	def __init__(self, layer_sizes, activations, seed, num_mlp=1, optimizer='Adam', lr=0.01, weight_decay=0,
				 batch_norm=False, weight_init=None):
		super().__init__(layer_sizes=layer_sizes, activations=activations, seed=seed, optimizer=optimizer, lr=lr,
						 weight_decay=weight_decay, batch_norm=batch_norm, weight_init=weight_init)

		# Build combined MLPs if necessary
		self.mlps = []
		params = []

		# Apply weight init to each mlp, gather the parameters
		for i in range(num_mlp):
			mlp = self._build_mlp(layer_sizes, activations, batch_norm)
			if weight_init is not None:
				mlp.apply(self.init_weights)
			self.mlps.append(mlp)
			params.append(mlp.parameters())

		# Modify MLP attributes to work with multiple MLPs
		comb_params = itertools.chain(*params)
		MLP.set_params(self, comb_params)
		MLP.set_optimizer(self, comb_params, optimizer, lr, weight_decay)

	def forward(self, x):
		if type(x) == np.ndarray:
			x = torch.from_numpy(x).float()
		x = x.view(-1, self.obs_dim)

		# Compute Q-val for each MLP
		q_vals = []
		for mlp in self.mlps:
			q_vals.append(mlp(x))

		# Deal with rank 1 tuples
		q_vals = tuple(q_vals)[0] if len(self.mlps) == 1 else tuple(q_vals)
		return q_vals

	def get_q_value(self, obs, act):
		return self.forward(torch.cat([obs, act], dim=-1))
