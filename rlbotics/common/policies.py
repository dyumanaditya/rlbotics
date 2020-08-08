import torch
import numpy as np
from torch.distributions import Categorical
from rlbotics.common.approximators import MLP


class MLPSoftmaxPolicy(MLP):
	def __init__(self, layer_sizes, activations, optimizer='Adam', lr=0.01):
		super().__init__(layer_sizes=layer_sizes, activations=activations, optimizer=optimizer, lr=lr)

	def get_policy(self, obs):
		act_logits = self.predict(obs)
		act_dist = Categorical(logits=act_logits)
		return act_dist

	def get_action(self, obs):
		return self.get_policy(obs).sample().item()

	def get_log_prob(self, obs, act):
		act_dist = self.get_policy(obs)
		#log_p = act_dist.probs[act]
		log_p = act_dist.log_prob(torch.as_tensor(act, dtype=torch.int32))
		return log_p


class MLPGaussian(MLP):
	def __init__(self, layer_sizes, activations, optimizer='Adam', lr=0.01):
		super().__init__(layer_sizes=layer_sizes, activations=activations, optimizer=optimizer, lr=lr)

	def get_policy(self, obs):
		pass

	def get_action(self, obs):
		pass

	def get_log_prob(self, obs, act):
		pass


class MLPEpsilonGreedy(MLP):
<<<<<<< Updated upstream
	def __init__(self, layer_sizes, activations, optimizer='Adam', lr=0.01):
		super().__init__(layer_sizes=layer_sizes, activations=activations, optimizer=optimizer, lr=lr)
		self.action_size = layer_sizes[-1]
=======
	def __init__(self, IO_sizes, hidden_sizes, activations, layer_types, optimizer='Adam', lr=0.01):
		super().__init__(IO_sizes=IO_sizes, hidden_sizes=hidden_sizes, activations=activations, layer_types=layer_types, optimizer=optimizer, lr=lr)
		self.act_dim = IO_sizes[1]
>>>>>>> Stashed changes

	def get_action(self, obs, epsilon):
		if np.random.rand() < epsilon:
			action = np.random.randint(0, self.act_dim)
		else:
			with torch.no_grad():
				act_logits = self.predict(obs)
				action = act_logits.argmax().item()
		return action


