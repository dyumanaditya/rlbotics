import torch
import random
from rlbotics.common.approximators import MLP
from torch.distributions import Categorical


class MLPSoftmaxPolicy(MLP):
	def __init__(self, layer_sizes, activations, optimizer='Adam', lr=0.01):
		super().__init__(layer_sizes=layer_sizes, activations=activations, optimizer=optimizer, lr=lr)

	def get_policy(self, obs):
		#with torch.no_grad():
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
	def __init__(self, layer_sizes, activations, optimizer='Adam', lr=0.01):
		super().__init__(layer_sizes=layer_sizes, activations=activations, optimizer=optimizer, lr=lr)
		self.action_size = layer_sizes[-1]

	def get_action(self, obs, epsilon):
		if random.random() < epsilon:
			action = random.randrange(self.action_size)
		else:
			with torch.no_grad():
				output = self.predict(obs)
				action = output.argmax().item()
		return action
