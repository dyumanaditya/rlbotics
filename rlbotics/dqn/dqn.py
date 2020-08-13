import torch
import torch.nn as nn

from rlbotics.dqn.replay_buffer import ReplayBuffer
import rlbotics.dqn.hyperparameters as h
from rlbotics.common.policies import MLPEpsilonGreedy
from rlbotics.common.approximators import MLP
from rlbotics.common.logger import Logger


class DQN:
	def __init__(self, env):
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.n

		# Replay buffer
		self.memory = ReplayBuffer(h.buffer_size)

		# Logger
		self.logger = Logger('DQN')

		# Decaying epsilon
		self.epsilon = h.epsilon

		# Device
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Loss function
		self.criterion = nn.MSELoss().to(self.device)

		# Build policies
		self._build_policy()

	def _build_policy(self):
		layer_sizes = [self.obs_dim] + h.hidden_sizes + [self.act_dim]
		self.policy = MLPEpsilonGreedy(layer_sizes=layer_sizes,
										activations=h.activations,
										optimizer=h.optimizer,
										lr=h.lr).to(self.device)

		self.target_policy = MLP(layer_sizes=layer_sizes, activations=h.activations).to(self.device)
		self.update_target_policy()

	def get_action(self, obs):
		# Decay epsilon
		if self.epsilon > h.min_epsilon:
			self.epsilon *= h.epsilon_decay
		return self.policy.get_action(obs, self.epsilon)

	def store_transition(self, obs, act, rew, new_obs, done):
		self.memory.add(obs, act, rew, new_obs, done)

		# Log Done, reward, epsilon data
		self.logger.save_tabular(done=done, rewards=rew, epsilon=self.epsilon)

	def update_policy(self):
		if len(self.memory) < h.batch_size:
			return

		# Sample batch of transitions
		transition_batch = self.memory.sample(h.batch_size)

		# Extract batches and convert to tensors
		obs_batch = torch.as_tensor(transition_batch.obs, dtype=torch.float).to(self.device)
		act_batch = torch.as_tensor(transition_batch.act).to(self.device)
		rew_batch = torch.as_tensor(transition_batch.rew).to(self.device)
		new_obs_batch = torch.as_tensor(transition_batch.new_obs, dtype=torch.float).to(self.device)
		done_batch = torch.as_tensor(transition_batch.done).to(self.device)
		not_done_batch = torch.logical_not(done_batch).to(self.device)

		# Update
		q_values = self.policy.predict(obs_batch).gather(1, act_batch.unsqueeze(1))
		target_values = torch.zeros(h.batch_size, device=self.device)
		target_values[not_done_batch] = self.target_policy.predict(new_obs_batch[not_done_batch]).max(1)[0].detach()

		expected_q_values = rew_batch + h.gamma * target_values
		expected_q_values = expected_q_values.unsqueeze(1)

		loss = self.criterion(q_values, expected_q_values)
		self.policy.learn(loss)

	def update_target_policy(self):
		self.target_policy.load_state_dict(self.policy.state_dict())
