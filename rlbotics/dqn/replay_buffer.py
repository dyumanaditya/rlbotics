from collections import deque
import random


class ReplayBuffer:
	def __init__(self, max_len, batch_size):
		"""
		Buffer Memory for a DQN agent with random sampling
		"""
		self.memory = deque(maxlen=max_len)
		self.batch_size = batch_size

	def store_sample(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def sample(self):
		mini_batch = random.sample(self.memory, self.batch_size)
		return mini_batch

	def __len__(self):
		return len(self.memory)

