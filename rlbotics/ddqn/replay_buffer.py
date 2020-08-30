from collections import namedtuple, deque
import numpy as np
import random


class ReplayBuffer:
	def __init__(self, buffer_size, seed):
		random.seed(seed)
		self.memory = deque(maxlen=int(buffer_size))
		self.transition = namedtuple("transition", field_names=["obs", "act", "rew", "new_obs", "done"])

	def add(self, obs, act, rew, new_obs, done):
		transition = self.transition(obs=obs, act=act, rew=rew, new_obs=new_obs, done=done)
		self.memory.append(transition)

	def sample(self, batch_size):
		transition_batch = random.sample(self.memory, batch_size)

		# Transpose transitions
		transition_batch = self.transition(*zip(*transition_batch))
		return transition_batch

	def __len__(self):
		return len(self.memory)


class SumTree:
	def __init__(self, capacity):
		self.data_pointer = 0
		self.capacity = capacity
		self.tree = np.zeros(2*capacity-1)              # Create tree
		self.data = np.zeros(capacity, dtype=object)    # Experiences
		self.n_entries = 0                              # Num of entries in tree (for IS)

	def add(self, priority, data):
		tree_index = self.data_pointer + self.capacity -1
		self.data[self.data_pointer] = data

		self._propagate(tree_index, priority)           # Update tree
		self.data_pointer += 1

		if self.data_pointer >= self.capacity:          # reset if capacity limit exceeded
			self.data_pointer = 0

		if self.n_entries < self.capacity:              # Increment num of entries
			self.n_entries += 1

	def _propagate(self, tree_index, priority):
		change = priority - self.tree[tree_index]       # Change = new priority score - former priority score
		self.tree[tree_index] = priority

		# Then propagate the change through the tree
		while tree_index != 0:
			# Add change to the parents
			tree_index = (tree_index - 1) // 2
			self.tree[tree_index] += change

	def update(self, tree_index, priority):
		self._propagate(tree_index, priority)

	def total(self):
		return self.tree[0]

	def get_leaf(self, val):
		parent_index = 0
		# Traverse tree
		while True:
			left_child_index = 2 * parent_index +1
			right_child_index = left_child_index +1

			# Bottom reached
			if left_child_index >= len(self.tree):
				leaf_index = parent_index
				break

			else:
				if val <= self.tree[left_child_index]:
					parent_index = left_child_index
				else:
					val -= self.tree[left_child_index]
					parent_index = right_child_index

		data_index = leaf_index - self.capacity + 1
		# Returns index in tree, value in tree (priority), data corresponding to val
		return leaf_index, self.tree[leaf_index], self.data[data_index]
