import os
import csv
import json
import torch
import shutil
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from rlbotics.common.utils import get_latest_ep_return


class Logger:
	def __init__(self, algo_name, env_name, seed, resume=False):
		"""
		:param algo_name: (str) name of file in which log will be stored
		:param env_name: (str) name of environment used in experiment
		:param seed: (int) random seed used in experiment
		:param resume: (bool) Resume training some model
		"""
		cur_dir = os.getcwd()
		self.log_dir = os.path.join(cur_dir, 'experiments', 'logs', algo_name + '_' + env_name + '_' + str(seed))
		self.model_dir = os.path.join(cur_dir, 'experiments', 'models', algo_name + '_' + env_name + '_' + str(seed))
		if os.path.exists(self.log_dir) and not resume:
			shutil.rmtree(self.log_dir)
		if os.path.exists(self.model_dir) and not resume:
			shutil.rmtree(self.model_dir)
		if not resume:
			os.makedirs(self.log_dir)
			os.makedirs(self.model_dir)

		self.resume = resume
		self.transition_keys, self.policy_update_keys = [], []

		# Tensor Board
		self.writer = SummaryWriter(log_dir=self.log_dir)

		# Counter to track number of policy updates
		self.tensorboard_updated = 0

		# Keeps track of returns from episodes for each epoch
		self.episode_returns = []

		if self.resume:
			self.resume_log()

	def log(self, name='params', **kwargs):
		if name == 'transitions':
			file = os.path.join(self.log_dir, 'transitions.csv')
			header = True if len(self.transition_keys) == 0 and not self.resume else False
			if header:
				self.transition_keys = list(kwargs.keys())
			self._save_tabular(file, header, **kwargs)

		elif name == 'policy_updates':
			file = os.path.join(self.log_dir, 'policy_updates.csv')
			header = True if len(self.policy_update_keys) == 0 and not self.resume else False
			if header:
				self.policy_update_keys = list(kwargs.keys())
			self._save_tabular(file, header, **kwargs)

		elif name == 'params':
			file = os.path.join(self.log_dir, 'params.json')
			with open(file, 'w') as f:
				json.dump(kwargs, f, indent=4)

		elif name == 'checkpoint':
			file = os.path.join(self.log_dir, 'checkpoint')
			with open(file, 'w') as f:
				json.dump(kwargs, f, indent=4)

	def _save_tabular(self, file, header, **kwargs):
		with open(file, 'a') as f:
			writer = csv.writer(f)
			if header:
				writer.writerow(kwargs.keys())
			writer.writerow(kwargs.values())

		self._write_tensorboard(file, **kwargs)

	def _write_tensorboard(self, file, **kwargs):
		if 'transitions.csv' in file:
			if kwargs.get('done'):
				latest_return = get_latest_ep_return(file)
				self.episode_returns.append(latest_return)
		else:
			for key, value in kwargs.items():
				self.writer.add_scalar(key, value, self.tensorboard_updated)

			if self.episode_returns:
				self.writer.add_scalar("mean reward/epoch", np.mean(self.episode_returns), self.tensorboard_updated)
				self.episode_returns.clear()

				self.tensorboard_updated += 1

	def log_model(self, mlp, name=''):
		file = os.path.join(self.model_dir, name + 'model.pth')
		torch.save(mlp, file)

	def log_state_dict(self, dct, name):
		file = os.path.join(self.model_dir, name)
		torch.save(dct, file)

	def resume_log(self):
		log_file = os.path.join(self.log_dir, 'transitions.csv')
		log = pd.read_csv(log_file)
		for i in range(len(log)):
			if log.loc[i, 'done']:
				self.tensorboard_updated += 1
