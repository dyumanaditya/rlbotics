import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class Logger:
	def __init__(self, log_name='data'):
		"""
		NOTE: You can only log once per time step
		:param log_name: (str) name of file in which log will be stored
		"""
		cur_dir = os.getcwd()
		log_dir = os.path.join(cur_dir, 'logs/')
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)

		self.keys = []
		self.filename = os.path.join(log_dir, 'log_' + log_name + '.csv')
		open(self.filename, 'w')

		# Tensor Board
		self.writer = SummaryWriter(log_dir=log_dir)

	def save_tabular(self, **kwargs):
		header = True if len(self.keys) == 0 else False
		self.keys = list(kwargs.keys())
		df = pd.DataFrame(kwargs, index=[0])
		self._write_file(df, header)

	def _write_file(self, df, header):
		df.to_csv(self.filename, header=header, mode='a', index=False)
