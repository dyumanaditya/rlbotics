import os
import csv


class Logger:
	def __init__(self, log_name):
		"""
		:param log_name: (str) name of file in which log will be stored
		"""
		cur_dir = os.getcwd()
		log_dir = cur_dir + '/logs/'
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)

		self.log_dict = {}
		self.filename = log_dir + 'log_' + log_name + '.csv'

	def log(self, **kwargs):
		for key, value in kwargs.items():
			if key in self.log_dict.keys():
				self.log_dict[key].append(value)
			else:
				self.log_dict[key] = [value]

		self._write_file()

	def _write_file(self):
		with open(self.filename, 'w') as f:
			writer = csv.writer(f)
			for k, v in self.log_dict.items():
				writer.writerow([k, v])

