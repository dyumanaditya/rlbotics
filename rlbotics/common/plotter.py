import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rlbotics.common.utils import get_return


class Plotter:
	def __init__(self):
		sns.set()
		self.cur_dir = os.getcwd()
		self.plt_dir = os.path.join(self.cur_dir, 'experiments', 'plots')
		if not os.path.exists(self.plt_dir):
			os.makedirs(self.plt_dir)

	def plot_individual(self, title, xlabel, ylabel, algo, env, seed, display=False):
		log_file = os.path.join(self.cur_dir, 'experiments', 'logs', algo + '_' + env + '_' + str(seed), 'transitions.csv')
		ep_returns = get_return(log_file)

		# Plot
		ax = sns.lineplot(x=list(range(len(ep_returns))), y=ep_returns)
		ax.axes.set_title(title, fontsize=20)
		ax.set_xlabel(xlabel, fontsize=15)
		ax.set_ylabel(ylabel, fontsize=15)

		filename = os.path.join(self.plt_dir, algo + '_' + env + '_plt.png')
		plt.savefig(filename)

		if display:
			plt.show()

	def combine_csv_files(self, algo, env, data='rewards', log_file_type='transitions'):
		# Combine all csv files with different seeds into x and y so we can plot
		x = []
		y = []
		num_seeds = 1

		if data == 'rewards':
			for seed in range(num_seeds):
				filename = os.path.join('experiments', 'logs', algo + '_' + env + '_' + str(seed), log_file_type + '.csv')
				returns = get_return(filename)
				x += list(range(len(returns)))
				y += returns
		else:
			for seed in range(num_seeds):
				filename = os.path.join('experiments', 'logs', algo + '_' + env + '_' + str(seed), log_file_type + '.csv')
				df = pd.read_csv(filename)
				col = list(df[data])
				x += list(range(len(col)))
				y += col

		return x, y

	def plot_combined(self, title, xlabel, ylabel, algo, env, display=False, data='rewards', log_file_type='transitions'):
		"""
		:param data: Either rewards or anything else you want to plot
		:param log_file_type: Either transitions or policy_updates
		"""
		filename = os.path.join(self.plt_dir, 'all_seeds', algo)
		if not os.path.exists(filename):
			os.makedirs(filename)

		x, y = self.combine_csv_files(algo, env, data, log_file_type)
		y = pd.Series(y).rolling(5, min_periods=1).mean()

		# Plot
		ax = sns.lineplot(x=x, y=y)
		ax.axes.set_title(title, fontsize=20)
		ax.set_xlabel(xlabel, fontsize=15)
		ax.set_ylabel(ylabel, fontsize=15)
		plt.legend([algo + '_' + env])
		plt.savefig(filename + '/' + env + '_all_seeds_plt.png')

		# Display
		if display:
			plt.show()
