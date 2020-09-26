import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rlbotics.common.utils import get_ep_returns


class Plotter:
	def __init__(self):
		sns.set()
		self.cur_dir = os.getcwd()
		self.plt_dir = os.path.join(self.cur_dir, 'experiments', 'plots')
		if not os.path.exists(self.plt_dir):
			os.makedirs(self.plt_dir)

	def plot_individual(self, title, xlabel, ylabel, algo, env, seed, epoch_iter=1, display=False):
		log_file = os.path.join(self.cur_dir, 'experiments', 'logs', f'{algo}_{env}_{seed}', 'transitions.csv')
		ep_returns = get_ep_returns(log_file, epoch_iter)

		# Plot
		ep_returns = pd.Series(ep_returns).rolling(10, min_periods=1).mean()
		ax = sns.lineplot(x=list(range(len(ep_returns))), y=ep_returns)
		ax.axes.set_title(title, fontsize=20)
		ax.set_xlabel(xlabel, fontsize=15)
		ax.set_ylabel(ylabel, fontsize=15)

		filename = os.path.join(self.plt_dir, f'{algo}_{env}_plt.png')
		plt.savefig(filename)

		if display:
			plt.show()

	def combine_csv_files(self, algo, env, epoch_iter, data='rewards', log_file_type='transitions'):
		# Combine all csv files with different seeds into x and y so we can plot
		x = []
		y = []
		num_seeds = 10

		if data == 'rewards':
			for seed in range(num_seeds):
				filename = os.path.join('experiments', 'logs', f'{algo}_{env}_{seed}', f'{log_file_type}.csv')
				returns = get_ep_returns(filename, epoch_iter)
				x += list(range(len(returns)))
				y += returns
		else:
			for seed in range(num_seeds):
				filename = os.path.join('experiments', 'logs', f'{algo}_{env}_{seed}', f'{log_file_type}.csv')
				df = pd.read_csv(filename)
				col = list(df[data])
				x += list(range(len(col)))
				y += col

		return x, y

	def plot_combined(self, title, xlabel, ylabel, algo, env, epoch_iter=1, display=False, data='rewards', log_file_type='transitions'):
		"""
		:param epoch_iter: number of iterations for 1 epoch. (hyperparameters:max_iterations)
		:param data: Either rewards or anything else you want to plot
		:param log_file_type: Either transitions or policy_updates
		"""
		filename = os.path.join(self.plt_dir, 'all_seeds', algo)
		if not os.path.exists(filename):
			os.makedirs(filename)

		x, y = self.combine_csv_files(algo, env, epoch_iter, data, log_file_type)
		y = pd.Series(y).rolling(5, min_periods=1).mean()

		# Plot
		ax = sns.lineplot(x=x, y=y, ci=95)
		ax.axes.set_title(title, fontsize=20)
		ax.set_xlabel(xlabel, fontsize=15)
		ax.set_ylabel(ylabel, fontsize=15)
		plt.legend([algo + '_' + env])
		plt.savefig(f'{filename}/{env}_all_seeds_plt.png')

		# Display
		if display:
			plt.show()

# p= Plotter()
# p.plot_combined('LunarLander DDPG', 'epochs', 'rewards', 'DDPG', 'LunarLanderContinuous-v2', display=True)
