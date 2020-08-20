import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from rlbotics.common.utils import get_return


def plot(algo_name, env_name, seed, xlabel=None, ylabel=None, display=False):
	cur_dir = os.getcwd()
	plt_dir = os.path.join(cur_dir, 'experiments', 'plots', algo_name + '_' + env_name + '_' + str(seed))
	log_file = os.path.join(cur_dir, 'experiments', 'logs', algo_name + '_' + env_name + '_' + str(seed), 'transitions.csv')
	if os.path.exists(plt_dir):
		shutil.rmtree(plt_dir)
	os.makedirs(plt_dir)

	ep_returns = get_return(log_file)
	ax = sns.lineplot(x=list(range(len(ep_returns))), y=ep_returns)
	ax.set(xlabel=xlabel, ylabel=ylabel)
	plt.savefig(plt_dir + '/' + algo_name + '_plt' + '.png')

	if display:
		plt.show()
