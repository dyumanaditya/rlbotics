from rlbotics.common.utils import get_return
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot(log_file, graph_name='graph', xlabel=None, ylabel=None, display=False):
	cur_dir = os.getcwd()
	plt_dir = os.path.join(cur_dir, 'plots/')
	if not os.path.exists(plt_dir):
		os.makedirs(plt_dir)

	ep_returns = get_return(log_file)
	ax = sns.lineplot(x=ep_returns, y=list(range(len(ep_returns))))
	ax.set(xlabel=xlabel, ylabel=ylabel)
	plt.savefig(plt_dir + graph_name + '_plt' + '.png')

	if display:
		plt.show()
