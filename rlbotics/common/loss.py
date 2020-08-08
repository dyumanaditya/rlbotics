import torch.nn as nn

def loss(loss_name):
	if loss_name == 'mse':
		return nn.MSELoss()
	else:
		raise NotImplementedError