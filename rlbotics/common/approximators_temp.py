import torch
import torch.nn as nn

class MLP:
	"""
	Multi-Layered Perceptron
	"""
	def __init__(self, IO_sizes, hidden_sizes, activations, layer_types, optimizer='Adam', lr=0.01):
		"""
		:param IO_sizes: (list) 2 elements: input size, output size
		:param hidden_sizes: (list) hidden layer sizes
		:param activations: (list)(strings) activations corresponding to each layer	e.g. ['relu', 'relu', None]
		:param layer_types: (list)(strings) e.g. ['conv', 'linear', 'linear']
		:param optimizer: (str) e.g. 'RMSprop'
		:param lr: (float) learning rate
		"""
		self.obs_dim = IO_sizes[0]
		# Build NN
		self.model = Net(IO_sizes, hidden_sizes, activations, layer_types)
		# Set optimizer
		if optimizer == 'Adam':
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
		elif optimizer == 'RMSprop':
			self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr)
		else:
			raise NameError(str(optimizer) + ' Optimizer not supported')

	def predict(self, x):
		x = torch.FloatTensor(x)
		x = x.view(-1, self.obs_dim)
		return self.model(x)

	def train(self, loss):
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def save_weights(self, dir_path):
		torch.save(self.model.state_dict(), dir_path)

	def load_weights(self, dir_path):
		self.model.load_state_dict(torch.load(dir_path))

	def save_model(self, dir_path):
		torch.save(self.model, dir_path)

	def load_model(self, dir_path):
		self.model = torch.load(dir_path)

	def summary(self):
		print(self.model)



class Net(nn.Module):
	def __init__(self, IO_sizes, hidden_sizes, activations, layer_types):
		super(Net, self).__init__()
		self.layers = nn.ModuleList()
		self.activations = nn.ModuleList()

		layer_sizes = [IO_sizes[0]] + hidden_sizes + [IO_sizes[1]]
		for i in range(len(layer_sizes)-1):
			if layer_types[i] == 'linear':
				self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
			elif layer_types[i] == 'conv':
				raise NotImplementedError('Conv nets not supported')
			elif layer_types[i] == 'lstm':
				raise NotImplementedError('LSTM nets not supported')
			else:
				raise NameError(str(layer_types[i]) + ' Layer is not supported')

		for activation in activations:
			if activation == 'relu':
				self.activations.append(nn.ReLU(inplace=True))
			elif activation == 'tanh':
				self.activations.append(nn.Tanh())
			elif activation == 'sigmoid':
				self.activations.append(nn.Sigmoid())
			elif activation == 'softmax':
				self.activations.append(nn.LogSoftmax())
			elif activation == None or activation == 'none':
				self.activations.append(nn.Identity())
			else:
				raise NameError(str(activation) + ' activation not supported')

	def forward(self, x):
		for i in range(len(self.layers)):
			x = self.activations[i](self.layers[i](x))
		return x