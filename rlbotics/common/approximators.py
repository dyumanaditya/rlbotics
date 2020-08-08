import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


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

        # TODO: Delete IO_sizes and get as input layer_sizes
        layer_sizes = [IO_sizes[0]] + hidden_sizes + [IO_sizes[1]]

        # Build NN
        self.model = MLPBase(layer_sizes, activations)

        # Set optimizer
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        elif optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr)
        else:
            raise NameError(str(optimizer) + ' Optimizer not supported')

    def predict(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float()
        x = x.view(-1, self.obs_dim)
        return self.model(x)

    def train(self, loss):
        self.loss = loss
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


class MLPBase(nn.Module):
    def __init__(self, layer_sizes, activations):
        super(MLPBase, self).__init__()
        activations_dict = nn.ModuleDict({
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "selu": nn.SELU(),
            "elu": nn.ELU(),
            "none": nn.Identity(),
        })

        self.mlp = nn.Sequential(*[nn.Sequential(nn.Linear(in_features, out_features), activations_dict[activation])
                                   for in_features, out_features, activation in
                                   zip(layer_sizes[:-1], layer_sizes[1:], activations)])

    def forward(self, x):
        return self.mlp(x)
