import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """
    Multi-Layered Perceptron
    """

    def __init__(self, layer_sizes, activations, seed, optimizer='Adam', lr=0.01, weight_decay=0, batch_norm=False, weight_init=None):
        """
        :param layer_sizes: (list) sizes of each layer (including IO layers)
        :param activations: (list)(strings) activations corresponding to each layer	e.g. ['relu', 'relu', 'none']
        :param optimizer: (str) e.g. 'RMSprop'
        :param lr: (float) learning rate
        """
        super(MLP, self).__init__()
        torch.manual_seed(seed)
        self.obs_dim = layer_sizes[0]
        self.weight_init = weight_init

        # Build NN
        self.activations_dict = nn.ModuleDict({
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.LogSoftmax(),
            "selu": nn.SELU(),
            "elu": nn.ELU(),
            "leakyrelu": nn.LeakyReLU(),
            "none": nn.Identity(),
        })

        # Build MLP and initialize weights if necessary
        self.mlp = self._make_mlp(layer_sizes, activations, batch_norm)
        if weight_init is not None:
            self.mlp.apply(self.init_weights)

        # Set optimizer
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr, weight_decay=weight_decay)
        elif optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.mlp.parameters(), lr, weight_decay=weight_decay)
        else:
            raise NameError(str(optimizer) + ' Optimizer not supported')

    def _make_mlp(self, layer_sizes, activations, batch_norm):
        layers = []
        for i in range(len(layer_sizes)-1):
            if batch_norm:
                layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), nn.BatchNorm1d(layer_sizes[i+1]),
                           self.activations_dict[activations[i]]]
            else:
                layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), self.activations_dict[activations[i]]]
        return nn.Sequential(*layers)

    def init_weights(self, mlp):
        if type(mlp) == nn.Linear:
            nn.init.uniform_(mlp.weight, -self.weight_init, self.weight_init)

    def forward(self, x):
        return self.mlp(x)

    def predict(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float()

        x = x.view(-1, self.obs_dim)
        return self.forward(x)

    def learn(self, loss, grad_clip=None):
        self.optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            for param in self.mlp.parameters():
                param.grad.data.clamp_(grad_clip[0], grad_clip[1])
        self.optimizer.step()

    def save_weights(self, dir_path):
        torch.save(self.mlp.state_dict(), dir_path)

    def load_weights(self, dir_path):
        self.mlp.load_state_dict(torch.load(dir_path))

    def save_model(self, dir_path):
        torch.save(self.mlp, dir_path)

    def load_model(self, dir_path):
        self.mlp = torch.load(dir_path)

    def summary(self):
        print(self.mlp)
