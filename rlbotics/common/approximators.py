import torch
import torch.nn as nn
import numpy as np
import itertools
import copy

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
        :param weight_decay: (float) L2 decay for optimizer
        :param weight_init: (None/float) uniform initialization for mlp params
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
        self.mlp = self._build_mlp(layer_sizes, activations, batch_norm)
        if weight_init is not None:
            self.mlp.apply(self.init_weights)

        # Set optimizer
        self.set_params(self.mlp.parameters())
        self.set_optimizer(self.params, optimizer, lr, weight_decay)

    def _build_mlp(self, layer_sizes, activations, batch_norm):
        layers = []
        for i in range(len(layer_sizes)-1):
            if batch_norm:
                layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), nn.BatchNorm1d(layer_sizes[i+1]),
                           self.activations_dict[activations[i]]]
            else:
                layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), self.activations_dict[activations[i]]]
        return nn.Sequential(*layers)

    def set_optimizer(self, params, optimizer, lr, weight_decay):
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(params, lr, weight_decay=weight_decay)
        elif optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(params, lr, weight_decay=weight_decay)
        else:
            raise NameError(str(optimizer) + ' Optimizer not supported')

    def set_params(self, params):
        self.params = itertools.chain(params)

    def init_weights(self, mlp):
        if type(mlp) == nn.Linear:
            nn.init.uniform_(mlp.weight, -self.weight_init, self.weight_init)

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float()

        x = x.view(-1, self.obs_dim)
        return self.mlp(x)

    def learn(self, loss, grad_clip=None):
        self.optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            for param in self.params:
                param.grad.data.clamp_(grad_clip[0], grad_clip[1])
        self.optimizer.step()

    def summary(self):
        print(self.mlp)
