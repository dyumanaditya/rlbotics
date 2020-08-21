import torch

def flat_grad(y, x):
    grads = torch.autograd.grad(y, x)
    return torch.cat([torch.reshape(g, [-1]) for g in grads], axis=0)
