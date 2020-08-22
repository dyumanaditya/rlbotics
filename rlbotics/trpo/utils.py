import torch

def flat_grad(y, x):
    g = torch.autograd.grad(y, x)
    g = torch.cat([t.view(-1) for t in g])
    return g
