import torch
import numpy as np


def get_episode_returns(rew_batch, done_batch, gamma):
    episode_returns = []
    episodes = []
    prev_index = 0

    indices = torch.nonzero(done_batch, as_tuple=True)[0]

    for index in indices:
        episodes.append(rew_batch[prev_index:index+1])
        prev_index = index + 1

    for ep in episodes:
        episode_returns += [ep.sum().item()] * len(ep)

    return torch.as_tensor(episode_returns, dtype=torch.float32)


def get_expected_return(rew, done, gamma, normalize_output=True):
    g = torch.zeros_like(rew, dtype=torch.float32)
    cumulative = 0.0
    for k in reversed(range(len(rew))):
        if done[k]:
            g[k] = 0
        cumulative = rew[k] + gamma * cumulative * (1.0 - done[k])
        g[k] = cumulative
    if normalize_output:
        normalize(g)
    return g


def normalize(x):
    return (x - torch.mean(x)) / torch.std(x)

