import pandas as pd

import torch
import numpy as np

def get_latest_return(log_file):
    logs = pd.read_csv(log_file)
    ep_return = 1

    for i in reversed(range(len(logs)-1)):
        if logs.loc[i,'done'] == False:
            ep_return += logs.loc[i, 'rewards']

        elif logs.loc[i,'done'] == True:
            break

    return ep_return


def get_return(log_file):
    logs = pd.read_csv(log_file)

    ep_sum = 0
    ep_returns = []

    for i in range(len(logs)):
        if logs.loc[i,'done'] == False:
            ep_sum += logs.loc[i, 'rewards']

        elif logs.loc[i,'done'] == True:
            ep_sum += logs.loc[i, 'rewards']
            ep_returns.append(ep_sum)
            ep_sum = 0

    return ep_returns


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


def finish_path(rew, done, val, adv, gamma, lam):

    indices = torch.nonzero(done, as_tuple=True)[0]

    last_done = max(indices)

    if last_done >= len(rew)-2:
        return adv

    last_ep_rew = rew[max(indices) + 1:]
    last_ep_done = done[max(indices) + 1:]
    last_ep_val = val[max(indices) + 1:]

    deltas = last_ep_rew[:-1] + gamma * torch.flatten(last_ep_val)[1:] - torch.flatten(last_ep_val)[:-1]
    last_ep_adv = get_expected_return(deltas, last_ep_done, gamma * lam)

    adv[max(indices) + 2:] = last_ep_adv

    return adv
