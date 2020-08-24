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
        cumulative = rew[k] + gamma * cumulative * (1.0 - done[k])
        g[k] = cumulative
    if normalize_output:
        normalize(g)
    return g


def GAE(rew, done, val, gamma, lam, normalize_output=True):
    rew = torch.cat((rew, torch.tensor([0], dtype=torch.float32)))
    val = torch.cat((val, torch.tensor([0], dtype=torch.float32)))

    td_errors = rew[:-1] + gamma * val[1:] * (1 - done) - val[:-1]

    adv = get_expected_return(td_errors, done, gamma*lam, normalize_output)

    return adv


def normalize(x):
    return (x - torch.mean(x)) / torch.std(x)
