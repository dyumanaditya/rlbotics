import torch
import pandas as pd
from statistics import mean


def get_latest_ep_return(log_file):
    logs = pd.read_csv(log_file)
    ep_return = 1

    for i in reversed(range(len(logs)-1)):
        if logs.loc[i,'done'] == False:
            ep_return += logs.loc[i, 'rewards']

        elif logs.loc[i,'done'] == True:
            break

    return ep_return


def get_ep_returns(log_file, epoch_iter=1):
    """
    :param log_file: file where transitions.csv is found
    :param epoch_iter: number of iterations for one epoch. (hyperparameters:max_iterations)
    :return: (list) of returns from each episode
    """
    logs = pd.read_csv(log_file)

    ep_sum = 0
    temp = []
    ep_returns = []

    for i in range(len(logs)):
        if logs.loc[i,'done'] == False:
            ep_sum += logs.loc[i, 'rewards']

        elif logs.loc[i,'done'] == True:
            ep_sum += logs.loc[i, 'rewards']
            temp.append(ep_sum)
            ep_sum = 0

        if i % epoch_iter == 0:
            ep_returns.append(mean(temp))

    return ep_returns


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
