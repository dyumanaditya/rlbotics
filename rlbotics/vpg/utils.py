import torch

def get_episode_returns(rew_batch, done_batch, gamma):
    episode_returns = []
    episodes = []
    prev_index = 0

    indeces = torch.nonzero(done_batch, as_tuple=True)[0]

    for index in indeces:
    	episodes.append(rew_batch[prev_index:index+1])
    	prev_index = index + 1

    for ep in episodes:
	     episode_returns += [ep.sum().item()] * len(ep)

    return torch.as_tensor(episode_returns, dtype=torch.float32)


def get_reward_to_go(rew_batch, done_batch, gamma):
    rgts = torch.zeros_like(rew_batch, dtype=torch.float32)
    cumulative = 0.0
    for k in reversed(range(len(rew_batch))):
        if done_batch[k]:
            rgts[k] = 0
        cumulative = rew_batch[k] + gamma * cumulative * (1.0 - done_batch[k])
        rgts[k] = cumulative

    return rgts

def get_expected_return(rew_batch, done_batch, gamma):
    rtgs = get_reward_to_go(rew_batch, done_batch, gamma)
    return normalize(rtgs)

def normalize(x):
    return (x - torch.mean(x)) / torch.std(x)
