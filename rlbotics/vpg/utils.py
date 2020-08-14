import torch

def get_episode_returns(obs_batch, act_batch, rew_batch, done_batch):
    episode_returns = []
    episodes = []
    prev_index = 0

    indeces = torch.nonzero(done_batch, as_tuple=True)[0]

    for index in indeces:
    	episodes.append(rew_batch[prev_index:index+1])
    	prev_index = index + 1

    for ep in episodes:
	     episode_returns += [ep.sum().item()] * len(ep)

    return obs_batch[:len(episode_returns)], act_batch[:len(episode_returns)], torch.as_tensor(episode_returns, dtype=torch.float32)

def get_reward_to_go(obs_batch, act_batch, rew_batch, done_batch):
    reward_to_go = []
    episodes = []
    prev_index = 0

    indeces = torch.nonzero(done_batch, as_tuple=True)[0]

    for index in indeces:
    	episodes.append(rew_batch[prev_index:index+1])
    	prev_index = index + 1

    for ep in episodes:
        n = len(ep)
        rtgs = torch.zeros_like(ep)
        for i in reversed(range(n)):
            rtgs[i] = ep[i] + (rtgs[i+1] if i+1 < n else 0)

        reward_to_go.append(rtgs)

    reward_to_go_tensor = torch.cat(reward_to_go)

    return obs_batch[:len(reward_to_go_tensor)], act_batch[:len(reward_to_go_tensor)], reward_to_go_tensor
