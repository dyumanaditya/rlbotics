import torch

def get_episode_returns(obs_batch, act_batch, rew_batch, done_batch):
    episode_returns = []
    steps = 0
    episode_sum = 0

    for i, done in enumerate(done_batch):
        if done == 0:
            steps += 1
            episode_sum += rew_batch[i]

        elif done == 1:
            steps += 1
            episode_sum += rew_batch[i]

            episode_returns += [episode_sum.item()] * steps
            episode_sum = 0
            steps = 0

    return obs_batch[:len(episode_returns)], act_batch[:len(episode_returns)], torch.as_tensor(episode_returns, dtype=torch.float32)
