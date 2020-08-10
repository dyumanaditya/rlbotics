from rlbotics.vpg.vpg import *
import gym
import time
import matplotlib.pyplot as plt

def train(batch_size, render=False):
    global env, model, episode_count

    obs_batch = []
    act_batch = []
    rew_batch = []

    episode_rewards = []

    new_obs = env.reset()
    done = False
    finished_rendering_this_epoch = False

    while True:
        if (not finished_rendering_this_epoch) and render:
            env.render()
            time.sleep(0.02)

        obs_batch.append(new_obs.copy())

        action = model.get_action(new_obs)
        new_obs, rew, done, info = env.step(action)

        act_batch.append(action)
        episode_rewards.append(rew)

        # failed or goal reached
        if done:
            episode_count += 1
            finished_rendering_this_epoch = True

            episode_return = sum(episode_rewards)
            episode_len = len(episode_rewards)

            model.logger.log(ep_rew=episode_return)
            model.logger.writer.add_scalar('Reward/Episode', episode_return, episode_count)

            rew_batch += [episode_return] * episode_len

            new_obs, done, episode_rewards = env.reset(), False, []

            if len(obs_batch) >= batch_size:
                break

    env.close()
    #model.update_policy(obs_batch, act_batch, rew_batch)
    model.update_policy(obs_batch=torch.as_tensor(obs_batch, dtype=torch.float32),
                        act_batch=torch.as_tensor(act_batch, dtype=torch.int32),
                        rew_batch=torch.as_tensor(rew_batch, dtype=torch.float32))


def main():
    render = False

    for epoch in range(100):
        if epoch % 10 == 0:
            print("epoch : ", epoch)
            render = False

        else:
            render = False

        train(1000, render=render)

    model.logger.writer.close()


if __name__ == "__main__":
    episode_count = 0
    env = gym.make('CartPole-v1')
    model = VPG(env)

    main()
