from rlbotics.vpg.vpg import *
import gym
import time
import matplotlib.pyplot as plt


def train(model, env, batch_size, render=False):
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
            finished_rendering_this_epoch = True

            episode_return = sum(episode_rewards)
            episode_len = len(episode_rewards)

            rew_batch += [episode_return] * episode_len

            new_obs, done, episode_rewards = env.reset(), False, []

            if len(obs_batch) >= batch_size:
                break

    env.close()
    model.update_policy(obs_batch, act_batch, rew_batch)

    # for testing
    return max(rew_batch)


def main():
    env = gym.make('CartPole-v1')

    model = VPG(env)

    render = False

    max_reward_per_epoch = []

    for epoch in range(1000):
        if epoch % 10 == 0:
            print("epoch : ", epoch)
            render = False

        else:
            render = False

        max_reward_per_epoch.append(train(model, env, 1000, render=render))

    plt.xlabel('epochs')
    plt.ylabel('max reward')

    plt.plot(max_reward_per_epoch, label = "max reward in epoch")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
