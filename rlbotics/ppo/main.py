import gym
import torch

from rlbotics.ppo.ppo import PPO
import rlbotics.vpg.hyperparameters as h
from rlbotics.common.visualization import plot


def main():
    # Set device
    gpu = 0
    device = torch.device(f"cuda:{gpu}"if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # Build environment
    env = gym.make(h.env_name)
    agent = PPO(env)
    obs = env.reset()

    # Episode related information
    ep_counter = 0
    ep_rew = 0

    for iteration in range(h.max_iterations):
        if h.render:
            env.render()

        # Take action
        act = agent.get_action(obs)
        new_obs, rew, done, _ = env.step(act)

        # Store experience
        agent.store_transition(obs, act, rew, new_obs, done)
        ep_rew += rew
        obs = new_obs

        # Episode done
        if done:
            obs = env.reset()
            # Display results
            print("episode: {}, total reward: {}".format(ep_counter, ep_rew))

            agent.logger.writer.add_scalar("return/episode", ep_rew, ep_counter)

            # Logging
            ep_counter += 1
            ep_rew = 0

        if iteration % h.batch_size == 0:
            # Update Policy
            agent.update_policy()

            # Update Value
            agent.update_value()

    # End
    env.close()
    plot(agent.logger.filename, 'vpg', 'episodes', 'rewards', True)


if __name__ == '__main__':
    main()
