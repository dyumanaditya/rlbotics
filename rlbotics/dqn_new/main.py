import gym
from rlbotics.dqn_new.dqn import *
import rlbotics.dqn_new.hyperparameters as h


def main():
    max_iterations = 100000
    env = gym.make("CartPole-v0")
    dqn = DQN(env)
    obs = env.reset()

    # Episode related information
    ep_counter = 0
    ep_rew = 0

    for iter in range(max_iterations):
        act = dqn.get_action(obs)
        new_obs, rew, done, _ = env.step(act)

        # Store transition in Replay Memory
        dqn.store_transition(obs, act, rew, new_obs, done)

        obs = new_obs
        ep_rew += rew
        if done:
            obs = env.reset()

            # Display results
            print("episode: {}, total reward: {}".format(ep_counter, ep_rew))

            # Logging
            ep_counter += 1
            ep_rew = 0
            continue

        # Update policy
        dqn.update_policy()

        # Update target policy
        if iter % h.update_target_freq == 0:
            dqn.update_target_policy()


if __name__ == "__main__":
    main()