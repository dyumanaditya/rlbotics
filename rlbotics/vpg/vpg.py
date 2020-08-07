from rlbotics.common.policies import *
import rlbotics.vpg.hyperparameters as h
import time

class Reinforce:
    def __init__(self, env):
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self.build_policy()

    def build_policy(self):
        self.policy = MLPSoftmaxPolicy([self.obs_dim, self.act_dim], h.hidden_layers, h.activations, h.layer_types, lr=h.lr)
        self.policy.summary()

    def compute_loss(self, obs, acts, rews):
        rews_tensor = torch.as_tensor(rews, dtype=torch.float32)
        logp = self.policy.get_log_prob(obs, acts)
        return -(logp * rews_tensor).mean()


    def update_policy(self, batch_obs, batch_acts, batch_rews):
        loss = self.compute_loss(batch_obs, batch_acts, batch_rews)
        self.policy.train(loss)

    def train(self, batch_size, render=False):
        batch_obs = []
        batch_acts = []
        batch_rewards = []

        episode_rewards = []

        obs = self.env.reset()
        done = False
        finished_rendering_this_epoch = False

        while True:
            if (not finished_rendering_this_epoch) and render:
                self.env.render()
                time.sleep(0.02)

            batch_obs.append(obs.copy())

            action = self.policy.get_action(obs)
            obs, reward, done, info = self.env.step(action)

            batch_acts.append(action)
            episode_rewards.append(reward)

            # failed or goal reached
            if done:
                finished_rendering_this_epoch = True

                episode_return = sum(episode_rewards)
                episode_len = len(episode_rewards)

                batch_rewards += [episode_return] * episode_len

                obs, done, episode_rewards = self.env.reset(), False, []

                if len(batch_obs) >= batch_size:
                    break

        self.env.close()

        self.update_policy(batch_obs, batch_acts, batch_rewards)
