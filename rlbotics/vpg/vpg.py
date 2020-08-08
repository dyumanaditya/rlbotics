from rlbotics.common.policies import *
import rlbotics.vpg.hyperparameters as h
import time


class VPG:
    def __init__(self, env):
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self._build_policy()

    def _build_policy(self):
        self.policy = MLPSoftmaxPolicy([self.obs_dim] + h.hidden_layers + [self.act_dim] , h.activations, lr=h.lr)
        self.policy.summary()

    def _loss(self, obs_batch, act_batch, rew_batch):
        rews_tensor = torch.as_tensor(rew_batch, dtype=torch.float32)
        logp = self.policy.get_log_prob(obs_batch, act_batch)
        return -(logp * rews_tensor).mean()

    def get_action(self, obs):
        return self.policy.get_action(obs)

    def update_policy(self, obs_batch, act_batch, rew_batch):
        loss = self._loss(obs_batch, act_batch, rew_batch)
        self.policy.train(loss)
