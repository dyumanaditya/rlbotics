from rlbotics.common.policies import *
import rlbotics.vpg.hyperparameters as h
from rlbotics.common.logger import Logger


class VPG:
    def __init__(self, env):
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._build_policy()

        self.logger = Logger('VPG')

    def _build_policy(self):
        self.policy = MLPSoftmaxPolicy([self.obs_dim] + h.hidden_layers + [self.act_dim] , h.activations, lr=h.lr).to(self.device)
        self.policy.summary()

    def _loss(self, obs_batch, act_batch, rew_batch):
        logp = self.policy.get_log_prob(obs_batch, act_batch)
        return -(logp * rew_batch).mean()

    def get_action(self, obs):
        return self.policy.get_action(obs)

    def update_policy(self, obs_batch, act_batch, rew_batch):
        obs_batch.to(self.device)
        act_batch.to(self.device)
        rew_batch.to(self.device)

        loss = self._loss(obs_batch, act_batch, rew_batch)
        self.policy.learn(loss)
