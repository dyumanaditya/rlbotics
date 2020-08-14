from rlbotics.common.policies import *
from rlbotics.vpg.utils import *
from rlbotics.vpg.memory import Memory
import rlbotics.vpg.hyperparameters as h
from rlbotics.common.logger import Logger


class VPG:
    def __init__(self, env):
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        # Replay buffer
        self.memory = Memory()

        # Logger
        self.logger = Logger('VPG')

        self._build_policy()

    def _build_policy(self):
        self.policy = MLPSoftmaxPolicy([self.obs_dim] + h.hidden_sizes + [self.act_dim] , h.activations, lr=h.lr)
        self.policy.summary()

    def _loss(self, obs_batch, act_batch, rew_batch):
        logp = self.policy.get_log_prob(obs_batch, act_batch)
        return -(logp * rew_batch).mean()

    def get_action(self, obs):
        return self.policy.get_action(obs)

    def store_transition(self, obs, act, rew, new_obs, done):
        self.memory.add(obs, act, rew, new_obs, done)

        # Log Done, reward
        #self.logger.save_tabular(done=done, rewards=rew)

    def update_policy(self):
        if len(self.memory) < h.batch_size:
            return

        # get batches from memory
        transition_batch = self.memory.get_batch()

        obs_batch = torch.as_tensor(transition_batch.obs, dtype=torch.float32)
        act_batch = torch.as_tensor(transition_batch.act, dtype=torch.int32)
        rew_batch = torch.as_tensor(transition_batch.rew, dtype=torch.float32)
        done_batch = torch.as_tensor(transition_batch.done, dtype=torch.int32)

        obs_batch, act_batch, ret_batch = get_episode_returns(obs_batch, act_batch, rew_batch, done_batch)

        loss = self._loss(obs_batch, act_batch, ret_batch)
        self.policy.learn(loss)

        self.memory.reset_memory()
