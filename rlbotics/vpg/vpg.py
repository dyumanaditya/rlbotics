import torch
import torch.nn as nn

from rlbotics.vpg.memory import Memory
import rlbotics.vpg.hyperparameters as h
from rlbotics.common.policies import MLPSoftmaxPolicy
from rlbotics.common.approximators import MLP
from rlbotics.common.logger import Logger
from rlbotics.vpg.utils import *


class VPG:
    def __init__(self, env):
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        # Replay buffer
        self.memory = Memory()

        # Logger
        self.logger = Logger('VPG')

        self._build_policy()
        self._build_value_function()

    def _build_policy(self):
        self.policy = MLPSoftmaxPolicy([self.obs_dim] + h.pi_hidden_sizes + [self.act_dim] , h.pi_activations, lr=h.pi_lr)
        self.policy.summary()

    def _build_value_function(self):
        self.value = MLP([self.obs_dim] + h.v_hidden_sizes , h.v_activations, lr=h.v_lr)
        self._loss_value = nn.MSELoss()
        self.value.summary()

    def _loss(self, obs_batch, act_batch, adv_batch):
        logp = self.policy.get_log_prob(obs_batch, act_batch)
        return -(logp * adv_batch).mean()

    def get_action(self, obs):
        return self.policy.get_action(obs)

    def store_transition(self, obs, act, rew, new_obs, done):
        self.memory.add(obs, act, rew, new_obs, done)

        # Log Done, reward
        self.logger.save_tabular(done=done, rewards=rew)

    def update_policy(self):
        if len(self.memory) < h.batch_size:
            return

        # get batches from memory
        transition_batch = self.memory.get_batch()

        obs_batch = torch.as_tensor(transition_batch.obs, dtype=torch.float32)
        act_batch = torch.as_tensor(transition_batch.act, dtype=torch.int32)
        rew_batch = torch.as_tensor(transition_batch.rew, dtype=torch.float32)
        done_batch = torch.as_tensor(transition_batch.done, dtype=torch.int32)

        #reward_to_go = get_reward_to_go(rew_batch, done_batch, h.gamma)
        expected_return = get_expected_return(rew_batch, done_batch, h.gamma)
        values = self.value(obs_batch)

        #adv_batch = reward_to_go - expected_return
        adv_batch = expected_return - values

        loss = self._loss(obs_batch, act_batch, adv_batch)
        self.policy.learn(loss)

        #self.memory.reset_memory()

    def update_value(self):
        if len(self.memory) < h.batch_size:
            return
        # get batches from memory
        transition_batch = self.memory.get_batch()

        obs_batch = torch.as_tensor(transition_batch.obs, dtype=torch.float32)
        rew_batch = torch.as_tensor(transition_batch.rew, dtype=torch.float32)
        done_batch = torch.as_tensor(transition_batch.done, dtype=torch.int32)

        expected_return = get_expected_return(rew_batch, done_batch, h.gamma)
        values = self.value(obs_batch)

        loss = self._loss_value(values.squeeze(1), expected_return)
        self.value.learn(loss)

        self.memory.reset_memory()
