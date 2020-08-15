import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.data = None

        # Logger
        self.logger = Logger('VPG')

        self._build_policy()
        self._build_value_function()

    def _build_policy(self):
        self.policy = MLPSoftmaxPolicy([self.obs_dim] + h.pi_hidden_sizes + [self.act_dim], h.pi_activations, lr=h.pi_lr)
        self.policy.summary()

    def _build_value_function(self):
        self.value = MLP([self.obs_dim] + h.v_hidden_sizes , h.v_activations, lr=h.v_lr)
        self.value.summary()

    def compute_policy_loss(self, obs_batch, act_batch, adv_batch):
        logp = self.policy.get_log_prob(obs_batch, act_batch)
        return -(logp * adv_batch).mean()

    def get_action(self, obs):
        return self.policy.get_action(obs)

    def store_transition(self, obs, act, rew, new_obs, done):
        self.memory.add(obs, act, rew, new_obs, done)

        # Log Done, reward
        self.logger.save_tabular(done=done, rewards=rew)

    def _get_data(self):
        # get batches from memory
        transition_batch = self.memory.get_batch()

        obs_batch = torch.as_tensor(transition_batch.obs, dtype=torch.float32)
        act_batch = torch.as_tensor(transition_batch.act, dtype=torch.int32)
        rew_batch = torch.as_tensor(transition_batch.rew, dtype=torch.float32)
        done_batch = torch.as_tensor(transition_batch.done, dtype=torch.int32)

        expected_return = get_expected_return(rew_batch, done_batch, h.gamma)
        values = self.value.predict(obs_batch)

        # adv_batch = reward_to_go - expected_return
        adv_batch = expected_return - values

        data = dict(obs=obs_batch,
                    act=act_batch,
                    val=values,
                    adv=adv_batch,
                    ret=expected_return)
        return data

    def update_policy(self):
        if len(self.memory) < h.batch_size:
            return
        self.data = self._get_data()
        obs, act, adv = self.data["obs"],  self.data["act"],  self.data["adv"]

        loss = self.compute_policy_loss(obs, act, adv)
        self.policy.learn(loss)

    def update_value(self):
        if len(self.memory) < h.batch_size:
            return
        for _ in range(h.num_v_iters):
            self.value.optimizer.zero_grad()
            val, ret = self.value.predict(self.data["obs"]).squeeze(1), self.data["ret"]
            loss = F.mse_loss(val, ret)
            loss.backward()
            self.value.optimizer.step()

        self.memory.reset_memory()
