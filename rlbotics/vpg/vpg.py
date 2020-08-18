import torch
import torch.nn.functional as F

from rlbotics.vpg.memory import Memory
from rlbotics.common.policies import MLPSoftmaxPolicy
from rlbotics.common.approximators import MLP
from rlbotics.common.logger import Logger
from rlbotics.common.utils import *


class VPG:
    def __init__(self, args, env):
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        # General parameters
        self.gamma = args.gamma
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.num_v_iters = args.num_v_iters

        # Policy Network
        self.pi_hidden_sizes = args.pi_hidden_sizes
        self.pi_activations = args.pi_activations
        self.pi_lr = args.pi_lr
        self.pi_optimizer = args.pi_optimizer

        # Value Network
        self.v_hidden_sizes = args.v_hidden_sizes
        self.v_activations = args.v_activations
        self.v_lr = args.v_lr
        self.v_optimizer = args.v_optimizer

        # Replay buffer
        self.memory = Memory()
        self.data = None

        # Logger
        self.logger = Logger('VPG', args.env_name, self.seed)

        self._build_policy()
        self._build_value_function()

        # Log parameter data
        total_params = sum(p.numel() for p in self.policy.parameters())
        trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        self.logger.log(hyperparameters=vars(args), total_params=total_params, trainable_params=trainable_params)

    def _build_policy(self):
        self.policy = MLPSoftmaxPolicy(layer_sizes=[self.obs_dim] + self.pi_hidden_sizes + [self.act_dim],
                                       activations=self.pi_activations,
                                       optimizer=self.pi_optimizer,
                                       lr=self.pi_lr)
        self.policy.summary()

    def _build_value_function(self):
        self.value = MLP(layer_sizes=[self.obs_dim] + self.v_hidden_sizes,
                         activations=self.v_activations,
                         optimizer=self.v_optimizer,
                         lr=self.v_lr)
        self.value.summary()

    def get_action(self, obs):
        return self.policy.get_action(obs)

    def store_transition(self, obs, act, rew, new_obs, done):
        self.memory.add(obs, act, rew, new_obs, done)

        # Log Done, reward
        self.logger.log(name='transitions', done=done, rewards=rew)

    def _get_data(self):
        # get batches from memory
        transition_batch = self.memory.get_batch()

        obs_batch = torch.as_tensor(transition_batch.obs, dtype=torch.float32)
        act_batch = torch.as_tensor(transition_batch.act, dtype=torch.int32)
        rew_batch = torch.as_tensor(transition_batch.rew, dtype=torch.float32)
        done_batch = torch.as_tensor(transition_batch.done, dtype=torch.int32)

        expected_return = get_expected_return(rew_batch, done_batch, self.gamma)
        values = self.value.predict(obs_batch)

        # adv_batch = reward_to_go - expected_return
        adv_batch = expected_return - values

        data = dict(obs=obs_batch,
                    act=act_batch,
                    val=values,
                    adv=adv_batch,
                    ret=expected_return)
        return data

    def compute_policy_loss(self, obs_batch, act_batch, adv_batch):
        logp = self.policy.get_log_prob(obs_batch, act_batch)
        return -(logp * adv_batch).mean()

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return

        self.data = self._get_data()
        obs, act, adv = self.data["obs"],  self.data["act"],  self.data["adv"]

        loss = self.compute_policy_loss(obs, act, adv)
        self.policy.learn(loss)

        self.logger.log(name='policy_updates', loss=loss.item())

        # Log Model
        self.logger.log_model(self.policy)

    def update_value(self):
        if len(self.memory) < self.batch_size:
            return

        for _ in range(self.num_v_iters):
            self.value.optimizer.zero_grad()
            val, ret = self.value.predict(self.data["obs"]).squeeze(1), self.data["ret"]
            loss = F.mse_loss(val, ret)
            loss.backward()
            self.value.optimizer.step()

        self.memory.reset_memory()
