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
        self.lam = args.lam
        self.seed = args.seed
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
                                       seed=self.seed,
                                       optimizer=self.pi_optimizer,
                                       lr=self.pi_lr)
        self.policy.summary()

    def _build_value_function(self):
        self.value = MLP(layer_sizes=[self.obs_dim] + self.v_hidden_sizes,
                         activations=self.v_activations,
                         seed=self.seed,
                         optimizer=self.v_optimizer,
                         lr=self.v_lr)
        self.value.summary()

    def get_action(self, obs):
        return self.policy.get_action(obs)

    def store_transition(self, obs, act, rew, new_obs, done):
        value = self.value.predict(obs)
        log_prob = self.policy.get_log_prob(obs, torch.tensor(act))

        self.memory.add(obs, act, rew, new_obs, done, log_prob, value)

        # Log Done, reward
        self.logger.log(name='transitions', done=done, rewards=rew)

    def _get_data(self):
        # get batches from memory
        transition_batch = self.memory.get_batch()

        obs_batch = torch.as_tensor(transition_batch.obs, dtype=torch.float32)
        act_batch = torch.as_tensor(transition_batch.act, dtype=torch.int32)
        rew_batch = torch.as_tensor(transition_batch.rew, dtype=torch.float32)
        done_batch = torch.as_tensor(transition_batch.done, dtype=torch.int32)

        log_prob = torch.cat(list(transition_batch.log_prob))
        values = torch.cat(transition_batch.val).squeeze(-1)

        expected_return = get_expected_return(rew_batch, done_batch, self.gamma)
        adv_batch = GAE(rew_batch, done_batch, values, self.gamma, self.lam)

        old_policy = self.policy.get_distribution(obs_batch)

        data = dict(obs=obs_batch,
                    act=act_batch,
                    val=values,
                    adv=adv_batch,
                    ret=expected_return,
                    old_log_prob=log_prob,
                    old_policy=old_policy)
        return data

    def compute_policy_loss(self, log_prob_batch, adv_batch):
        return -(log_prob_batch * adv_batch).mean()

    def update_policy(self):
        self.data = self._get_data()

        adv = self.data["adv"]
        log_prob = self.data["old_log_prob"]

        loss = self.compute_policy_loss(log_prob, adv)
        self.policy.learn(loss)

        self.logger.log(name='policy_updates', loss=loss.item())

        # Log Model
        self.logger.log_model(self.policy)

    def update_value(self):
        for _ in range(self.num_v_iters):
            val= self.value.predict(self.data["obs"]).squeeze(1)
            ret = self.data["ret"]

            loss = F.mse_loss(val, ret)

            self.value.optimizer.zero_grad()
            loss.backward()
            self.value.optimizer.step()

        self.memory.reset_memory()
