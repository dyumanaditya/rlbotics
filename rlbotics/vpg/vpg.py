import torch
import torch.nn.functional as F

from rlbotics.vpg.memory import Memory
from rlbotics.common.policies import MLPSoftmaxPolicy
from rlbotics.common.approximators import MLP
from rlbotics.common.logger import Logger
from rlbotics.common.utils import GAE, get_expected_return


class VPG:
    def __init__(self, args, env):
        self.env = env

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
        continuous = False if len(self.env.action_space.shape) == 0 else True

        if continuous:
            self.obs_dim = self.env.observation_space.shape[0]
            self.act_dim = self.env.action_space.shape[0]
            self.act_lim = self.env.action_space.high[0]

            self.policy = MLPGaussianPolicy(act_lim=self.act_lim,
                                           layer_sizes=[self.obs_dim] + self.pi_hidden_sizes + [self.act_dim],
                                           activations=self.pi_activations,
                                           seed=self.seed)
        else:
            self.obs_dim = self.env.observation_space.shape[0]
            self.act_dim = self.env.action_space.n

            self.policy = MLPSoftmaxPolicy(layer_sizes=[self.obs_dim] + self.pi_hidden_sizes + [self.act_dim],
                                           activations=self.pi_activations,
                                           seed=self.seed)
        self.policy.summary()

        # Set Optimizer
        if self.pi_optimizer == 'Adam':
            self.pi_optim = torch.optim.Adam(self.policy.parameters(), lr=self.pi_lr)
        elif self.pi_optimizer == 'RMSprop':
            self.pi_optim = torch.optim.RMSprop(self.policy.parameters(), lr=self.pi_lr)
        else:
            raise NameError(str(self.pi_optimizer) + ' Optimizer not supported')

    def _build_value_function(self):
        self.value = MLP(layer_sizes=[self.obs_dim] + self.v_hidden_sizes,
                         activations=self.v_activations,
                         seed=self.seed)
        self.value.summary()

        # Set Optimizer
        if self.v_optimizer == 'Adam':
            self.v_optim = torch.optim.Adam(self.value.parameters(), lr=self.v_lr)
        elif self.v_optimizer == 'RMSprop':
            self.v_optim = torch.optim.RMSprop(self.value.parameters(), lr=self.v_lr)
        else:
            raise NameError(str(self.v_optimizer) + ' Optimizer not supported')

    def get_action(self, obs):
        return self.policy.get_action(obs)

    def store_transition(self, obs, act, rew, new_obs, done):
        value = self.value(obs)
        log_prob = self.policy.get_log_prob(obs, torch.tensor(act))

        self.memory.add(obs, act, rew, new_obs, done, log_prob, value)

        # Log Done, reward
        self.logger.log(name='transitions', done=done, rewards=rew)

    def _get_data(self):
        # get batches from memory
        transition_batch = self.memory.get_batch()

        # print(len(self.memory))

        obs_batch = torch.as_tensor(transition_batch.obs, dtype=torch.float)
        act_batch = torch.as_tensor(transition_batch.act, dtype=torch.int)
        rew_batch = torch.as_tensor(transition_batch.rew, dtype=torch.float)
        done_batch = torch.as_tensor(transition_batch.done, dtype=torch.int)

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

        # update policy
        self.pi_optim.zero_grad()
        loss.backward()
        self.pi_optim.step()

        self.logger.log(name='policy_updates', loss=loss.item())

        # Log Model
        self.logger.log_model(self.policy, name='pi')

    def update_value(self):
        for _ in range(self.num_v_iters):
            val = self.value(self.data["obs"]).squeeze(1)
            ret = self.data["ret"]

            loss = F.mse_loss(val, ret)

            # update value
            self.v_optim.zero_grad()
            loss.backward()
            self.v_optim.step()

        self.memory.reset_memory()
