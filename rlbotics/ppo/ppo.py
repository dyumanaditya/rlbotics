import torch
import torch.nn.functional as F

from rlbotics.ppo.memory import Memory
from rlbotics.common.policies import MLPSoftmaxPolicy
from rlbotics.common.approximators import MLP
from rlbotics.common.logger import Logger
from rlbotics.common.utils import GAE, get_expected_return


class PPO:
    def __init__(self, args, env):
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        # General parameters
        self.gamma = args.gamma
        self.lam = args.lam
        self.seed = args.seed
        self.num_value_iters = args.num_value_iters
        self.num_policy_iters = args.num_policy_iters

        # PPO specific hyperparameters
        self.kl_target = args.kl_target
        self.clip_ratio = args.clip_ratio

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
        self.logger = Logger('PPO', args.env_name, self.seed)

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
        value = self.value(obs)
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

    def compute_policy_loss(self):
        obs = self.data["obs"].detach()
        act = self.data["act"].detach()
        adv = self.data["adv"].detach()

        # New and old policies
        new_policy = self.policy.get_distribution(obs)
        old_policy = self.data["old_policy"]

        # Log probabilities and probability ratio of new and old policies
        log_prob = self.policy.get_log_prob(obs, act)
        old_log_prob = self.data["old_log_prob"].detach()
        prob_ratio = torch.exp(log_prob - old_log_prob)

        # Advantage clipping
        clip_prob_ratio = torch.clamp(prob_ratio, 1-self.clip_ratio, 1+self.clip_ratio)

        # Clipped loss
        clipped_loss = -torch.min(prob_ratio * adv, clip_prob_ratio * adv).mean()

        # Useful information
        kl = torch.distributions.kl.kl_divergence(old_policy, new_policy).mean().item()
        ent = new_policy.entropy().mean().item()
        logging_info = dict(kl=kl, entropy=ent)

        return clipped_loss, logging_info

    def update_policy(self):
        self.data = self._get_data()

        for _ in range(self.num_policy_iters):
            policy_loss, logging_info = self.compute_policy_loss()

            if logging_info["kl"] > 1.5 * self.kl_target:
                break

            self.policy.optimizer.zero_grad()
            policy_loss.backward()
            self.policy.optimizer.step()

        self.logger.log(name='policy_updates', loss=policy_loss.item(), kl=logging_info["kl"], entropy=logging_info["entropy"])

        # Log Model
        self.logger.log_model(self.policy, name='pi')

    def update_value(self):
        for _ in range(self.num_value_iters):
            val = self.value(self.data["obs"]).squeeze(1)
            ret = self.data["ret"]

            loss = F.mse_loss(val, ret)

            self.value.optimizer.zero_grad()
            loss.backward()
            self.value.optimizer.step()

        self.memory.reset_memory()
