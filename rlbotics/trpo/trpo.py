import torch
import torch.nn.functional as F

from rlbotics.trpo.memory import Memory
from rlbotics.trpo.utils import *
from rlbotics.common.policies import MLPSoftmaxPolicy
from rlbotics.common.approximators import MLP
from rlbotics.common.logger import Logger
from rlbotics.common.utils import *


class TRPO:
    def __init__(self, args, env):
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        # General parameters
        self.gamma = args.gamma
        self.lam = args.lam
        self.seed = args.seed
        self.num_v_iters = args.num_v_iters

        # TRPO specific hyperparameters
        self.kl_target = args.kl_target

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
        self.logger = Logger('TRPO', args.env_name, self.seed)

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

        old_log_prob = self.policy.get_log_prob(obs_batch, act_batch)
        old_policy = self.policy(obs_batch)
        old_policy = torch.distributions.utils.clamp_probs(old_policy)

        expected_return = get_expected_return(rew_batch, done_batch, self.gamma)
        values = torch.flatten(self.value.predict(obs_batch))

        adv_batch = expected_return - values

        adv_batch = finish_path(rew_batch, done_batch, values, adv_batch, self.gamma, self.lam)

        data = dict(obs=obs_batch,
                    act=act_batch,
                    val=values,
                    adv=adv_batch,
                    ret=expected_return,
                    old_log_prob=old_log_prob,
                    old_policy=old_policy)
        return data

    def compute_policy_loss(self):
        # return max_step
        obs = self.data["obs"]
        act = self.data["act"]
        adv = self.data["adv"]

        new_policy = self.policy(obs)
        new_policy = torch.nn.Softmax(dim=1)(new_policy)
        new_policy = torch.distributions.utils.clamp_probs(new_policy)
        #new_policy = self.policy.get_distribution(obs)
        old_policy = self.data["old_policy"]

        new_log_prob = self.policy.get_log_prob(obs, act)
        old_log_prob = self.data["old_log_prob"]


        L = torch.mul(torch.exp(new_log_prob - old_log_prob.detach()), adv).mean()
        kl = kl_div(old_policy, new_policy)
        #kl = torch.distributions.kl.kl_divergence(old_policy, new_policy).mean()

        #ent = new_policy.entropy().mean()

        #print(L, kl, ent)
        #logging_info = dict(kl=kl, entropy=ent)

        return L, kl#, ent

    def update_policy(self):
        self.data = self._get_data()

        L, kl = self.compute_policy_loss()

        parameters = list(self.policy.parameters())

        print("first kl", kl)

        g = flat_grad(L, parameters, retain_graph=True)
        d_kl = flat_grad(kl, parameters, create_graph=True)

        def HVP(v):
            return flat_grad(torch.dot(d_kl, v), parameters, retain_graph=True)

        search_dir = conjugate_gradient(HVP, g)
        print("search dir = ", search_dir)
        print("denom = ", torch.dot(search_dir, HVP(search_dir)))
        max_length = torch.sqrt(2 * self.kl_target / torch.dot(search_dir, HVP(search_dir)))
        print("max_length = ", max_length)
        max_step = max_length * search_dir
        print("max_step = ", max_step)

        i = 0
        while not self.take_step((0.9 ** i) * max_step, L) and i < 10:
            i += 1

        self.logger.log(name='policy_updates', loss=L.item())

        # Log Model
        self.logger.log_model(self.policy)

    def take_step(self, step, L):
        apply_update(self.policy, step)

        L_new, kl_new = self.compute_policy_loss()

        print("kl_new = ", kl_new)

        L_improvement = L_new - L

        if L_improvement > 0 and kl_new <= self.kl_target:
            return True

        apply_update(self.policy, -step)
        return False


    def update_value(self):
        for _ in range(self.num_v_iters):
            self.value.optimizer.zero_grad()
            val, ret = self.value.predict(self.data["obs"]).squeeze(1), self.data["ret"]
            loss = F.mse_loss(val, ret)
            loss.backward()
            self.value.optimizer.step()

        self.memory.reset_memory()