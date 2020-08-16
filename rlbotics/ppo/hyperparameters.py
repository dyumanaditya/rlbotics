# General Parameters:
env_name       = 'CartPole-v1'
gamma          = 0.99
max_iterations = 1000000
render         = False
batch_size     = 512
num_value_iters = 80
num_policy_iters = 20

# PPO specific hyperparameters
kl_target = 0.003
clip_ratio = 0.2

# Policy Network:
pi_hidden_sizes   = [64, 64]
pi_activations    = ['relu', 'relu', 'none']
pi_lr             = 1e-3

# Policy Network:
v_hidden_sizes   = [64, 64, 1]
v_activations    = ['relu', 'relu', 'relu', 'none']
v_lr             = 1e-3
