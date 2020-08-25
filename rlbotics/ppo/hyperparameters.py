# General Parameters:
env_name            = 'CartPole-v1'
gamma               = 0.99
lam                 = 0.95
max_iterations      = 1024
max_epochs          = 400
render              = False
num_value_iters     = 80
num_policy_iters    = 20
seed                = 0

# PPO specific hyperparameters
kl_target   = 0.003
clip_ratio  = 0.2

# Policy Network:
pi_hidden_sizes   = [128, 128]
pi_activations    = ['relu', 'relu', 'none']
pi_lr             = 3e-5
pi_optimizer        = 'Adam'

# Value Network:
v_hidden_sizes   = [128, 128, 1]
v_activations    = ['relu', 'relu', 'relu', 'none']
v_lr             = 1e-4
v_optimizer        = 'Adam'
