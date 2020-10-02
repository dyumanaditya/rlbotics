# General Parameters:
env_name            = 'LunarLanderContinuous-v2'
gamma               = 0.99
lam                 = 0.95
max_iterations      = 1000
max_epochs          = 1000
render              = False
seed                = 0
memory_size         = 1000

# PPO specific hyperparameters
kl_target   = 0.003
clip_ratio  = 0.2

# Policy Network:
pi_hidden_sizes     = [64, 64]
pi_activations      = ['tanh', 'tanh', 'none']
pi_lr               = 5e-4
pi_optimizer        = 'Adam'
num_pi_iters    = 20

# Value Network:
v_hidden_sizes      = [64, 64, 1]
v_activations       = ['tanh', 'tanh', 'none']
v_lr                = 1e-3
v_optimizer         = 'Adam'
num_v_iters     = 80
