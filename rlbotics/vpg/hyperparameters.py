# General Parameters:
env_name        = 'CartPole-v1'
gamma           = 0.99
lam             = 0.99
max_iterations  = 512
max_epochs      = 1000
render          = False
num_v_iters     = 1
seed            = 0

# Policy Network:
pi_hidden_sizes     = [128, 128]
pi_activations      = ['relu', 'relu', 'none']
pi_lr               = 0.01
pi_optimizer        = 'Adam'

# Value Network:
v_hidden_sizes      = [128, 128, 1]
v_activations       = ['relu', 'relu', 'relu', 'none']
v_lr                = 0.01
v_optimizer        = 'Adam'
