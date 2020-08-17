# General Parameters:
env_name        = 'CartPole-v1'
gamma           = 0.99
max_iterations  = 1000
max_epochs      = 50
render          = False
batch_size      = 512
num_v_iters     = 80
seed            = 0

# Policy Network:
pi_hidden_sizes     = [64, 64]
pi_activations      = ['relu', 'relu', 'none']
pi_lr               = 1e-3
pi_optimizer        = 'Adam'

# Value Network:
v_hidden_sizes      = [64, 64, 1]
v_activations       = ['relu', 'relu', 'relu', 'none']
v_lr                = 1e-3
v_optimizer        = 'Adam'
