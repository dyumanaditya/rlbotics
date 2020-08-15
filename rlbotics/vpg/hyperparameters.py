# General Parameters:
env_name       = 'CartPole-v1'
gamma          = 0.99
max_iterations = 1000000
render         = False
batch_size     = 1000


# Policy Network:
pi_hidden_sizes   = [64, 64]
pi_activations    = ['relu', 'relu', 'none']
pi_lr             = 1e-3

# Policy Network:
v_hidden_sizes   = [64, 64, 1]
v_activations    = ['relu', 'relu', 'relu', 'none']
v_lr             = 1e-3
