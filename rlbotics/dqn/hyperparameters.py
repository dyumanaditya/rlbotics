# General Parameters:
env_name       = 'CartPole-v1'
gamma          = 0.95
lr             = 1e-3
max_iterations = 1000000
render         = False
seed           = 0

# DQN Specific:
batch_size    = 512
buffer_size   = 6000

# Exp. epsilon decay
epsilon 	  = 1.0
min_epsilon   = 0.01
exp_decay     = 200

# Linear epsilon decay
linear_decay   = 0.001

# Policy Network:
hidden_sizes   = [64, 64] 	# Dimensions have to be 1 less than activations
activations    = ['relu', 'relu', 'none']
optimizer      = 'Adam'
loss_type      = 'mse'

# Policy Target Network:
update_target_freq = 10	# Update target network per _ timesteps

# Gradient clipping
use_grad_clip = True
