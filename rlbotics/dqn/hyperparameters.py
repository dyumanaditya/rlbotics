# General Parameters:
env_name       = 'CartPole-v1'
gamma          = 1.0
lr             = 6e-5
max_iterations = 1000000
render         = False
seed           = 0

# DQN Specific:
batch_size    = 32
buffer_size   = 50000

# Exp. epsilon decay
epsilon 	  = 1.0
min_epsilon   = 0.001
exp_decay     = 200

# Linear epsilon decay
linear_decay   = 0.001

# Policy Network:
hidden_sizes   = [64, 64] 	# Dimensions have to be 1 less than activations
activations    = ['relu', 'relu', 'none']
optimizer      = 'Adam'
loss_type      = 'mse'

# Policy Target Network:
update_target_freq = 20	# Update target network per _ episodes

# Gradient clipping
use_grad_clip = True
