# General Parameters:
env_name       = 'CartPole-v1'
gamma          = 1.0
lr             = 1e-4
max_iterations = 200000
render         = False
seed           = 0

# DDQN Specific:
batch_size    = 512
buffer_size   = 100000

# Exp. epsilon decay
epsilon 	  = 1.0
min_epsilon   = 0.01
exp_decay     = 0.0001

# Linear epsilon decay
linear_decay   = 0.001

# Policy Network:
hidden_sizes   = [64, 64] 	# Dimensions have to be 1 less than activations
activations    = ['relu', 'relu', 'none']
optimizer      = 'Adam'
loss_type      = 'mse'

# Policy Target Network:
update_target_freq = 10	# Update target network per _ episodes

# Gradient clipping
use_grad_clip = False
