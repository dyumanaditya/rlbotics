# General Parameters:
env_name       = 'CartPole-v1'
gamma          = 0.95
lr             = 0.001
max_iterations = 200000
render         = False

# DQN Specific:
epsilon 	  = 1.0
min_epsilon   = 0.01
epsilon_decay = 0.995
batch_size    = 128
buffer_size   = 6000

# Policy Network:
hidden_sizes   = [64, 64] 	# Dimensions have to be 1 less than activations
activations    = ['relu', 'relu', 'none']
optimizer      = 'Adam'
loss           = 'mse'

# Policy Target Network:
update_target_freq = 10	# Update target network per _ timesteps

