# General Parameters:
env_name       = 'CartPole-v1'
gamma          = 0.95
lr             = 0.0006
max_iterations = 1000000
render         = False


# DDQN Specific:
batch_size    = 512
buffer_size   = 6000

# Exp. epsilon decay
epsilon 	  = 1.0
min_epsilon   = 0.01
epsilon_decay = 0.999

# Linear epsilon decay
linear_decay   = 0.001


# Policy Network:
hidden_sizes   = [64, 64] 	# Dimensions have to be 1 less than activations
activations    = ['relu', 'relu', 'none']
optimizer      = 'Adam'
loss           = 'mse'

# Policy Target Network:
update_target_freq = 30	# Update target network per _ timesteps

