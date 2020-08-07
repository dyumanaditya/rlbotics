# General Parameters:
env_name      = 'CartPole-v1'
gamma         = 0.99
lr            = 0.01
num_episodes  = 100
render        = False

# DQN Specific:
epsilon 	  = 1
min_epsilon   = 0.1
epsilon_decay = 0.995
batch_size    = 64
memory_limit  = 2000

# Policy Network:
start_learning = 500        # Start learning after some experience
hidden_sizes   = [64, 64]	# Dimensions have to be 1 less than activations/layer_types
activations    = ['relu', 'relu', 'softmax']
layer_types    = ['linear', 'linear', 'linear']
optimizer      = 'Adam'
loss           = 'mse'

# Policy Target Network:
update_target_net = 500	# Update target network per _ episodes

