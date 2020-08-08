# General Parameters:
env_name      = 'CartPole-v1'
gamma         = 0.99
lr            = 0.01
num_episodes  = 300
render        = False

# DQN Specific:
epsilon  	  = 1
min_epsilon   = 0.05
epsilon_decay = 0.99
batch_size    = 128
buffer_size  = 2000

# Policy Network:
start_learning = 500        # Start learning after some experience
hidden_sizes   = [64, 64]	# Dimensions have to be 1 less than activations/layer_types
activations    = ['tanh', 'tanh', 'none']
layer_types    = ['linear', 'linear', 'linear']
optimizer      = 'RMSprop'
loss           = 'mse'

# Policy Target Network:
update_target_freq = 500	 # Update target network per _ episodes

