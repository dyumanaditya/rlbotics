# General Parameters:
env_name      = 'CartPole-v1'
<<<<<<< Updated upstream
gamma         = 0.95
lr            = 0.001
num_episodes  = 1000
=======
gamma         = 0.99
lr            = 0.01
num_episodes  = 300
>>>>>>> Stashed changes
render        = False

# DQN Specific:
epsilon 	  = 1.0
min_epsilon   = 0.01
epsilon_decay = 0.995
batch_size    = 64
memory_limit  = 2000

# Policy Network:
<<<<<<< Updated upstream
start_learning = 400        # Start learning after some experience
hidden_sizes   = [120, 120, 120]	# Dimensions have to be 1 less than activations
activations    = ['relu', 'relu', 'relu', 'none']
=======
start_learning = 500        # Start learning after some experience
hidden_sizes   = [64, 64]	# Dimensions have to be 1 less than activations/layer_types
activations    = ['tanh', 'tanh', 'none']
layer_types    = ['linear', 'linear', 'linear']
>>>>>>> Stashed changes
optimizer      = 'Adam'
loss           = 'mse'

# Policy Target Network:
update_target_net = 150	# Update target network per _ timesteps

