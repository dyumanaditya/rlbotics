# https://spinningup.openai.com/en/latest/algorithms/ddpg.html
# General Parameters:
env_name       = 'Humanoid-v2'
gamma          = 0.99
max_iterations = 1500000
render         = False
seed           = 0
use_grad_clip  = False
save_freq      = 3000		  # Freq to save policy and q models

# DDPG Specific:
batch_size     = 100
buffer_size    = 1e6
polyak         = 0.001		  # Soft update for target network
act_noise      = 0.1		  # Stddev for Gaussian exploration noise added to policy at training time
noise_type     = 'gaussian'   # Gaussian or OU noise
random_steps   = 1e4		  # Random actions before training for exploration
update_after   = 1000		  # Number of env interactions to collect before training. Ensures replay buffer is full
update_every   = 50           # Number of iteration to pass before doing an update.
# Note: Regardless of how long you wait between updates, the ratio of env steps to gradient steps is locked to 1.

# Policy Network Parameters
pi_lr           = 1e-4
pi_hidden_sizes = [256, 256, 256] 	  # Dimensions have to be 1 less than activations
pi_activations  = ['relu', 'relu', 'relu', 'tanh']
pi_optimizer    = 'Adam'

# Q Network Parameters
q_lr      	    = 1e-3
q_hidden_sizes  = [256, 256, 256]	  # Dimensions have to be 1 less than activations
q_activations   = ['relu', 'relu', 'relu', 'none']
q_optimizer     = 'Adam'
q_loss_type     = 'mse'
weight_decay    = 0

# Both Networks
weight_init     = None
batch_norm      = False
