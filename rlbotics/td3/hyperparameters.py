# https://spinningup.openai.com/en/latest/algorithms/td3.html
# General Parameters:
env_name       = 'Humanoid-v2'
gamma          = 0.99
max_iterations = 1500000
render         = False
seed           = 0
use_grad_clip  = False
save_freq      = 2000		  # Freq to save policy and q models
resume         = False

# TD3 Specific:
batch_size     = 256
buffer_size    = 1e6
polyak         = 0.005		  # Soft update for target network
act_noise      = 0.1		  # Stddev for Gaussian exploration noise added to policy at training time
random_steps   = 1e4		  # Random actions before training for exploration
update_after   = 1000		  # Number of env interactions to collect before training. Ensures replay buffer is full

# Policy Network Parameters
pi_lr           = 3e-4
pi_hidden_sizes = [256, 256]  # Dimensions have to be 1 less than activations
pi_activations  = ['relu', 'relu', 'tanh']
pi_optimizer    = 'Adam'
pi_update_delay = 2			  # Policy update every __ q updates
pi_targ_noise   = 0.2		  # Noise added to target
noise_clip		= 0.5		  # Clip after adding targ_noise

# Q Network Parameters
q_lr      	    = 3e-4
q_hidden_sizes  = [256, 256]  # Dimensions have to be 1 less than activations
q_activations   = ['relu', 'relu', 'none']
q_optimizer     = 'Adam'
q_loss_type     = 'mse'
weight_decay    = 0			  # L2 weight decay for the parameters

# Both Networks
weight_init     = None		  # Initialise MLP params from (-x, x) either float or None
batch_norm      = False
