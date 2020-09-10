# https://spinningup.openai.com/en/latest/algorithms/ddpg.html
# General Parameters:
env_name       = 'Humanoid-v2'
seed           = 0
gamma          = 0.99
max_iterations = 1500000
render         = False
use_grad_clip  = False
save_freq      = 2000		  # Freq to save policy and q models
resume         = False		  # Resume training

# DDPG Specific:
batch_size     = 128
buffer_size    = 1e6
polyak         = 0.005		  # Soft update for target network
act_noise      = 0.05		  # Stddev for Gaussian exploration noise added to policy at training time
noise_type     = 'gaussian'   # Gaussian or ou noise
random_steps   = 1e4		  # Random actions before training for exploration
update_after   = 1000		  # Number of env interactions to collect before training. Ensures replay buffer is full

# Policy Network Parameters
pi_lr           = 1e-4
pi_hidden_sizes = [256] 	  # Dimensions have to be 1 less than activations
pi_activations  = ['relu', 'tanh']
pi_optimizer    = 'Adam'

# Q Network Parameters
q_lr      	    = 1e-3
q_hidden_sizes  = [256, 256, 128]	  # Dimensions have to be 1 less than activations
q_activations   = ['relu', 'relu', 'relu', 'none']
q_optimizer     = 'Adam'
q_loss_type     = 'mse'
weight_decay    = 1e-4				# L2 weight decay for the parameters

# Both Networks
weight_init     = 3e-3			# Initialise MLP params from (-x, x) either float or None
batch_norm      = False
