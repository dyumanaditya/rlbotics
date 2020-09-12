# General Parameters:
env_name       = 'LunarLanderContinuous-v2'
seed           = 0
gamma          = 0.99
max_iterations = 1500000
render         = False
alpha          = 0.2
update_every   = 50
save_freq      = 1

batch_size     = 100
buffer_size    = 1e6
polyak         = 0.995
start_steps    = 10000
update_after   = 1000

pi_lr              = 1e-3
pi_hidden_sizes = [256, 256]
pi_activations  = ['relu', 'relu', 'tanh']
pi_optimizer    = 'Adam'

# Q Network Parameters
q_lr      	    = 3e-4
q_hidden_sizes  = [256, 256]  # Dimensions have to be 1 less than activations
q_activations   = ['relu', 'relu', 'none']
q_optimizer     = 'Adam'
q_loss_type     = 'mse'
