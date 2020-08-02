### RLbotics

Reinforcement Learning Toolbox developed in Pytorch.

## Toolbox Structure
All the algorithms are in the `rlbotics` directory. Each algorithm specified above has an individual directory.

### Common
The directory `common` contains common modular classes to easily build new algorithms.
- `approximators`: Basic Deep Neural Networks (Dense, Conv, LSTM).
- `data_collection`: Performs rollouts and collect observations and rewards
- `logger`: Log training data and other information
- `plotter`: Plot graphs
- `policies`: Common policies such as Random, Softmax, Parametrized Softmax and Gaussian Policy
- `utils`: Functions to compute the expected return, the Generalized Advantage Estimation (GAE), etc.

### Algorithm Directories
Each algorithm directory contains at least 3 files:
- `main.py`: Main script to run the algorithm
- `hyperparameters.py`: File to contain the default hyperparameters
- `<algo>.py`: Implementation of the algorithm
- `utils.py`: (Optional) File containing some utility functions

Some algorithm directories may have additional files specific to the algorithm.

## Contributing
To contribute to this package, it is recommended to follow this structure:
- The new algorithm directory should at least contain the 3 files mentioned above.
- `main.py` should contain at least the following functions:
  - `main`: Parses input argument, builds the environment and agent, and train the agent.
  - `argparse`: Parses input argument and loads default hyperparameters from `hyperparameter.py`.
- `<algo>.py` should contain at least the following methods:
  - `__init__`: Initializes the classes
  - `_build_policy`: Build policy
  - `_build_value_function`: Build value function
  - `_loss`: Build policy loss function
  - `train`: Main training loop called by `main.py`
  - `update_policy`: Update the policy
  - `update_value`: Update the value function
  - `process_paths`: (optional) Process collected trajectories 
