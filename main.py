import torch
import yaml
import os

from Environment import Environment
from train import train

# hyperparameter_to_tune = {'layers': [[128, 128], [256, 256], [64, 64, 64], [64, 128], [128, 64]]}
# hyperparameter_to_tune = {'timesteps_per_batch': [1000, 8000]}
# hyperparameter_to_tune = {'standard_deviation': [0.01, 0.05, 0.1, 0.3]}
# hyperparameter_to_tune = {'discount_factor': [0.99, 0.95, 0.9]}
# hyperparameter_to_tune = {'clip_ratio': [0.1, 0.2, 0.3]}
# hyperparameter_to_tune = {'lr': [1e-4, 3e-4, 1e-3]}
hyperparameter_to_tune = {'updates_per_iteration': [3, 5, 10]}

# This can speed up training if you have a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load hyperparameters
with open('config.yaml', 'r') as file:
    hyperparameters = yaml.safe_load(file)

# ENVIRONMENT
env = Environment()
parameter = list(hyperparameter_to_tune.keys())[0]

for value in hyperparameter_to_tune[parameter]:
    if parameter == 'lr':
        hyperparameters['lr_actor'] = value
        hyperparameters['lr_critic'] = value
    else:
        hyperparameters[parameter] = value

    print(hyperparameters.values())

    print('********************************')
    print(f'Tuning {parameter} = {value}')

    if not os.path.exists(f'logs/{parameter}'):
        os.makedirs(f'logs/{parameter}')
    if not os.path.exists(f'models/{parameter}'):
        os.makedirs(f'models/{parameter}')

    if parameter == 'layers':
        file_name = '_'.join(map(str, value))

        log_path = f'logs/{parameter}/{file_name}.csv'
        ckpt_path = f'models/{parameter}/{file_name}.pth'
    else:
        log_path = f'logs/{parameter}/{value}.csv'
        ckpt_path = f'models/{parameter}/{value}.pth'

    train(env, hyperparameters, log_path, ckpt_path, device)
    print('********************************\n\n')
        