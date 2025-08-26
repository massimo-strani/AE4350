import torch
import yaml
import numpy as np
import os
from datetime import datetime

import gymnasium as gym

from Environment import Environment
from PPO import PPO

# This can speed up training if you have a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load hyperparameters
with open('config.yaml', 'r') as file:
    hyperparameters = yaml.safe_load(file)

# ENVIRONMENT
env = Environment()