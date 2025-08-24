import torch
import yaml
import numpy as np
import os
from datetime import datetime

import gymnasium as gym
from gym_satellite_ca.envs import sat_data_class

from PPO import PPO

RANDOM_SEED = 42

# This can speed up training if you have a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# ENVIRONMENT
primary = sat_data_class.SatelliteData(**config['satellite'])
primary.change_angles_to_radians()
primary.set_random_tran()

env = gym.make('CollisionAvoidanceEnv-v0', satellite=primary)