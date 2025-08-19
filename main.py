import torch
import yaml
import numpy as np

from environment import Environment
from ppo import PPO

# This can speed up training if you have a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# ENVIRONMENT
env = Environment(config['environment'])

obs_dim = env.observation_space.shape[0] # TODO: implement observation and action spaces in Environment class
act = env.action_space.shape[0]

# TRAINING LOOP
# load hyperparameters
hyperparameters = config['training']

# initialize PPO agent
ppo_agent = PPO(obs_dim, act, **hyperparameters)

time_step = 0

while time_step <= hyperparameters['total_timesteps']:
    # Reset environment and get initial observation
    obs = env.reset()
    done = False

    episode_reward = 0
    episode_length = 0

    for t in range(1, hyperparameters['timesteps_per_episode'] + 1):
        time_step += 1

        # Select action from policy
        action = ppo_agent.get_action(obs) # saves action, observation and log probability in buffer
        obs, reward, done, _ = env.step(action)

        # save reward into buffer and is_done flag (to distinguish between episodes)
        ppo_agent.buffer.rews.append(reward)
        ppo_agent.buffer.dones.append(done)

        # update PPO agent
        if time_step % hyperparameters['timesteps_per_batch'] == 0:
            ppo_agent.update()

        if done:
            break

    # Update policy
    ppo_agent.update()
