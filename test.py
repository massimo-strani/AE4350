import torch
import yaml
import numpy as np

from Environment import Environment
from PPO import PPO

from utils import *


def test_policy(env: Environment, hyperparameters: dict, model_path: str, num_episodes: int, device: torch.device = torch.device('cpu')):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    ppo_agent = PPO(obs_dim, act_dim, device=device, **hyperparameters)

    print('loading model from : ' + model_path)
    ppo_agent.load(model_path)

    test_reward = 0
    for i_episode in range(1, num_test_episodes + 1):
        obs = env.reset()
        episode_reward = 0

        for time_step in range(1, hyperparameters['timesteps_per_episode'] + 1):
            act = ppo_agent.get_action(obs)
            obs, reward, done, info = env.step(act)

            episode_reward += reward

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_reward += episode_reward
        print('\n------------------- Episode {} -----------------'.format(i_episode))
        print('Reward: {}'.format(episode_reward))
        episode_reward = 0

        print('Termination reason: ', info['termination reason'])
        print('Miss distance: ', info['relative distance'])
        print('Orbital elements: ' + ', '.join(f'{key}: {value:.4f}' for key, value in info['orbital elements'].items()))
        print('Cumulative delta V: ', info['cumulative deltaV'])

        fig, ax = plot_nominal_orbits(env)
        sat_history = np.zeros((len(env.state_history), 2))
        
        for i, orbital_elements in enumerate(env.state_history):
            sat_pos, _ = env._keplerian2cartesian(**orbital_elements)
            sat_history[i,:] = sat_pos

        ax.plot(sat_history[:,0], sat_history[:,1], '-', label='Satellite Trajectory', color='tab:blue')
        plt.legend()
        plt.show()

    print('\nAverage Reward over {} episodes: {}'.format(num_test_episodes, test_reward / num_test_episodes))


if __name__ == '__main__':
    plot_log('logs/test.csv', show=True)
    exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = 'models/ppo_test.pth'
    num_test_episodes = 10

    with open('config.yaml', 'r') as file:
        hyperparameters = yaml.safe_load(file)
    
    env = Environment()
    test_policy(env, hyperparameters, model_path, num_test_episodes, device=device)
    