import torch
import yaml
import numpy as np
import argparse
import os

from Environment import Environment
from PPO import PPO

from utils import *

def test_policy(env: Environment, hyperparameters: dict, model_path: str, num_episodes: int, device: torch.device = torch.device('cpu'), show: bool = False, save: bool = False, plot_divergence: bool = False):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    ppo_agent = PPO(obs_dim, act_dim, device=device, **hyperparameters)

    print('loading model from : ' + model_path)
    ppo_agent.load(model_path)

    test_reward = 0
    test_deltaV = 0
    test_duration = 0
    for i_episode in range(1, num_episodes + 1):
        obs = env.reset()
        episode_reward = 0

        epochs = []
        for _ in range(1, hyperparameters['timesteps_per_episode'] + 1):
            act = ppo_agent.get_action(obs)
            obs, reward, done, info = env.step(act)

            episode_reward += reward
            epochs.append(env.epoch)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        print('Episode {} \t Reward: {:.2f} \t DeltaV: {:.3f} km/s \t Duration: {:.3f} min'.format(i_episode, episode_reward, info['cumulative deltaV'], epochs[-1]/60))

        # store statistics
        test_reward += episode_reward
        episode_reward = 0

        test_deltaV += info['cumulative deltaV']
        test_duration += epochs[-1]/60

        # compute trajectory and orbit divergence
        sat_history = np.zeros((len(env.state_history), 2))
        orbit_divergence = []

        for i, orbital_elements in enumerate(env.state_history):
            sat_pos, _ = env._keplerian2cartesian(**orbital_elements)
            sat_history[i,:] = sat_pos

            orbit_divergence.append({
                'sma': (orbital_elements['sma'] - env.nominal_orbit['sma']),
                'ecc': (orbital_elements['ecc'] - env.nominal_orbit['ecc']),
                'aop': (orbital_elements['aop'] - env.nominal_orbit['aop']),
            })

        if plot_divergence:
            fig, ax = plt.subplots(3,1,figsize=(6,8), sharex=True)

            for i, key in enumerate(['sma', 'ecc', 'aop']):
                data = [d[key] for d in orbit_divergence[:-1]]
                ax[i].plot(epochs, data)
                ax[i].set_ylabel(['$\Delta$ a [km]', '$\Delta$ e [-]', '$\Delta$ ω [rad]'][i])
                ax[i].grid()
                
            ax[2].set_xlabel('Time [s]')

            fig.tight_layout()
            
            if show:
                plt.show()
            
            if save:
                fig.savefig('plots/test_episode_{}_divergence.pdf'.format(i_episode))
                plt.close()


        if show or save:
            fig, ax = plot_nominal_orbits(env)
            ax.plot(sat_history[:,0], sat_history[:,1], '-', label='Satellite Trajectory', color='tab:blue')
            ax.plot(sat_history[-1,0], sat_history[-1,1], '^', label='Satellite @ TCA', color='tab:blue')
            plt.legend(loc='upper right')

            fig.tight_layout()

        if show:
            plt.show()

        if save:
            fig.savefig('plots/test_episode_{}.pdf'.format(i_episode))
            plt.close()

    avg_reward = test_reward / num_episodes
    avg_deltaV = test_deltaV / num_episodes
    avg_duration = test_duration / num_episodes

    return avg_reward, avg_deltaV, avg_duration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--show', type=str, default='True')
    parser.add_argument('--action', type=str, default='plot', choices=['plot', 'test'])
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--save', type=str, default='True')
    parser.add_argument('--plot_divergence', type=str, default='False')

    args = parser.parse_args()

    for flag in [args.show, args.save, args.plot_divergence]:
        if flag.lower() == 'true':
            vars(args)[flag] = True
        elif flag.lower() == 'false':
            vars(args)[flag] = False

    if args.action == 'plot':
        directory = 'logs/' + args.model
        files = os.listdir(directory)

        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:olive', 'tab:brown', 'tab:magenta', 'tab:cyan', 'tab:crimson','tab:gray', 'tab:black']

        fig, ax = plt.subplots(figsize=(7,6))
        for i, file in enumerate(files):
            if args.model == 'layers':
                file_name = '[' + file.split('.csv')[0].replace('_', ', ') + ']'
            else:
                file_name = file.split('.csv')[0]

            plot_log(directory + '/' + file, show=False, figure=(fig, ax), label=file_name, color=colors[i])

        plt.show()
        fig.savefig('plots/' + args.model + '.pdf')
        plt.close()

    elif args.action == 'test':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_path = 'models/' + args.model + '.pth'

        with open('config.yaml', 'r') as file:
            hyperparameters = yaml.safe_load(file)
        
        env = Environment()
        avg_reward, avg_deltaV, avg_duration = test_policy(env, hyperparameters, model_path, args.num_episodes, device=device, show=args.show, save=args.save, plot_divergence=args.plot_divergence)

        print(f'Average Reward over {args.num_episodes} episodes: {avg_reward}')
        print(f'Average DeltaV over {args.num_episodes} episodes: {avg_deltaV} km/s')
        print(f'Average Duration over {args.num_episodes} episodes: {avg_duration} min')
