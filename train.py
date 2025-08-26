import torch
import yaml
import numpy as np
import os
from datetime import datetime

from Environment import Environment
from PPO import PPO

def train(env: Environment, hyperparameters: dict, log_path: str, ckpt_path: str, device: torch.device = torch.device('cpu')):
    # logging and printing frequency
    log_frequency = hyperparameters['timesteps_per_episode'] * 2
    print_frequency = hyperparameters['timesteps_per_episode'] * 4

    log_f = open(log_path, 'w')
    log_f.write('episode,timestep,reward\n')

    print_reward = 0
    print_episodes = 0

    log_reward = 0
    log_episodes = 0

    # initialize PPO agent
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ppo_agent = PPO(obs_dim, act_dim, device=device, **hyperparameters)

    # convergence criteria
    window_rewards = []
    window_size = 50
    mean_reward = None
    convergence_threshold = 0.02
    std_threshold = 2500
    converged_windows = 0
    required_converged_windows = 5

    # tracking time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    time_step = 0
    i_episode = 0
    while time_step <= hyperparameters['total_timesteps']:
        i_episode += 1

        # Reset environment and get initial observation
        obs = env.reset()
        done = False

        episode_reward = 0
        episode_length = 0
        for t in range(1, hyperparameters['timesteps_per_episode'] + 1):
            time_step += 1
            episode_length += 1

            # Select action from policy
            action = ppo_agent.get_action(obs) # saves action, observation and log probability in buffer
            obs, reward, done, _ = env.step(action)

            episode_reward += reward

            # save reward into buffer and is_done flag (to distinguish between episodes)
            ppo_agent.buffer.rews.append(reward)
            ppo_agent.buffer.is_done.append(done)

            # update PPO agent
            if time_step % hyperparameters['timesteps_per_batch'] == 0:
                ppo_agent.update()

            # log
            if time_step % log_frequency == 0:
                log_avg_reward = log_reward / log_episodes
                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_reward = 0
                log_episodes = 0

            # print
            if time_step % print_frequency == 0:
                print_avg_reward = print_reward / print_episodes
                print('Episode: {}, Timestep: {}, Reward: {}'.format(i_episode, time_step, print_avg_reward))

                print_reward = 0
                print_episodes = 0

            # save
            if time_step % hyperparameters['save_frequency'] == 0:
                print('-------------------------------------------------------------')
                print('saving model at : ' + ckpt_path)
                ppo_agent.save(ckpt_path)
                print('model saved')
                print('time elapsed : ', datetime.now().replace(microsecond=0) - start_time)
                print('-------------------------------------------------------------')

            if done:
                break

        log_reward += episode_reward
        log_episodes += 1

        print_reward += episode_reward
        print_episodes += 1

        # check for convergence
        window_rewards.append(episode_reward)
        if len(window_rewards) > window_size:
            window_rewards.pop(0)

        if len(window_rewards) == window_size:
            window_avg = np.mean(window_rewards)
            window_std = np.std(window_rewards)

            if mean_reward is not None:
                relative_change = np.abs(window_avg - mean_reward) / (np.abs(mean_reward) + 1e-8)

                if relative_change < convergence_threshold and window_std < std_threshold:
                    converged_windows += 1
                else:
                    converged_windows = 0

                if converged_windows >= required_converged_windows:
                    print('Convergence achieved at episode {}, timestep {}!'.format(i_episode, time_step))
                    print('Final average reward: {:.2f} ± {:.2f}'.format(window_avg, window_std))
                    print('Total training time: ', datetime.now().replace(microsecond=0) - start_time)
                    break
            
            mean_reward = window_avg

    log_f.close()


if __name__ == '__main__':
    # This can speed up training if you have a GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ENVIRONMENT
    env = Environment()

    # LOGGING
    # create directory
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # create log file (overwrites previous logs)
    log_path = log_dir + '/test.csv'
    print('logging at : ' + log_path)

    # CHECKPOINTING
    ckpt_dir = 'models'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # create checkpoint path (overwrites previous checkpoints)
    ckpt_path = ckpt_dir + '/ppo_test.pth'
    print('save checkpoint path : ' + ckpt_path)

    # TRAINING LOOP
    # Load hyperparameters
    with open('config.yaml', 'r') as file:
        hyperparameters = yaml.safe_load(file)

    train(env, hyperparameters, log_path, ckpt_path, device=device)