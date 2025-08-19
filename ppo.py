import torch
import torch.nn as nn
import numpy as np

from network import ActorCritic

class RolloutBuffer:
    def __init__(self):
        self.obss = []
        self.acts = []
        self.rews = []
        self.vals = []
        self.logprob = []
        self.is_done = []

    def clear(self):
        self.obss = []
        self.acts = []
        self.rews = []
        self.vals = []
        self.logprob = []
        self.is_done = []


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, device: torch.device = torch.device('cpu'), **kwargs):
        self.device = device
        
        # Extract environment information
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.gamma = kwargs.get('gamma', 0.95)  # Discount factor
        self.updates_per_iteration = kwargs.get('updates_per_iteration', 5)  # Number of updates per iteration
        self.clip = kwargs.get('clip', 0.2)  # Clipping parameter for PPO

        # Initialize actor-critic network
        self.policy = ActorCritic(self.obs_dim, self.act_dim, device=device).to(device)
        self.buffer = RolloutBuffer()

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': kwargs.get('lr_actor', 5e-3)},
            {'params': self.policy.critic.parameters(), 'lr': kwargs.get('lr_critic', 5e-3)}
        ])

    def get_action(self, obs: np.ndarray):
        ''' Appends observation, action and log probability to the buffer and returns the action. '''
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            action, log_prob, value = self.policy.act(obs) # returns action, log probability and value that are already detached

        self.buffer.obss.append(obs)
        self.buffer.acts.append(action)
        self.buffer.vals.append(value)
        self.buffer.logprobs.append(log_prob)

        return action.cpu().numpy() # Convert action back to numpy array for environment step

    def update(self):
        ''' Update the policy using the collected data in the buffer. '''
        
        # Compute rewards-to-go
        rewards_to_go = []
        discounted_reward = 0

        for reward, is_done in zip(reversed(self.buffer.rews), reversed(self.buffer.is_done)):
            # the discounted reward is reset to 0 if the episode is done
            if is_done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards_to_go.insert(0, discounted_reward)

        # Convert rewards-to-go to tensor
        rewards_to_go = torch.FloatTensor(rewards_to_go).to(self.device)

        # Convert list of tensors (from buffer) to a single tensor
        batch_obss = torch.squeeze(torch.stack(self.buffer.obss), dim=0).to(self.device)
        batch_acts = torch.squeeze(torch.stack(self.buffer.acts), dim=0).to(self.device)
        batch_vals = torch.squeeze(torch.stack(self.buffer.vals), dim=0).to(self.device)
        batch_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs), dim=0).to(self.device)

        # Compute (normalized) advantages
        # Note: can try to normalize rtgs before computing advantages
        advantages = rewards_to_go - batch_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        for _ in range(self.updates_per_iteration):
            vals, logprobs = self.policy.evaluate(batch_obss, batch_acts)

            # compute ratio of new and old probabilities
            ratios = torch.exp(logprobs - batch_logprobs)

            # compute surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages # ensures that the ratio is within the clipping range and doesn't step too far during stochastic gradient ascent

            # compute policy loss
            # we want to maximize the surrogate loss, but since Adam minimizes the loss, we take the negative
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss(vals, rewards_to_go) # - 0.01 * dist_entropy # Optional entropy term for exploration

            # calculate gradient and perform backward propagation
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Clear the buffer after updating
        self.buffer.clear()

        




            