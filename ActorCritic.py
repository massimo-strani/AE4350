import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, std: float, layers: list[int], device: torch.device = torch.device('cpu')):
        super(ActorCritic, self).__init__()

        # Standard deviation for the action distribution is fixed for simplicity
        # This can be made learnable or adaptive if needed
        self.act_var = torch.full(size=(act_dim,), fill_value=std).to(device) # Variance for the action distribution
        self.act_mat = torch.diag(self.act_var).to(device)                    # Covariance matrix for the action distribution

        self.act_dim = act_dim
        
        self.actor = self._build_mlp(obs_dim, act_dim, layers, output_activation=nn.Tanh())
        self.critic = self._build_mlp(obs_dim, 1, layers, output_activation=None)


    def forward(self):
        raise NotImplementedError
    
    def _build_mlp(self, input_dim: int, output_dim: int, layer_sizes: list[int], output_activation: nn.Module = None):
        layers = []
        prev_dim = input_dim
        
        for size in layer_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.Tanh())
            prev_dim = size
        
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation:
            layers.append(output_activation)

        return nn.Sequential(*layers)

    def act(self, obs: torch.Tensor):
        ''' Returns an action and its log probability based on the current observation. '''
        act_mean = self.actor(obs) 
        dist = MultivariateNormal(act_mean, self.act_mat)

        # sample an action from the distribution and get its log probability
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Evaluate the value function for the current observation
        value = self.critic(obs)

        return action.detach(), log_prob.detach(), value.detach()

    def evaluate(self, batch_obs: torch.Tensor, batch_acts: torch.Tensor):
        ''' Returns the value of the current observation. '''
        act_mean = self.actor(batch_obs)
        dist = MultivariateNormal(act_mean, self.act_mat)

        # query critic network for a value for each observation in the batch
        values = self.critic(batch_obs)
        logprobs = dist.log_prob(batch_acts)
        dist_entropy = dist.entropy()  # Optional: can be used for exploration bonus

        return values, logprobs, dist_entropy
