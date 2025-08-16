import torch
import torch.nn as nn
import torch.optim as optim

import random

import numpy as np

class DQN(nn.Module):
  def __init__(self, state_dim, action_dim):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(state_dim, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, action_dim)
    )

  def forward(self, x):
    return self.net(x)
  

class ReplayBuffer:
  def __init__(self, capacity):
    self.capacity = capacity
    self.buffer = []
    
  def add(self, transition):
    if len(self.buffer) >= self.capacity:
      self.buffer.pop(0)
    self.buffer.append(transition)

  def sample(self, batch_size):
    return random.sample(self.buffer, batch_size)