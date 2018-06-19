import torch
import os
import gym

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

from tqdm import trange
from itertools import count


class PolicyNetwork(nn.Module):
    """docstring for PolicyNetwork."""
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.log_prob_reward = []
        self.net = nn.Sequential(nn.Linear(4, 128),
                               nn.ReLU(),
                               nn.Linear(128, 2),
                               nn.Softmax(1))

    def forward(self, x):
       return self.net(x)




policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters())

env = gym.make("CartPole-v0")


for i in trange(500):
    state = env.reset()
    for t in count():
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = policy(state)
        distrib = torch.distributions.Categorical(probs)
        action = distrib.sample()
        state, reward, done, _ = env.step(action.item())

        policy.log_prob_reward.append(-distrib.log_prob(action) * reward)

        env.render()
        if done:
            break

    optimizer.zero_grad()
    loss = torch.cat(policy.log_prob_reward).sum()
    loss.backward()
    optimizer.step()
    policy.log_prob_reward = []
env.close()
