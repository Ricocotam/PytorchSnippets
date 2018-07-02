import torch
import gym

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm


env = gym.make("CartPole-v0")
device = torch.device("cpu:0")
nb_epi = 1000


print("Building network")
model = nn.Sequential(
                nn.Linear(4, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, 2)).double()



optimizer = optim.Adam(model.parameters())
loss_function = nn.MSELoss()

print("Training")
cum_rewards = []
y = torch.rand((2,), requires_grad=False).double()
for i in range(nb_epi):
    prev_state = env.reset()
    env.render()
    done = False
    prev_action_values = model(torch.from_numpy(prev_state))
    cum_r = 0
    while not done:
        # Play
        action = random.randint(0, env.action_space.n-1) if random.random() < eps else prev_action_values.argmax().tolist()  # This is scalar
        state, reward, done, _ = env.step(action)
        env.render()

        # Learn
        action_values = model(torch.from_numpy(state))

        prev_action_values = prev_action_values.double()
        y.copy_(prev_action_values.detach())  # Set older action values as targets
        y[action] = reward + (0 if done else gamma * action_values.max().tolist())

        loss = loss_function(prev_action_values, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prev_action_values = action_values
        cum_r += reward
    cum_rewards.append(cum_r)

plt.plot(len(cum_reward), cum_rewards)
