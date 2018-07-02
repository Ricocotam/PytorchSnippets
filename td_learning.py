import gym
import random
import numpy as np

gamma = .99
nb_epi = 100
eps = .05
alpha = 1

env = gym.make("FrozenLake-v0")
env.reset()

# TD-learning
V = np.zeros(env.observation_space.n)

def get_policy(state):
    arr = []
    indices = []
    if state % 4:
        arr.append(V[state-1])
        indices.append(0)
    if state % 4 < 3:
        arr.append(V[state + 1])
        indices.append(2)
    if state // 4 != 0:
        arr.append(V[state - 4])
        indices.append(3)
    if state // 4 != 3:
        arr.append(V[state + 4])
        indices.append(1)

    return indices[np.argmax(arr)]


for i in range(nb_epi):
    prev_state = env.reset()
    done = False
    while not done:
        action = random.randint(0, env.action_space.n-1) if random.random() < eps else get_policy(prev_state)
        state, reward, done, _ = env.step(action)
        V[prev_state] = V[prev_state] + alpha * (reward + gamma * V[state] - V[prev_state])
        prev_state = state
