import gym
import random
import numpy as np

gamma = .99
nb_epi = 10000
eps = .05
alpha = 0.1

env = gym.make("FrozenLake-v0")
env.reset()

# TD-learning
Q = np.zeros((env.observation_space.n, env.action_space.n))

def get_policy(state):
    return np.argmax(Q[state])


for i in range(nb_epi):
    prev_state = env.reset()
    prev_action = random.randint(0, env.action_space.n-1)
    done = False
    while not done:
        action = random.randint(0, env.action_space.n-1) if random.random() < eps else get_policy(prev_state)
        state, reward, done, _ = env.step(action)
        Q[prev_state, prev_action] = Q[prev_state, prev_action] + alpha * (reward + gamma * Q[state, action] - Q[prev_state, prev_action])
        prev_state = state
        prev_action = action
    for action in range(env.action_space.n):
        Q[state, action] = Q[state, action] + alpha * (reward - Q[state, action])
