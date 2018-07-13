import torch
import gym
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import util

from collections import deque

from models import Model
from policies import SoftmaxPolicy, Greedy,EpsDecay
from deep_agent import Agent, CompleteBuffer

from gym.envs.registration import register

register(
    id='MO-LunarLander-v2',
    entry_point='envs:LunarLander',
)

psis = []
def psi(x, ref, lambdaa, epsilon):
    temp = lambdaa * (ref - x).abs()
    res = temp.max(2)[0] + epsilon * temp.sum(2)
    psis.append(res.detach().numpy().mean())
    return res


class MORL(Model):
    """docstring for MORL."""
    def __init__(self, reward_weights, epsilon, net_structure, gamma, optim, loss_function, device, optim_param=[], tau=1):
        super(MORL, self).__init__(gamma, optim, loss_function, device, optim_param)

        self.nb_action = net_structure[-1]
        q_shape = list(net_structure)
        q_shape[-1] *= 2

        self.q = util.model_from_structure(q_shape).to(self.device)
        self.q_target = util.model_from_structure(q_shape).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.v = util.model_from_structure(net_structure).to(self.device)
        self.v_target = util.model_from_structure(net_structure).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.q_optimizer = self.optim(self.q.parameters(), *optim_param)
        self.v_optimizer = self.optim(self.v.parameters(), *optim_param)

        self.update_counter = 0

        self.epsilon = epsilon
        self.tau = 1

        self.ideal_value = -float("Inf") * torch.ones(len(reward_weights))
        self.min_value = +float("Inf") * torch.ones(len(reward_weights))
        self.weights = torch.tensor(reward_weights)

    def __call__(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.v.eval()
        with torch.no_grad():
            action_values = self.v(state)
        self.v.train()
        return action_values

    def learn(self, sample):
        states, actions, rewards, next_states, next_actions, dones = sample
        batch_size = len(states)
        reward_size = len(self.weights)

        stacked_actions = torch.stack([actions, actions], dim=2)
        q_out_view = (batch_size, self.nb_action, reward_size)
        # Learn Q
        q_target_values = self.q_target(next_states).view(*q_out_view)  # Dont gather we need later
        gathered_q_target = q_target_values.gather(1, stacked_actions).detach()
        expected_q_values = rewards.view(batch_size, 1, reward_size) + (self.gamma * gathered_q_target * (1-dones).view(batch_size, 1, 1))

        actual_q_values = self.q(states).view(*q_out_view).gather(1, stacked_actions)

        q_loss = self.loss_function(actual_q_values, expected_q_values)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Learn V
        # q_target_values.max(0)[0].size() = 4x2
        # q_target_values.max(0)[0].max(0)[0].size() = 2
        self.ideal_value = torch.stack([self.ideal_value, q_target_values.max(0)[0].max(0)[0].squeeze()]).max(0)[0]
        self.min_value = torch.stack([self.min_value, q_target_values.min(0)[0].min(0)[0].squeeze()]).min(0)[0]
        temp = (self.ideal_value - self.min_value).abs()
        lambdaa = self.weights / temp

        v_target_values = self.v_target(next_states)
        expected_v_values = psi(q_target_values, self.ideal_value, lambdaa, self.epsilon).gather(1, actions).detach()

        actual_v_values = self.v(states).gather(1, actions)

        v_loss = self.loss_function(actual_v_values, expected_v_values)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

    def update(self):
        if self.update_counter > 0:
            for target_param, predict_param in zip(self.q_target.parameters(), self.q.parameters()):
                target_param.data.copy_(self.tau*predict_param.data + (1.0-self.tau)*target_param.data)

            for target_param, predict_param in zip(self.v_target.parameters(), self.v.parameters()):
                target_param.data.copy_(self.tau*predict_param.data + (1.0-self.tau)*target_param.data)
        self.update_counter += 1

env = gym.make("MO-LunarLander-v2")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

nb_epi_max = 500
max_step = 1000
gamma = torch.tensor([.99, .99])

alpha = 1e-5
eps_start = 1
eps_decay = 0.9995
eps_min = 0.01

batch_size = 64
memory_size = int(1e5)

average_goal = [9999, 9999]
goal_size = 100

device = torch.device("cpu")

seed = 1651
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)

model = MORL(reward_weights=(5.0, 1.0), epsilon=1e-3, net_structure=(state_size, 256, 256, action_size), gamma=gamma, optim=optim.Adam, optim_param=[alpha],
            loss_function=nn.MSELoss(), tau=1, device=device)

buffer = CompleteBuffer(memory_size, batch_size, device)
learning_policy = SoftmaxPolicy() #EpsDecay(eps_start, eps_min, eps_decay, env.action_space.n)
playing_policy = Greedy()
agent = Agent(model=model, buffer=buffer, learn_every=4, update_every=4, policy_learning=learning_policy,
              policy_playing=playing_policy)


# model.ideal_value = torch.tensor([200.0, 0.0])
# model.min_value = torch.tensor([0.0, -100.0])

scores = [(0, 0)]
scores_window = [deque(maxlen=goal_size), deque(maxlen=goal_size)]
for i in range(nb_epi_max):
    state = env.reset()
    action = None
    score = np.zeros(2)
    for _ in range(max_step):
        next_action = agent.act(state)
        next_state, reward, done, info = env.step(next_action)
        if action is not None:
            agent.step((state, action, reward, next_state, next_action, done))
        score += np.array(reward)
        state = next_state
        action = next_action

        if done:
            break
    learning_policy.update()
    scores.append(score)
    scores_window[0].append(score[0])
    scores_window[1].append(score[1])

    print('\rEpisode {}\tAverage Score: {:.2f}, {:.2f}'.format(i, np.mean(scores_window[0]), np.mean(scores_window[1])), end="")
    if i % goal_size == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}, {:.2f}'.format(i, np.mean(scores_window[0]), np.mean(scores_window[1])))
    if np.mean(scores_window[0])>= average_goal[0] and np.mean(scores_window[1])>= average_goal[1]:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}, {:.2f}'.format(i, np.mean(scores_window[1]), np.mean(scores_window[0])))
        torch.save(agent.model.predict.state_dict(), 'checkpoint.pth')
        break

land, fuel = zip(*scores)
plt.plot(range(len(land)), land, label="land")
plt.plot(range(len(fuel)), fuel, label="fuel")
plt.legend()
plt.show()
plt.plot(range(len(psis)), psis)
plt.show()
