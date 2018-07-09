import random
import torch
import numpy as np


def greedy_policy(state, model):
    return model(state).argmax().tolist()  # It returns a scalar

def eps_greedy_policy(eps, action_size):
    return lambda state, model: random.randint(0, action_size-1) if random.random() < eps else greedy_policy(state, model)


class Agent(object):
    """A base class for deep RL agents."""
    def __init__(self, model, buffer, update_every, policy_learning, policy_playing=greedy_policy):
        """Initialize an agent.

        Parameters
        -------------
            model : Model
                The model you want to use

            buffer : ReplayBuffer
                The buffer you want to use for your episodes

            update_every : int
                The step gap between two learning phase

            policy_learning : function(state, model)
                The policy you use during learning

            policy_playing : function(state, model)
                The policy used during playing phase
        """
        self.model = model
        self.buffer = buffer
        self.learning_strategy = policy_learning
        self.playing_strategy = policy_playing
        self.update_every = update_every
        self.update_counter = 0
        self.learning = True

    def act(self, state):
        """Get the action to play."""
        if self.learning:
            self.learning_strategy(state, model)
        else:
            self.playing_strategy(state, model)

    def step(self, state, action, reward, nest_state, done):
        """Do a step for the agent. Memorize and learn if needed."""
        self.buffer.add(state, action, reward, nex_state, done)
        self.update_counter = (self.update_counter + 1) % self.update_every

        if self.update_counter == 0:
            if self.buffer.can_sample():
                sample = self.buffer.sample()
                model.learn(*sample)

    def learning(self):
        """Set learning policy."""
        self.learning = True

    def playing(self):
        """Set playing policy."""
        self.learning = False


class Model(object):
    """A base class for deep RL models."""
    def __call__(self, state):
        """Return action values."""
        pass

    def learn(self, states, actions, rewards, next_states, dones):
        """Learn base on a mini-batch."""
        pass



class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        --------
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def can_sample(self):
        return len(self.memory) > self.batch_size

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
