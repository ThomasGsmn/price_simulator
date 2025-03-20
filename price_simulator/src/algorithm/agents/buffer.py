import random
from collections import deque, namedtuple

import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size=None):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            seed (int): random seed
        """
        if buffer_size is None:
            self.buffer_size = 10000
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])

    def add(self, state, action, reward, next_state):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state)
        self.memory.append(e)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        return states, actions, rewards, next_states

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



class SequentialReplayBuffer:
    """Replay buffer that stores sequences of states, actions, and rewards.
    Basic idea here: store the states, actions, and rewards as an array.
    When needed, sample based on a random index"""

    def __init__(self, buffer_size=None):
        """Initialize a SequentialReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
        """
        if buffer_size is None:
            buffer_size = 10000
        self.buffer_size = buffer_size
        self.states = deque(maxlen=buffer_size)  # Stores individual states
        self.actions = deque(maxlen=buffer_size)  # Stores actions
        self.rewards = deque(maxlen=buffer_size)  # Stores rewards

    def add(self, state, action, reward):
        """Add a new experience to memory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def sample(self, batch_size, sequence_length):
        """Sample a batch of sequential experiences from memory.
        Params
        ======
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence of states
        Returns
        =======
            sampled_states: Batch of sequences of states
            sampled_actions: Actions corresponding to the end of sampled_states sequence
            sampled_rewards: Rewards corresponding to the end of sampled_states sequence
            next_states: Batch of sequences of next states, shifted by one step
        """
        if len(self.states) < sequence_length + 1:
            raise ValueError("Not enough data in the buffer to sample sequences.")

        max_start_index = len(self.states) - sequence_length - 1
        indices = random.sample(range(max_start_index), batch_size)

        sampled_states = []
        sampled_actions = []
        sampled_rewards = []
        sampled_next_states = []

        for idx in indices:
            # Sequence of states
            sampled_states.append(list(self.states)[idx:idx + sequence_length])
            # Action and reward at the end of the sequence
            sampled_actions.append(self.actions[idx + sequence_length - 1])
            sampled_rewards.append(self.rewards[idx + sequence_length - 1])
            # Next sequence of states
            sampled_next_states.append(list(self.states)[idx + 1:idx + sequence_length + 1])

        return (
            np.array(sampled_states),
            np.array(sampled_actions),
            np.array(sampled_rewards),
            np.array(sampled_next_states),
        )

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.states)