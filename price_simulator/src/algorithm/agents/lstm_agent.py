import attr
import torch
import random
import numpy as np
from torch import nn, optim
from typing import List, Tuple

from price_simulator.src.algorithm.agents.buffer import SequentialReplayBuffer
from price_simulator.src.algorithm.agents.simple import AgentStrategy
from price_simulator.src.algorithm.policies import EpsilonGreedy, ExplorationStrategy

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out

@attr.s
class SimpleLSTMAgent(AgentStrategy):
    """Simplified LSTM Agent using sequences of past states"""

    # LSTM Network
    lstm: LSTMModel = attr.ib(default=None)
    hidden_nodes: int = attr.ib(default=32)
    sequence_length: int = attr.ib(default=5)  # Number of past states to use
    state_history: List[Tuple[float, ...]] = attr.ib(factory=list)

    # General
    decision: ExplorationStrategy = attr.ib(factory=EpsilonGreedy)
    discount: float = attr.ib(default=0.95)
    learning_rate: float = attr.ib(default=0.1)

    loss_history: List[float] = attr.ib(factory=list, init=False)

    @discount.validator
    def check_discount(self, attribute, value):
        if not 0 <= value <= 1:
            raise ValueError("Discount factor must lie in [0,1]")

    @learning_rate.validator
    def check_learning_rate(self, attribute, value):
        if not 0 <= value < 1:
            raise ValueError("Learning rate must lie in [0,1)")

    def who_am_i(self) -> str:
        return type(self).__name__ + " (gamma: {}, alpha: {}, policy: {}, quality: {}, mc: {})".format(
            self.discount, self.learning_rate, self.decision.who_am_i(), self.quality, self.marginal_cost
        )

    def update_state_history(self, state: Tuple[float]):
        #ToDo: make more versatile to use in other agents
        if not isinstance(self.state_history, list):
            self.state_history = []
        """Update the history of states with the new state."""
        self.state_history.append(state)
        if len(self.state_history) > self.sequence_length:
            self.state_history.pop(0)

    def play_price(self, state: Tuple[float], action_space: List[float], n_period: int, t: int) -> float:
        """Returns an action by either following greedy policy or experimentation."""

        # Update state history
        self.update_state_history(state)

        # Initialize LSTM network if necessary
        if not self.lstm:
            self.lstm = self.initialize_network(len(state), len(action_space))

        # Play action
        if self.decision.explore(n_period, t):
            return random.choice(action_space)
        else:
            # Use state history as input to the LSTM network
            states_input = torch.tensor(self.scale_sequence(self.state_history, action_space)).float().unsqueeze(0)
            action_values = self.lstm(states_input).detach().numpy()
            if sum(np.isclose(action_values[0], action_values[0].max())) > 1:
                optimal_action_index = np.random.choice(
                    np.flatnonzero(np.isclose(action_values[0], action_values[0].max()))
                )
            else:
                optimal_action_index = np.argmax(action_values[0])
            return action_space[optimal_action_index]
        
    def learn(
        self,
        previous_reward: float,
        reward: float,
        previous_action: float,
        action: float,
        action_space: List[float],
        previous_state: Tuple[float],
        state: Tuple[float],
        next_state: Tuple[float],
    ):
        """Update the LSTM network based on the observed rewards and actions."""
        # Update state history with the current state
        self.update_state_history(state)

        # Create a sequence of state history length for the next state
        next_state_history = self.state_history[-self.sequence_length:] + [next_state]
        # Scale the input sequences
        states_input = torch.tensor(self.scale_sequence(self.state_history, action_space)).float().unsqueeze(0)
        next_states_input = torch.tensor(self.scale_sequence(next_state_history, action_space)).float().unsqueeze(0)

        # Compute the target Q-values using the Bellman equation
        next_optimal_q = self.lstm(next_states_input).max().item()
        target = reward + self.discount * next_optimal_q

        # Get the local estimates from the LSTM network
        local_estimates = self.lstm(states_input)
        action_idx = np.atleast_1d(action_space == action).nonzero()[0]
        target_tensor = local_estimates.clone().detach()
        target_tensor[0, action_idx] = target

        # Update the LSTM network using backpropagation
        optimizer = optim.Adam(self.lstm.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss = nn.MSELoss()(local_estimates, target_tensor)
        loss.backward()
        optimizer.step()
        # Store the loss value
        self.loss_history.append(loss.item())
        # Debugging: Print loss and target values
        print(f"Loss: {loss.item()}, Target: {target}, Local Estimates: {local_estimates[0, action_idx].item()}")

    def initialize_network(self, n_agents: int, n_actions: int):
        """Create a neural network with one output node per possible action"""
        return LSTMModel(input_size=n_agents, hidden_size=self.hidden_nodes, output_size=n_actions)

    def scale_sequence(self, sequences: List[Tuple], action_space: List) -> np.array:
        """Scale float input sequences to range from 0 to 1."""
        max_action = max(action_space)
        min_action = min(action_space)
        return np.array([
            np.multiply(np.divide(np.array(seq) - min_action, max_action - min_action), 1) for seq in sequences
        ])
    

@attr.s
class LSTMReplayAgent(SimpleLSTMAgent):
    """LSTM Agent with Replay Buffer and Target Network"""

    # Replay Buffer
    replay_buffer: SequentialReplayBuffer = attr.ib(factory=lambda: SequentialReplayBuffer(buffer_size=10000))
    batch_size: int = attr.ib(default=32)

    # Target and Local Networks
    target_network: LSTMModel = attr.ib(default=None)
    update_target_after: int = attr.ib(default=100)
    update_counter: int = attr.ib(default=0)

    def initialize_network(self, n_agents: int, n_actions: int):
        """Initialize both local and target networks."""
        local_network = super().initialize_network(n_agents, n_actions)
        target_network = super().initialize_network(n_agents, n_actions)
        target_network.load_state_dict(local_network.state_dict())  # Synchronize weights
        self.target_network = target_network
        return local_network

    def add_experience(self, state_sequence, action, reward):
        """Add a new experience to the replay buffer."""
        self.replay_buffer.add(state_sequence, action, reward)

    def sample_experiences(self):
        """Sample a batch of experiences from the replay buffer."""
        return self.replay_buffer.sample(self.batch_size, self.sequence_length)

    def learn(
        self,
        previous_reward: float,
        reward: float,
        previous_action: float,
        action: float,
        action_space: List[float],
        previous_state: Tuple[float],
        state: Tuple[float],
        next_state: Tuple[float],
    ):
        """Train the LSTM network using replay buffer and target network."""
        # Add the current experience to the replay buffer
        self.replay_buffer.add(state, action, reward)
        
        # Ensure the buffer has enough data to sample sequences
        required_size = self.batch_size + self.sequence_length
        if len(self.replay_buffer) < required_size:
            print(f"Warm-up: Not enough experiences in the buffer to train. Current size: {len(self.replay_buffer)}")
            return

        # Sample a batch of experiences
        states, actions, rewards, next_states = self.sample_experiences()
        
        # Scale the sampled states and next states
        states = self.scale_sequence(states, self.action_space)
        next_states = self.scale_sequence(next_states, self.action_space)
        
        # Convert to tensors
        states_input = torch.tensor(states).float()
        next_states_input = torch.tensor(next_states).float()
        actions_tensor = torch.tensor([self.action_space.index(a) for a in actions])
        rewards_tensor = torch.tensor(rewards).float()

        # Compute target Q-values using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states_input).max(dim=1)[0]
            targets = rewards_tensor + self.discount * next_q_values

        # Get current Q-values from the local network
        local_q_values = self.lstm(states_input)
        predicted_q_values = local_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze()

        # Compute loss
        loss = nn.MSELoss()(predicted_q_values, targets)

        # Perform gradient descent step
        optimizer = optim.Adam(self.lstm.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the target network periodically
        self.update_counter += 1
        if self.update_counter >= self.update_target_after:
            self.target_network.load_state_dict(self.lstm.state_dict())
            self.update_counter = 0
            print("Target network updated.")

        # Store the loss value
        self.loss_history.append(loss.item())
        print(f"Loss: {loss.item()}")

    def play_price(self, state: Tuple[float], action_space: List[float], n_period: int, t: int) -> float:
        """Returns an action by either following greedy policy or experimentation."""
        self.update_state_history(state)

        if not self.lstm:
            self.lstm = self.initialize_network(len(state), len(action_space))
            self.action_space = action_space  # Save action space for indexing

        if self.decision.explore(n_period, t):
            return random.choice(action_space)
        else:
            states_input = torch.tensor(self.scale_sequence(self.state_history, action_space)).float().unsqueeze(0)
            action_values = self.lstm(states_input).detach().numpy()
            return action_space[np.argmax(action_values)]