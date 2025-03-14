import attr
import torch
import random
import numpy as np
from torch import nn, optim
from typing import List, Tuple

# from price_simulator.src.algorithm.agents.buffer import ReplayBuffer
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