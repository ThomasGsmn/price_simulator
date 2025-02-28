import attr
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from price_simulator.src.algorithm.agents.buffer import ReplayBuffer
from price_simulator.src.algorithm.agents.simple import AgentStrategy
from price_simulator.src.algorithm.policies import ExplorationStrategy, DecreasingEpsilonGreedy

@attr.s
class LSTM_Agent(AgentStrategy):
    qnetwork_local = attr.ib(default=None)
    qnetwork_target = attr.ib(default=None)
    replay_memory = attr.ib(factory=ReplayBuffer)
    update_target_after = attr.ib(default=100)
    batch_size = attr.ib(default=32)
    update_counter = attr.ib(default=0)
    hidden_nodes = attr.ib(default=32)
    decision = attr.ib(factory=DecreasingEpsilonGreedy)
    discount = attr.ib(default=0.95)
    learning_rate = attr.ib(default=0.001)
    
    def __attrs_post_init__(self):
        if not self.qnetwork_local or not self.qnetwork_target:
            self.qnetwork_local = self.initialize_network()
            self.qnetwork_target = self.initialize_network()
            self.qnetwork_target.set_weights(self.qnetwork_local.get_weights())

    def initialize_network(self):
        model = Sequential()
        model.add(LSTM(self.hidden_nodes, input_shape=(1, self.hidden_nodes), activation='relu'))  # Adjust input shape
        model.add(Dense(self.hidden_nodes, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def play_price(self, state, action_space, n_period, t):
        if self.decision.explore(n_period, t):
            action = random.choice(action_space)
        else:
            state = np.reshape(state, [1, 1, len(state)])  # Reshape for LSTM input
            action_values = self.qnetwork_local.predict(state)
            action = action_space[np.argmax(action_values[0])]
        return action

    def learn(
        self,
        previous_reward,
        reward,
        previous_action,
        action,
        action_space,
        previous_state,
        state,
        next_state,
    ):
        try:
            action = np.where(np.array(action_space) == action)[0][0]
            previous_state = np.reshape(previous_state, [1, 1, len(previous_state)])  # Reshape for LSTM input
            state = np.reshape(state, [1, 1, len(state)])  # Reshape for LSTM input
            next_state = np.reshape(next_state, [1, 1, len(next_state)])  # Reshape for LSTM input
            self.replay_memory.add(previous_state, action, reward, next_state)
            
            if len(self.replay_memory) > self.batch_size:
                states, actions, rewards, next_states = self.replay_memory.sample(self.batch_size)
                targets = rewards + self.discount * np.amax(self.qnetwork_target.predict(next_states), axis=1)
                targets_full = self.qnetwork_local.predict(states)
                ind = np.array([i for i in range(self.batch_size)])
                targets_full[[ind], [actions]] = targets
                self.qnetwork_local.fit(states, targets_full, epochs=1, verbose=0)
                self.update_counter += 1
                if self.update_counter == self.update_target_after:
                    self.qnetwork_target.set_weights(self.qnetwork_local.get_weights())
                    self.update_counter = 0
        except Exception as e:
            print(f"An error occurred during learning: {e}")
