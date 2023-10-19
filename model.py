from PIL import Image, ImageGrab
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random

# The replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQNAgent:
    def __init__(self, model, buffer_capacity=10000, batch_size=32):
        self.model = model
        self.memory = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.last_action = set()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            actions = ['up', 'down', 'left', 'right']
            chosen_actions = set(np.random.choice(actions, size=np.random.randint(1, 5), replace=False))
            return chosen_actions
        
        state = np.expand_dims(state, axis=0)  # Need this for model prediction
        q_values = self.model.predict(state)
        action_index = np.argmax(q_values[0])
        return set([['up', 'down', 'left', 'right'][action_index]])

    def train(self, gamma=0.99):
        if len(self.memory) < self.batch_size:
            return
        
        mini_batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        
        states = np.array(states)
        next_states = np.array(next_states)
        targets = self.model.predict(states)
        next_q_values = self.model.predict(next_states)
        
        for i in range(self.batch_size):
            action_index = ['up', 'down', 'left', 'right'].index(list(actions[i])[0])  # Convert action to index
            if dones[i]:
                targets[i][action_index] = rewards[i]
            else:
                targets[i][action_index] = rewards[i] + gamma * np.max(next_q_values[i])
        
        self.model.train_on_batch(states, targets)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)



def create_dqn_model(input_shape, n_actions):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(layers.Flatten())

    # Dense layers
    model.add(layers.Dense(512, activation='relu'))
    
    # Output layer
    model.add(layers.Dense(n_actions, activation='linear'))

    # Compile with a suitable optimizer, like Adam
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model