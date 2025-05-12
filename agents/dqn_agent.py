# agents/dqn_agent.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import os
import numpy as np

class DQNAgent:
    """Old Deep Q-Learning agent for traffic signal control."""
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=2000,
                 batch_size=32, model_path=None):
        """Initialize the DQN agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate at which exploration rate decays
            memory_size: Size of replay memory
            batch_size: Batch size for training
            model_path: Path to pre-trained model
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            self.model = self._build_model()
    
    def _build_model(self):
        """Build a neural network model for DQN."""
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, explore=True):
        """Select action using epsilon-greedy policy."""
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(np.reshape(state, [1, self.state_size]), verbose=0) # 1 is for the batch size
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=None):
        """Train on batch of experiences."""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        def get_vectorized_minibatch():
            state_list, action_list, reward_list, next_state_list, done_list = zip(*minibatch)
            state_batch      = tf.stack(state_list)
            action_batch     = tf.stack(action_list).numpy().astype(int)
            reward_batch     = tf.stack(reward_list).numpy()
            next_state_batch = tf.stack(next_state_list)
            done_batch       = tf.stack(done_list).numpy().astype(bool)
            
            return state_batch, action_batch, reward_batch, next_state_batch, done_batch

        vectorized_minibatch = get_vectorized_minibatch()

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = vectorized_minibatch

        target_q    = self.model.predict(state_batch, verbose=0)
        next_q = self.model.predict(next_state_batch, verbose=0)

        batch_idx = np.arange(batch_size)
        max_next_q = next_q.max(axis=1) 
        target_q[batch_idx, action_batch] = reward_batch + (1.0 - done_batch.astype(float)) * self.gamma * max_next_q
        
        self.model.fit(state_batch, target_q, epochs=1, verbose=0)
        
        #decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save the model to disk."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")