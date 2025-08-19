"""
Epsilon-Greedy Reinforcement Learning Strategy

Implementation of epsilon-greedy Q-learning algorithm for algorithmic trading.
Uses exploration-exploitation tradeoff with Q-table or neural network approximation.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Tuple, Optional
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class EpsilonGreedyStrategy(BaseStrategy):
    """
    Epsilon-Greedy Q-learning strategy for trading.
    
    Implements Deep Q-Network (DQN) with epsilon-greedy exploration
    for learning optimal trading policies.
    """
    
    def __init__(self, state_dim: int = 13, action_dim: int = 3,
                 learning_rate: float = 0.001, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 gamma: float = 0.95, memory_size: int = 10000,
                 hidden_units: List[int] = [128, 64], **kwargs):
        """
        Initialize Epsilon-Greedy strategy.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (3: sell, hold, buy)
            learning_rate: Learning rate for neural network
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            gamma: Discount factor for future rewards
            memory_size: Size of experience replay buffer
            hidden_units: Hidden layer sizes for Q-network
            **kwargs: Additional parameters
        """
        super().__init__("EpsilonGreedy", **kwargs)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.memory_size = memory_size
        self.hidden_units = hidden_units
        
        # Experience replay buffer
        self.memory = []
        self.memory_index = 0
        
        # Build Q-networks
        self.q_network = self._build_q_network()
        self.target_network = self._build_q_network()
        self.optimizer = keras.optimizers.Adam(learning_rate)
        
        # Update target network
        self._update_target_network()
        
        logger.info("Epsilon-Greedy strategy initialized")
    
    def _build_q_network(self) -> keras.Model:
        """
        Build Q-network for value function approximation.
        
        Returns:
            Q-network neural network model
        """
        inputs = layers.Input(shape=(self.state_dim,))
        x = inputs
        
        # Hidden layers
        for units in self.hidden_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer for Q-values
        outputs = layers.Dense(self.action_dim, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def _update_target_network(self) -> None:
        """Update target network weights with main network weights."""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def train(self, training_data: pd.DataFrame, episodes: int = 1000,
              batch_size: int = 32, target_update_freq: int = 100,
              **kwargs) -> None:
        """
        Train the Epsilon-Greedy strategy.
        
        Args:
            training_data: Historical market data
            episodes: Number of training episodes
            batch_size: Batch size for training
            target_update_freq: Frequency to update target network
            **kwargs: Additional training parameters
        """
        logger.info(f"Training Epsilon-Greedy strategy for {episodes} episodes")
        
        # Preprocess data
        features, returns = self.preprocess_data(training_data)
        
        episode_rewards = []
        losses = []
        
        for episode in range(episodes):
            episode_reward = 0
            
            # Run episode
            for t in range(1, len(features) - 1):
                state = features[t]
                next_state = features[t + 1]
                market_return = returns[t + 1]
                
                # Get action using epsilon-greedy policy
                action = self._get_action(state)
                
                # Calculate reward
                reward = self.calculate_reward(action - 1, market_return)  # Convert to -1, 0, 1
                
                # Store experience in replay buffer
                self._store_experience(state, action, reward, next_state, False)
                
                episode_reward += reward
                
                # Train network if enough experience
                if len(self.memory) >= batch_size:
                    loss = self._replay(batch_size)
                    losses.append(loss)
            
            episode_rewards.append(episode_reward)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Update target network
            if episode % target_update_freq == 0:
                self._update_target_network()
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_loss = np.mean(losses[-100:]) if losses else 0
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, "
                          f"Avg Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.3f}")
        
        self.is_trained = True
        logger.info("Epsilon-Greedy training completed")
    
    def _get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Current market state
            training: Whether in training mode (for exploration)
            
        Returns:
            Action index (0: sell, 1: hold, 2: buy)
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(0, self.action_dim)
        else:
            # Exploitation: best action according to Q-network
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            q_values = self.q_network(state_tensor)
            return int(tf.argmax(q_values[0]).numpy())
    
    def _store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.memory) < self.memory_size:
            self.memory.append(experience)
        else:
            # Circular buffer
            self.memory[self.memory_index] = experience
            self.memory_index = (self.memory_index + 1) % self.memory_size
    
    def _replay(self, batch_size: int) -> float:
        """
        Train Q-network using experience replay.
        
        Args:
            batch_size: Size of training batch
            
        Returns:
            Training loss
        """
        # Sample random batch from memory
        batch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        
        # Unpack batch
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        # Convert to tensors
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Calculate target Q-values
        next_q_values = self.target_network(next_states_tensor)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        target_q_values = rewards_tensor + self.gamma * max_next_q_values * (1 - dones_tensor)
        
        # Train Q-network
        with tf.GradientTape() as tape:
            # Get current Q-values
            current_q_values = self.q_network(states_tensor)
            
            # Select Q-values for taken actions
            actions_one_hot = tf.one_hot(actions, self.action_dim)
            selected_q_values = tf.reduce_sum(current_q_values * actions_one_hot, axis=1)
            
            # Calculate loss (Huber loss for stability)
            loss = tf.reduce_mean(tf.keras.losses.huber(target_q_values, selected_q_values))
        
        # Apply gradients
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        return float(loss.numpy())
    
    def predict(self, market_state: np.ndarray) -> int:
        """
        Make trading decision based on current market state.
        
        Args:
            market_state: Current market state features
            
        Returns:
            Trading action (-1: sell, 0: hold, 1: buy)
        """
        if not self.is_trained:
            logger.warning("Strategy not trained, using random action")
            return np.random.choice([-1, 0, 1])
        
        action = self._get_action(market_state, training=False)
        return action - 1  # Convert from 0,1,2 to -1,0,1
    
    def update(self, market_state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray) -> None:
        """
        Update strategy based on observed outcome (online learning).
        
        Args:
            market_state: Previous market state
            action: Action taken
            reward: Reward received
            next_state: New market state
        """
        # Convert action from -1,0,1 to 0,1,2
        action_idx = action + 1
        
        # Store experience
        self._store_experience(market_state, action_idx, reward, next_state, False)
        
        # Train if enough experience
        if len(self.memory) >= 32:
            self._replay(32)
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions in given state.
        
        Args:
            state: Market state
            
        Returns:
            Q-values for [sell, hold, buy] actions
        """
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        q_values = self.q_network(state_tensor)
        return q_values[0].numpy()
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained Q-network.
        
        Args:
            filepath: Filepath for saving model
        """
        self.q_network.save(f"{filepath}_q_network.h5")
        
        # Save additional parameters
        params = {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'is_trained': self.is_trained
        }
        np.save(f"{filepath}_params.npy", params)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained Q-network.
        
        Args:
            filepath: Filepath for loading model
        """
        self.q_network = keras.models.load_model(f"{filepath}_q_network.h5")
        self.target_network = keras.models.load_model(f"{filepath}_q_network.h5")
        
        # Load additional parameters
        try:
            params = np.load(f"{filepath}_params.npy", allow_pickle=True).item()
            self.epsilon = params.get('epsilon', self.epsilon_min)
            self.is_trained = params.get('is_trained', True)
        except FileNotFoundError:
            logger.warning("Parameters file not found, using default values")
            self.epsilon = self.epsilon_min
            self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def analyze_exploration_exploitation(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze exploration vs exploitation behavior.
        
        Args:
            test_data: Test data for analysis
            
        Returns:
            Dictionary with exploration/exploitation metrics
        """
        features, _ = self.preprocess_data(test_data)
        
        exploration_actions = 0
        exploitation_actions = 0
        
        # Temporarily store original epsilon
        original_epsilon = self.epsilon
        
        for t in range(len(features)):
            state = features[t]
            
            # Check what action would be taken with epsilon=0 (pure exploitation)
            self.epsilon = 0
            exploit_action = self._get_action(state, training=True)
            
            # Check what action would be taken with current epsilon
            self.epsilon = original_epsilon
            actual_action = self._get_action(state, training=True)
            
            if actual_action == exploit_action:
                exploitation_actions += 1
            else:
                exploration_actions += 1
        
        # Restore original epsilon
        self.epsilon = original_epsilon
        
        total_actions = exploration_actions + exploitation_actions
        exploration_rate = exploration_actions / total_actions if total_actions > 0 else 0
        exploitation_rate = exploitation_actions / total_actions if total_actions > 0 else 0
        
        return {
            'exploration_rate': exploration_rate,
            'exploitation_rate': exploitation_rate,
            'current_epsilon': self.epsilon,
            'total_actions_analyzed': total_actions
        }