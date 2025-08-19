"""
Actor-Critic Reinforcement Learning Strategy

Implementation of Actor-Critic algorithm for algorithmic trading.
The actor learns the policy while the critic evaluates the value function.
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


class ActorCriticStrategy(BaseStrategy):
    """
    Actor-Critic reinforcement learning strategy for trading.
    
    Uses deep neural networks for both actor (policy) and critic (value function).
    Suitable for continuous learning in dynamic market environments.
    """
    
    def __init__(self, state_dim: int = 13, action_dim: int = 3, 
                 learning_rate: float = 0.001, gamma: float = 0.95,
                 hidden_units: List[int] = [128, 64], **kwargs):
        """
        Initialize Actor-Critic strategy.
        
        Args:
            state_dim: Dimension of state space (number of features)
            action_dim: Dimension of action space (3: buy, hold, sell)
            learning_rate: Learning rate for neural networks
            gamma: Discount factor for future rewards
            hidden_units: Hidden layer sizes for networks
            **kwargs: Additional parameters
        """
        super().__init__("ActorCritic", **kwargs)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.hidden_units = hidden_units
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        # Build networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate)
        
        logger.info("Actor-Critic strategy initialized")
    
    def _build_actor(self) -> keras.Model:
        """
        Build actor network (policy network).
        
        Returns:
            Actor neural network model
        """
        inputs = layers.Input(shape=(self.state_dim,))
        x = inputs
        
        # Hidden layers
        for units in self.hidden_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer with softmax for action probabilities
        outputs = layers.Dense(self.action_dim, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def _build_critic(self) -> keras.Model:
        """
        Build critic network (value function network).
        
        Returns:
            Critic neural network model
        """
        inputs = layers.Input(shape=(self.state_dim,))
        x = inputs
        
        # Hidden layers
        for units in self.hidden_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer for state value
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def train(self, training_data: pd.DataFrame, episodes: int = 1000,
              batch_size: int = 32, **kwargs) -> None:
        """
        Train the Actor-Critic strategy.
        
        Args:
            training_data: Historical market data
            episodes: Number of training episodes
            batch_size: Batch size for training
            **kwargs: Additional training parameters
        """
        logger.info(f"Training Actor-Critic strategy for {episodes} episodes")
        
        # Preprocess data
        features, returns = self.preprocess_data(training_data)
        
        episode_rewards = []
        
        for episode in range(episodes):
            episode_reward = 0
            self._clear_experience_buffer()
            
            # Run episode
            for t in range(1, len(features) - 1):
                state = features[t]
                next_state = features[t + 1]
                market_return = returns[t + 1]
                
                # Get action
                action = self._get_action(state)
                
                # Calculate reward
                reward = self.calculate_reward(action - 1, market_return)  # Convert to -1, 0, 1
                
                # Store experience
                self._store_experience(state, action, reward, next_state, False)
                
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            
            # Train networks
            if len(self.states) >= batch_size:
                self._train_networks()
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"Episode {episode}, Average Reward: {avg_reward:.4f}")
        
        self.is_trained = True
        logger.info("Actor-Critic training completed")
    
    def _get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Get action from actor network.
        
        Args:
            state: Current market state
            training: Whether in training mode (for exploration)
            
        Returns:
            Action index (0: sell, 1: hold, 2: buy)
        """
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        action_probs = self.actor(state_tensor)
        
        if training:
            # Sample from probability distribution
            action = tf.random.categorical(tf.math.log(action_probs), 1)[0, 0]
        else:
            # Take most probable action
            action = tf.argmax(action_probs[0])
        
        return int(action.numpy())
    
    def _store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        """Store experience in buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def _clear_experience_buffer(self) -> None:
        """Clear experience buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def _train_networks(self) -> None:
        """Train actor and critic networks."""
        # Convert experience to tensors
        states = tf.convert_to_tensor(self.states, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(self.rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(self.next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(self.dones, dtype=tf.float32)
        
        # Calculate advantages
        with tf.GradientTape() as tape:
            # Get current state values
            values = tf.squeeze(self.critic(states))
            next_values = tf.squeeze(self.critic(next_states))
            
            # Calculate target values (TD target)
            targets = rewards + self.gamma * next_values * (1 - dones)
            
            # Calculate advantages
            advantages = targets - values
            
            # Critic loss (MSE)
            critic_loss = tf.reduce_mean(tf.square(advantages))
        
        # Update critic
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        # Train actor
        with tf.GradientTape() as tape:
            # Get action probabilities
            action_probs = self.actor(states)
            
            # Convert actions to one-hot
            actions_one_hot = tf.one_hot(actions, self.action_dim)
            
            # Calculate log probabilities
            selected_action_probs = tf.reduce_sum(action_probs * actions_one_hot, axis=1)
            log_probs = tf.math.log(selected_action_probs + 1e-8)
            
            # Actor loss (policy gradient)
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))
            
            # Add entropy bonus for exploration
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)
            actor_loss -= 0.01 * tf.reduce_mean(entropy)
        
        # Update actor
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
    
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
        if len(self.states) >= 32:
            self._train_networks()
            self._clear_experience_buffer()
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained models.
        
        Args:
            filepath: Base filepath for saving models
        """
        self.actor.save(f"{filepath}_actor.h5")
        self.critic.save(f"{filepath}_critic.h5")
        logger.info(f"Models saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained models.
        
        Args:
            filepath: Base filepath for loading models
        """
        self.actor = keras.models.load_model(f"{filepath}_actor.h5")
        self.critic = keras.models.load_model(f"{filepath}_critic.h5")
        self.is_trained = True
        logger.info(f"Models loaded from {filepath}")
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for given state.
        
        Args:
            state: Market state
            
        Returns:
            Action probabilities [sell, hold, buy]
        """
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        action_probs = self.actor(state_tensor)
        return action_probs[0].numpy()
    
    def get_state_value(self, state: np.ndarray) -> float:
        """
        Get state value from critic network.
        
        Args:
            state: Market state
            
        Returns:
            State value estimate
        """
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        value = self.critic(state_tensor)
        return float(value[0, 0].numpy())