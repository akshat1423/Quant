"""
Deep Option Network (DON) Strategy

Implementation of Deep Option Network for algorithmic trading.
Combines deep learning with option pricing theory for enhanced trading decisions.
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


class DONStrategy(BaseStrategy):
    """
    Deep Option Network (DON) strategy for trading.
    
    Integrates option pricing signals with deep learning to make trading decisions.
    Uses implied volatility, Greeks, and market microstructure features.
    """
    
    def __init__(self, state_dim: int = 20, option_features_dim: int = 7,
                 learning_rate: float = 0.001, hidden_units: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.3, **kwargs):
        """
        Initialize DON strategy.
        
        Args:
            state_dim: Dimension of market state features
            option_features_dim: Dimension of option-related features
            learning_rate: Learning rate for neural network
            hidden_units: Hidden layer sizes
            dropout_rate: Dropout rate for regularization
            **kwargs: Additional parameters
        """
        super().__init__("DeepOptionNetwork", **kwargs)
        
        self.state_dim = state_dim
        self.option_features_dim = option_features_dim
        self.total_input_dim = state_dim + option_features_dim
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        # Build networks
        self.option_encoder = self._build_option_encoder()
        self.market_encoder = self._build_market_encoder()
        self.decision_network = self._build_decision_network()
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate)
        
        # Training data storage
        self.training_states = []
        self.training_option_features = []
        self.training_targets = []
        
        logger.info("Deep Option Network strategy initialized")
    
    def _build_option_encoder(self) -> keras.Model:
        """
        Build option feature encoder network.
        
        Processes option-related features like Greeks, implied volatility, etc.
        
        Returns:
            Option encoder neural network
        """
        inputs = layers.Input(shape=(self.option_features_dim,))
        
        # Option-specific feature processing
        x = layers.Dense(64, activation='relu', name='option_dense_1')(inputs)
        x = layers.BatchNormalization(name='option_bn_1')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(32, activation='relu', name='option_dense_2')(x)
        x = layers.BatchNormalization(name='option_bn_2')(x)
        
        outputs = layers.Dense(16, activation='tanh', name='option_encoded')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='option_encoder')
        return model
    
    def _build_market_encoder(self) -> keras.Model:
        """
        Build market feature encoder network.
        
        Processes market state features like technical indicators, price movements, etc.
        
        Returns:
            Market encoder neural network
        """
        inputs = layers.Input(shape=(self.state_dim,))
        
        # Market feature processing with attention mechanism
        x = layers.Dense(128, activation='relu', name='market_dense_1')(inputs)
        x = layers.BatchNormalization(name='market_bn_1')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(64, activation='relu', name='market_dense_2')(x)
        x = layers.BatchNormalization(name='market_bn_2')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Self-attention layer for feature importance
        attention = layers.Dense(64, activation='softmax', name='market_attention')(x)
        x = layers.Multiply(name='market_attention_applied')([x, attention])
        
        outputs = layers.Dense(32, activation='tanh', name='market_encoded')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='market_encoder')
        return model
    
    def _build_decision_network(self) -> keras.Model:
        """
        Build decision network that combines encoded features.
        
        Returns:
            Decision neural network
        """
        # Inputs for encoded features
        option_encoded = layers.Input(shape=(16,), name='option_encoded_input')
        market_encoded = layers.Input(shape=(32,), name='market_encoded_input')
        
        # Combine encoded features
        combined = layers.Concatenate(name='feature_combination')([option_encoded, market_encoded])
        
        # Decision layers
        x = layers.Dense(128, activation='relu', name='decision_dense_1')(combined)
        x = layers.BatchNormalization(name='decision_bn_1')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(64, activation='relu', name='decision_dense_2')(x)
        x = layers.BatchNormalization(name='decision_bn_2')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(32, activation='relu', name='decision_dense_3')(x)
        
        # Output layers for different trading decisions
        position_output = layers.Dense(3, activation='softmax', name='position_decision')(x)  # Buy/Hold/Sell
        confidence_output = layers.Dense(1, activation='sigmoid', name='confidence_score')(x)  # Confidence
        expected_return = layers.Dense(1, activation='linear', name='expected_return')(x)  # Expected return
        
        model = keras.Model(
            inputs=[option_encoded, market_encoded],
            outputs=[position_output, confidence_output, expected_return],
            name='decision_network'
        )
        return model
    
    def _extract_option_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract option-related features from market data.
        
        Args:
            data: Market data
            
        Returns:
            Option features array
        """
        option_features = []
        
        # Calculate implied volatility proxy (20-day rolling volatility)
        returns = data['close'].pct_change()
        implied_vol = returns.rolling(window=20).std() * np.sqrt(252)
        option_features.append(implied_vol.fillna(method='bfill').values)
        
        # Calculate option Greeks proxies
        prices = data['close'].values
        
        # Delta proxy (price momentum)
        delta_proxy = pd.Series(prices).pct_change(5).fillna(0).values
        option_features.append(delta_proxy)
        
        # Gamma proxy (acceleration of price changes)
        gamma_proxy = pd.Series(delta_proxy).diff().fillna(0).values
        option_features.append(gamma_proxy)
        
        # Theta proxy (time decay - negative of days to expiration effect)
        theta_proxy = -np.ones_like(prices) * 0.01  # Constant time decay
        option_features.append(theta_proxy)
        
        # Vega proxy (volatility sensitivity)
        vega_proxy = implied_vol * prices / 100  # Simplified vega calculation
        option_features.append(vega_proxy)
        
        # Rho proxy (interest rate sensitivity)
        rho_proxy = prices * 0.05 / 100  # Simplified rho with 5% interest rate
        option_features.append(rho_proxy)
        
        # Option moneyness proxy (assuming at-the-money options)
        moneyness = np.ones_like(prices)  # ATM options
        option_features.append(moneyness)
        
        # Convert to array and transpose
        option_matrix = np.array(option_features).T
        
        # Normalize features
        option_matrix = self._normalize_features(option_matrix)
        
        return option_matrix
    
    def train(self, training_data: pd.DataFrame, validation_split: float = 0.2,
              epochs: int = 100, batch_size: int = 64, **kwargs) -> Dict[str, List[float]]:
        """
        Train the DON strategy.
        
        Args:
            training_data: Historical market data
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            **kwargs: Additional training parameters
            
        Returns:
            Training history
        """
        logger.info(f"Training DON strategy for {epochs} epochs")
        
        # Preprocess data
        market_features, returns = self.preprocess_data(training_data)
        option_features = self._extract_option_features(training_data)
        
        # Create targets (future returns classification)
        targets = self._create_targets(returns)
        
        # Split data
        split_idx = int(len(market_features) * (1 - validation_split))
        
        train_market = market_features[:split_idx]
        train_option = option_features[:split_idx]
        train_targets = targets[:split_idx]
        
        val_market = market_features[split_idx:]
        val_option = option_features[split_idx:]
        val_targets = targets[split_idx:]
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(train_market, train_option, train_targets, batch_size)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self._validate_epoch(val_market, val_option, val_targets)
            val_losses.append(val_loss)
            
            # Log progress
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        self.is_trained = True
        logger.info("DON training completed")
        
        return {'train_loss': train_losses, 'val_loss': val_losses}
    
    def _create_targets(self, returns: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create training targets from returns.
        
        Args:
            returns: Historical returns
            
        Returns:
            Dictionary with different target types
        """
        # Future returns (shifted by 1)
        future_returns = np.roll(returns, -1)
        future_returns[-1] = 0  # Last value
        
        # Position targets (0: sell, 1: hold, 2: buy)
        position_targets = np.zeros(len(future_returns))
        position_targets[future_returns > 0.001] = 2  # Buy if return > 0.1%
        position_targets[future_returns < -0.001] = 0  # Sell if return < -0.1%
        position_targets[(future_returns >= -0.001) & (future_returns <= 0.001)] = 1  # Hold otherwise
        
        # Convert to one-hot
        position_one_hot = tf.keras.utils.to_categorical(position_targets, num_classes=3)
        
        # Confidence targets (absolute return magnitude)
        confidence_targets = np.abs(future_returns)
        confidence_targets = np.clip(confidence_targets * 10, 0, 1)  # Scale and clip
        
        # Expected return targets
        expected_return_targets = future_returns
        
        return {
            'position': position_one_hot,
            'confidence': confidence_targets,
            'expected_return': expected_return_targets
        }
    
    def _train_epoch(self, market_features: np.ndarray, option_features: np.ndarray,
                    targets: Dict[str, np.ndarray], batch_size: int) -> float:
        """Train for one epoch."""
        n_samples = len(market_features)
        n_batches = n_samples // batch_size
        epoch_loss = 0
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_market = market_features[start_idx:end_idx]
            batch_option = option_features[start_idx:end_idx]
            batch_targets = {
                'position': targets['position'][start_idx:end_idx],
                'confidence': targets['confidence'][start_idx:end_idx],
                'expected_return': targets['expected_return'][start_idx:end_idx]
            }
            
            loss = self._train_step(batch_market, batch_option, batch_targets)
            epoch_loss += loss
        
        return epoch_loss / n_batches
    
    def _train_step(self, market_features: np.ndarray, option_features: np.ndarray,
                   targets: Dict[str, np.ndarray]) -> float:
        """Single training step."""
        with tf.GradientTape() as tape:
            # Encode features
            market_encoded = self.market_encoder(market_features)
            option_encoded = self.option_encoder(option_features)
            
            # Make predictions
            position_pred, confidence_pred, return_pred = self.decision_network(
                [option_encoded, market_encoded]
            )
            
            # Calculate losses
            position_loss = tf.keras.losses.categorical_crossentropy(
                targets['position'], position_pred
            )
            confidence_loss = tf.keras.losses.binary_crossentropy(
                targets['confidence'], confidence_pred
            )
            return_loss = tf.keras.losses.mse(
                targets['expected_return'], tf.squeeze(return_pred)
            )
            
            # Combined loss
            total_loss = tf.reduce_mean(position_loss + confidence_loss + return_loss)
        
        # Apply gradients
        all_variables = (self.option_encoder.trainable_variables +
                        self.market_encoder.trainable_variables +
                        self.decision_network.trainable_variables)
        
        gradients = tape.gradient(total_loss, all_variables)
        self.optimizer.apply_gradients(zip(gradients, all_variables))
        
        return float(total_loss.numpy())
    
    def _validate_epoch(self, market_features: np.ndarray, option_features: np.ndarray,
                       targets: Dict[str, np.ndarray]) -> float:
        """Validate for one epoch."""
        # Encode features
        market_encoded = self.market_encoder(market_features)
        option_encoded = self.option_encoder(option_features)
        
        # Make predictions
        position_pred, confidence_pred, return_pred = self.decision_network(
            [option_encoded, market_encoded]
        )
        
        # Calculate losses
        position_loss = tf.keras.losses.categorical_crossentropy(
            targets['position'], position_pred
        )
        confidence_loss = tf.keras.losses.binary_crossentropy(
            targets['confidence'], confidence_pred
        )
        return_loss = tf.keras.losses.mse(
            targets['expected_return'], tf.squeeze(return_pred)
        )
        
        # Combined loss
        total_loss = tf.reduce_mean(position_loss + confidence_loss + return_loss)
        
        return float(total_loss.numpy())
    
    def predict(self, market_state: np.ndarray, option_state: Optional[np.ndarray] = None) -> int:
        """
        Make trading decision based on current market state.
        
        Args:
            market_state: Current market state features
            option_state: Option-related features (if None, will be estimated)
            
        Returns:
            Trading action (-1: sell, 0: hold, 1: buy)
        """
        if not self.is_trained:
            logger.warning("Strategy not trained, using random action")
            return np.random.choice([-1, 0, 1])
        
        # Handle single state prediction
        if len(market_state.shape) == 1:
            market_state = market_state.reshape(1, -1)
        
        if option_state is None:
            # Create dummy option features if not provided
            option_state = np.zeros((market_state.shape[0], self.option_features_dim))
        elif len(option_state.shape) == 1:
            option_state = option_state.reshape(1, -1)
        
        # Encode features
        market_encoded = self.market_encoder(market_state)
        option_encoded = self.option_encoder(option_state)
        
        # Make prediction
        position_pred, confidence_pred, return_pred = self.decision_network(
            [option_encoded, market_encoded]
        )
        
        # Get action with highest probability
        action = int(tf.argmax(position_pred[0]).numpy())
        return action - 1  # Convert from 0,1,2 to -1,0,1
    
    def update(self, market_state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray) -> None:
        """
        Update strategy based on observed outcome.
        
        For DON, this could implement online learning, but for now it's a placeholder.
        """
        # DON typically uses batch training, so online updates are minimal
        pass
    
    def get_prediction_details(self, market_state: np.ndarray, 
                              option_state: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get detailed prediction information.
        
        Returns:
            Dictionary with prediction details including probabilities and confidence
        """
        if not self.is_trained:
            return {'error': 'Strategy not trained'}
        
        # Handle single state prediction
        if len(market_state.shape) == 1:
            market_state = market_state.reshape(1, -1)
        
        if option_state is None:
            option_state = np.zeros((market_state.shape[0], self.option_features_dim))
        elif len(option_state.shape) == 1:
            option_state = option_state.reshape(1, -1)
        
        # Encode features
        market_encoded = self.market_encoder(market_state)
        option_encoded = self.option_encoder(option_state)
        
        # Make prediction
        position_pred, confidence_pred, return_pred = self.decision_network(
            [option_encoded, market_encoded]
        )
        
        return {
            'sell_probability': float(position_pred[0][0].numpy()),
            'hold_probability': float(position_pred[0][1].numpy()),
            'buy_probability': float(position_pred[0][2].numpy()),
            'confidence_score': float(confidence_pred[0][0].numpy()),
            'expected_return': float(return_pred[0][0].numpy())
        }
    
    def save_model(self, filepath: str) -> None:
        """Save all DON models."""
        self.option_encoder.save(f"{filepath}_option_encoder.h5")
        self.market_encoder.save(f"{filepath}_market_encoder.h5")
        self.decision_network.save(f"{filepath}_decision_network.h5")
        logger.info(f"DON models saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load all DON models."""
        self.option_encoder = keras.models.load_model(f"{filepath}_option_encoder.h5")
        self.market_encoder = keras.models.load_model(f"{filepath}_market_encoder.h5")
        self.decision_network = keras.models.load_model(f"{filepath}_decision_network.h5")
        self.is_trained = True
        logger.info(f"DON models loaded from {filepath}")