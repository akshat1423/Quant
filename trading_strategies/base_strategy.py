"""
Base Strategy Framework

Provides the abstract base class and common functionality for all trading strategies.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Defines the interface and common functionality that all trading strategies
    must implement.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize base strategy.
        
        Args:
            name: Name of the strategy
            **kwargs: Additional strategy-specific parameters
        """
        self.name = name
        self.parameters = kwargs
        self.is_trained = False
        self.performance_metrics = {}
        self.trades = []
        
    @abstractmethod
    def train(self, training_data: pd.DataFrame, **kwargs) -> None:
        """
        Train the strategy on historical data.
        
        Args:
            training_data: Historical market data for training
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, market_state: np.ndarray) -> int:
        """
        Make trading decision based on current market state.
        
        Args:
            market_state: Current market state features
            
        Returns:
            Trading action (0: hold, 1: buy, -1: sell)
        """
        pass
    
    @abstractmethod
    def update(self, market_state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray) -> None:
        """
        Update strategy based on observed outcome.
        
        Args:
            market_state: Previous market state
            action: Action taken
            reward: Reward received
            next_state: New market state
        """
        pass
    
    def calculate_reward(self, action: int, price_change: float, 
                        transaction_cost: float = 0.001) -> float:
        """
        Calculate reward based on action and market movement.
        
        Args:
            action: Trading action taken
            price_change: Relative price change
            transaction_cost: Transaction cost rate
            
        Returns:
            Reward value
        """
        # Basic reward calculation
        if action == 1:  # Buy
            reward = price_change - transaction_cost
        elif action == -1:  # Sell
            reward = -price_change - transaction_cost
        else:  # Hold
            reward = 0
        
        return reward
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess market data for training/prediction.
        
        Args:
            data: Raw market data
            
        Returns:
            Tuple of (features, returns)
        """
        # Calculate technical indicators
        features = self._calculate_technical_indicators(data)
        
        # Calculate returns
        returns = data['close'].pct_change().fillna(0).values
        
        return features, returns
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate technical indicators from price data.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            Feature matrix
        """
        features = []
        
        # Price-based features
        features.append(data['close'].values)
        features.append(data['high'].values)
        features.append(data['low'].values)
        features.append(data['volume'].values)
        
        # Moving averages
        features.append(data['close'].rolling(window=5).mean().fillna(method='bfill').values)
        features.append(data['close'].rolling(window=10).mean().fillna(method='bfill').values)
        features.append(data['close'].rolling(window=20).mean().fillna(method='bfill').values)
        
        # RSI
        rsi = self._calculate_rsi(data['close'], window=14)
        features.append(rsi)
        
        # MACD
        macd, signal = self._calculate_macd(data['close'])
        features.append(macd)
        features.append(signal)
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(data['close'])
        features.append(bb_upper)
        features.append(bb_lower)
        
        # Volatility
        volatility = data['close'].rolling(window=20).std().fillna(method='bfill').values
        features.append(volatility)
        
        # Convert to array and transpose
        feature_matrix = np.array(features).T
        
        # Normalize features
        feature_matrix = self._normalize_features(feature_matrix)
        
        return feature_matrix
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, 
                       signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD and signal line."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd.fillna(0).values, macd_signal.fillna(0).values
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, 
                                  num_std: float = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band.fillna(method='bfill').values, lower_band.fillna(method='bfill').values
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using z-score normalization."""
        means = np.nanmean(features, axis=0)
        stds = np.nanstd(features, axis=0)
        stds[stds == 0] = 1  # Avoid division by zero
        
        normalized = (features - means) / stds
        # Replace NaN values with 0
        normalized = np.nan_to_num(normalized)
        
        return normalized
    
    def backtest(self, test_data: pd.DataFrame, initial_capital: float = 10000,
                transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Backtest the strategy on historical data.
        
        Args:
            test_data: Historical data for backtesting
            initial_capital: Initial capital amount
            transaction_cost: Transaction cost rate
            
        Returns:
            Dictionary with performance metrics
        """
        if not self.is_trained:
            raise ValueError("Strategy must be trained before backtesting")
        
        # Preprocess data
        features, returns = self.preprocess_data(test_data)
        
        # Initialize tracking variables
        capital = initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        portfolio_values = [capital]
        trades = []
        
        # Run backtest
        for i in range(1, len(features)):
            current_state = features[i-1]
            next_return = returns[i]
            
            # Get trading signal
            action = self.predict(current_state)
            
            # Execute trade
            if action != position:
                # Close existing position
                if position != 0:
                    capital *= (1 + position * next_return - transaction_cost)
                    trades.append({
                        'timestamp': test_data.index[i],
                        'action': 'close',
                        'position': position,
                        'return': position * next_return,
                        'capital': capital
                    })
                
                # Open new position
                if action != 0:
                    position = action
                    trades.append({
                        'timestamp': test_data.index[i],
                        'action': 'open',
                        'position': position,
                        'capital': capital
                    })
                else:
                    position = 0
            else:
                # Hold position
                if position != 0:
                    capital *= (1 + position * next_return)
            
            portfolio_values.append(capital)
        
        # Calculate performance metrics
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        metrics = self._calculate_performance_metrics(
            portfolio_returns, portfolio_values, initial_capital
        )
        
        self.trades = trades
        self.performance_metrics = metrics
        
        return metrics
    
    def _calculate_performance_metrics(self, returns: np.ndarray, 
                                     portfolio_values: List[float],
                                     initial_capital: float) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Annualized metrics (assuming daily data)
        trading_days = 252
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (trading_days / n_periods) - 1
        
        volatility = np.std(returns) * np.sqrt(trading_days)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        positive_returns = returns[returns > 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_capital': final_value,
            'number_of_trades': len(self.trades)
        }