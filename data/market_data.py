"""
Market Data Processor

Provides advanced market data processing utilities including technical indicators,
feature engineering, and data preparation for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MarketDataProcessor:
    """
    Advanced market data processor for quantitative finance applications.
    
    Provides comprehensive data processing capabilities including technical indicators,
    statistical features, and ML-ready data preparation.
    """
    
    def __init__(self):
        """Initialize market data processor."""
        self.processed_data = None
        self.feature_names = []
        
    def process_ohlcv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process OHLCV data and add comprehensive technical indicators.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added technical indicators
        """
        logger.info("Processing OHLCV data with technical indicators")
        
        processed = data.copy()
        
        # Basic price features
        processed = self._add_price_features(processed)
        
        # Moving averages
        processed = self._add_moving_averages(processed)
        
        # Momentum indicators
        processed = self._add_momentum_indicators(processed)
        
        # Volatility indicators
        processed = self._add_volatility_indicators(processed)
        
        # Volume indicators
        processed = self._add_volume_indicators(processed)
        
        # Pattern recognition
        processed = self._add_pattern_features(processed)
        
        # Statistical features
        processed = self._add_statistical_features(processed)
        
        # Clean data
        processed = self._clean_processed_data(processed)
        
        self.processed_data = processed
        self.feature_names = [col for col in processed.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'adj_close']]
        
        logger.info(f"Added {len(self.feature_names)} technical features")
        
        return processed
    
    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-derived features."""
        # Returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Price ratios
        data['hl_ratio'] = data['high'] / data['low']
        data['oc_ratio'] = data['open'] / data['close']
        
        # Price ranges
        data['daily_range'] = (data['high'] - data['low']) / data['close']
        data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        
        # Typical price
        data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
        
        return data
    
    def _add_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add moving average indicators."""
        periods = [5, 10, 20, 50, 100]
        
        for period in periods:
            # Simple moving average
            data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            
            # Exponential moving average
            data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            
            # Price relative to moving average
            data[f'price_sma_{period}_ratio'] = data['close'] / data[f'sma_{period}']
            data[f'price_ema_{period}_ratio'] = data['close'] / data[f'ema_{period}']
        
        # Moving average convergence divergence (MACD)
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        return data
    
    def _add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators."""
        # RSI (Relative Strength Index)
        data['rsi'] = self._calculate_rsi(data['close'])
        
        # Stochastic Oscillator
        data['stoch_k'], data['stoch_d'] = self._calculate_stochastic(data)
        
        # Williams %R
        data['williams_r'] = self._calculate_williams_r(data)
        
        # Rate of Change
        for period in [5, 10, 20]:
            data[f'roc_{period}'] = ((data['close'] - data['close'].shift(period)) / data['close'].shift(period)) * 100
        
        # Momentum
        for period in [5, 10, 20]:
            data[f'momentum_{period}'] = data['close'] / data['close'].shift(period)
        
        return data
    
    def _add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators."""
        # Historical volatility
        for window in [10, 20, 30]:
            data[f'volatility_{window}'] = data['returns'].rolling(window=window).std() * np.sqrt(252)
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = data['close'].rolling(window=period).mean()
            std = data['close'].rolling(window=period).std()
            
            data[f'bb_upper_{period}'] = sma + (2 * std)
            data[f'bb_lower_{period}'] = sma - (2 * std)
            data[f'bb_width_{period}'] = (data[f'bb_upper_{period}'] - data[f'bb_lower_{period}']) / sma
            data[f'bb_position_{period}'] = (data['close'] - data[f'bb_lower_{period}']) / (data[f'bb_upper_{period}'] - data[f'bb_lower_{period}'])
        
        # Average True Range (ATR)
        data['atr'] = self._calculate_atr(data)
        
        # Volatility ratio
        data['vol_ratio'] = data['volatility_10'] / data['volatility_30']
        
        return data
    
    def _add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume moving averages
        for period in [10, 20, 50]:
            data[f'volume_sma_{period}'] = data['volume'].rolling(window=period).mean()
            data[f'volume_ratio_{period}'] = data['volume'] / data[f'volume_sma_{period}']
        
        # On Balance Volume (OBV)
        data['obv'] = self._calculate_obv(data)
        
        # Volume Price Trend (VPT)
        data['vpt'] = self._calculate_vpt(data)
        
        # Accumulation/Distribution Line
        data['ad_line'] = self._calculate_ad_line(data)
        
        # Volume-weighted average price (VWAP)
        data['vwap'] = self._calculate_vwap(data)
        
        return data
    
    def _add_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features."""
        # Candlestick patterns (simplified)
        data['doji'] = self._identify_doji(data)
        data['hammer'] = self._identify_hammer(data)
        data['shooting_star'] = self._identify_shooting_star(data)
        
        # Price patterns
        data['higher_high'] = (data['high'] > data['high'].shift(1)).astype(int)
        data['lower_low'] = (data['low'] < data['low'].shift(1)).astype(int)
        
        # Gap analysis
        data['gap_up'] = ((data['open'] > data['close'].shift(1)) & (data['low'] > data['high'].shift(1))).astype(int)
        data['gap_down'] = ((data['open'] < data['close'].shift(1)) & (data['high'] < data['low'].shift(1))).astype(int)
        
        return data
    
    def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        # Rolling statistics
        for window in [10, 20, 30]:
            data[f'skewness_{window}'] = data['returns'].rolling(window=window).skew()
            data[f'kurtosis_{window}'] = data['returns'].rolling(window=window).kurt()
            data[f'autocorr_{window}'] = data['returns'].rolling(window=window).apply(lambda x: x.autocorr(lag=1))
        
        # Percentile ranks
        for window in [20, 50]:
            data[f'price_percentile_{window}'] = data['close'].rolling(window=window).rank(pct=True)
            data[f'volume_percentile_{window}'] = data['volume'].rolling(window=window).rank(pct=True)
        
        # Z-scores
        for window in [20, 50]:
            rolling_mean = data['close'].rolling(window=window).mean()
            rolling_std = data['close'].rolling(window=window).std()
            data[f'price_zscore_{window}'] = (data['close'] - rolling_mean) / rolling_std
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = data['low'].rolling(window=k_window).min()
        highest_high = data['high'].rolling(window=k_window).max()
        
        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return k_percent, d_percent
    
    def _calculate_williams_r(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = data['high'].rolling(window=window).max()
        lowest_low = data['low'].rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - data['close']) / (highest_high - lowest_low))
        return williams_r
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close_prev = np.abs(data['high'] - data['close'].shift(1))
        low_close_prev = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume."""
        obv = np.where(data['close'] > data['close'].shift(1), data['volume'],
               np.where(data['close'] < data['close'].shift(1), -data['volume'], 0))
        return pd.Series(obv, index=data.index).cumsum()
    
    def _calculate_vpt(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend."""
        vpt = (data['volume'] * data['returns']).cumsum()
        return vpt
    
    def _calculate_ad_line(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        clv = clv.fillna(0)
        ad_line = (clv * data['volume']).cumsum()
        return ad_line
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return vwap
    
    def _identify_doji(self, data: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """Identify Doji candlestick pattern."""
        body = np.abs(data['close'] - data['open'])
        range_val = data['high'] - data['low']
        doji = (body / range_val) < threshold
        return doji.astype(int)
    
    def _identify_hammer(self, data: pd.DataFrame) -> pd.Series:
        """Identify Hammer candlestick pattern."""
        body = np.abs(data['close'] - data['open'])
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        
        hammer = (lower_shadow > 2 * body) & (upper_shadow < body)
        return hammer.astype(int)
    
    def _identify_shooting_star(self, data: pd.DataFrame) -> pd.Series:
        """Identify Shooting Star candlestick pattern."""
        body = np.abs(data['close'] - data['open'])
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        
        shooting_star = (upper_shadow > 2 * body) & (lower_shadow < body)
        return shooting_star.astype(int)
    
    def _clean_processed_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean processed data by handling NaN values and infinite values."""
        # Replace infinite values with NaN
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Forward fill NaN values
        data.fillna(method='ffill', inplace=True)
        
        # Backward fill remaining NaN values
        data.fillna(method='bfill', inplace=True)
        
        # Drop any remaining NaN values
        data.dropna(inplace=True)
        
        return data
    
    def prepare_ml_features(self, data: pd.DataFrame, target_column: str = 'returns',
                           feature_columns: Optional[List[str]] = None,
                           lookback_window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for machine learning models.
        
        Args:
            data: Processed data with technical indicators
            target_column: Column to use as target variable
            feature_columns: Specific columns to use as features
            lookback_window: Number of periods to look back for features
            
        Returns:
            Tuple of (features, targets)
        """
        if feature_columns is None:
            feature_columns = self.feature_names
        
        # Create feature matrix with lookback
        features = []
        targets = []
        
        for i in range(lookback_window, len(data)):
            # Features: lookback window of feature columns
            feature_window = data[feature_columns].iloc[i-lookback_window:i].values.flatten()
            features.append(feature_window)
            
            # Target: future return
            if target_column in data.columns:
                targets.append(data[target_column].iloc[i])
            else:
                # Calculate future return
                future_return = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
                targets.append(future_return)
        
        features = np.array(features)
        targets = np.array(targets)
        
        # Normalize features
        features = self._normalize_features(features)
        
        logger.info(f"Prepared ML features: {features.shape}, targets: {targets.shape}")
        
        return features, targets
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using z-score normalization."""
        mean = np.nanmean(features, axis=0)
        std = np.nanstd(features, axis=0)
        
        # Avoid division by zero
        std[std == 0] = 1
        
        normalized = (features - mean) / std
        
        # Replace NaN with 0
        normalized = np.nan_to_num(normalized)
        
        return normalized
    
    def calculate_feature_importance(self, data: pd.DataFrame, target_column: str = 'returns') -> pd.DataFrame:
        """
        Calculate feature importance using correlation analysis.
        
        Args:
            data: Processed data
            target_column: Target column for importance calculation
            
        Returns:
            DataFrame with feature importance scores
        """
        if target_column not in data.columns:
            # Calculate returns if not present
            data[target_column] = data['close'].pct_change()
        
        # Calculate correlations
        correlations = data[self.feature_names].corrwith(data[target_column]).abs()
        
        # Sort by importance
        importance_df = pd.DataFrame({
            'feature': correlations.index,
            'importance': correlations.values
        }).sort_values('importance', ascending=False)
        
        # Remove NaN values
        importance_df = importance_df.dropna()
        
        return importance_df
    
    def generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate basic trading signals based on technical indicators.
        
        Args:
            data: Processed data with technical indicators
            
        Returns:
            DataFrame with trading signals
        """
        signals = data.copy()
        
        # Moving average crossover signals
        signals['ma_signal'] = np.where(signals['ema_10'] > signals['ema_20'], 1, -1)
        
        # RSI signals
        signals['rsi_signal'] = np.where(signals['rsi'] < 30, 1, 
                                np.where(signals['rsi'] > 70, -1, 0))
        
        # Bollinger Band signals
        signals['bb_signal'] = np.where(signals['close'] < signals['bb_lower_20'], 1,
                              np.where(signals['close'] > signals['bb_upper_20'], -1, 0))
        
        # MACD signals
        signals['macd_signal'] = np.where(signals['macd'] > signals['macd_signal'], 1, -1)
        
        # Combined signal
        signals['combined_signal'] = (signals['ma_signal'] + 
                                    signals['rsi_signal'] + 
                                    signals['bb_signal'] + 
                                    signals['macd_signal']) / 4
        
        # Discretize combined signal
        signals['final_signal'] = np.where(signals['combined_signal'] > 0.25, 1,
                                 np.where(signals['combined_signal'] < -0.25, -1, 0))
        
        return signals
    
    def get_feature_summary(self) -> Dict[str, any]:
        """
        Get summary of generated features.
        
        Returns:
            Dictionary with feature summary information
        """
        if self.processed_data is None:
            return {"error": "No processed data available"}
        
        summary = {
            "total_features": len(self.feature_names),
            "feature_categories": {
                "price_features": len([f for f in self.feature_names if any(x in f for x in ['returns', 'ratio', 'range', 'gap'])]),
                "moving_averages": len([f for f in self.feature_names if any(x in f for x in ['sma', 'ema', 'macd'])]),
                "momentum": len([f for f in self.feature_names if any(x in f for x in ['rsi', 'stoch', 'williams', 'roc', 'momentum'])]),
                "volatility": len([f for f in self.feature_names if any(x in f for x in ['volatility', 'bb_', 'atr'])]),
                "volume": len([f for f in self.feature_names if any(x in f for x in ['volume', 'obv', 'vpt', 'ad_line', 'vwap'])]),
                "patterns": len([f for f in self.feature_names if any(x in f for x in ['doji', 'hammer', 'shooting', 'higher', 'lower', 'gap'])]),
                "statistical": len([f for f in self.feature_names if any(x in f for x in ['skewness', 'kurtosis', 'autocorr', 'percentile', 'zscore'])])
            },
            "data_shape": self.processed_data.shape,
            "date_range": {
                "start": str(self.processed_data.index.min()),
                "end": str(self.processed_data.index.max())
            }
        }
        
        return summary