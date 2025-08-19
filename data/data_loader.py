"""
Data Loader Module

Provides functionality to load market data from various sources including
APIs, CSV files, and database connections.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader for financial market data.
    
    Supports loading data from Yahoo Finance, CSV files, and other sources.
    """
    
    def __init__(self, cache_dir: str = "./data_cache"):
        """
        Initialize data loader.
        
        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def load_yahoo_data(self, symbols: Union[str, List[str]], 
                       start_date: str, end_date: str,
                       use_cache: bool = True) -> pd.DataFrame:
        """
        Load data from Yahoo Finance.
        
        Args:
            symbols: Stock symbol(s) to load
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        cache_file = f"{self.cache_dir}/yahoo_{'_'.join(symbols)}_{start_date}_{end_date}.csv"
        
        # Check cache first
        if use_cache and os.path.exists(cache_file):
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        logger.info(f"Downloading data for {symbols} from {start_date} to {end_date}")
        
        try:
            # Download data
            if len(symbols) == 1:
                data = yf.download(symbols[0], start=start_date, end=end_date)
                data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            else:
                data = yf.download(symbols, start=start_date, end=end_date)
            
            # Clean and standardize column names
            data = self._clean_yahoo_data(data)
            
            # Cache the data
            if use_cache:
                data.to_csv(cache_file)
                logger.info(f"Data cached to {cache_file}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
    
    def load_csv_data(self, filepath: str, date_column: str = 'Date',
                     required_columns: List[str] = None) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            date_column: Name of date column
            required_columns: List of required columns
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from CSV: {filepath}")
        
        try:
            data = pd.read_csv(filepath)
            
            # Set date index
            if date_column in data.columns:
                data[date_column] = pd.to_datetime(data[date_column])
                data.set_index(date_column, inplace=True)
            
            # Check required columns
            if required_columns:
                missing_cols = [col for col in required_columns if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Standardize column names
            data = self._standardize_column_names(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise
    
    def generate_sample_data(self, symbol: str = "AAPL", days: int = 1000,
                           start_price: float = 100.0, volatility: float = 0.2,
                           drift: float = 0.05) -> pd.DataFrame:
        """
        Generate sample market data using geometric Brownian motion.
        
        Args:
            symbol: Symbol name for the data
            days: Number of days to generate
            start_price: Starting price
            volatility: Annual volatility
            drift: Annual drift rate
            
        Returns:
            DataFrame with simulated OHLCV data
        """
        logger.info(f"Generating {days} days of sample data for {symbol}")
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate price series using GBM
        dt = 1/252  # Daily time step
        n_steps = len(dates)
        
        # Random shocks
        shocks = np.random.normal(0, 1, n_steps)
        
        # Generate price path
        prices = np.zeros(n_steps)
        prices[0] = start_price
        
        for i in range(1, n_steps):
            prices[i] = prices[i-1] * np.exp(
                (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks[i]
            )
        
        # Generate OHLC from close prices
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        
        # Generate intraday OHLC with realistic patterns
        daily_volatility = volatility / np.sqrt(252)
        
        # Open prices (gap from previous close)
        open_gaps = np.random.normal(0, daily_volatility * 0.5, n_steps)
        data['open'] = data['close'].shift(1) * (1 + open_gaps)
        data['open'].iloc[0] = start_price
        
        # High and low prices
        intraday_range = np.random.exponential(daily_volatility, n_steps)
        high_factor = 1 + intraday_range * np.random.uniform(0.3, 0.7, n_steps)
        low_factor = 1 - intraday_range * np.random.uniform(0.3, 0.7, n_steps)
        
        data['high'] = np.maximum(data['open'], data['close']) * high_factor
        data['low'] = np.minimum(data['open'], data['close']) * low_factor
        
        # Volume (correlated with price movements)
        returns = data['close'].pct_change().fillna(0)
        base_volume = 1000000
        volume_multiplier = 1 + 2 * np.abs(returns)  # Higher volume on big moves
        data['volume'] = base_volume * volume_multiplier * np.random.lognormal(0, 0.3, n_steps)
        data['volume'] = data['volume'].astype(int)
        
        # Adjusted close (same as close for simplicity)
        data['adj_close'] = data['close']
        
        # Reorder columns
        data = data[['open', 'high', 'low', 'close', 'adj_close', 'volume']]
        
        return data
    
    def _clean_yahoo_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize Yahoo Finance data."""
        # Standardize column names
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        
        data = data.rename(columns=column_mapping)
        
        # Remove any completely empty rows
        data = data.dropna(how='all')
        
        # Forward fill missing values
        data = data.fillna(method='ffill')
        
        # Remove any remaining NaN values
        data = data.dropna()
        
        return data
    
    def _standardize_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase with underscores."""
        # Common column name mappings
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low', 
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Adjusted Close': 'adj_close',
            'Volume': 'volume',
            'Date': 'date',
            'Timestamp': 'timestamp'
        }
        
        # Apply mappings
        data = data.rename(columns=column_mapping)
        
        # Convert all column names to lowercase and replace spaces with underscores
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        
        return data
    
    def load_multiple_assets(self, symbols: List[str], start_date: str, end_date: str,
                           use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple assets.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                data_dict[symbol] = self.load_yahoo_data(symbol, start_date, end_date, use_cache)
                logger.info(f"Loaded data for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
        
        return data_dict
    
    def combine_asset_data(self, data_dict: Dict[str, pd.DataFrame], 
                          price_column: str = 'close') -> pd.DataFrame:
        """
        Combine data from multiple assets into a single DataFrame.
        
        Args:
            data_dict: Dictionary mapping symbols to DataFrames
            price_column: Column to extract for combination
            
        Returns:
            DataFrame with combined price data
        """
        combined_data = pd.DataFrame()
        
        for symbol, data in data_dict.items():
            if price_column in data.columns:
                combined_data[symbol] = data[price_column]
        
        # Forward fill and backward fill to handle missing data
        combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
        
        return combined_data
    
    def get_data_info(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Get information about loaded data.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with data information
        """
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'start_date': data.index.min(),
            'end_date': data.index.max(),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict()
        }
        
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            info['price_stats'] = {
                'min_price': data['close'].min(),
                'max_price': data['close'].max(),
                'mean_price': data['close'].mean(),
                'std_price': data['close'].std(),
                'mean_return': returns.mean(),
                'std_return': returns.std(),
                'annualized_volatility': returns.std() * np.sqrt(252)
            }
        
        return info