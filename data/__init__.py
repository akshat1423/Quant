"""
Data Management Module

This module provides data loading and processing utilities for the quantitative finance project.
"""

from .data_loader import DataLoader
from .market_data import MarketDataProcessor

__all__ = [
    'DataLoader',
    'MarketDataProcessor'
]