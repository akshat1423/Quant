"""
Monte Carlo Engine for Option Pricing

Provides the core Monte Carlo simulation infrastructure for option pricing models.
"""

import numpy as np
from typing import Tuple, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for financial derivatives pricing.
    
    This class provides the fundamental simulation infrastructure used by
    various option pricing models.
    """
    
    def __init__(self, n_simulations: int = 100000, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo engine.
        
        Args:
            n_simulations: Number of Monte Carlo paths to simulate
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_stock_paths(self, S0: float, r: float, sigma: float, T: float, 
                           n_steps: int) -> np.ndarray:
        """
        Generate stock price paths using geometric Brownian motion.
        
        Args:
            S0: Initial stock price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            n_steps: Number of time steps
            
        Returns:
            Array of shape (n_simulations, n_steps+1) containing stock paths
        """
        dt = T / n_steps
        
        # Generate random increments
        dW = np.random.normal(0, np.sqrt(dt), (self.n_simulations, n_steps))
        
        # Initialize paths array
        paths = np.zeros((self.n_simulations, n_steps + 1))
        paths[:, 0] = S0
        
        # Generate paths using Euler scheme
        for i in range(n_steps):
            paths[:, i + 1] = paths[:, i] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * dW[:, i]
            )
        
        return paths
    
    def generate_correlated_paths(self, S0: float, r: float, sigma: float, T: float,
                                n_steps: int, correlation_matrix: np.ndarray) -> np.ndarray:
        """
        Generate correlated stock price paths for multi-asset options.
        
        Args:
            S0: Initial stock prices (array)
            r: Risk-free rate
            sigma: Volatilities (array)  
            T: Time to maturity
            n_steps: Number of time steps
            correlation_matrix: Correlation matrix between assets
            
        Returns:
            Array of correlated stock paths
        """
        n_assets = len(S0)
        dt = T / n_steps
        
        # Cholesky decomposition for correlation
        L = np.linalg.cholesky(correlation_matrix)
        
        # Generate independent random variables
        Z = np.random.normal(0, 1, (self.n_simulations, n_steps, n_assets))
        
        # Apply correlation
        dW = np.zeros_like(Z)
        for i in range(self.n_simulations):
            for j in range(n_steps):
                dW[i, j, :] = L @ Z[i, j, :] * np.sqrt(dt)
        
        # Initialize paths
        paths = np.zeros((self.n_simulations, n_steps + 1, n_assets))
        paths[:, 0, :] = S0
        
        # Generate correlated paths
        for i in range(n_steps):
            for k in range(n_assets):
                paths[:, i + 1, k] = paths[:, i, k] * np.exp(
                    (r - 0.5 * sigma[k]**2) * dt + sigma[k] * dW[:, i, k]
                )
        
        return paths
    
    def discount_factor(self, r: float, T: float) -> float:
        """Calculate discount factor."""
        return np.exp(-r * T)
    
    def european_payoff(self, S: np.ndarray, K: float, option_type: str = 'put') -> np.ndarray:
        """
        Calculate European option payoff.
        
        Args:
            S: Stock prices at maturity
            K: Strike price
            option_type: 'put' or 'call'
            
        Returns:
            Option payoffs
        """
        if option_type.lower() == 'put':
            return np.maximum(K - S, 0)
        elif option_type.lower() == 'call':
            return np.maximum(S - K, 0)
        else:
            raise ValueError("option_type must be 'put' or 'call'")
    
    def estimate_greeks(self, pricing_function: Callable, S0: float, *args, 
                       bump_size: float = 0.01) -> dict:
        """
        Estimate option Greeks using finite differences.
        
        Args:
            pricing_function: Function that returns option price
            S0: Current stock price
            *args: Additional arguments for pricing function
            bump_size: Size of finite difference bump
            
        Returns:
            Dictionary containing estimated Greeks
        """
        # Base price
        price_base = pricing_function(S0, *args)
        
        # Delta (sensitivity to stock price)
        price_up = pricing_function(S0 + bump_size, *args)
        price_down = pricing_function(S0 - bump_size, *args)
        delta = (price_up - price_down) / (2 * bump_size)
        
        # Gamma (second derivative w.r.t. stock price)
        gamma = (price_up - 2 * price_base + price_down) / (bump_size**2)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'price': price_base
        }