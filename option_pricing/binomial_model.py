"""
Binomial Model for Option Pricing

Implementation of the binomial tree model for American and European option pricing.
Used as a benchmark for comparison with Monte Carlo methods.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BinomialPricer:
    """
    Binomial tree model for option pricing.
    
    Provides accurate pricing for American and European options
    using the Cox-Ross-Rubinstein binomial tree method.
    """
    
    def __init__(self, n_steps: int = 1000):
        """
        Initialize binomial pricer.
        
        Args:
            n_steps: Number of time steps in the binomial tree
        """
        self.n_steps = n_steps
        
    def price_american_put(self, S0: float, K: float, T: float, r: float, 
                          sigma: float) -> Tuple[float, np.ndarray]:
        """
        Price American put option using binomial tree.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Tuple of (option_price, exercise_boundary)
        """
        logger.info(f"Pricing American put (Binomial): S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
        
        # Calculate binomial parameters
        dt = T / self.n_steps
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
        discount = np.exp(-r * dt)  # Discount factor
        
        # Initialize stock price tree
        stock_prices = np.zeros((self.n_steps + 1, self.n_steps + 1))
        
        # Fill stock price tree
        for i in range(self.n_steps + 1):
            for j in range(i + 1):
                stock_prices[j, i] = S0 * (u ** (i - j)) * (d ** j)
        
        # Initialize option value tree
        option_values = np.zeros((self.n_steps + 1, self.n_steps + 1))
        
        # Set option values at maturity (final column)
        for j in range(self.n_steps + 1):
            option_values[j, self.n_steps] = max(K - stock_prices[j, self.n_steps], 0)
        
        # Exercise boundary tracking
        exercise_boundary = np.zeros(self.n_steps + 1)
        exercise_boundary[-1] = K
        
        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Continuation value
                continuation = discount * (p * option_values[j, i + 1] + 
                                         (1 - p) * option_values[j + 1, i + 1])
                
                # Intrinsic value
                intrinsic = max(K - stock_prices[j, i], 0)
                
                # American option: max of intrinsic and continuation
                option_values[j, i] = max(intrinsic, continuation)
            
            # Find exercise boundary at this time step
            # Exercise boundary is the highest stock price where exercise is optimal
            boundary = 0
            for j in range(i + 1):
                intrinsic = max(K - stock_prices[j, i], 0)
                continuation = discount * (p * option_values[j, i + 1] + 
                                         (1 - p) * option_values[j + 1, i + 1])
                
                if intrinsic > continuation and stock_prices[j, i] > boundary:
                    boundary = stock_prices[j, i]
            
            exercise_boundary[i] = boundary
        
        option_price = option_values[0, 0]
        
        logger.info(f"American put price (Binomial): {option_price:.4f}")
        
        return option_price, exercise_boundary
    
    def price_european_put(self, S0: float, K: float, T: float, r: float, 
                          sigma: float) -> float:
        """
        Price European put option using binomial tree.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            European put option price
        """
        logger.info(f"Pricing European put (Binomial): S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
        
        # Calculate binomial parameters
        dt = T / self.n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        discount = np.exp(-r * dt)
        
        # Initialize option values at maturity
        option_values = np.zeros(self.n_steps + 1)
        
        # Set payoffs at maturity
        for j in range(self.n_steps + 1):
            S_T = S0 * (u ** (self.n_steps - j)) * (d ** j)
            option_values[j] = max(K - S_T, 0)
        
        # Backward induction (European - no early exercise)
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                option_values[j] = discount * (p * option_values[j] + 
                                             (1 - p) * option_values[j + 1])
        
        option_price = option_values[0]
        
        logger.info(f"European put price (Binomial): {option_price:.4f}")
        
        return option_price
    
    def price_american_call(self, S0: float, K: float, T: float, r: float, 
                           sigma: float) -> Tuple[float, np.ndarray]:
        """
        Price American call option using binomial tree.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Tuple of (option_price, exercise_boundary)
        """
        logger.info(f"Pricing American call (Binomial): S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
        
        # Calculate binomial parameters
        dt = T / self.n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        discount = np.exp(-r * dt)
        
        # Initialize stock price tree
        stock_prices = np.zeros((self.n_steps + 1, self.n_steps + 1))
        
        # Fill stock price tree
        for i in range(self.n_steps + 1):
            for j in range(i + 1):
                stock_prices[j, i] = S0 * (u ** (i - j)) * (d ** j)
        
        # Initialize option value tree
        option_values = np.zeros((self.n_steps + 1, self.n_steps + 1))
        
        # Set option values at maturity
        for j in range(self.n_steps + 1):
            option_values[j, self.n_steps] = max(stock_prices[j, self.n_steps] - K, 0)
        
        # Exercise boundary tracking
        exercise_boundary = np.zeros(self.n_steps + 1)
        exercise_boundary[-1] = K
        
        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Continuation value
                continuation = discount * (p * option_values[j, i + 1] + 
                                         (1 - p) * option_values[j + 1, i + 1])
                
                # Intrinsic value
                intrinsic = max(stock_prices[j, i] - K, 0)
                
                # American option: max of intrinsic and continuation
                option_values[j, i] = max(intrinsic, continuation)
            
            # Find exercise boundary
            boundary = np.inf
            for j in range(i + 1):
                intrinsic = max(stock_prices[j, i] - K, 0)
                continuation = discount * (p * option_values[j, i + 1] + 
                                         (1 - p) * option_values[j + 1, i + 1])
                
                if intrinsic > continuation and stock_prices[j, i] < boundary:
                    boundary = stock_prices[j, i]
            
            exercise_boundary[i] = boundary if boundary != np.inf else 0
        
        option_price = option_values[0, 0]
        
        logger.info(f"American call price (Binomial): {option_price:.4f}")
        
        return option_price, exercise_boundary
    
    def calculate_greeks(self, S0: float, K: float, T: float, r: float, 
                        sigma: float, option_type: str = 'put', 
                        american: bool = True) -> dict:
        """
        Calculate option Greeks using finite differences.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'put' or 'call'
            american: True for American, False for European
            
        Returns:
            Dictionary containing Greeks
        """
        bump_size = 0.01
        
        # Base price
        if american:
            if option_type == 'put':
                base_price, _ = self.price_american_put(S0, K, T, r, sigma)
            else:
                base_price, _ = self.price_american_call(S0, K, T, r, sigma)
        else:
            base_price = self.price_european_put(S0, K, T, r, sigma)
        
        # Delta
        if american:
            if option_type == 'put':
                price_up, _ = self.price_american_put(S0 + bump_size, K, T, r, sigma)
                price_down, _ = self.price_american_put(S0 - bump_size, K, T, r, sigma)
            else:
                price_up, _ = self.price_american_call(S0 + bump_size, K, T, r, sigma)
                price_down, _ = self.price_american_call(S0 - bump_size, K, T, r, sigma)
        else:
            price_up = self.price_european_put(S0 + bump_size, K, T, r, sigma)
            price_down = self.price_european_put(S0 - bump_size, K, T, r, sigma)
        
        delta = (price_up - price_down) / (2 * bump_size)
        gamma = (price_up - 2 * base_price + price_down) / (bump_size ** 2)
        
        # Theta
        if T > 0.01:
            if american:
                if option_type == 'put':
                    price_theta, _ = self.price_american_put(S0, K, T - 0.01, r, sigma)
                else:
                    price_theta, _ = self.price_american_call(S0, K, T - 0.01, r, sigma)
            else:
                price_theta = self.price_european_put(S0, K, T - 0.01, r, sigma)
            
            theta = -(price_theta - base_price) / 0.01
        else:
            theta = 0
        
        # Vega
        if american:
            if option_type == 'put':
                price_vega, _ = self.price_american_put(S0, K, T, r, sigma + 0.01)
            else:
                price_vega, _ = self.price_american_call(S0, K, T, r, sigma + 0.01)
        else:
            price_vega = self.price_european_put(S0, K, T, r, sigma + 0.01)
        
        vega = (price_vega - base_price) / 0.01
        
        # Rho
        if american:
            if option_type == 'put':
                price_rho, _ = self.price_american_put(S0, K, T, r + 0.01, sigma)
            else:
                price_rho, _ = self.price_american_call(S0, K, T, r + 0.01, sigma)
        else:
            price_rho = self.price_european_put(S0, K, T, r + 0.01, sigma)
        
        rho = (price_rho - base_price) / 0.01
        
        return {
            'price': base_price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }