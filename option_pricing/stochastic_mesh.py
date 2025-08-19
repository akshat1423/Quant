"""
Stochastic Mesh Method for American Option Pricing

Implementation of the Stochastic Mesh method for pricing American options,
particularly effective for American put options.
"""

import numpy as np
from typing import Tuple, Optional
import logging
from .monte_carlo_engine import MonteCarloEngine

logger = logging.getLogger(__name__)


class StochasticMeshPricer(MonteCarloEngine):
    """
    Stochastic Mesh method for pricing American options.
    
    This method uses a mesh of simulated paths and estimates continuation values
    using basis functions and regression.
    """
    
    def __init__(self, n_simulations: int = 100000, n_time_steps: int = 50,
                 random_seed: Optional[int] = None):
        """
        Initialize Stochastic Mesh pricer.
        
        Args:
            n_simulations: Number of Monte Carlo paths
            n_time_steps: Number of time steps in the mesh
            random_seed: Random seed for reproducibility
        """
        super().__init__(n_simulations, random_seed)
        self.n_time_steps = n_time_steps
        
    def price_american_put(self, S0: float, K: float, T: float, r: float, 
                          sigma: float) -> Tuple[float, float]:
        """
        Price American put option using Stochastic Mesh method.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Tuple of (option_price, standard_error)
        """
        logger.info(f"Pricing American put: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
        
        # Generate stock price paths
        paths = self.generate_stock_paths(S0, r, sigma, T, self.n_time_steps)
        dt = T / self.n_time_steps
        
        # Initialize option values at maturity
        option_values = self.european_payoff(paths[:, -1], K, 'put')
        
        # Backward induction through the mesh
        for t in range(self.n_time_steps - 1, 0, -1):
            current_prices = paths[:, t]
            intrinsic_values = self.european_payoff(current_prices, K, 'put')
            
            # Discount previous option values
            discounted_future_values = option_values * np.exp(-r * dt)
            
            # Only consider in-the-money options for continuation value estimation
            itm_mask = intrinsic_values > 0
            
            if np.sum(itm_mask) > 10:  # Need sufficient points for regression
                # Estimate continuation values using regression
                continuation_values = np.zeros_like(intrinsic_values)
                continuation_values[itm_mask] = self._estimate_continuation_value(
                    current_prices[itm_mask], 
                    discounted_future_values[itm_mask], 
                    r, dt
                )
                continuation_values[~itm_mask] = discounted_future_values[~itm_mask]
                
                # American option: max of intrinsic and continuation
                option_values = np.maximum(intrinsic_values, continuation_values)
            else:
                # Not enough ITM options, use intrinsic value
                option_values = intrinsic_values
        
        # Calculate price and standard error
        option_price = np.mean(option_values)
        standard_error = np.std(option_values) / np.sqrt(self.n_simulations)
        
        logger.info(f"American put price: {option_price:.4f} ± {standard_error:.4f}")
        
        return option_price, standard_error
    
    def _estimate_continuation_value(self, stock_prices: np.ndarray, 
                                   future_values: np.ndarray, r: float, 
                                   dt: float) -> np.ndarray:
        """
        Estimate continuation value using basis function regression.
        
        Args:
            stock_prices: Current stock prices
            future_values: Future option values (already discounted)
            r: Risk-free rate
            dt: Time step
            
        Returns:
            Estimated continuation values
        """
        # Create basis functions
        basis_functions = self._create_basis_functions(stock_prices)
        
        # Regression to estimate continuation value
        try:
            # Use least squares regression
            coefficients = np.linalg.lstsq(basis_functions, future_values, rcond=None)[0]
            continuation_values = basis_functions @ coefficients
            
            # Ensure non-negative continuation values
            continuation_values = np.maximum(continuation_values, 0)
            
        except np.linalg.LinAlgError:
            # Fallback to average if regression fails
            logger.warning("Regression failed, using average continuation value")
            continuation_values = np.full_like(stock_prices, np.mean(future_values))
        
        return continuation_values
    
    def _create_basis_functions(self, stock_prices: np.ndarray) -> np.ndarray:
        """
        Create basis functions for regression.
        
        Uses weighted Laguerre polynomials as basis functions.
        
        Args:
            stock_prices: Stock prices to create basis for
            
        Returns:
            Matrix of basis function values
        """
        n_paths = len(stock_prices)
        n_basis = 3  # Reduced number of basis functions for stability
        
        # Normalize stock prices
        S_normalized = stock_prices / np.mean(stock_prices)
        
        # Weight function (commonly used in Longstaff-Schwartz)
        weights = np.exp(-S_normalized / 2)
        
        # Basis functions: weighted Laguerre polynomials
        basis = np.zeros((n_paths, n_basis))
        basis[:, 0] = weights  # L0
        basis[:, 1] = weights * (1 - S_normalized)  # L1
        basis[:, 2] = weights * (1 - 2*S_normalized + S_normalized**2/2)  # L2
        
        return basis
    
    def calculate_mse_vs_european(self, S0: float, K: float, T: float, r: float, 
                                sigma: float, european_price: float) -> float:
        """
        Calculate Mean Squared Error compared to European option price.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            european_price: European option price for comparison
            
        Returns:
            Mean squared error
        """
        american_price, _ = self.price_american_put(S0, K, T, r, sigma)
        mse = (american_price - european_price) ** 2
        
        logger.info(f"MSE vs European: {mse:.6f}")
        return mse
    
    def sensitivity_analysis(self, S0: float, K: float, T: float, r: float, 
                           sigma: float) -> dict:
        """
        Perform sensitivity analysis for the American put option.
        
        Returns:
            Dictionary containing sensitivities (Greeks)
        """
        def pricing_func(S, K, T, r, sigma):
            return self.price_american_put(S, K, T, r, sigma)[0]
        
        # Calculate Greeks using finite differences
        greeks = self.estimate_greeks(pricing_func, S0, K, T, r, sigma)
        
        # Additional sensitivities
        base_price = greeks['price']
        
        # Theta (time decay)
        if T > 0.01:  # Avoid division by very small numbers
            price_theta = pricing_func(S0, K, T - 0.01, r, sigma)
            theta = -(price_theta - base_price) / 0.01
            greeks['theta'] = theta
        
        # Vega (volatility sensitivity)
        price_vega = pricing_func(S0, K, T, r, sigma + 0.01)
        vega = (price_vega - base_price) / 0.01
        greeks['vega'] = vega
        
        # Rho (interest rate sensitivity)  
        price_rho = pricing_func(S0, K, T, r + 0.01, sigma)
        rho = (price_rho - base_price) / 0.01
        greeks['rho'] = rho
        
        return greeks