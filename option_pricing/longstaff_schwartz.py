"""
Longstaff-Schwartz Algorithm for American Option Pricing

Implementation of the Longstaff-Schwartz method for pricing American options
using least squares Monte Carlo simulation.
"""

import numpy as np
from typing import Tuple, Optional, List
import logging
from .monte_carlo_engine import MonteCarloEngine

logger = logging.getLogger(__name__)


class LongstaffSchwartzPricer(MonteCarloEngine):
    """
    Longstaff-Schwartz method for pricing American options.
    
    This method uses least squares regression to estimate continuation values
    and determine optimal exercise strategies.
    """
    
    def __init__(self, n_simulations: int = 100000, n_time_steps: int = 50,
                 polynomial_degree: int = 4, random_seed: Optional[int] = None):
        """
        Initialize Longstaff-Schwartz pricer.
        
        Args:
            n_simulations: Number of Monte Carlo paths
            n_time_steps: Number of time steps
            polynomial_degree: Degree of polynomials for regression
            random_seed: Random seed for reproducibility
        """
        super().__init__(n_simulations, random_seed)
        self.n_time_steps = n_time_steps
        self.polynomial_degree = polynomial_degree
        
    def price_american_put(self, S0: float, K: float, T: float, r: float, 
                          sigma: float) -> Tuple[float, float, np.ndarray]:
        """
        Price American put option using Longstaff-Schwartz method.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Tuple of (option_price, standard_error, exercise_boundary)
        """
        logger.info(f"Pricing American put (LS): S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
        
        # Generate stock price paths
        paths = self.generate_stock_paths(S0, r, sigma, T, self.n_time_steps)
        dt = T / self.n_time_steps
        
        # Initialize cash flows with payoff at maturity
        cash_flows = np.zeros_like(paths)
        cash_flows[:, -1] = self.european_payoff(paths[:, -1], K, 'put')
        
        # Exercise boundary tracking
        exercise_boundary = np.zeros(self.n_time_steps + 1)
        exercise_boundary[-1] = K  # At maturity, exercise boundary is strike
        
        # Backward induction
        for t in range(self.n_time_steps - 1, 0, -1):
            current_prices = paths[:, t]
            intrinsic_values = self.european_payoff(current_prices, K, 'put')
            
            # Find in-the-money paths
            itm_mask = intrinsic_values > 0
            
            if np.sum(itm_mask) == 0:
                # No paths are in-the-money
                exercise_boundary[t] = 0
                continue
                
            # Get future cash flows for in-the-money paths
            future_cash_flows = cash_flows[itm_mask, t+1:]
            discounted_cash_flows = self._discount_cash_flows(future_cash_flows, r, dt)
            
            # Estimate continuation value using regression
            continuation_values = self._estimate_continuation_value_ls(
                current_prices[itm_mask], 
                discounted_cash_flows
            )
            
            # Exercise decision: exercise if intrinsic > continuation
            exercise_mask = intrinsic_values[itm_mask] > continuation_values
            
            # Update cash flows
            cash_flows[itm_mask, t] = np.where(
                exercise_mask,
                intrinsic_values[itm_mask],  # Exercise
                0  # Hold (cash flow occurs later)
            )
            
            # Clear future cash flows for exercised paths
            exercised_paths = np.where(itm_mask)[0][exercise_mask]
            cash_flows[exercised_paths, t+1:] = 0
            
            # Estimate exercise boundary
            if np.sum(exercise_mask) > 0:
                exercise_boundary[t] = np.min(current_prices[itm_mask][exercise_mask])
            else:
                exercise_boundary[t] = 0
        
        # Calculate option values
        total_cash_flows = np.sum(cash_flows, axis=1)
        option_values = total_cash_flows * self.discount_factor(r, dt)
        
        # Calculate price and standard error
        option_price = np.mean(option_values)
        standard_error = np.std(option_values) / np.sqrt(self.n_simulations)
        
        logger.info(f"American put price (LS): {option_price:.4f} ± {standard_error:.4f}")
        
        return option_price, standard_error, exercise_boundary
    
    def _discount_cash_flows(self, cash_flows: np.ndarray, r: float, 
                           dt: float) -> np.ndarray:
        """
        Discount future cash flows to present value.
        
        Args:
            cash_flows: Future cash flows matrix
            r: Risk-free rate
            dt: Time step
            
        Returns:
            Present value of cash flows
        """
        n_paths, n_steps = cash_flows.shape
        discount_factors = np.array([self.discount_factor(r, (i+1) * dt) 
                                   for i in range(n_steps)])
        
        # Apply discount factors
        discounted = cash_flows * discount_factors[np.newaxis, :]
        
        return np.sum(discounted, axis=1)
    
    def _estimate_continuation_value_ls(self, stock_prices: np.ndarray, 
                                      continuation_payoffs: np.ndarray) -> np.ndarray:
        """
        Estimate continuation value using least squares regression.
        
        Args:
            stock_prices: Current stock prices
            continuation_payoffs: Discounted continuation payoffs
            
        Returns:
            Estimated continuation values
        """
        # Create basis functions (weighted Laguerre polynomials)
        basis_matrix = self._create_laguerre_basis(stock_prices)
        
        try:
            # Least squares regression
            coefficients = np.linalg.lstsq(basis_matrix, continuation_payoffs, rcond=None)[0]
            continuation_values = basis_matrix @ coefficients
            
            # Ensure non-negative continuation values
            continuation_values = np.maximum(continuation_values, 0)
            
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to average if regression fails
            logger.warning("LS regression failed, using average continuation value")
            continuation_values = np.full_like(stock_prices, np.mean(continuation_payoffs))
        
        return continuation_values
    
    def _create_laguerre_basis(self, stock_prices: np.ndarray) -> np.ndarray:
        """
        Create weighted Laguerre polynomial basis functions.
        
        Args:
            stock_prices: Stock prices
            
        Returns:
            Basis function matrix
        """
        n_paths = len(stock_prices)
        
        # Normalize stock prices
        S_mean = np.mean(stock_prices)
        x = stock_prices / S_mean
        
        # Weight function
        w = np.exp(-x / 2)
        
        # Laguerre polynomials L_0, L_1, L_2, L_3, L_4
        basis = np.zeros((n_paths, self.polynomial_degree + 1))
        
        if self.polynomial_degree >= 0:
            basis[:, 0] = w  # L_0(x) = 1
        if self.polynomial_degree >= 1:
            basis[:, 1] = w * (1 - x)  # L_1(x) = 1 - x
        if self.polynomial_degree >= 2:
            basis[:, 2] = w * (1 - 2*x + x**2/2)  # L_2(x) = 1 - 2x + x²/2
        if self.polynomial_degree >= 3:
            basis[:, 3] = w * (1 - 3*x + 3*x**2/2 - x**3/6)  # L_3(x)
        if self.polynomial_degree >= 4:
            basis[:, 4] = w * (1 - 4*x + 3*x**2 - 2*x**3/3 + x**4/24)  # L_4(x)
        
        return basis
    
    def calculate_early_exercise_premium(self, S0: float, K: float, T: float, 
                                       r: float, sigma: float, 
                                       european_price: float) -> float:
        """
        Calculate the early exercise premium of American option.
        
        Args:
            S0: Initial stock price
            K: Strike price  
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            european_price: European option price for comparison
            
        Returns:
            Early exercise premium
        """
        american_price, _, _ = self.price_american_put(S0, K, T, r, sigma)
        premium = american_price - european_price
        
        logger.info(f"Early exercise premium: {premium:.4f}")
        return premium
    
    def simulate_exercise_decisions(self, S0: float, K: float, T: float, 
                                  r: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate and analyze exercise decisions.
        
        Returns:
            Tuple of (exercise_times, exercise_stock_prices)
        """
        # Generate paths and get exercise boundary
        _, _, exercise_boundary = self.price_american_put(S0, K, T, r, sigma)
        paths = self.generate_stock_paths(S0, r, sigma, T, self.n_time_steps)
        
        exercise_times = []
        exercise_prices = []
        
        dt = T / self.n_time_steps
        time_grid = np.linspace(0, T, self.n_time_steps + 1)
        
        for path_idx in range(self.n_simulations):
            path = paths[path_idx, :]
            
            # Check exercise at each time step
            for t_idx in range(1, self.n_time_steps):
                stock_price = path[t_idx]
                boundary = exercise_boundary[t_idx]
                
                # Exercise if stock price is below boundary (for put option)
                if stock_price <= boundary and boundary > 0:
                    exercise_times.append(time_grid[t_idx])
                    exercise_prices.append(stock_price)
                    break
        
        return np.array(exercise_times), np.array(exercise_prices)
    
    def compare_with_binomial(self, S0: float, K: float, T: float, r: float, 
                            sigma: float, binomial_price: float) -> dict:
        """
        Compare Longstaff-Schwartz price with binomial model.
        
        Returns:
            Dictionary with comparison metrics
        """
        ls_price, ls_std_err, _ = self.price_american_put(S0, K, T, r, sigma)
        
        price_diff = ls_price - binomial_price
        percentage_diff = (price_diff / binomial_price) * 100
        mse = price_diff ** 2
        
        comparison = {
            'longstaff_schwartz_price': ls_price,
            'longstaff_schwartz_std_error': ls_std_err,
            'binomial_price': binomial_price,
            'price_difference': price_diff,
            'percentage_difference': percentage_diff,
            'mse': mse,
            'within_tolerance': mse < 0.05
        }
        
        logger.info(f"LS vs Binomial comparison: MSE = {mse:.6f}")
        
        return comparison