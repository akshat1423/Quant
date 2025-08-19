"""
European Option Pricing Models

Implementation of analytical and numerical methods for European option pricing,
including Black-Scholes model and Monte Carlo simulation.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional
import logging
from .monte_carlo_engine import MonteCarloEngine

logger = logging.getLogger(__name__)


class EuropeanOptionPricer(MonteCarloEngine):
    """
    European option pricing using analytical and Monte Carlo methods.
    
    Provides both Black-Scholes analytical solutions and Monte Carlo
    simulation for comparison and validation.
    """
    
    def __init__(self, n_simulations: int = 100000, random_seed: Optional[int] = None):
        """
        Initialize European option pricer.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        """
        super().__init__(n_simulations, random_seed)
    
    def black_scholes_put(self, S0: float, K: float, T: float, r: float, 
                         sigma: float) -> float:
        """
        Calculate European put option price using Black-Scholes formula.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            European put option price
        """
        if T <= 0:
            return max(K - S0, 0)
        
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        
        logger.info(f"Black-Scholes put price: {put_price:.4f}")
        return put_price
    
    def black_scholes_call(self, S0: float, K: float, T: float, r: float, 
                          sigma: float) -> float:
        """
        Calculate European call option price using Black-Scholes formula.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            European call option price
        """
        if T <= 0:
            return max(S0 - K, 0)
        
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        logger.info(f"Black-Scholes call price: {call_price:.4f}")
        return call_price
    
    def monte_carlo_european_put(self, S0: float, K: float, T: float, r: float, 
                                sigma: float) -> Tuple[float, float]:
        """
        Calculate European put option price using Monte Carlo simulation.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Tuple of (option_price, standard_error)
        """
        logger.info(f"MC European put: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
        
        # Generate terminal stock prices
        Z = np.random.standard_normal(self.n_simulations)
        S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs
        payoffs = self.european_payoff(S_T, K, 'put')
        
        # Discount to present value
        discounted_payoffs = payoffs * self.discount_factor(r, T)
        
        # Calculate price and standard error
        option_price = np.mean(discounted_payoffs)
        standard_error = np.std(discounted_payoffs) / np.sqrt(self.n_simulations)
        
        logger.info(f"MC European put price: {option_price:.4f} ± {standard_error:.4f}")
        
        return option_price, standard_error
    
    def monte_carlo_european_call(self, S0: float, K: float, T: float, r: float, 
                                 sigma: float) -> Tuple[float, float]:
        """
        Calculate European call option price using Monte Carlo simulation.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Tuple of (option_price, standard_error)
        """
        logger.info(f"MC European call: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
        
        # Generate terminal stock prices
        Z = np.random.standard_normal(self.n_simulations)
        S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs
        payoffs = self.european_payoff(S_T, K, 'call')
        
        # Discount to present value
        discounted_payoffs = payoffs * self.discount_factor(r, T)
        
        # Calculate price and standard error
        option_price = np.mean(discounted_payoffs)
        standard_error = np.std(discounted_payoffs) / np.sqrt(self.n_simulations)
        
        logger.info(f"MC European call price: {option_price:.4f} ± {standard_error:.4f}")
        
        return option_price, standard_error
    
    def black_scholes_greeks(self, S0: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'put') -> dict:
        """
        Calculate analytical Greeks for European options.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'put' or 'call'
            
        Returns:
            Dictionary containing analytical Greeks
        """
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Common terms
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        cdf_neg_d1 = norm.cdf(-d1)
        cdf_neg_d2 = norm.cdf(-d2)
        
        if option_type.lower() == 'put':
            # Put option Greeks
            delta = cdf_neg_d1 - 1
            theta = (-S0 * pdf_d1 * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * cdf_neg_d2)
            rho = -K * T * np.exp(-r * T) * cdf_neg_d2
            
            price = self.black_scholes_put(S0, K, T, r, sigma)
            
        else:  # call option
            delta = cdf_d1
            theta = (-S0 * pdf_d1 * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * cdf_d2)
            rho = K * T * np.exp(-r * T) * cdf_d2
            
            price = self.black_scholes_call(S0, K, T, r, sigma)
        
        # Gamma and Vega are the same for puts and calls
        gamma = pdf_d1 / (S0 * sigma * np.sqrt(T))
        vega = S0 * pdf_d1 * np.sqrt(T)
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def implied_volatility(self, market_price: float, S0: float, K: float, 
                          T: float, r: float, option_type: str = 'put',
                          max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Observed market price
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: 'put' or 'call'
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility
        """
        # Initial guess
        sigma = 0.2
        
        for i in range(max_iterations):
            if option_type.lower() == 'put':
                price = self.black_scholes_put(S0, K, T, r, sigma)
            else:
                price = self.black_scholes_call(S0, K, T, r, sigma)
            
            # Calculate vega for Newton-Raphson
            greeks = self.black_scholes_greeks(S0, K, T, r, sigma, option_type)
            vega = greeks['vega']
            
            # Newton-Raphson update
            price_diff = price - market_price
            
            if abs(price_diff) < tolerance:
                logger.info(f"Implied volatility converged: {sigma:.4f}")
                return sigma
            
            if vega == 0:
                logger.warning("Vega is zero, cannot calculate implied volatility")
                return sigma
            
            sigma = sigma - price_diff / vega
            
            # Ensure volatility stays positive
            sigma = max(sigma, 0.001)
        
        logger.warning(f"Implied volatility did not converge after {max_iterations} iterations")
        return sigma
    
    def compare_pricing_methods(self, S0: float, K: float, T: float, r: float, 
                              sigma: float, option_type: str = 'put') -> dict:
        """
        Compare Black-Scholes and Monte Carlo pricing methods.
        
        Returns:
            Dictionary with comparison results
        """
        # Black-Scholes price
        if option_type.lower() == 'put':
            bs_price = self.black_scholes_put(S0, K, T, r, sigma)
            mc_price, mc_std_err = self.monte_carlo_european_put(S0, K, T, r, sigma)
        else:
            bs_price = self.black_scholes_call(S0, K, T, r, sigma)
            mc_price, mc_std_err = self.monte_carlo_european_call(S0, K, T, r, sigma)
        
        # Calculate differences
        price_diff = abs(mc_price - bs_price)
        percentage_diff = (price_diff / bs_price) * 100
        
        # Check if MC is within confidence interval
        confidence_interval = 1.96 * mc_std_err  # 95% confidence
        within_ci = price_diff <= confidence_interval
        
        comparison = {
            'black_scholes_price': bs_price,
            'monte_carlo_price': mc_price,
            'monte_carlo_std_error': mc_std_err,
            'price_difference': price_diff,
            'percentage_difference': percentage_diff,
            'confidence_interval': confidence_interval,
            'mc_within_confidence': within_ci,
            'n_simulations': self.n_simulations
        }
        
        logger.info(f"Pricing comparison: BS={bs_price:.4f}, MC={mc_price:.4f} ± {mc_std_err:.4f}")
        
        return comparison
    
    def sensitivity_analysis(self, S0: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'put') -> dict:
        """
        Perform comprehensive sensitivity analysis.
        
        Returns:
            Dictionary with sensitivity results
        """
        # Get analytical Greeks
        analytical_greeks = self.black_scholes_greeks(S0, K, T, r, sigma, option_type)
        
        # Calculate numerical Greeks using finite differences
        def pricing_func(S, K, T, r, sigma):
            if option_type.lower() == 'put':
                return self.black_scholes_put(S, K, T, r, sigma)
            else:
                return self.black_scholes_call(S, K, T, r, sigma)
        
        numerical_greeks = self.estimate_greeks(pricing_func, S0, K, T, r, sigma)
        
        return {
            'analytical_greeks': analytical_greeks,
            'numerical_greeks': numerical_greeks
        }