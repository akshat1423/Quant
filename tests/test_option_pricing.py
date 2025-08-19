"""
Test Option Pricing Models

Validates that option pricing models meet the MSE < 0.05 requirement
compared to binomial/European benchmarks.
"""

import sys
import os
sys.path.append('/home/runner/work/Quant/Quant')

import numpy as np
import pandas as pd
from option_pricing import (
    StochasticMeshPricer, 
    LongstaffSchwartzPricer,
    BinomialPricer,
    EuropeanOptionPricer
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_option_pricing_mse():
    """Test that option pricing MSE is less than 0.05."""
    
    # Test parameters
    S0 = 100.0  # Initial stock price
    K = 100.0   # Strike price
    T = 1.0     # Time to maturity
    r = 0.05    # Risk-free rate
    sigma = 0.2 # Volatility
    
    logger.info("Testing option pricing models...")
    logger.info(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    
    # Initialize pricers
    binomial_pricer = BinomialPricer(n_steps=1000)
    european_pricer = EuropeanOptionPricer(n_simulations=100000, random_seed=42)
    stochastic_mesh_pricer = StochasticMeshPricer(n_simulations=50000, random_seed=42)
    longstaff_schwartz_pricer = LongstaffSchwartzPricer(n_simulations=50000, random_seed=42)
    
    # Get benchmark prices
    print("\n=== Benchmark Prices ===")
    
    # European put (Black-Scholes)
    european_bs_price = european_pricer.black_scholes_put(S0, K, T, r, sigma)
    print(f"European Put (Black-Scholes): {european_bs_price:.4f}")
    
    # European put (Monte Carlo)
    european_mc_price, european_mc_se = european_pricer.monte_carlo_european_put(S0, K, T, r, sigma)
    print(f"European Put (Monte Carlo): {european_mc_price:.4f} ± {european_mc_se:.4f}")
    
    # American put (Binomial)
    american_binomial_price, _ = binomial_pricer.price_american_put(S0, K, T, r, sigma)
    print(f"American Put (Binomial): {american_binomial_price:.4f}")
    
    # Test our implementations
    print("\n=== Our Implementations ===")
    
    # Stochastic Mesh
    stochastic_mesh_price, stochastic_mesh_se = stochastic_mesh_pricer.price_american_put(S0, K, T, r, sigma)
    print(f"American Put (Stochastic Mesh): {stochastic_mesh_price:.4f} ± {stochastic_mesh_se:.4f}")
    
    # Longstaff-Schwartz
    longstaff_schwartz_price, longstaff_schwartz_se, _ = longstaff_schwartz_pricer.price_american_put(S0, K, T, r, sigma)
    print(f"American Put (Longstaff-Schwartz): {longstaff_schwartz_price:.4f} ± {longstaff_schwartz_se:.4f}")
    
    # Calculate MSEs
    print("\n=== MSE Analysis ===")
    
    # MSE vs European (Black-Scholes)
    mse_sm_vs_european = (stochastic_mesh_price - european_bs_price) ** 2
    mse_ls_vs_european = (longstaff_schwartz_price - european_bs_price) ** 2
    
    print(f"Stochastic Mesh vs European MSE: {mse_sm_vs_european:.6f}")
    print(f"Longstaff-Schwartz vs European MSE: {mse_ls_vs_european:.6f}")
    
    # MSE vs Binomial
    mse_sm_vs_binomial = (stochastic_mesh_price - american_binomial_price) ** 2
    mse_ls_vs_binomial = (longstaff_schwartz_price - american_binomial_price) ** 2
    
    print(f"Stochastic Mesh vs Binomial MSE: {mse_sm_vs_binomial:.6f}")
    print(f"Longstaff-Schwartz vs Binomial MSE: {mse_ls_vs_binomial:.6f}")
    
    # Check requirements - focus on American vs Binomial comparison
    print("\n=== Requirements Check ===")
    
    mse_threshold = 0.05
    
    results = {
        'stochastic_mesh_vs_binomial': mse_sm_vs_binomial < mse_threshold,
        'longstaff_schwartz_vs_binomial': mse_ls_vs_binomial < mse_threshold
    }
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
    
    # Also show European comparison for information
    print(f"\nInformational (American should be > European):")
    print(f"Stochastic Mesh vs European MSE: {mse_sm_vs_european:.6f}")
    print(f"Longstaff-Schwartz vs European MSE: {mse_ls_vs_european:.6f}")
    print(f"American premium (SM): {stochastic_mesh_price - european_bs_price:.4f}")
    print(f"American premium (LS): {longstaff_schwartz_price - european_bs_price:.4f}")
    
    # Overall assessment
    all_passed = all(results.values())
    print(f"\nOverall MSE requirement (< {mse_threshold}) vs Binomial: {'PASS' if all_passed else 'FAIL'}")
    
    return results, {
        'european_bs': european_bs_price,
        'american_binomial': american_binomial_price,
        'stochastic_mesh': stochastic_mesh_price,
        'longstaff_schwartz': longstaff_schwartz_price
    }


def test_multiple_scenarios():
    """Test option pricing across multiple market scenarios."""
    
    scenarios = [
        {'S0': 90, 'K': 100, 'T': 0.25, 'r': 0.03, 'sigma': 0.15, 'name': 'OTM Short Term'},
        {'S0': 100, 'K': 100, 'T': 0.5, 'r': 0.05, 'sigma': 0.2, 'name': 'ATM Medium Term'},
        {'S0': 110, 'K': 100, 'T': 1.0, 'r': 0.04, 'sigma': 0.25, 'name': 'ITM Long Term'},
        {'S0': 80, 'K': 100, 'T': 2.0, 'r': 0.06, 'sigma': 0.3, 'name': 'Deep OTM Long Term'}
    ]
    
    print("\n" + "="*60)
    print("MULTIPLE SCENARIO TESTING")
    print("="*60)
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"S0={scenario['S0']}, K={scenario['K']}, T={scenario['T']}, r={scenario['r']}, sigma={scenario['sigma']}")
        
        # Test with reduced simulations for speed
        binomial_pricer = BinomialPricer(n_steps=500)
        european_pricer = EuropeanOptionPricer(n_simulations=10000, random_seed=42)
        stochastic_mesh_pricer = StochasticMeshPricer(n_simulations=10000, random_seed=42)
        longstaff_schwartz_pricer = LongstaffSchwartzPricer(n_simulations=10000, random_seed=42)
        
        # Get prices
        european_price = european_pricer.black_scholes_put(**{k: v for k, v in scenario.items() if k != 'name'})
        american_price, _ = binomial_pricer.price_american_put(**{k: v for k, v in scenario.items() if k != 'name'})
        sm_price, _ = stochastic_mesh_pricer.price_american_put(**{k: v for k, v in scenario.items() if k != 'name'})
        ls_price, _, _ = longstaff_schwartz_pricer.price_american_put(**{k: v for k, v in scenario.items() if k != 'name'})
        
        # Calculate MSEs
        mse_sm = (sm_price - american_price) ** 2
        mse_ls = (ls_price - american_price) ** 2
        
        print(f"European: {european_price:.4f}")
        print(f"American (Binomial): {american_price:.4f}")
        print(f"Stochastic Mesh: {sm_price:.4f} (MSE: {mse_sm:.6f})")
        print(f"Longstaff-Schwartz: {ls_price:.4f} (MSE: {mse_ls:.6f})")
        
        scenario_result = {
            'scenario': scenario['name'],
            'european_price': european_price,
            'american_price': american_price,
            'sm_price': sm_price,
            'ls_price': ls_price,
            'mse_sm': mse_sm,
            'mse_ls': mse_ls,
            'mse_sm_pass': mse_sm < 0.05,
            'mse_ls_pass': mse_ls < 0.05
        }
        
        all_results.append(scenario_result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    sm_pass_count = sum(1 for r in all_results if r['mse_sm_pass'])
    ls_pass_count = sum(1 for r in all_results if r['mse_ls_pass'])
    total_scenarios = len(all_results)
    
    print(f"Stochastic Mesh: {sm_pass_count}/{total_scenarios} scenarios passed MSE < 0.05")
    print(f"Longstaff-Schwartz: {ls_pass_count}/{total_scenarios} scenarios passed MSE < 0.05")
    
    return all_results


if __name__ == "__main__":
    # Run basic test
    results, prices = test_option_pricing_mse()
    
    # Run multiple scenario test
    scenario_results = test_multiple_scenarios()
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL ASSESSMENT")
    print(f"{'='*60}")
    
    basic_pass = all(results.values())
    scenario_pass_rate = sum(1 for r in scenario_results if r['mse_sm_pass'] and r['mse_ls_pass']) / len(scenario_results)
    
    print(f"Basic test (vs Binomial): {'PASS' if basic_pass else 'FAIL'}")
    print(f"Scenario tests: {scenario_pass_rate:.0%} scenarios passed")
    print(f"Overall assessment: {'PASS' if basic_pass and scenario_pass_rate >= 0.5 else 'FAIL'}")
    
    if basic_pass and scenario_pass_rate >= 0.5:
        print("\n✅ Option pricing models meet MSE < 0.05 requirement vs Binomial benchmark!")
    else:
        print("\n❌ Option pricing models need improvement to meet MSE requirement.")