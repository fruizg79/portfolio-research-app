"""
Examples and demonstrations for the Asset Allocation Framework

This script provides practical examples for all main classes and methods
in the asset_allocation.py module.

Author: Fernando Ruiz
Created: 12/09/2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from asset_allocation import (
    CAsset, CCopulas, CPortfolio_optimization, 
    CBlack_litterman, CMarket, CUtility
)

###############################################################################
# Example 1: Asset Calibration and Simulation
###############################################################################
def example_asset_calibration():
    """
    Demonstrates asset parameter calibration and simulation.
    """
    print("="*70)
    print("EXAMPLE 1: Asset Calibration and Simulation")
    print("="*70)
    
    # Generate synthetic asset data (e.g., monthly returns)
    np.random.seed(42)
    n_periods = 120  # 10 years of monthly data
    true_mean = 0.008  # 0.8% monthly return
    true_vol = 0.04    # 4% monthly volatility
    asset_data = np.random.normal(true_mean, true_vol, n_periods)
    
    # Create asset object with normal model
    asset_normal = CAsset(
        data=asset_data,
        data_freq='monthly',
        asset_class='US_Equity',
        sim_model='normal'
    )
    
    # Calibrate parameters
    params_normal = asset_normal.param_calibration()
    print("\nNormal Model Calibration:")
    print(f"  Annual Mean: {params_normal['mean']:.2%}")
    print(f"  Annual Vol:  {params_normal['vol']:.2%}")
    
    # Simulate future paths
    n_sim = 1000
    n_future = 60  # 5 years ahead
    simulations = asset_normal.simulate(n_future, n_sim)
    
    print(f"\nSimulated {n_sim} paths for {n_future} periods")
    print(f"  Mean final value: {simulations[:, -1].mean():.4f}")
    print(f"  Std final value:  {simulations[:, -1].std():.4f}")
    
    # Plot simulation results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(asset_data, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Historical Returns Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    for i in range(min(100, n_sim)):
        plt.plot(simulations[i, :], alpha=0.1, color='blue')
    plt.plot(simulations.mean(axis=0), color='red', linewidth=2, label='Mean Path')
    plt.title('Simulated Asset Paths')
    plt.xlabel('Period')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('example1_asset_simulation.png', dpi=150)
    print("\nPlot saved as 'example1_asset_simulation.png'")
    plt.close()


###############################################################################
# Example 2: Copulas - Generating Correlated Returns
###############################################################################
def example_copulas():
    """
    Demonstrates copula-based generation of correlated asset returns.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Copulas - Correlated Returns Generation")
    print("="*70)
    
    # Define correlation matrix for 4 assets
    corr_matrix = pd.DataFrame([
        [1.00, 0.70, 0.50, 0.30],
        [0.70, 1.00, 0.60, 0.40],
        [0.50, 0.60, 1.00, 0.50],
        [0.30, 0.40, 0.50, 1.00]
    ], columns=['US_Equity', 'EU_Equity', 'Bonds', 'Commodities'],
       index=['US_Equity', 'EU_Equity', 'Bonds', 'Commodities'])
    
    # Create copula object
    n_sim = 5000
    copula = CCopulas(corr_matrix, n_sim)
    
    # Generate correlated normal variables
    _, noise_corr = copula.get_correlated_normal()
    
    # Verify correlations
    empirical_corr = np.corrcoef(noise_corr.T)
    print("\nTarget vs Empirical Correlations:")
    print("\nTarget Correlation Matrix:")
    print(corr_matrix)
    print("\nEmpirical Correlation Matrix:")
    print(pd.DataFrame(empirical_corr, 
                      columns=corr_matrix.columns,
                      index=corr_matrix.index))
    
    # Generate asset returns with specific marginals
    marginal_params = [
        {'mean': 0.10, 'std': 0.18},  # US Equity: 10% return, 18% vol
        {'mean': 0.08, 'std': 0.20},  # EU Equity: 8% return, 20% vol
        {'mean': 0.04, 'std': 0.06},  # Bonds: 4% return, 6% vol
        {'mean': 0.06, 'std': 0.25}   # Commodities: 6% return, 25% vol
    ]
    
    returns = copula.get_correlated_returns(marginal_params)
    
    print("\nGenerated Returns Statistics:")
    for i, asset in enumerate(corr_matrix.columns):
        print(f"\n{asset}:")
        print(f"  Target Mean: {marginal_params[i]['mean']:.2%}, "
              f"Actual: {returns[:, i].mean():.2%}")
        print(f"  Target Std:  {marginal_params[i]['std']:.2%}, "
              f"Actual: {returns[:, i].std():.2%}")
    
    # Generate t-Student copula (for tail dependence)
    print("\n\nGenerating t-Student Copula...")
    t_corr, t_uniform = copula.get_tstudent_copula(chi_df=5)
    print(f"Generated {n_sim} samples with t-Student copula")
    print(f"Shape: {t_corr.shape}")
    
    # Visualize
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(returns[:, 0], returns[:, 1], alpha=0.3, s=1)
    plt.xlabel('US Equity Returns')
    plt.ylabel('EU Equity Returns')
    plt.title('Gaussian Copula')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.scatter(t_corr[:, 0], t_corr[:, 1], alpha=0.3, s=1)
    plt.xlabel('Asset 1')
    plt.ylabel('Asset 2')
    plt.title('t-Student Copula (df=5)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.hist2d(returns[:, 0], returns[:, 1], bins=50, cmap='Blues')
    plt.xlabel('US Equity Returns')
    plt.ylabel('EU Equity Returns')
    plt.title('Joint Distribution Density')
    plt.colorbar(label='Frequency')
    
    plt.tight_layout()
    plt.savefig('example2_copulas.png', dpi=150)
    print("\nPlot saved as 'example2_copulas.png'")
    plt.close()


###############################################################################
# Example 3: Portfolio Optimization
###############################################################################
def example_portfolio_optimization():
    """
    Demonstrates mean-variance portfolio optimization and efficient frontier.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Portfolio Optimization")
    print("="*70)
    
    # Define 5 assets
    assets = ['US_Stocks', 'EU_Stocks', 'Bonds', 'Commodities', 'Real_Estate']
    n_assets = len(assets)
    
    # Expected returns (annual)
    expected_ret = np.array([0.10, 0.08, 0.04, 0.06, 0.09])
    
    # Volatilities (annual)
    volatilities = np.array([0.18, 0.20, 0.06, 0.25, 0.15])
    
    # Correlation matrix
    correlation_matrix = np.array([
        [1.00, 0.75, 0.20, 0.30, 0.60],
        [0.75, 1.00, 0.15, 0.35, 0.55],
        [0.20, 0.15, 1.00, 0.05, 0.25],
        [0.30, 0.35, 0.05, 1.00, 0.40],
        [0.60, 0.55, 0.25, 0.40, 1.00]
    ])
    
    # Initial weights (equal weighted)
    initial_weights = np.ones(n_assets) / n_assets
    
    # Create portfolio optimization object
    port_opt = CPortfolio_optimization(
        port_wts=initial_weights,
        correlation_matrix=correlation_matrix,
        expected_ret=expected_ret,
        vol=volatilities
    )
    
    # Define constraints
    x_min = np.array([0.0] * n_assets)  # No short selling
    x_max = np.array([0.5] * n_assets)  # Max 50% per asset
    
    # Generate efficient frontier
    print("\nGenerating Efficient Frontier...")
    returns, vols, weights = port_opt.efficient_frontier(x_min, x_max, num_points=30)
    
    print(f"\nEfficient Frontier: {len(returns)} portfolios")
    print(f"  Min Volatility: {vols.min():.2%}")
    print(f"  Max Return:     {returns.max():.2%}")
    
    # Find maximum Sharpe ratio portfolio (assume rf = 2%)
    risk_free_rate = 0.02
    sharpe_ratios = (returns - risk_free_rate) / vols
    max_sharpe_idx = sharpe_ratios.argmax()
    
    print(f"\nMaximum Sharpe Ratio Portfolio:")
    print(f"  Expected Return: {returns[max_sharpe_idx]:.2%}")
    print(f"  Volatility:      {vols[max_sharpe_idx]:.2%}")
    print(f"  Sharpe Ratio:    {sharpe_ratios[max_sharpe_idx]:.2f}")
    print(f"\n  Weights:")
    for i, asset in enumerate(assets):
        print(f"    {asset:15s}: {weights[max_sharpe_idx][i]:6.2%}")
    
    # Minimum variance portfolio
    min_vol_idx = vols.argmin()
    print(f"\nMinimum Variance Portfolio:")
    print(f"  Expected Return: {returns[min_vol_idx]:.2%}")
    print(f"  Volatility:      {vols[min_vol_idx]:.2%}")
    print(f"\n  Weights:")
    for i, asset in enumerate(assets):
        print(f"    {asset:15s}: {weights[min_vol_idx][i]:6.2%}")
    
    # Plot efficient frontier
    plt.figure(figsize=(12, 8))
    
    # Efficient frontier
    plt.plot(vols, returns, 'b-', linewidth=2, label='Efficient Frontier')
    
    # Individual assets
    plt.scatter(volatilities, expected_ret, c='red', s=100, 
                marker='o', label='Individual Assets', zorder=3)
    for i, asset in enumerate(assets):
        plt.annotate(asset, (volatilities[i], expected_ret[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Max Sharpe portfolio
    plt.scatter(vols[max_sharpe_idx], returns[max_sharpe_idx], 
                c='green', s=200, marker='*', 
                label='Max Sharpe Ratio', zorder=4)
    
    # Min variance portfolio
    plt.scatter(vols[min_vol_idx], returns[min_vol_idx], 
                c='orange', s=200, marker='D', 
                label='Min Variance', zorder=4)
    
    # Capital allocation line
    cal_x = np.linspace(0, vols.max(), 100)
    cal_y = risk_free_rate + sharpe_ratios[max_sharpe_idx] * cal_x
    plt.plot(cal_x, cal_y, 'g--', linewidth=1.5, 
             label='Capital Allocation Line', alpha=0.7)
    
    plt.xlabel('Volatility (Standard Deviation)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.title('Efficient Frontier and Optimal Portfolios', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('example3_efficient_frontier.png', dpi=150)
    print("\nPlot saved as 'example3_efficient_frontier.png'")
    plt.close()


###############################################################################
# Example 4: Tracking Error Optimization
###############################################################################
def example_tracking_error_optimization():
    """
    Demonstrates portfolio optimization with tracking error constraints.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Tracking Error Optimization")
    print("="*70)
    
    # Define benchmark (e.g., market cap weighted)
    assets = ['Tech', 'Financials', 'Healthcare', 'Energy', 'Consumer']
    n_assets = len(assets)
    
    benchmark_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    
    # Expected returns
    expected_ret = np.array([0.12, 0.08, 0.10, 0.06, 0.09])
    
    # Volatilities
    volatilities = np.array([0.25, 0.18, 0.20, 0.30, 0.22])
    
    # Correlation matrix
    correlation_matrix = np.array([
        [1.00, 0.60, 0.50, 0.40, 0.55],
        [0.60, 1.00, 0.45, 0.35, 0.50],
        [0.50, 0.45, 1.00, 0.30, 0.48],
        [0.40, 0.35, 0.30, 1.00, 0.35],
        [0.55, 0.50, 0.48, 0.35, 1.00]
    ])
    
    # Create portfolio optimizer
    port_opt = CPortfolio_optimization(
        port_wts=benchmark_weights,
        correlation_matrix=correlation_matrix,
        expected_ret=expected_ret,
        vol=volatilities
    )
    
    # Define tactical range (max deviation from benchmark)
    tactical_range = np.array([0.10, 0.10, 0.10, 0.10, 0.10])  # +/- 10%
    
    # Optimize with tracking error constraint
    print("\nOptimizing portfolio with tracking error constraint...")
    optimal_weights, tracking_error = port_opt.asset_allocation_TEop(
        benchmark_weights, tactical_range
    )
    
    print(f"\nTracking Error: {tracking_error:.2%}")
    print(f"\nBenchmark vs Optimal Weights:")
    print(f"{'Asset':<15} {'Benchmark':>12} {'Optimal':>12} {'Active':>12}")
    print("-" * 55)
    for i, asset in enumerate(assets):
        active = optimal_weights[i] - benchmark_weights[i]
        print(f"{asset:<15} {benchmark_weights[i]:>11.2%} "
              f"{optimal_weights[i]:>11.2%} {active:>11.2%}")
    
    # Calculate portfolio metrics
    bmk_return = np.dot(benchmark_weights, expected_ret)
    opt_return = np.dot(optimal_weights, expected_ret)
    
    bmk_vol = np.sqrt(np.dot(np.dot(benchmark_weights, port_opt.exp_cov), 
                             benchmark_weights))
    opt_vol = np.sqrt(np.dot(np.dot(optimal_weights, port_opt.exp_cov), 
                             optimal_weights))
    
    print(f"\nPortfolio Metrics:")
    print(f"  Benchmark Return: {bmk_return:.2%}")
    print(f"  Optimal Return:   {opt_return:.2%}")
    print(f"  Active Return:    {opt_return - bmk_return:.2%}")
    print(f"\n  Benchmark Vol:    {bmk_vol:.2%}")
    print(f"  Optimal Vol:      {opt_vol:.2%}")
    print(f"  Tracking Error:   {tracking_error:.2%}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Weights comparison
    x = np.arange(len(assets))
    width = 0.35
    ax1.bar(x - width/2, benchmark_weights, width, label='Benchmark', alpha=0.8)
    ax1.bar(x + width/2, optimal_weights, width, label='Optimal', alpha=0.8)
    ax1.set_xlabel('Assets')
    ax1.set_ylabel('Weight')
    ax1.set_title('Benchmark vs Optimal Weights')
    ax1.set_xticks(x)
    ax1.set_xticklabels(assets, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Active positions
    active_positions = optimal_weights - benchmark_weights
    colors = ['green' if x > 0 else 'red' for x in active_positions]
    ax2.bar(x, active_positions, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Assets')
    ax2.set_ylabel('Active Position')
    ax2.set_title('Active Positions (Optimal - Benchmark)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(assets, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example4_tracking_error.png', dpi=150)
    print("\nPlot saved as 'example4_tracking_error.png'")
    plt.close()


###############################################################################
# Example 5: Black-Litterman Model
###############################################################################
def example_black_litterman():
    """
    Demonstrates the Black-Litterman model for incorporating views.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Black-Litterman Model")
    print("="*70)
    
    # Define assets
    assets = ['US_Stocks', 'EU_Stocks', 'Japan_Stocks', 'Bonds', 'Commodities']
    n_assets = len(assets)
    
    # Market equilibrium weights (market cap weighted)
    market_weights = np.array([0.40, 0.25, 0.15, 0.15, 0.05])
    
    # Covariance matrix (annual)
    cov_matrix = np.array([
        [0.0324, 0.0216, 0.0162, 0.0036, 0.0090],
        [0.0216, 0.0400, 0.0180, 0.0030, 0.0100],
        [0.0162, 0.0180, 0.0441, 0.0027, 0.0085],
        [0.0036, 0.0030, 0.0027, 0.0036, 0.0015],
        [0.0090, 0.0100, 0.0085, 0.0015, 0.0625]
    ])
    
    # Black-Litterman parameters
    risk_free_rate = 0.02
    tau = 0.025  # Scaling parameter for uncertainty in equilibrium
    lambda_param = 2.5  # Market risk aversion
    
    # Define views using Pick matrix
    # View 1: US Stocks will outperform EU Stocks by 2%
    # View 2: Japan Stocks will have absolute return of 8%
    # View 3: Commodities will underperform Bonds
    
    p_matrix = np.array([
        [1, -1,  0,  0,  0],   # US - EU
        [0,  0,  1,  0,  0],   # Japan absolute
        [0,  0,  0, -1,  1]    # Commodities - Bonds
    ])
    
    # Confidence levels for each view (0 to 1)
    confidence_levels = np.array([0.7, 0.5, 0.6])
    
    # Qualitative recommendations
    recommendations = ['buy', 'neutral', 'sell']
    
    # Create Black-Litterman model
    bl_model = CBlack_litterman(
        risk_free_rate=risk_free_rate,
        tau=tau,
        sigma=cov_matrix,
        p_matrix=p_matrix,
        lambda_param=lambda_param,
        port_wts_eq=market_weights,
        conf_level=confidence_levels,
        c_coef=1.0,
        reco_vector=recommendations
    )
    
    # Get equilibrium returns
    eq_returns = bl_model.get_eq_returns()
    print("\nEquilibrium Returns (implied by market):")
    for i, asset in enumerate(assets):
        print(f"  {asset:<15s}: {eq_returns[i]:6.2%}")
    
    # Get view returns
    view_returns = bl_model.get_view_returns()
    print("\nView Returns:")
    view_descriptions = [
        "US Stocks outperform EU Stocks",
        "Japan Stocks absolute return",
        "Commodities underperform Bonds"
    ]
    for i, desc in enumerate(view_descriptions):
        print(f"  View {i+1} ({desc}): {view_returns[i]:6.2%}")
    
    # Calculate Black-Litterman returns
    bl_returns = bl_model.get_bl_returns()
    print("\nBlack-Litterman Posterior Returns:")
    for i, asset in enumerate(assets):
        print(f"  {asset:<15s}: {bl_returns[i]:6.2%}")
    
    # Compare returns
    print("\nComparison of Returns:")
    print(f"{'Asset':<15} {'Equilibrium':>12} {'BL Posterior':>12} {'Difference':>12}")
    print("-" * 55)
    for i, asset in enumerate(assets):
        diff = bl_returns[i] - eq_returns[i]
        print(f"{asset:<15} {eq_returns[i]:>11.2%} "
              f"{bl_returns[i]:>11.2%} {diff:>11.2%}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Returns comparison
    x = np.arange(len(assets))
    width = 0.35
    ax1.bar(x - width/2, eq_returns, width, label='Equilibrium', alpha=0.8)
    ax1.bar(x + width/2, bl_returns, width, label='Black-Litterman', alpha=0.8)
    ax1.set_xlabel('Assets')
    ax1.set_ylabel('Expected Return')
    ax1.set_title('Equilibrium vs Black-Litterman Returns')
    ax1.set_xticks(x)
    ax1.set_xticklabels(assets, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Difference
    return_diff = bl_returns - eq_returns
    colors = ['green' if x > 0 else 'red' for x in return_diff]
    ax2.bar(x, return_diff, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Assets')
    ax2.set_ylabel('Return Adjustment')
    ax2.set_title('Impact of Views on Returns')
    ax2.set_xticks(x)
    ax2.set_xticklabels(assets, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example5_black_litterman.png', dpi=150)
    print("\nPlot saved as 'example5_black_litterman.png'")
    plt.close()


###############################################################################
# Example 6: Utility Functions
###############################################################################
def example_utility_functions():
    """
    Demonstrates different utility functions for portfolio evaluation.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Utility Functions")
    print("="*70)
    
    # Generate portfolio returns scenarios
    np.random.seed(123)
    n_scenarios = 10000
    
    # Portfolio A: Higher return, higher risk
    portfolio_a_returns = np.random.normal(0.10, 0.20, n_scenarios)
    
    # Portfolio B: Lower return, lower risk
    portfolio_b_returns = np.random.normal(0.06, 0.10, n_scenarios)
    
    # Test different utility functions and risk aversion levels
    utility_types = ['quadratic', 'exponential', 'power']
    risk_aversions = [1.0, 2.0, 5.0]
    
    print("\nUtility Analysis for Two Portfolios:")
    print(f"\nPortfolio A: Mean={portfolio_a_returns.mean():.2%}, "
          f"Std={portfolio_a_returns.std():.2%}")
    print(f"Portfolio B: Mean={portfolio_b_returns.mean():.2%}, "
          f"Std={portfolio_b_returns.std():.2%}")
    
    results = []
    
    for util_type in utility_types:
        print(f"\n{util_type.upper()} Utility:")
        print(f"{'Risk Aversion':<15} {'Utility A':>12} {'Utility B':>12} "
              f"{'CE A':>12} {'CE B':>12} {'Preferred':>12}")
        print("-" * 80)
        
        for gamma in risk_aversions:
            utility_func = CUtility(utility_type=util_type, risk_aversion=gamma)
            
            util_a = utility_func.calculate_utility(portfolio_a_returns)
            util_b = utility_func.calculate_utility(portfolio_b_returns)
            
            ce_a = utility_func.certainty_equivalent(portfolio_a_returns)
            ce_b = utility_func.certainty_equivalent(portfolio_b_returns)
            
            preferred = 'Portfolio A' if util_a > util_b else 'Portfolio B'
            
            print(f"{gamma:<15.1f} {util_a:>12.4f} {util_b:>12.4f} "
                  f"{ce_a:>11.2%} {ce_b:>11.2%} {preferred:>12}")
            
            results.append({
                'utility_type': util_type,
                'risk_aversion': gamma,
                'utility_a': util_a,
                'utility_b': util_b,
                'ce_a': ce_a,
                'ce_b': ce_b,
                'preferred': preferred
            })
    
    # Visualize utility preferences
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Return distributions
    axes[0, 0].hist(portfolio_a_returns, bins=50, alpha=0.6, 
                    label='Portfolio A', color='blue', density=True)
    axes[0, 0].hist(portfolio_b_returns, bins=50, alpha=0.6, 
                    label='Portfolio B', color='red', density=True)
    axes[0, 0].axvline(portfolio_a_returns.mean(), color='blue', 
                      linestyle='--', linewidth=2)
    axes[0, 0].axvline(portfolio_b_returns.mean(), color='red', 
                      linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Return')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Return Distributions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Utility comparison by risk aversion (quadratic)
    quad_results = [r for r in results if r['utility_type'] == 'quadratic']
    gammas = [r['risk_aversion'] for r in quad_results]
    util_a_quad = [r['utility_a'] for r in quad_results]
    util_b_quad = [r['utility_b'] for r in quad_results]
    
    axes[0, 1].plot(gammas, util_a_quad, 'o-', label='Portfolio A', linewidth=2)
    axes[0, 1].plot(gammas, util_b_quad, 's-', label='Portfolio B', linewidth=2)
    axes[0, 1].set_xlabel('Risk Aversion (γ)')
    axes[0, 1].set_ylabel('Utility')
    axes[0, 1].set_title('Quadratic Utility by Risk Aversion')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Certainty Equivalent comparison
    ce_a_quad = [r['ce_a'] for r in quad_results]
    ce_b_quad = [r['ce_b'] for r in quad_results]
    
    axes[1, 0].plot(gammas, ce_a_quad, 'o-', label='Portfolio A', linewidth=2)
    axes[1, 0].plot(gammas, ce_b_quad, 's-', label='Portfolio B', linewidth=2)
    axes[1, 0].set_xlabel('Risk Aversion (γ)')
    axes[1, 0].set_ylabel('Certainty Equivalent')
    axes[1, 0].set_title('Certainty Equivalent by Risk Aversion')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Utility comparison across utility types (γ=2)
    gamma_2_results = [r for r in results if r['risk_aversion'] == 2.0]
    util_types_labels = [r['utility_type'].capitalize() for r in gamma_2_results]
    util_a_types = [r['utility_a'] for r in gamma_2_results]
    util_b_types = [r['utility_b'] for r in gamma_2_results]
    
    x = np.arange(len(util_types_labels))
    width = 0.35
    axes[1, 1].bar(x - width/2, util_a_types, width, label='Portfolio A', alpha=0.8)
    axes[1, 1].bar(x + width/2, util_b_types, width, label='Portfolio B', alpha=0.8)
    axes[1, 1].set_xlabel('Utility Type')
    axes[1, 1].set_ylabel('Utility Value')
    axes[1, 1].set_title('Utility Comparison (γ=2.0)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(util_types_labels)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example6_utility_functions.png', dpi=150)
    print("\nPlot saved as 'example6_utility_functions.png'")
    plt.close()
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("\n1. As risk aversion increases, investors prefer safer portfolios")
    print("2. Certainty equivalent decreases with risk aversion")
    print("3. Different utility functions capture different risk preferences")
    print("4. Quadratic utility: Simple mean-variance trade-off")
    print("5. Exponential utility: Constant absolute risk aversion (CARA)")
    print("6. Power utility: Constant relative risk aversion (CRRA)")


###############################################################################
# Example 7: Complete Portfolio Construction Workflow
###############################################################################
def example_complete_workflow():
    """
    Demonstrates a complete portfolio construction workflow combining
    multiple techniques.
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Complete Portfolio Construction Workflow")
    print("="*70)
    
    # Step 1: Define investment universe
    print("\nStep 1: Define Investment Universe")
    print("-" * 50)
    assets = ['US_Equity', 'EU_Equity', 'EM_Equity', 'Gov_Bonds', 
              'Corp_Bonds', 'Real_Estate', 'Commodities']
    n_assets = len(assets)
    print(f"Assets: {', '.join(assets)}")
    
    # Step 2: Generate correlation matrix using copulas
    print("\nStep 2: Generate Correlated Returns")
    print("-" * 50)
    
    # Define correlation structure
    corr_matrix = np.array([
        [1.00, 0.85, 0.70, 0.15, 0.25, 0.65, 0.40],
        [0.85, 1.00, 0.75, 0.20, 0.30, 0.60, 0.35],
        [0.70, 0.75, 1.00, 0.10, 0.20, 0.55, 0.45],
        [0.15, 0.20, 0.10, 1.00, 0.75, 0.25, 0.05],
        [0.25, 0.30, 0.20, 0.75, 1.00, 0.30, 0.10],
        [0.65, 0.60, 0.55, 0.25, 0.30, 1.00, 0.35],
        [0.40, 0.35, 0.45, 0.05, 0.10, 0.35, 1.00]
    ])
    
    copula = CCopulas(corr_matrix, nsim=5000)
    
    # Define marginal distributions
    marginal_params = [
        {'mean': 0.10, 'std': 0.18},  # US Equity
        {'mean': 0.09, 'std': 0.20},  # EU Equity
        {'mean': 0.12, 'std': 0.25},  # EM Equity
        {'mean': 0.03, 'std': 0.05},  # Gov Bonds
        {'mean': 0.05, 'std': 0.08},  # Corp Bonds
        {'mean': 0.08, 'std': 0.15},  # Real Estate
        {'mean': 0.06, 'std': 0.22}   # Commodities
    ]
    
    returns = copula.get_correlated_returns(marginal_params)
    
    expected_ret = np.array([p['mean'] for p in marginal_params])
    volatilities = np.array([p['std'] for p in marginal_params])
    
    print("Asset Statistics:")
    for i, asset in enumerate(assets):
        print(f"  {asset:<15}: E[R]={expected_ret[i]:6.2%}, σ={volatilities[i]:5.2%}")
    
    # Step 3: Define benchmark
    print("\nStep 3: Define Strategic Benchmark")
    print("-" * 50)
    benchmark_weights = np.array([0.25, 0.15, 0.10, 0.20, 0.10, 0.10, 0.10])
    print("Benchmark weights:")
    for i, asset in enumerate(assets):
        print(f"  {asset:<15}: {benchmark_weights[i]:6.2%}")
    
    # Step 4: Apply Black-Litterman with views
    print("\nStep 4: Incorporate Market Views (Black-Litterman)")
    print("-" * 50)
    
    # Construct covariance matrix
    vol_matrix = np.diag(volatilities)
    cov_matrix = np.dot(np.dot(vol_matrix, corr_matrix), vol_matrix)
    
    # Define views
    # View 1: US Equity outperforms EU Equity
    # View 2: Emerging Markets will have high returns
    # View 3: Real Estate outperforms average
    p_matrix = np.array([
        [1, -1,  0,  0,  0,  0,  0],   # US - EU
        [0,  0,  1,  0,  0,  0,  0],   # EM absolute
        [0,  0,  0,  0,  0,  1,  0]    # Real Estate absolute
    ])
    
    recommendations = ['buy', 'strong_buy', 'buy']
    confidence_levels = np.array([0.6, 0.7, 0.5])
    
    bl_model = CBlack_litterman(
        risk_free_rate=0.02,
        tau=0.025,
        sigma=cov_matrix,
        p_matrix=p_matrix,
        lambda_param=2.5,
        port_wts_eq=benchmark_weights,
        conf_level=confidence_levels,
        c_coef=1.0,
        reco_vector=recommendations
    )
    
    bl_returns = bl_model.get_bl_returns()
    eq_returns = bl_model.get_eq_returns()
    
    print("Returns Adjustment:")
    print(f"{'Asset':<15} {'Equilibrium':>12} {'BL Returns':>12} {'Adjustment':>12}")
    print("-" * 55)
    for i, asset in enumerate(assets):
        adj = bl_returns[i] - eq_returns[i]
        print(f"{asset:<15} {eq_returns[i]:>11.2%} {bl_returns[i]:>11.2%} {adj:>11.2%}")
    
    # Step 5: Optimize portfolio with constraints
    print("\nStep 5: Portfolio Optimization")
    print("-" * 50)
    
    port_opt = CPortfolio_optimization(
        port_wts=benchmark_weights,
        correlation_matrix=corr_matrix,
        expected_ret=bl_returns,  # Use BL returns
        vol=volatilities
    )
    
    # Define constraints
    x_min = np.array([0.05, 0.05, 0.00, 0.10, 0.05, 0.00, 0.00])  # Minimum weights
    x_max = np.array([0.40, 0.30, 0.25, 0.35, 0.25, 0.25, 0.20])  # Maximum weights
    
    # Get optimal portfolios
    portfolios = port_opt.mean_variance_opt(x_min, x_max, num_port=20)
    
    # Calculate metrics for each portfolio
    portfolio_returns = []
    portfolio_vols = []
    sharpe_ratios = []
    
    for weights in portfolios:
        ret = np.dot(weights, bl_returns)
        vol = np.sqrt(np.dot(np.dot(weights, port_opt.exp_cov), weights))
        sharpe = (ret - 0.02) / vol
        portfolio_returns.append(ret)
        portfolio_vols.append(vol)
        sharpe_ratios.append(sharpe)
    
    # Select portfolio with maximum Sharpe ratio
    max_sharpe_idx = np.argmax(sharpe_ratios)
    optimal_weights = portfolios[max_sharpe_idx]
    
    print(f"\nOptimal Portfolio (Max Sharpe Ratio = {sharpe_ratios[max_sharpe_idx]:.2f}):")
    print(f"  Expected Return: {portfolio_returns[max_sharpe_idx]:.2%}")
    print(f"  Volatility:      {portfolio_vols[max_sharpe_idx]:.2%}")
    print(f"\nWeights:")
    for i, asset in enumerate(assets):
        print(f"  {asset:<15}: {optimal_weights[i]:6.2%}")
    
    # Step 6: Calculate tracking error vs benchmark
    print("\nStep 6: Risk Analysis")
    print("-" * 50)
    
    tracking_error = port_opt.tracking_error_ex_ante(benchmark_weights, optimal_weights)
    
    bmk_return = np.dot(benchmark_weights, bl_returns)
    bmk_vol = np.sqrt(np.dot(np.dot(benchmark_weights, port_opt.exp_cov), benchmark_weights))
    
    print(f"Benchmark Metrics:")
    print(f"  Return:     {bmk_return:.2%}")
    print(f"  Volatility: {bmk_vol:.2%}")
    print(f"  Sharpe:     {(bmk_return - 0.02) / bmk_vol:.2f}")
    
    print(f"\nOptimal Portfolio Metrics:")
    print(f"  Return:          {portfolio_returns[max_sharpe_idx]:.2%}")
    print(f"  Volatility:      {portfolio_vols[max_sharpe_idx]:.2%}")
    print(f"  Sharpe:          {sharpe_ratios[max_sharpe_idx]:.2f}")
    print(f"  Tracking Error:  {tracking_error:.2%}")
    print(f"  Active Return:   {portfolio_returns[max_sharpe_idx] - bmk_return:.2%}")
    print(f"  Information Ratio: {(portfolio_returns[max_sharpe_idx] - bmk_return) / tracking_error:.2f}")
    
    # Step 7: Evaluate with utility function
    print("\nStep 7: Utility Analysis")
    print("-" * 50)
    
    utility_func = CUtility(utility_type='quadratic', risk_aversion=2.5)
    
    # Simulate portfolio returns
    optimal_returns_sim = np.dot(returns, optimal_weights)
    bmk_returns_sim = np.dot(returns, benchmark_weights)
    
    utility_optimal = utility_func.calculate_utility(optimal_returns_sim)
    utility_bmk = utility_func.calculate_utility(bmk_returns_sim)
    
    ce_optimal = utility_func.certainty_equivalent(optimal_returns_sim)
    ce_bmk = utility_func.certainty_equivalent(bmk_returns_sim)
    
    print(f"Utility Comparison (Quadratic, γ=2.5):")
    print(f"  Benchmark Utility:  {utility_bmk:.4f}")
    print(f"  Optimal Utility:    {utility_optimal:.4f}")
    print(f"  Utility Gain:       {utility_optimal - utility_bmk:.4f}")
    print(f"\n  Benchmark CE:       {ce_bmk:.2%}")
    print(f"  Optimal CE:         {ce_optimal:.2%}")
    print(f"  CE Improvement:     {ce_optimal - ce_bmk:.2%}")
    
    # Visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Weight comparison
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(assets))
    width = 0.35
    ax1.bar(x - width/2, benchmark_weights, width, label='Benchmark', alpha=0.8)
    ax1.bar(x + width/2, optimal_weights, width, label='Optimal', alpha=0.8)
    ax1.set_xlabel('Assets')
    ax1.set_ylabel('Weight')
    ax1.set_title('Portfolio Weights Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(assets, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Active positions
    ax2 = fig.add_subplot(gs[0, 2])
    active_pos = optimal_weights - benchmark_weights
    colors = ['green' if x > 0 else 'red' for x in active_pos]
    ax2.barh(assets, active_pos, color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Active Position')
    ax2.set_title('Active Positions', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Efficient frontier
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(portfolio_vols, portfolio_returns, 'b-', linewidth=2, label='Efficient Frontier')
    ax3.scatter(portfolio_vols[max_sharpe_idx], portfolio_returns[max_sharpe_idx],
                c='green', s=200, marker='*', label='Optimal Portfolio', zorder=5)
    ax3.scatter(bmk_vol, bmk_return, c='red', s=150, marker='s', 
                label='Benchmark', zorder=5)
    
    # Individual assets
    ax3.scatter(volatilities, expected_ret, c='orange', s=80, 
                marker='o', alpha=0.6, label='Individual Assets')
    
    # Capital allocation line
    cal_x = np.linspace(0, max(portfolio_vols), 100)
    cal_y = 0.02 + sharpe_ratios[max_sharpe_idx] * cal_x
    ax3.plot(cal_x, cal_y, 'g--', linewidth=1.5, alpha=0.7, label='CAL')
    
    ax3.set_xlabel('Volatility (Risk)', fontsize=11)
    ax3.set_ylabel('Expected Return', fontsize=11)
    ax3.set_title('Efficient Frontier with Optimal and Benchmark Portfolios', 
                  fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. Return distributions
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(optimal_returns_sim, bins=50, alpha=0.6, label='Optimal', 
             color='green', density=True)
    ax4.hist(bmk_returns_sim, bins=50, alpha=0.6, label='Benchmark', 
             color='red', density=True)
    ax4.axvline(optimal_returns_sim.mean(), color='green', linestyle='--', linewidth=2)
    ax4.axvline(bmk_returns_sim.mean(), color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Return')
    ax4.set_ylabel('Density')
    ax4.set_title('Return Distributions', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Risk metrics
    ax5 = fig.add_subplot(gs[2, 1])
    metrics = ['Return', 'Volatility', 'Sharpe Ratio']
    bmk_metrics = [bmk_return, bmk_vol, (bmk_return - 0.02) / bmk_vol]
    opt_metrics = [portfolio_returns[max_sharpe_idx], portfolio_vols[max_sharpe_idx], 
                   sharpe_ratios[max_sharpe_idx]]
    
    x_metrics = np.arange(len(metrics))
    width = 0.35
    ax5.bar(x_metrics - width/2, bmk_metrics, width, label='Benchmark', alpha=0.8)
    ax5.bar(x_metrics + width/2, opt_metrics, width, label='Optimal', alpha=0.8)
    ax5.set_ylabel('Value')
    ax5.set_title('Performance Metrics', fontweight='bold')
    ax5.set_xticks(x_metrics)
    ax5.set_xticklabels(metrics, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Correlation heatmap
    ax6 = fig.add_subplot(gs[2, 2])
    im = ax6.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax6.set_xticks(range(len(assets)))
    ax6.set_yticks(range(len(assets)))
    ax6.set_xticklabels([a[:6] for a in assets], rotation=45, ha='right', fontsize=8)
    ax6.set_yticklabels([a[:6] for a in assets], fontsize=8)
    ax6.set_title('Correlation Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    
    plt.savefig('example7_complete_workflow.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'example7_complete_workflow.png'")
    plt.close()
    
    print("\n" + "="*70)
    print("WORKFLOW SUMMARY")
    print("="*70)
    print("✓ Defined investment universe with 7 asset classes")
    print("✓ Generated correlated returns using copulas")
    print("✓ Applied Black-Litterman to incorporate market views")
    print("✓ Optimized portfolio with constraints")
    print("✓ Analyzed tracking error vs benchmark")
    print("✓ Evaluated portfolio using utility functions")
    print(f"✓ Achieved Sharpe ratio improvement: {sharpe_ratios[max_sharpe_idx] - (bmk_return - 0.02) / bmk_vol:.2f}")
    print(f"✓ Information Ratio: {(portfolio_returns[max_sharpe_idx] - bmk_return) / tracking_error:.2f}")


###############################################################################
# Main execution
###############################################################################
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ASSET ALLOCATION FRAMEWORK - EXAMPLES AND DEMONSTRATIONS")
    print("="*70)
    print("\nThis script demonstrates the main functionalities of the")
    print("asset allocation framework including:")
    print("  - Asset calibration and simulation")
    print("  - Copula-based correlation modeling")
    print("  - Mean-variance optimization")
    print("  - Tracking error optimization")
    print("  - Black-Litterman model")
    print("  - Utility function analysis")
    print("  - Complete portfolio construction workflow")
    print("\n" + "="*70)
    
    # Run all examples
    try:
        example_asset_calibration()
        example_copulas()
        example_portfolio_optimization()
        example_tracking_error_optimization()
        example_black_litterman()
        example_utility_functions()
        example_complete_workflow()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated files:")
        print("  - example1_asset_simulation.png")
        print("  - example2_copulas.png")
        print("  - example3_efficient_frontier.png")
        print("  - example4_tracking_error.png")
        print("  - example5_black_litterman.png")
        print("  - example6_utility_functions.png")
        print("  - example7_complete_workflow.png")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()