"""
ML Portfolio Enhancement - Comprehensive Examples
==================================================

This script demonstrates all features of the ml_portfolio_enhancement module
with real-world examples and use cases.

Author: Fernando Ruiz
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the ML enhancement module
try:
    from ml_portfolio_enhancement import MLPortfolioEnhancer
    ML_AVAILABLE = True
except ImportError:
    print("Warning: ml_portfolio_enhancement.py not found in current directory")
    print("Please ensure the file is in the same folder as this script")
    ML_AVAILABLE = False
    exit(1)


###############################################################################
# Helper Functions
###############################################################################

def generate_synthetic_data(n_days=1000, n_assets=5, seed=42):
    """
    Generate synthetic market data for testing.
    
    Args:
        n_days (int): Number of trading days
        n_assets (int): Number of assets
        seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Historical returns
    """
    np.random.seed(seed)
    
    # Asset names
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(n_days * 1.5))  # Account for weekends
    dates = pd.date_range(start=start_date, end=end_date, freq='B')[:n_days]
    
    # Base parameters
    base_returns = np.random.uniform(0.0001, 0.0008, n_assets)  # Daily returns
    base_vols = np.random.uniform(0.01, 0.03, n_assets)  # Daily volatilities
    
    # Generate returns with some structure
    returns_data = {}
    
    for i, asset in enumerate(asset_names):
        # Random walk with drift
        returns = np.random.normal(base_returns[i], base_vols[i], n_days)
        
        # Add momentum
        for j in range(1, n_days):
            returns[j] += returns[j-1] * 0.05
        
        # Add mean reversion
        ma = pd.Series(returns).rolling(20).mean()
        for j in range(20, n_days):
            returns[j] -= (returns[j] - ma[j]) * 0.1
        
        # Add some regime changes (volatility clustering)
        high_vol_periods = np.random.choice(n_days, size=int(n_days * 0.2), replace=False)
        returns[high_vol_periods] *= 2
        
        returns_data[asset] = returns
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    return returns_df


def plot_results(results_dict, title="Results"):
    """Helper function to plot results nicely."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # You can customize this based on what's in results_dict
    
    plt.tight_layout()
    return fig


###############################################################################
# Example 1: Basic ML Return Prediction
###############################################################################

def example_1_return_prediction():
    """
    Example 1: Predict future returns using different ML models.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: ML Return Prediction")
    print("="*80)
    
    # Generate synthetic data
    print("\n[Step 1] Generating synthetic market data...")
    returns_df = generate_synthetic_data(n_days=500, n_assets=5)
    print(f"Generated {len(returns_df)} days of data for {len(returns_df.columns)} assets")
    
    # Create ML enhancer
    print("\n[Step 2] Initializing ML Portfolio Enhancer...")
    enhancer = MLPortfolioEnhancer()
    
    # Compare different prediction methods
    methods = ['random_forest', 'gradient_boosting', 'ensemble']
    
    print("\n[Step 3] Comparing prediction methods...\n")
    
    results = {}
    for method in methods:
        print(f"Training {method} model...")
        predictions = enhancer.predict_returns(
            returns_df,
            method=method,
            forecast_days=21  # 1 month ahead
        )
        
        # Annualize predictions
        predictions_annual = predictions * 252
        results[method] = predictions_annual
        
        print(f"\n{method.upper()} Predictions (Annual):")
        for asset, pred in predictions_annual.items():
            print(f"  {asset}: {pred:>7.2%}")
    
    # Compare with historical average
    historical_avg = returns_df.mean() * 252
    
    print("\n" + "-"*80)
    print("COMPARISON: ML vs Historical Average")
    print("-"*80)
    
    comparison_df = pd.DataFrame({
        'Historical_Avg': historical_avg,
        'Random_Forest': results['random_forest'],
        'Gradient_Boosting': results['gradient_boosting'],
        'Ensemble': results['ensemble']
    })
    
    print(comparison_df.to_string(float_format=lambda x: f'{x:.2%}'))
    
    # Visualization
    print("\n[Step 4] Creating visualization...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(returns_df.columns))
    width = 0.2
    
    ax.bar(x - 1.5*width, historical_avg, width, label='Historical', alpha=0.8)
    ax.bar(x - 0.5*width, results['random_forest'], width, label='Random Forest', alpha=0.8)
    ax.bar(x + 0.5*width, results['gradient_boosting'], width, label='Gradient Boosting', alpha=0.8)
    ax.bar(x + 1.5*width, results['ensemble'], width, label='Ensemble', alpha=0.8)
    
    ax.set_xlabel('Assets', fontsize=12)
    ax.set_ylabel('Annual Return', fontsize=12)
    ax.set_title('Return Predictions: ML vs Historical Average', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(returns_df.columns)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    plt.tight_layout()
    plt.savefig('example1_ml_predictions.png', dpi=150, bbox_inches='tight')
    print("Chart saved as 'example1_ml_predictions.png'")
    plt.close()
    
    return results


###############################################################################
# Example 2: Robust Covariance Estimation
###############################################################################

def example_2_robust_covariance():
    """
    Example 2: Estimate covariance matrix with noise reduction.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Robust Covariance Estimation")
    print("="*80)
    
    # Generate data
    print("\n[Step 1] Generating synthetic data...")
    returns_df = generate_synthetic_data(n_days=500, n_assets=5)
    
    # Create enhancer
    enhancer = MLPortfolioEnhancer()
    
    # Standard covariance
    print("\n[Step 2] Calculating standard covariance...")
    standard_cov = returns_df.cov() * 252  # Annualized
    
    # PCA-based robust covariance
    print("\n[Step 3] Estimating robust covariance with PCA...")
    robust_cov, explained_var = enhancer.estimate_robust_covariance(
        returns_df,
        method='pca',
        n_components=3
    )
    robust_cov = robust_cov * 252  # Annualized
    
    print(f"\nVariance explained by top 3 components: {explained_var.sum():.2%}")
    print("Individual components:")
    for i, var in enumerate(explained_var):
        print(f"  Component {i+1}: {var:.2%}")
    
    # Compare volatilities
    print("\n" + "-"*80)
    print("VOLATILITY COMPARISON")
    print("-"*80)
    
    standard_vols = np.sqrt(np.diag(standard_cov))
    robust_vols = np.sqrt(np.diag(robust_cov))
    
    vol_comparison = pd.DataFrame({
        'Standard': standard_vols,
        'Robust (PCA)': robust_vols,
        'Difference': robust_vols - standard_vols
    }, index=returns_df.columns)
    
    print(vol_comparison.to_string(float_format=lambda x: f'{x:.2%}'))
    
    # Visualize correlation matrices
    print("\n[Step 4] Creating correlation heatmaps...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Standard correlation
    standard_corr = standard_cov / np.outer(standard_vols, standard_vols)
    im1 = ax1.imshow(standard_corr, cmap='RdYlGn', vmin=-1, vmax=1)
    ax1.set_title('Standard Correlation Matrix', fontweight='bold')
    ax1.set_xticks(range(len(returns_df.columns)))
    ax1.set_yticks(range(len(returns_df.columns)))
    ax1.set_xticklabels(returns_df.columns, rotation=45, ha='right')
    ax1.set_yticklabels(returns_df.columns)
    
    for i in range(len(returns_df.columns)):
        for j in range(len(returns_df.columns)):
            ax1.text(j, i, f'{standard_corr[i, j]:.2f}',
                    ha='center', va='center', color='black', fontsize=10)
    
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Robust correlation
    robust_corr = robust_cov / np.outer(robust_vols, robust_vols)
    im2 = ax2.imshow(robust_corr, cmap='RdYlGn', vmin=-1, vmax=1)
    ax2.set_title('Robust Correlation Matrix (PCA)', fontweight='bold')
    ax2.set_xticks(range(len(returns_df.columns)))
    ax2.set_yticks(range(len(returns_df.columns)))
    ax2.set_xticklabels(returns_df.columns, rotation=45, ha='right')
    ax2.set_yticklabels(returns_df.columns)
    
    for i in range(len(returns_df.columns)):
        for j in range(len(returns_df.columns)):
            ax2.text(j, i, f'{robust_corr[i, j]:.2f}',
                    ha='center', va='center', color='black', fontsize=10)
    
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('example2_robust_covariance.png', dpi=150, bbox_inches='tight')
    print("Chart saved as 'example2_robust_covariance.png'")
    plt.close()
    
    return {'standard_cov': standard_cov, 'robust_cov': robust_cov}


###############################################################################
# Example 3: Market Regime Detection
###############################################################################

def example_3_regime_detection():
    """
    Example 3: Detect different market regimes automatically.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Market Regime Detection")
    print("="*80)
    
    # Generate data
    print("\n[Step 1] Generating synthetic data...")
    returns_df = generate_synthetic_data(n_days=1000, n_assets=5)
    
    # Create enhancer
    enhancer = MLPortfolioEnhancer()
    
    # Detect regimes
    print("\n[Step 2] Detecting market regimes...")
    regimes, regime_stats = enhancer.detect_market_regimes(
        returns_df,
        n_regimes=3
    )
    
    print(f"\nIdentified {len(regime_stats)} distinct market regimes")
    
    # Display regime statistics
    print("\n" + "-"*80)
    print("REGIME STATISTICS")
    print("-"*80)
    
    regime_df = pd.DataFrame({
        'Regime': list(regime_stats.keys()),
        'Days': [stats['n_days'] for stats in regime_stats.values()],
        'Percentage': [f"{stats['pct_days']:.1%}" for stats in regime_stats.values()],
        'Avg Return': [f"{stats['avg_return']:.2%}" for stats in regime_stats.values()],
        'Volatility': [f"{stats['volatility']:.2%}" for stats in regime_stats.values()]
    })
    
    print(regime_df.to_string(index=False))
    
    # Classify regimes
    print("\n[Step 3] Classifying regimes...")
    regime_names = {}
    for regime, stats in regime_stats.items():
        if stats['avg_return'] > 0.1 and stats['volatility'] < 0.15:
            regime_names[regime] = 'Bull Market (Low Vol)'
        elif stats['avg_return'] > 0.05:
            regime_names[regime] = 'Bull Market (High Vol)'
        elif stats['avg_return'] < 0:
            regime_names[regime] = 'Bear Market'
        else:
            regime_names[regime] = 'Sideways Market'
    
    print("\nRegime Classification:")
    for regime, name in regime_names.items():
        print(f"  Regime {regime}: {name}")
    
    # Visualize regimes over time
    print("\n[Step 4] Creating regime timeline...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Portfolio value over time
    cumulative_returns = (1 + returns_df.mean(axis=1)).cumprod()
    ax1.plot(returns_df.index, cumulative_returns, linewidth=2, color='black', label='Portfolio Value')
    
    # Color background by regime
    regime_colors = {0: 'lightgreen', 1: 'lightblue', 2: 'lightcoral'}
    current_regime = regimes[0]
    start_idx = 0
    
    for i in range(1, len(regimes)):
        if regimes[i] != current_regime or i == len(regimes) - 1:
            ax1.axvspan(returns_df.index[start_idx], returns_df.index[i-1],
                       alpha=0.3, color=regime_colors.get(current_regime, 'gray'),
                       label=f'Regime {current_regime}' if start_idx == 0 else '')
            ax2.axvspan(returns_df.index[start_idx], returns_df.index[i-1],
                       alpha=0.3, color=regime_colors.get(current_regime, 'gray'))
            current_regime = regimes[i]
            start_idx = i
    
    ax1.set_ylabel('Cumulative Value', fontsize=12)
    ax1.set_title('Portfolio Performance with Market Regimes', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Volatility over time
    rolling_vol = returns_df.mean(axis=1).rolling(21).std() * np.sqrt(252)
    ax2.plot(returns_df.index, rolling_vol, linewidth=2, color='red')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Rolling Volatility (21d)', fontsize=12)
    ax2.set_title('Volatility Across Regimes', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    plt.tight_layout()
    plt.savefig('example3_regime_detection.png', dpi=150, bbox_inches='tight')
    print("Chart saved as 'example3_regime_detection.png'")
    plt.close()
    
    return {'regimes': regimes, 'stats': regime_stats}


###############################################################################
# Example 4: Anomaly Detection
###############################################################################

def example_4_anomaly_detection():
    """
    Example 4: Identify and filter anomalous trading days.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Anomaly Detection")
    print("="*80)
    
    # Generate data
    print("\n[Step 1] Generating synthetic data...")
    returns_df = generate_synthetic_data(n_days=500, n_assets=5)
    
    # Add some artificial anomalies
    print("\n[Step 2] Injecting artificial anomalies...")
    anomaly_indices = np.random.choice(len(returns_df), size=15, replace=False)
    for idx in anomaly_indices:
        returns_df.iloc[idx] *= 5  # Make these days extreme
    
    print(f"Injected {len(anomaly_indices)} artificial anomalies")
    
    # Create enhancer
    enhancer = MLPortfolioEnhancer()
    
    # Detect anomalies
    print("\n[Step 3] Detecting anomalies...")
    clean_returns, anomaly_dates = enhancer.filter_anomalies(
        returns_df,
        contamination=0.05  # Expect 5% anomalies
    )
    
    print(f"\nDetected {len(anomaly_dates)} anomalous days")
    print(f"Removed {len(returns_df) - len(clean_returns)} days from analysis")
    
    # Compare statistics
    print("\n" + "-"*80)
    print("STATISTICS COMPARISON")
    print("-"*80)
    
    print("\nWith Anomalies:")
    print(f"  Mean Return:  {returns_df.mean().mean() * 252:>7.2%}")
    print(f"  Volatility:   {returns_df.std().mean() * np.sqrt(252):>7.2%}")
    print(f"  Min Return:   {returns_df.min().min():>7.2%}")
    print(f"  Max Return:   {returns_df.max().max():>7.2%}")
    
    print("\nWithout Anomalies (Clean):")
    print(f"  Mean Return:  {clean_returns.mean().mean() * 252:>7.2%}")
    print(f"  Volatility:   {clean_returns.std().mean() * np.sqrt(252):>7.2%}")
    print(f"  Min Return:   {clean_returns.min().min():>7.2%}")
    print(f"  Max Return:   {clean_returns.max().max():>7.2%}")
    
    # Show some detected anomalies
    if len(anomaly_dates) > 0:
        print(f"\nSample of detected anomaly dates:")
        for date in anomaly_dates[:5]:
            returns_on_date = returns_df.loc[date]
            print(f"  {date.strftime('%Y-%m-%d')}: {returns_on_date.to_dict()}")
    
    # Visualize
    print("\n[Step 4] Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Returns distribution (with vs without anomalies)
    ax1 = axes[0, 0]
    all_returns = returns_df.values.flatten()
    clean_all_returns = clean_returns.values.flatten()
    
    ax1.hist(all_returns, bins=50, alpha=0.5, label='With Anomalies', color='red', density=True)
    ax1.hist(clean_all_returns, bins=50, alpha=0.5, label='Clean Data', color='green', density=True)
    ax1.set_xlabel('Daily Return')
    ax1.set_ylabel('Density')
    ax1.set_title('Return Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time series with anomalies highlighted
    ax2 = axes[0, 1]
    portfolio_returns = returns_df.mean(axis=1)
    cumulative = (1 + portfolio_returns).cumprod()
    
    ax2.plot(returns_df.index, cumulative, linewidth=1.5, color='blue', label='Portfolio')
    
    # Highlight anomalies
    for date in anomaly_dates:
        if date in returns_df.index:
            idx = returns_df.index.get_loc(date)
            ax2.scatter(date, cumulative.iloc[idx], color='red', s=50, zorder=5)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Value')
    ax2.set_title('Portfolio with Anomalies Highlighted', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volatility comparison
    ax3 = axes[1, 0]
    assets = returns_df.columns
    original_vols = returns_df.std() * np.sqrt(252)
    clean_vols = clean_returns.std() * np.sqrt(252)
    
    x = np.arange(len(assets))
    width = 0.35
    
    ax3.bar(x - width/2, original_vols, width, label='With Anomalies', alpha=0.8, color='red')
    ax3.bar(x + width/2, clean_vols, width, label='Clean Data', alpha=0.8, color='green')
    ax3.set_xlabel('Assets')
    ax3.set_ylabel('Annual Volatility')
    ax3.set_title('Volatility Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(assets, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Plot 4: Detection accuracy
    ax4 = axes[1, 1]
    
    # Count true vs false positives (approximation)
    detected_anomalies = set([d.strftime('%Y-%m-%d') for d in anomaly_dates])
    injected_anomalies = set([returns_df.index[i].strftime('%Y-%m-%d') for i in anomaly_indices])
    
    true_positives = len(detected_anomalies & injected_anomalies)
    false_positives = len(detected_anomalies - injected_anomalies)
    false_negatives = len(injected_anomalies - detected_anomalies)
    
    categories = ['True\nPositives', 'False\nPositives', 'False\nNegatives']
    values = [true_positives, false_positives, false_negatives]
    colors = ['green', 'orange', 'red']
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.7)
    ax4.set_ylabel('Count')
    ax4.set_title('Anomaly Detection Performance', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('example4_anomaly_detection.png', dpi=150, bbox_inches='tight')
    print("Chart saved as 'example4_anomaly_detection.png'")
    plt.close()
    
    return {'clean_returns': clean_returns, 'anomaly_dates': anomaly_dates}


###############################################################################
# Example 5: Complete ML-Enhanced Optimization
###############################################################################

def example_5_complete_optimization():
    """
    Example 5: Full ML-enhanced portfolio optimization pipeline.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Complete ML-Enhanced Optimization")
    print("="*80)
    
    # Generate data
    print("\n[Step 1] Generating synthetic data...")
    returns_df = generate_synthetic_data(n_days=750, n_assets=5)
    
    # Create enhancer
    enhancer = MLPortfolioEnhancer()
    
    # Run complete optimization
    print("\n[Step 2] Running ML-enhanced optimization...")
    print("This includes:")
    print("  - Anomaly filtering")
    print("  - ML return prediction")
    print("  - Robust covariance estimation")
    print("  - Portfolio optimization")
    
    result = enhancer.optimize_with_ml_inputs(
        returns_df,
        risk_free_rate=0.02,
        use_ml_returns=True,
        use_robust_cov=True
    )
    
    # Display results
    print("\n" + "-"*80)
    print("OPTIMIZATION RESULTS")
    print("-"*80)
    
    print(f"\nPortfolio Metrics:")
    print(f"  Expected Return:  {result['expected_return']:>7.2%}")
    print(f"  Volatility:       {result['volatility']:>7.2%}")
    print(f"  Sharpe Ratio:     {result['sharpe_ratio']:>7.2f}")
    print(f"  Anomalies Filtered: {result['anomalies_detected']}")
    
    print(f"\nOptimal Weights:")
    for asset, weight in result['weights'].items():
        print(f"  {asset}: {weight:>7.2%}")
    
    # Compare with equal-weight and historical-based optimization
    print("\n[Step 3] Comparing with traditional approaches...")
    
    # Equal weight
    equal_weights = pd.Series(1/len(returns_df.columns), index=returns_df.columns)
    eq_return = np.dot(equal_weights, returns_df.mean() * 252)
    cov_matrix = returns_df.cov() * 252
    eq_vol = np.sqrt(np.dot(np.dot(equal_weights, cov_matrix), equal_weights))
    eq_sharpe = (eq_return - 0.02) / eq_vol
    
    print(f"\nEqual-Weight Portfolio:")
    print(f"  Expected Return:  {eq_return:>7.2%}")
    print(f"  Volatility:       {eq_vol:>7.2%}")
    print(f"  Sharpe Ratio:     {eq_sharpe:>7.2f}")
    
    # Visualization
    print("\n[Step 4] Creating comparison charts...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Weights comparison
    ax1 = fig.add_subplot(gs[0, :])
    assets = list(result['weights'].index)
    x = np.arange(len(assets))
    width = 0.35
    
    ax1.bar(x - width/2, equal_weights, width, label='Equal Weight', alpha=0.8, color='gray')
    ax1.bar(x + width/2, result['weights'], width, label='ML-Optimized', alpha=0.8, color='blue')
    ax1.set_ylabel('Weight')
    ax1.set_title('Portfolio Weights: Equal vs ML-Optimized', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(assets)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Plot 2: Performance metrics comparison
    ax2 = fig.add_subplot(gs[1, 0])
    metrics = ['Return', 'Volatility', 'Sharpe Ratio']
    eq_metrics = [eq_return, eq_vol, eq_sharpe]
    ml_metrics = [result['expected_return'], result['volatility'], result['sharpe_ratio']]
    
    x_metrics = np.arange(len(metrics))
    ax2.bar(x_metrics - width/2, eq_metrics, width, label='Equal Weight', alpha=0.8, color='gray')
    ax2.bar(x_metrics + width/2, ml_metrics, width, label='ML-Optimized', alpha=0.8, color='blue')
    ax2.set_ylabel('Value')
    ax2.set_title('Performance Metrics Comparison', fontweight='bold')
    ax2.set_xticks(x_metrics)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Risk-Return scatter
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(eq_vol, eq_return, s=200, color='gray', marker='o', label='Equal Weight', zorder=5)
    ax3.scatter(result['volatility'], result['expected_return'], s=200, color='blue', 
               marker='*', label='ML-Optimized', zorder=5)
    
    # Individual assets
    for asset in assets:
        asset_return = returns_df[asset].mean() * 252
        asset_vol = returns_df[asset].std() * np.sqrt(252)
        ax3.scatter(asset_vol, asset_return, s=50, alpha=0.6)
        ax3.annotate(asset, (asset_vol, asset_return), fontsize=8, 
                    xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('Volatility (Risk)')
    ax3.set_ylabel('Expected Return')
    ax3.set_title('Risk-Return Profile', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Plot 4: Feature importance (if available)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.text(0.5, 0.5, 'ML Optimization Complete!\n\n' +
             f'Sharpe Ratio Improvement: {result["sharpe_ratio"] - eq_sharpe:+.3f}\n' +
             f'Return Improvement: {result["expected_return"] - eq_return:+.2%}\n' +
             f'Risk Reduction: {eq_vol - result["volatility"]:+.2%}\n' +
             f'Anomalies Filtered: {result["anomalies_detected"]} days',
             ha='center', va='center', fontsize=14, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax4.axis('off')
    
    plt.savefig('example5_complete_optimization.png', dpi=150, bbox_inches='tight')
    print("Chart saved as 'example5_complete_optimization.png'")
    plt.close()
    
    return result


###############################################################################
# Example 6: Backtesting
###############################################################################

def example_6_backtesting():
    """
    Example 6: Backtest ML-optimized portfolio.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Portfolio Backtesting")
    print("="*80)
    
    # Generate data
    print("\n[Step 1] Generating synthetic data...")
    returns_df = generate_synthetic_data(n_days=500, n_assets=5)
    
    # Create enhancer and optimize
    print("\n[Step 2] Optimizing portfolio...")
    enhancer = MLPortfolioEnhancer()
    
    result = enhancer.optimize_with_ml_inputs(
        returns_df.iloc[:400],  # Use first 400 days for optimization
        risk_free_rate=0.02,
        use_ml_returns=True,
        use_robust_cov=True
    )
    
    optimal_weights = result['weights']
    
    # Backtest on remaining data
    print("\n[Step 3] Running backtest on out-of-sample data...")
    
    backtest_data = returns_df.iloc[400:]
    
    # Test different rebalancing frequencies
    frequencies = ['monthly', 'quarterly']
    
    backtest_results = {}
    for freq in frequencies:
        print(f"\nBacktesting with {freq} rebalancing...")
        bt_result = enhancer.backtest_strategy(
            backtest_data,
            optimal_weights,
            rebalance_frequency=freq
        )
        backtest_results[freq] = bt_result
        
        print(f"  Total Return:     {bt_result['total_return']:>7.2f}%")
        print(f"  Annual Return:    {bt_result['annual_return']:>7.2f}%")
        print(f"  Annual Vol:       {bt_result['annual_volatility']:>7.2f}%")
        print(f"  Sharpe Ratio:     {bt_result['sharpe_ratio']:>7.2f}")
        print(f"  Max Drawdown:     {bt_result['max_drawdown']:>7.2f}%")
    
    # Compare with buy-and-hold equal weight
    print("\n[Step 4] Comparing with buy-and-hold strategy...")
    
    equal_weights = pd.Series(1/len(returns_df.columns), index=returns_df.columns)
    bh_result = enhancer.backtest_strategy(
        backtest_data,
        equal_weights,
        rebalance_frequency='monthly'  # For fair comparison
    )
    
    print(f"\nBuy-and-Hold Equal Weight:")
    print(f"  Total Return:     {bh_result['total_return']:>7.2f}%")
    print(f"  Annual Return:    {bh_result['annual_return']:>7.2f}%")
    print(f"  Annual Vol:       {bh_result['annual_volatility']:>7.2f}%")
    print(f"  Sharpe Ratio:     {bh_result['sharpe_ratio']:>7.2f}")
    print(f"  Max Drawdown:     {bh_result['max_drawdown']:>7.2f}%")
    
    # Visualize
    print("\n[Step 5] Creating backtest charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Cumulative performance
    ax1 = axes[0, 0]
    for freq, bt in backtest_results.items():
        ax1.plot(range(len(bt['portfolio_values'])), bt['portfolio_values'], 
                linewidth=2, label=f'ML-Optimized ({freq})')
    ax1.plot(range(len(bh_result['portfolio_values'])), bh_result['portfolio_values'], 
            linewidth=2, label='Equal Weight', linestyle='--')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Portfolio Value')
    ax1.set_title('Cumulative Performance', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Drawdown
    ax2 = axes[0, 1]
    for freq, bt in backtest_results.items():
        values = bt['portfolio_values']
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak * 100
        ax2.plot(range(len(drawdown)), drawdown, linewidth=2, label=f'ML-Optimized ({freq})')
    
    bh_values = bh_result['portfolio_values']
    bh_peak = np.maximum.accumulate(bh_values)
    bh_drawdown = (bh_values - bh_peak) / bh_peak * 100
    ax2.plot(range(len(bh_drawdown)), bh_drawdown, linewidth=2, 
            label='Equal Weight', linestyle='--')
    
    ax2.fill_between(range(len(drawdown)), 0, drawdown, alpha=0.3)
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Portfolio Drawdown', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance metrics bar chart
    ax3 = axes[1, 0]
    metrics_names = ['Total\nReturn', 'Sharpe\nRatio', 'Max\nDrawdown']
    
    monthly_metrics = [
        backtest_results['monthly']['total_return'],
        backtest_results['monthly']['sharpe_ratio'] * 20,  # Scale for visibility
        abs(backtest_results['monthly']['max_drawdown'])
    ]
    
    bh_metrics = [
        bh_result['total_return'],
        bh_result['sharpe_ratio'] * 20,
        abs(bh_result['max_drawdown'])
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax3.bar(x - width/2, monthly_metrics, width, label='ML-Optimized', alpha=0.8, color='blue')
    ax3.bar(x + width/2, bh_metrics, width, label='Equal Weight', alpha=0.8, color='gray')
    ax3.set_ylabel('Value')
    ax3.set_title('Performance Metrics Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    summary_text = f"""
    BACKTEST SUMMARY (Out-of-Sample)
    
    ML-Optimized (Monthly Rebalance):
      Total Return: {backtest_results['monthly']['total_return']:.2f}%
      Sharpe Ratio: {backtest_results['monthly']['sharpe_ratio']:.2f}
      Max Drawdown: {backtest_results['monthly']['max_drawdown']:.2f}%
    
    Equal Weight Benchmark:
      Total Return: {bh_result['total_return']:.2f}%
      Sharpe Ratio: {bh_result['sharpe_ratio']:.2f}
      Max Drawdown: {bh_result['max_drawdown']:.2f}%
    
    Outperformance:
      Return: {backtest_results['monthly']['total_return'] - bh_result['total_return']:+.2f}%
      Sharpe: {backtest_results['monthly']['sharpe_ratio'] - bh_result['sharpe_ratio']:+.2f}
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('example6_backtesting.png', dpi=150, bbox_inches='tight')
    print("Chart saved as 'example6_backtesting.png'")
    plt.close()
    
    return backtest_results


###############################################################################
# Main Execution
###############################################################################

def main():
    """Run all examples."""
    
    print("\n" + "="*80)
    print("ML PORTFOLIO ENHANCEMENT - COMPREHENSIVE EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates all features of ml_portfolio_enhancement.py")
    print("Charts will be saved in the current directory")
    print("\n" + "="*80)
    
    try:
        # Run all examples
        example_1_return_prediction()
        example_2_robust_covariance()
        example_3_regime_detection()
        example_4_anomaly_detection()
        example_5_complete_optimization()
        example_6_backtesting()
        
        # Summary
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated files:")
        print("  - example1_ml_predictions.png")
        print("  - example2_robust_covariance.png")
        print("  - example3_regime_detection.png")
        print("  - example4_anomaly_detection.png")
        print("  - example5_complete_optimization.png")
        print("  - example6_backtesting.png")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()