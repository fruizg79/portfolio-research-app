# Asset Allocation & Portfolio Optimization - Core Modules

**Professional-grade Python implementation of Modern Portfolio Theory, Black-Litterman model, and advanced portfolio optimization techniques.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📚 Overview

This repository contains two core Python modules for quantitative portfolio management:

1. **`asset_allocation.py`** - Comprehensive portfolio optimization framework
2. **`ml_portfolio_enhancement.py`** - Machine learning enhancements for portfolio optimization

These modules provide production-ready implementations of classic and modern portfolio optimization techniques, suitable for both academic research and professional portfolio management.

---

## 🎯 Key Features

### `asset_allocation.py`

#### Asset Modeling & Simulation
- **Stochastic Process Calibration**: Vasicek and Normal models for asset evolution
- **Monte Carlo Simulation**: Generate thousands of potential future scenarios
- **Parameter Estimation**: MLE-based calibration from historical data

#### Correlation & Copulas
- **Gaussian Copula**: Model linear dependencies between assets
- **t-Student Copula**: Capture tail dependencies and extreme co-movements
- **Cholesky Decomposition**: Efficient correlation structure implementation

#### Portfolio Optimization
- **Mean-Variance Optimization**: Markowitz efficient frontier
- **Tracking Error Optimization**: Active portfolio management vs benchmark
- **Custom Constraints**: Min/max weights, sector limits, turnover constraints
- **Multiple Objectives**: Sharpe ratio, return targeting, risk parity

#### Black-Litterman Model
- **View Integration**: Combine market equilibrium with investor opinions
- **Bayesian Framework**: Rigorous statistical foundation
- **Qualitative Views**: Convert ratings (Buy/Sell) to quantitative inputs
- **Confidence Levels**: Weight views by analyst confidence

#### Risk Metrics & Utilities
- **Sharpe Ratio**: Risk-adjusted performance measurement
- **Sortino Ratio**: Downside risk focus
- **Value at Risk (VaR)**: Maximum expected loss at confidence level
- **Conditional VaR**: Expected shortfall beyond VaR
- **Maximum Drawdown**: Peak-to-trough decline analysis
- **Utility Functions**: Quadratic, exponential, and power utilities

### `ml_portfolio_enhancement.py`

#### Return Prediction
- **Random Forest Regression**: Non-linear pattern recognition
- **Gradient Boosting**: Sequential learning for improved accuracy
- **Ensemble Methods**: Combine multiple models for robust predictions
- **Feature Engineering**: Momentum, volatility, trend indicators

#### Covariance Estimation
- **PCA Denoising**: Reduce noise in covariance matrix
- **Shrinkage Methods**: Ledoit-Wolf optimal shrinkage
- **Regime-Based**: Different covariances for different market states

#### Market Regime Detection
- **K-Means Clustering**: Automatic regime identification
- **Multi-dimensional Features**: Returns, volatility, correlations
- **Regime Statistics**: Performance metrics per market state

#### Anomaly Detection
- **Isolation Forest**: Identify outlier trading days
- **Robust Statistics**: Estimate parameters excluding anomalies
- **Quality Control**: Automatic data cleaning

#### Backtesting Framework
- **Historical Simulation**: Test strategies on past data
- **Multiple Rebalancing**: Daily, weekly, monthly, quarterly
- **Performance Attribution**: Detailed metrics and attribution
- **Transaction Costs**: Realistic simulation with costs

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/fruizg79/portfolio-optimization.git
cd portfolio-optimization

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Asset Calibration and Simulation

```python
from asset_allocation import CAsset
import numpy as np

# Historical returns data
returns = np.random.randn(252) * 0.01 + 0.0005  # Daily returns

# Create and calibrate asset
asset = CAsset(
    data=returns,
    data_freq='daily',
    asset_class='US_Equity',
    sim_model='normal'
)

# Calibrate parameters
params = asset.param_calibration()
print(f"Annual Return: {params['mean']:.2%}")
print(f"Annual Volatility: {params['vol']:.2%}")

# Simulate future paths
simulations = asset.simulate(n_periods=252, n_simulations=1000)
print(f"Mean final value: {simulations[:, -1].mean():.2f}")
```

#### 2. Portfolio Optimization

```python
from asset_allocation import CPortfolio_optimization
import numpy as np

# Define assets
expected_returns = np.array([0.10, 0.08, 0.06, 0.12])  # 10%, 8%, 6%, 12%
volatilities = np.array([0.18, 0.15, 0.08, 0.22])       # Annual volatilities

# Correlation matrix
correlation = np.array([
    [1.00, 0.70, 0.30, 0.60],
    [0.70, 1.00, 0.40, 0.50],
    [0.30, 0.40, 1.00, 0.20],
    [0.60, 0.50, 0.20, 1.00]
])

# Create optimizer
optimizer = CPortfolio_optimization(
    port_wts=np.array([0.25, 0.25, 0.25, 0.25]),
    correlation_matrix=correlation,
    expected_ret=expected_returns,
    vol=volatilities
)

# Find optimal portfolios
optimal_portfolios = optimizer.mean_variance_opt(
    x_min=np.array([0.0, 0.0, 0.0, 0.0]),  # No short selling
    x_max=np.array([0.5, 0.5, 0.5, 0.5]),  # Max 50% per asset
    num_port=20
)

# Generate efficient frontier
returns, vols, weights = optimizer.efficient_frontier(
    x_min=np.array([0.0, 0.0, 0.0, 0.0]),
    x_max=np.array([0.5, 0.5, 0.5, 0.5]),
    num_points=50
)

print(f"Efficient Frontier: {len(returns)} portfolios")
print(f"Risk range: {vols.min():.2%} to {vols.max():.2%}")
print(f"Return range: {returns.min():.2%} to {returns.max():.2%}")
```

#### 3. Black-Litterman Model

```python
from asset_allocation import CBlack_litterman
import numpy as np

# Market parameters
n_assets = 4
cov_matrix = np.array([...])  # Your covariance matrix
market_weights = np.array([0.40, 0.30, 0.20, 0.10])

# Define views
# View 1: Asset 0 will outperform
# View 2: Asset 2 absolute return
p_matrix = np.array([
    [1, -1,  0,  0],   # Asset 0 vs Asset 1
    [0,  0,  1,  0]    # Asset 2 absolute
])

# Qualitative recommendations
recommendations = ['buy', 'neutral']
confidence_levels = np.array([0.7, 0.5])

# Create Black-Litterman model
bl_model = CBlack_litterman(
    risk_free_rate=0.02,
    tau=0.025,
    sigma=cov_matrix,
    p_matrix=p_matrix,
    lambda_param=2.5,
    port_wts_eq=market_weights,
    conf_level=confidence_levels,
    c_coef=1.0,
    reco_vector=recommendations
)

# Get Black-Litterman returns
bl_returns = bl_model.get_bl_returns()
print("Black-Litterman Expected Returns:")
print(bl_returns)
```

#### 4. ML-Enhanced Optimization

```python
from ml_portfolio_enhancement import MLPortfolioEnhancer
import pandas as pd

# Historical returns DataFrame
returns_df = pd.DataFrame({
    'Stock_A': [...],
    'Stock_B': [...],
    'Bond': [...]
})

# Create enhancer
enhancer = MLPortfolioEnhancer()

# Predict future returns using ML
ml_returns = enhancer.predict_returns(
    returns_df, 
    method='ensemble',
    forecast_days=21
)
print("ML Predicted Returns:")
print(ml_returns)

# Robust covariance estimation
cov_matrix, explained_var = enhancer.estimate_robust_covariance(
    returns_df,
    method='pca',
    n_components=5
)
print(f"Variance explained: {explained_var.sum():.2%}")

# Detect market regimes
regimes, stats = enhancer.detect_market_regimes(returns_df, n_regimes=3)
print("\nMarket Regimes:")
for regime, regime_stats in stats.items():
    print(f"Regime {regime}: {regime_stats['pct_days']:.1%} of days")

# Full ML optimization
result = enhancer.optimize_with_ml_inputs(
    returns_df,
    risk_free_rate=0.02,
    use_ml_returns=True,
    use_robust_cov=True
)

print(f"\nML-Optimized Portfolio:")
print(f"Expected Return: {result['expected_return']:.2%}")
print(f"Volatility: {result['volatility']:.2%}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
print(f"Optimal Weights:\n{result['weights']}")
```

---

## 📊 Module Architecture

### `asset_allocation.py` - Class Hierarchy

```
asset_allocation.py
│
├── CAsset
│   ├── param_calibration()      # Estimate model parameters
│   └── simulate()                # Monte Carlo simulation
│
├── CCopulas
│   ├── get_correlated_normal()   # Gaussian copula
│   ├── get_tstudent_copula()     # t-Student copula
│   └── get_correlated_returns()  # Returns with marginals
│
├── CPortfolio_optimization
│   ├── mean_variance_opt()       # Mean-variance optimization
│   ├── asset_allocation_TEop()   # Tracking error optimization
│   ├── tracking_error_ex_ante()  # Calculate tracking error
│   └── efficient_frontier()      # Generate frontier
│
├── CBlack_litterman
│   ├── get_bl_returns()          # Posterior returns
│   ├── get_eq_returns()          # Equilibrium returns
│   ├── get_omega()               # View uncertainty
│   ├── get_view_returns()        # Transform views
│   └── get_posterior_covariance() # Posterior covariance
│
├── CMarket
│   ├── get_mahalanobis()         # Distance metric
│   ├── get_cdf()                 # Empirical CDF
│   └── get_summary_statistics()  # Descriptive stats
│
├── CUtility
│   ├── calculate_utility()       # Utility value
│   └── certainty_equivalent()    # CE return
│
└── Helper Functions
    ├── calculate_sharpe_ratio()
    ├── calculate_sortino_ratio()
    ├── calculate_max_drawdown()
    ├── calculate_value_at_risk()
    └── calculate_conditional_var()
```

### `ml_portfolio_enhancement.py` - Class Structure

```
ml_portfolio_enhancement.py
│
└── MLPortfolioEnhancer
    ├── predict_returns()              # ML return prediction
    ├── estimate_robust_covariance()   # PCA/shrinkage
    ├── detect_market_regimes()        # Clustering
    ├── filter_anomalies()             # Outlier detection
    ├── optimize_with_ml_inputs()      # Full pipeline
    └── backtest_strategy()            # Performance testing
```

---

## 🔬 Mathematical Background

### Mean-Variance Optimization

**Objective:**
```
max  μ'w - (λ/2)w'Σw
s.t. Σw = 1
     w_min ≤ w ≤ w_max
```

Where:
- `w` = portfolio weights
- `μ` = expected returns
- `Σ` = covariance matrix
- `λ` = risk aversion parameter

### Black-Litterman Model

**Posterior Returns:**
```
E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) × [(τΣ)^(-1)Π + P'Ω^(-1)Q]
```

Where:
- `Π` = implied equilibrium returns = λΣw_mkt
- `P` = pick matrix (defines views)
- `Q` = view returns
- `Ω` = diagonal matrix of view uncertainties
- `τ` = scalar (typically 0.025 - 0.05)

### Sharpe Ratio

```
Sharpe = (R_p - R_f) / σ_p
```

Where:
- `R_p` = portfolio return
- `R_f` = risk-free rate
- `σ_p` = portfolio standard deviation

### Value at Risk (VaR)

```
VaR_α = -Percentile(returns, α)
```

95% VaR = maximum loss expected with 95% confidence

### Conditional VaR (CVaR)

```
CVaR_α = E[R | R ≤ -VaR_α]
```

Expected loss given that VaR is exceeded

---

## 📖 API Reference

### CAsset

```python
class CAsset(data, data_freq, asset_class, sim_model)
```

**Parameters:**
- `data` (array-like): Historical asset data
- `data_freq` (str): 'daily', 'weekly', or 'monthly'
- `asset_class` (str): Asset identifier
- `sim_model` (str): 'vasicek' or 'normal'

**Methods:**
- `param_calibration()` → dict: Estimate model parameters
- `simulate(n_periods, n_simulations, initial_value)` → np.array: Monte Carlo simulation

### CPortfolio_optimization

```python
class CPortfolio_optimization(port_wts, correlation_matrix, expected_ret, vol)
```

**Parameters:**
- `port_wts` (array-like): Initial weights
- `correlation_matrix` (array-like): Correlation matrix
- `expected_ret` (array-like): Expected returns
- `vol` (array-like): Volatilities

**Methods:**
- `mean_variance_opt(x_min, x_max, num_port)` → list: Optimal portfolios
- `efficient_frontier(x_min, x_max, num_points)` → tuple: (returns, vols, weights)
- `tracking_error_ex_ante(wts_bmk, wts_port)` → float: Tracking error

### CBlack_litterman

```python
class CBlack_litterman(risk_free_rate, tau, sigma, p_matrix, lambda_param, 
                       port_wts_eq, conf_level, c_coef, reco_vector)
```

**Methods:**
- `get_bl_returns()` → np.array: Posterior expected returns
- `get_eq_returns()` → np.array: Equilibrium returns
- `get_view_returns()` → np.array: View-implied returns

### MLPortfolioEnhancer

```python
class MLPortfolioEnhancer()
```

**Methods:**
- `predict_returns(returns_df, method, forecast_days)` → pd.Series
- `estimate_robust_covariance(returns_df, method, n_components)` → tuple
- `detect_market_regimes(returns_df, n_regimes)` → tuple
- `filter_anomalies(returns_df, contamination)` → tuple
- `optimize_with_ml_inputs(returns_df, risk_free_rate, ...)` → dict
- `backtest_strategy(returns_df, weights, rebalance_frequency)` → dict

---

## 🧪 Testing

Run the example scripts to verify installation:

```python
# Test asset_allocation.py
python -c "from asset_allocation import CAsset; print('✓ asset_allocation.py loaded')"

# Test ml_portfolio_enhancement.py  
python -c "from ml_portfolio_enhancement import MLPortfolioEnhancer; print('✓ ML module loaded')"
```

### Unit Tests (if available)

```bash
pytest tests/
```

---

## 📚 Examples

See the `examples/` directory for detailed examples:

- `example_basic_optimization.py` - Simple mean-variance optimization
- `example_black_litterman.py` - Black-Litterman with views
- `example_ml_prediction.py` - ML-enhanced return prediction
- `example_backtesting.py` - Strategy backtesting
- `portfolio_examples.py` - Comprehensive demonstration

---

## 🔧 Dependencies

### Required
```
numpy>=1.26.3        # Numerical computing
pandas>=2.1.4        # Data manipulation
scipy>=1.11.4        # Scientific computing
cvxopt>=1.3.2        # Convex optimization
```

### Optional (for ML features)
```
scikit-learn>=1.4.0  # Machine learning
matplotlib>=3.8.2    # Plotting
```

---

## 📈 Performance Considerations

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Mean-variance optimization | O(n³) | Where n = number of assets |
| Monte Carlo simulation | O(s×p) | s = simulations, p = periods |
| PCA denoising | O(n²×m) | m = observations |
| Black-Litterman | O(n³) | Matrix inversions |

### Optimization Tips

**For large portfolios (100+ assets):**
```python
# Use sparse matrices
from scipy.sparse import csr_matrix

# Reduce optimization iterations
num_portfolios = 10  # Instead of 50

# Use PCA for dimension reduction
n_components = 20  # Instead of all assets
```

---

## 🐛 Known Limitations

1. **cvxopt availability**: Requires C++ compiler on some systems
2. **Memory usage**: Monte Carlo with 10,000+ simulations can be memory-intensive
3. **ML data requirements**: Need 500+ observations for reliable ML predictions
4. **Optimization convergence**: May not converge with extreme constraints

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Code Style
- Follow PEP 8
- Use type hints where appropriate
- Add docstrings to all public methods
- Include unit tests for new features

### Submitting Changes
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Fernando Ruiz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🎓 References

### Academic Papers

1. **Markowitz, H. (1952)**. "Portfolio Selection". *The Journal of Finance*, 7(1), 77-91.
   - Foundation of Modern Portfolio Theory

2. **Black, F., & Litterman, R. (1992)**. "Global Portfolio Optimization". *Financial Analysts Journal*, 48(5), 28-43.
   - Black-Litterman model

3. **Sharpe, W. F. (1964)**. "Capital Asset Prices: A Theory of Market Equilibrium". *The Journal of Finance*, 19(3), 425-442.
   - Sharpe ratio and CAPM

4. **DeMiguel, V., Garlappi, L., & Uppal, R. (2009)**. "Optimal Versus Naive Diversification". *Review of Financial Studies*, 22(5), 1915-1953.
   - Critique of mean-variance optimization

### Books

- **Active Portfolio Management** - Grinold & Kahn
- **Modern Portfolio Theory and Investment Analysis** - Elton et al.
- **Machine Learning for Asset Managers** - López de Prado
- **Quantitative Portfolio Management** - Qian, Hua, & Sorensen

---

## 👤 Author

**Fernando Ruiz**
- Website: [portfolio-research.com](https://fruizg79.wixsite.com/portfolio-research)
- GitHub: [@fruizg79](https://github.com/fruizg79)
- Email: fruizg79@gmail.com

---

## 🙏 Acknowledgments

- Modern Portfolio Theory by Harry Markowitz
- Black-Litterman model by Fischer Black and Robert Litterman
- Scientific Python community (NumPy, SciPy, pandas)
- Machine learning community (scikit-learn)

---

## 📞 Support

### Getting Help

- 📖 Check the [Usage Examples](#-quick-start)
- 🐛 [Report bugs](https://github.com/fruizg79/portfolio-optimization/issues)
- 💡 [Request features](https://github.com/fruizg79/portfolio-optimization/issues)
- 💬 [Discussions](https://github.com/fruizg79/portfolio-optimization/discussions)

### Professional Consulting

For custom implementations or consulting:
- Email: consulting@portfolio-research.com
- Website: [portfolio-research.com/](https://portfolio-research.com)

---

## 📊 Project Status

![GitHub last commit](https://img.shields.io/github/last-commit/fruizg79/portfolio-optimization)
![GitHub issues](https://img.shields.io/github/issues/fruizg79/portfolio-optimization)
![GitHub pull requests](https://img.shields.io/github/issues-pr/fruizg79/portfolio-optimization)

---

<div align="center">

**⭐ If you find this useful, please star the repository! ⭐**

Built with ❤️ for quantitative portfolio managers

[Report Bug](https://github.com/fruizg79/portfolio-optimization/issues) · [Request Feature](https://github.com/fruizg79/portfolio-optimization/issues) · [View Examples](examples/)

</div>