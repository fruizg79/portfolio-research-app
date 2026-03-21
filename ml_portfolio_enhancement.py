"""
ML Portfolio Enhancement Module
================================

Ready-to-use ML components for portfolio optimization.
Can be integrated directly into the Streamlit app.

Usage in Streamlit:
    from ml_portfolio_enhancement import MLPortfolioEnhancer
    
    enhancer = MLPortfolioEnhancer()
    predictions = enhancer.predict_returns(historical_data)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


class MLPortfolioEnhancer:
    """
    Main class for ML-enhanced portfolio optimization.
    Combines all ML techniques in one easy-to-use interface.
    """
    
    def __init__(self):
        self.return_predictor = None
        self.cov_estimator = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        
    def predict_returns(self, returns_df, method='ensemble', forecast_days=21):
        """
        Predict future returns using ML.
        
        Args:
            returns_df: Historical returns (DataFrame)
            method: 'random_forest', 'gradient_boosting', or 'ensemble'
            forecast_days: Days ahead to predict
        
        Returns:
            Predicted returns for each asset
        """
        features = self._create_features(returns_df)
        
        if features is None or len(features) < 100:
            # Fallback to historical mean if not enough data
            return returns_df.mean() * forecast_days
        
        predictions = {}
        
        for asset in returns_df.columns:
            # Target: future returns
            y = returns_df[asset].shift(-forecast_days)
            
            # Align features and target
            X = features.loc[y.dropna().index]
            y_clean = y.dropna()
            
            if len(X) < 50:
                predictions[asset] = returns_df[asset].mean() * forecast_days
                continue
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            if method == 'random_forest':
                model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                model.fit(X_scaled[:-forecast_days], y_clean[:-forecast_days])
                pred = model.predict(X_scaled[-1:])
                
            elif method == 'gradient_boosting':
                model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
                model.fit(X_scaled[:-forecast_days], y_clean[:-forecast_days])
                pred = model.predict(X_scaled[-1:])
                
            else:  # ensemble
                rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
                
                rf.fit(X_scaled[:-forecast_days], y_clean[:-forecast_days])
                gb.fit(X_scaled[:-forecast_days], y_clean[:-forecast_days])
                
                pred_rf = rf.predict(X_scaled[-1:])
                pred_gb = gb.predict(X_scaled[-1:])
                pred = (pred_rf + pred_gb) / 2
            
            predictions[asset] = pred[0]
        
        return pd.Series(predictions)
    
    def _create_features(self, returns_df):
        """Create ML features from returns data."""
        try:
            features = pd.DataFrame(index=returns_df.index)
            
            # Market-wide features
            features['market_return'] = returns_df.mean(axis=1)
            features['market_vol'] = returns_df.std(axis=1)
            features['market_skew'] = returns_df.skew(axis=1)
            
            # Rolling statistics
            for window in [5, 10, 21, 63]:
                features[f'return_{window}d'] = returns_df.mean(axis=1).rolling(window).mean()
                features[f'vol_{window}d'] = returns_df.std(axis=1).rolling(window).mean()
            
            # Momentum indicators
            features['momentum_5d'] = returns_df.mean(axis=1).rolling(5).sum()
            features['momentum_21d'] = returns_df.mean(axis=1).rolling(21).sum()
            
            # Volatility indicators
            features['vol_ratio'] = (
                returns_df.std(axis=1).rolling(5).mean() / 
                returns_df.std(axis=1).rolling(21).mean()
            )
            
            return features.dropna()
        except:
            return None
    
    def estimate_robust_covariance(self, returns_df, method='pca', n_components=None):
        """
        Estimate covariance matrix with ML denoising.
        
        Args:
            returns_df: Historical returns
            method: 'pca' (Principal Component Analysis) or 'shrinkage'
            n_components: Number of PCA components (None = auto)
        
        Returns:
            Denoised covariance matrix
        """
        if method == 'pca':
            if n_components is None:
                n_components = min(5, len(returns_df.columns) - 1)
            
            pca = PCA(n_components=n_components)
            returns_transformed = pca.fit_transform(returns_df)
            returns_denoised = pca.inverse_transform(returns_transformed)
            
            cov_matrix = np.cov(returns_denoised.T)
            
            return cov_matrix, pca.explained_variance_ratio_
        
        elif method == 'shrinkage':
            # Ledoit-Wolf shrinkage
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(returns_df)
            return lw.covariance_, None
    
    def detect_market_regimes(self, returns_df, n_regimes=3):
        """
        Detect different market regimes using clustering.
        
        Args:
            returns_df: Historical returns
            n_regimes: Number of regimes to detect
        
        Returns:
            regime_labels: Array of regime assignments
            regime_stats: Statistics for each regime
        """
        # Features for regime detection
        features = pd.DataFrame()
        features['avg_return'] = returns_df.mean(axis=1).rolling(21).mean()
        features['volatility'] = returns_df.std(axis=1).rolling(21).mean()
        features['correlation'] = returns_df.rolling(21).corr().mean().mean()
        features['max_drawdown'] = (
            returns_df.cumsum(axis=1).max(axis=1) - returns_df.cumsum(axis=1).min(axis=1)
        )
        
        features = features.fillna(method='bfill').fillna(0)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        regime_labels = kmeans.fit_predict(features)
        
        # Calculate statistics per regime
        regime_stats = {}
        for regime in range(n_regimes):
            mask = regime_labels == regime
            regime_returns = returns_df[mask]
            
            regime_stats[regime] = {
                'avg_return': regime_returns.mean().mean() * 252,
                'volatility': regime_returns.std().mean() * np.sqrt(252),
                'n_days': mask.sum(),
                'pct_days': mask.sum() / len(returns_df)
            }
        
        return regime_labels, regime_stats
    
    def filter_anomalies(self, returns_df, contamination=0.05):
        """
        Detect and filter anomalous days.
        
        Args:
            returns_df: Historical returns
            contamination: Expected proportion of anomalies
        
        Returns:
            clean_returns: Returns with anomalies removed
            anomaly_dates: Dates identified as anomalies
        """
        # Fit anomaly detector
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        
        anomalies = iso_forest.fit_predict(returns_df)
        is_anomaly = anomalies == -1
        
        clean_returns = returns_df[~is_anomaly]
        anomaly_dates = returns_df.index[is_anomaly]
        
        return clean_returns, anomaly_dates
    
    def optimize_with_ml_inputs(self, returns_df, risk_free_rate=0.02, 
                                use_ml_returns=True, use_robust_cov=True):
        """
        Complete ML-enhanced optimization pipeline.
        
        Args:
            returns_df: Historical returns
            risk_free_rate: Risk-free rate
            use_ml_returns: Use ML-predicted returns
            use_robust_cov: Use ML-denoised covariance
        
        Returns:
            Dictionary with optimal weights and metrics
        """
        # 1. Filter anomalies
        clean_returns, anomaly_dates = self.filter_anomalies(returns_df)
        
        # 2. Predict returns
        if use_ml_returns:
            expected_returns = self.predict_returns(clean_returns)
            expected_returns_annual = expected_returns * 252
        else:
            expected_returns_annual = clean_returns.mean() * 252
        
        # 3. Estimate covariance
        if use_robust_cov:
            cov_matrix, explained_var = self.estimate_robust_covariance(
                clean_returns, method='pca'
            )
            cov_matrix_annual = cov_matrix * 252
        else:
            cov_matrix_annual = clean_returns.cov() * 252
        
        # 4. Simple mean-variance optimization
        # (In practice, use cvxopt or scipy.optimize)
        n_assets = len(returns_df.columns)
        
        # Equal weight as baseline
        weights = np.ones(n_assets) / n_assets
        
        # Iterative improvement (simplified)
        for _ in range(100):
            # Calculate gradients
            ret = np.dot(weights, expected_returns_annual)
            vol = np.sqrt(np.dot(weights, np.dot(cov_matrix_annual, weights)))
            sharpe = (ret - risk_free_rate) / vol
            
            # Adjust weights toward higher Sharpe
            grad = (expected_returns_annual - risk_free_rate) / (vol ** 2)
            weights += 0.01 * grad
            weights = np.maximum(weights, 0)
            weights /= weights.sum()
        
        # Calculate final metrics
        portfolio_return = np.dot(weights, expected_returns_annual)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix_annual, weights)))
        portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
        
        return {
            'weights': pd.Series(weights, index=returns_df.columns),
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': portfolio_sharpe,
            'anomalies_detected': len(anomaly_dates),
            'ml_returns_used': use_ml_returns,
            'robust_cov_used': use_robust_cov
        }
    
    def backtest_strategy(self, returns_df, weights, rebalance_frequency='monthly'):
        """
        Backtest a portfolio strategy.
        
        Args:
            returns_df: Historical returns
            weights: Portfolio weights
            rebalance_frequency: 'daily', 'weekly', 'monthly', 'quarterly'
        
        Returns:
            Performance metrics
        """
        # Rebalance periods
        if rebalance_frequency == 'daily':
            rebalance_days = 1
        elif rebalance_frequency == 'weekly':
            rebalance_days = 5
        elif rebalance_frequency == 'monthly':
            rebalance_days = 21
        else:  # quarterly
            rebalance_days = 63
        
        # Initialize portfolio value
        portfolio_values = [100]
        current_weights = weights.copy()
        
        for i in range(len(returns_df)):
            # Daily return
            daily_return = np.dot(current_weights, returns_df.iloc[i])
            portfolio_values.append(portfolio_values[-1] * (1 + daily_return))
            
            # Update weights due to market movement
            current_weights = current_weights * (1 + returns_df.iloc[i])
            current_weights = current_weights / current_weights.sum()
            
            # Rebalance
            if (i + 1) % rebalance_days == 0:
                current_weights = weights.copy()
        
        portfolio_values = np.array(portfolio_values)
        
        # Calculate metrics
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        annual_return = np.mean(daily_returns) * 252 * 100
        annual_vol = np.std(daily_returns) * np.sqrt(252) * 100
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': portfolio_values[-1],
            'portfolio_values': portfolio_values
        }


###############################################################################
# Quick Usage Examples
###############################################################################

def example_usage():
    """Quick example of how to use MLPortfolioEnhancer."""
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    assets = ['Stock_A', 'Stock_B', 'Stock_C', 'Bond', 'Gold']
    
    returns_df = pd.DataFrame(
        np.random.randn(500, 5) * 0.01 + 0.0002,
        index=dates,
        columns=assets
    )
    
    # Initialize enhancer
    enhancer = MLPortfolioEnhancer()
    
    print("ML Portfolio Enhancement Example")
    print("=" * 60)
    
    # 1. Predict returns
    print("\n1. ML Return Predictions:")
    ml_returns = enhancer.predict_returns(returns_df, method='ensemble')
    print(ml_returns * 252)  # Annualized
    
    # 2. Detect regimes
    print("\n2. Market Regime Detection:")
    regimes, stats = enhancer.detect_market_regimes(returns_df, n_regimes=3)
    for regime, regime_stats in stats.items():
        print(f"\nRegime {regime}:")
        print(f"  Days: {regime_stats['n_days']} ({regime_stats['pct_days']:.1%})")
        print(f"  Return: {regime_stats['avg_return']:.2%}")
        print(f"  Vol: {regime_stats['volatility']:.2%}")
    
    # 3. Full optimization
    print("\n3. ML-Enhanced Optimization:")
    result = enhancer.optimize_with_ml_inputs(returns_df)
    
    print(f"\nOptimal Weights:")
    print(result['weights'])
    print(f"\nExpected Return: {result['expected_return']:.2%}")
    print(f"Volatility: {result['volatility']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    
    # 4. Backtest
    print("\n4. Backtest Results:")
    backtest = enhancer.backtest_strategy(
        returns_df, 
        result['weights'],
        rebalance_frequency='monthly'
    )
    
    print(f"Total Return: {backtest['total_return']:.2f}%")
    print(f"Annual Return: {backtest['annual_return']:.2f}%")
    print(f"Annual Volatility: {backtest['annual_volatility']:.2f}%")
    print(f"Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest['max_drawdown']:.2f}%")


if __name__ == "__main__":
    example_usage()