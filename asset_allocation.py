"""
Asset Allocation and Portfolio Optimization Framework

This module provides a comprehensive set of tools for portfolio management,
including asset modeling, copula-based simulations, portfolio optimization,
and the Black-Litterman model.

Author: Fernando Ruiz
Created: 02/08/2017
Updated: 20/03/2026
Copyright: (c) portfolio-research.com - 2017
"""

from cvxopt import matrix, solvers
import math
import numpy as np
import os
import pandas as pd
from scipy import stats, spatial
import sklearn as sk

# Suppress cvxopt output
solvers.options['show_progress'] = False


###############################################################################
# Asset Evolution
###############################################################################
class CAsset(object):
    """
    Models asset evolution using stochastic processes.
    
    Supports calibration of parameters for different simulation models
    (Vasicek, Normal) based on historical data.
    
    Attributes:
        data (array-like): Historical asset data
        data_freq (str): Data frequency ('monthly', 'daily', 'weekly')
        asset_class (str): Asset class identifier
        sim_model (str): Simulation model ('vasicek', 'normal')
        calibrated_param (dict): Calibrated model parameters
    """
    
    def __init__(self, data, data_freq, asset_class, sim_model):
        """
        Initialize CAsset object.
        
        Args:
            data (array-like): Historical asset data
            data_freq (str): Data frequency ('monthly', 'daily', 'weekly')
            asset_class (str): Asset class identifier
            sim_model (str): Simulation model ('vasicek', 'normal')
        """
        self.data = np.array(data)
        self.asset_class = asset_class
        self.sim_model = sim_model
        self.data_freq = data_freq
        self.calibrated_param = {}
    
    def param_calibration(self):
        """
        Calibrate model parameters based on historical data.
        
        For Vasicek model: Estimates mean reversion level, speed of reversion,
        and volatility.
        For Normal model: Estimates mean and volatility.
        
        Returns:
            dict: Dictionary containing calibrated parameters
        """
        # Determine time scaling factor based on data frequency
        if self.data_freq == 'monthly':
            dt = 12
        elif self.data_freq == 'daily':
            dt = 250
        elif self.data_freq == 'weekly':
            dt = 52
        else:
            raise ValueError(f'Unknown data frequency: {self.data_freq}')
        
        if self.sim_model == 'vasicek':
            # Vasicek model: dr = κ(θ - r)dt + σdW
            # Using OLS regression: r(t+1) = a + b*r(t) + ε
            x = self.data[:-1]
            y = self.data[1:]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Calculate Vasicek parameters
            mean_rev = intercept / (1 - slope)  # θ (long-term mean)
            speed_rev = -math.log(slope) * dt   # κ (speed of reversion)
            vol = std_err * math.sqrt(-2 * math.log(slope) / (1 - slope**2)) * math.sqrt(dt)
            
            self.calibrated_param = {
                'sim_model': self.sim_model,
                'mean_rev': mean_rev,
                'speed_rev': speed_rev,
                'data_freq': self.data_freq,
                'vol': vol,
                'r_squared': r_value**2
            }
            
        elif self.sim_model == 'normal':
            # Normal model: Simple geometric Brownian motion
            mu = np.mean(self.data) * dt
            vol = np.std(self.data) * math.sqrt(dt)
            
            self.calibrated_param = {
                'sim_model': self.sim_model,
                'mean': mu,
                'vol': vol,
                'data_freq': self.data_freq
            }
        else:
            raise ValueError(f'Model {self.sim_model} is not implemented')
        
        return self.calibrated_param
    
    def simulate(self, n_periods, n_simulations=1000, initial_value=None):
        """
        Simulate future asset paths based on calibrated model.
        
        Args:
            n_periods (int): Number of periods to simulate
            n_simulations (int): Number of simulation paths
            initial_value (float): Starting value (uses last data point if None)
        
        Returns:
            np.array: Array of simulated paths (n_simulations x n_periods)
        """
        if not self.calibrated_param:
            self.param_calibration()
        
        if initial_value is None:
            initial_value = self.data[-1]
        
        simulations = np.zeros((n_simulations, n_periods + 1))
        simulations[:, 0] = initial_value
        
        if self.sim_model == 'vasicek':
            theta = self.calibrated_param['mean_rev']
            kappa = self.calibrated_param['speed_rev']
            sigma = self.calibrated_param['vol']
            dt = 1.0  # One period step
            
            for t in range(n_periods):
                dW = np.random.randn(n_simulations)
                simulations[:, t + 1] = (simulations[:, t] + 
                                        kappa * (theta - simulations[:, t]) * dt + 
                                        sigma * math.sqrt(dt) * dW)
        
        elif self.sim_model == 'normal':
            mu = self.calibrated_param['mean']
            sigma = self.calibrated_param['vol']
            
            for t in range(n_periods):
                dW = np.random.randn(n_simulations)
                simulations[:, t + 1] = simulations[:, t] + mu + sigma * dW
        
        return simulations


###############################################################################
# Copulas
###############################################################################
class CCopulas(object):
    """
    Generates correlated random variables using copula methods.
    
    Implements Gaussian and t-Student copulas for modeling dependence
    structures between assets.
    
    Attributes:
        correlation_matrix (np.array): Correlation matrix between assets
        chol_matrix (np.array): Cholesky decomposition of correlation matrix
        num_assets (int): Number of assets
        nsim (int): Number of simulations
    """
    
    def __init__(self, correlation_matrix, nsim):
        """
        Initialize CCopulas object.
        
        Args:
            correlation_matrix (pd.DataFrame or np.array): Correlation matrix
            nsim (int): Number of simulations to generate
        """
        if isinstance(correlation_matrix, pd.DataFrame):
            self.correlation_matrix = correlation_matrix.values
        else:
            self.correlation_matrix = np.array(correlation_matrix)
        
        self.chol_matrix = np.linalg.cholesky(self.correlation_matrix)
        self.num_assets = int(math.sqrt(self.correlation_matrix.size))
        self.nsim = nsim
        self.random_noise = None
        self.random_noise_corr = None
    
    def get_correlated_normal(self):
        """
        Generate correlated normal random variables.
        
        Uses Cholesky decomposition to induce correlation structure.
        
        Returns:
            tuple: (uncorrelated_noise, correlated_noise)
        """
        self.random_noise = np.random.randn(self.nsim, self.num_assets)
        self.random_noise_corr = np.dot(self.random_noise, self.chol_matrix.T)
        return self.random_noise, self.random_noise_corr
    
    def get_tstudent_copula(self, chi_df=5):
        """
        Generate correlated variables using t-Student copula.
        
        The t-Student copula allows for tail dependence, making it suitable
        for modeling joint extreme events.
        
        Args:
            chi_df (int): Degrees of freedom for the t-Student distribution
        
        Returns:
            tuple: (correlated_normal, correlated_uniform)
        """
        # Generate correlated normal variables
        normal_sample, _ = self.get_correlated_normal()
        
        # Generate chi-square samples
        chi_sample = np.random.chisquare(chi_df, size=self.nsim)
        
        mc_sim_nor_corr = []
        mc_sim_uniform_corr = []
        
        for sim in range(self.nsim):
            # Transform to t-Student
            aux = math.sqrt(chi_df) * normal_sample[sim, :] / math.sqrt(chi_sample[sim])
            mc_sim_nor_corr.append(aux)
            
            # Transform to uniform [0,1] via CDF
            u = stats.t.cdf(aux, df=chi_df)
            mc_sim_uniform_corr.append(u)
        
        self.mc_sim_nor_corr = np.array(mc_sim_nor_corr)
        self.mc_sim_uniform_corr = np.array(mc_sim_uniform_corr)
        
        return self.mc_sim_nor_corr, self.mc_sim_uniform_corr
    
    def get_correlated_returns(self, marginal_params):
        """
        Generate correlated asset returns with specified marginal distributions.
        
        Args:
            marginal_params (list): List of dicts with 'mean' and 'std' for each asset
        
        Returns:
            np.array: Correlated returns (nsim x num_assets)
        """
        _, noise_corr = self.get_correlated_normal()
        returns = np.zeros_like(noise_corr)
        
        for i, params in enumerate(marginal_params):
            returns[:, i] = params['mean'] + params['std'] * noise_corr[:, i]
        
        return returns


###############################################################################
# Portfolio Optimization
###############################################################################
class CPortfolio_optimization(object):
    """
    Portfolio optimization using mean-variance and tracking error approaches.
    
    Implements quadratic programming for finding optimal portfolio weights
    subject to various constraints.
    
    Attributes:
        port_wts (np.array): Portfolio weights
        correlation_matrix (np.array): Asset correlation matrix
        expected_ret (np.array): Expected returns for each asset
        vol (np.array): Volatility for each asset
        exp_cov (np.array): Expected covariance matrix
    """
    
    def __init__(self, port_wts, correlation_matrix, expected_ret, vol):
        """
        Initialize portfolio optimization object.
        
        Args:
            port_wts (array-like): Initial portfolio weights
            correlation_matrix (array-like): Correlation matrix
            expected_ret (array-like): Expected returns
            vol (array-like): Asset volatilities
        """
        self.port_wts = np.array(port_wts)
        
        if isinstance(correlation_matrix, pd.DataFrame):
            self.correlation_matrix = correlation_matrix.values
        else:
            self.correlation_matrix = np.array(correlation_matrix)
        
        self.expected_ret = np.array(expected_ret)
        self.vol = np.array(vol)
        
        # Calculate covariance matrix from correlation and volatilities
        vol_matrix = np.diag(self.vol)
        self.exp_cov = np.dot(np.dot(vol_matrix, self.correlation_matrix), vol_matrix)
    
    def mean_variance_opt(self, x_min, x_max, num_port=20):
        """
        Mean-variance portfolio optimization for different risk aversion levels.
        
        Solves: min λ*w'Σw - w'μ
        subject to: sum(w) = 1, x_min <= w <= x_max
        
        Args:
            x_min (array-like): Minimum weight for each asset
            x_max (array-like): Maximum weight for each asset
            num_port (int): Number of portfolios to generate
        
        Returns:
            list: List of optimal portfolio weights for different risk levels
        """
        lambda_vector = np.linspace(0.5, 2, num_port)
        
        x_min = np.array(x_min)
        x_max = np.array(x_max)
        
        # Min and Max weight constraints
        h = matrix(np.concatenate([x_max, -x_min], 0))
        nr = len(self.expected_ret)
        aux_max = np.eye(nr)
        aux_min = -np.eye(nr)
        G = matrix(np.concatenate([aux_max, aux_min], 0))
        
        # Budget constraint: sum(weights) = 1
        A = matrix(np.ones(nr).astype('double')).T
        b = matrix(1.0, tc='d')
        
        # Optimize for different risk tolerance levels (lambda)
        PortWts = []
        q = matrix(-self.expected_ret.astype('double'))
        
        for lam in lambda_vector:
            P = matrix(lam * self.exp_cov.astype('double'))
            sol = solvers.qp(P, q, G, h, A, b)
            PortWts.append(np.array(sol['x']).flatten())
        
        return PortWts
    
    def asset_allocation_TEop(self, wts_bmk, tactical_range):
        """
        Portfolio optimization minimizing tracking error.
        
        Finds optimal active positions relative to a benchmark to maximize
        expected return while controlling tracking error.
        
        Args:
            wts_bmk (array-like): Benchmark weights
            tactical_range (array-like): Maximum deviation from benchmark
        
        Returns:
            tuple: (optimal_weights, tracking_error)
        """
        wts_bmk = np.array(wts_bmk)
        tactical_range = np.array(tactical_range)
        
        # Min and max constraints on active positions
        x_min = -tactical_range
        x_max = tactical_range
        h = matrix(np.concatenate([x_max, -x_min], 0))
        
        nr = len(self.expected_ret)
        aux_max = np.eye(nr)
        aux_min = -np.eye(nr)
        G = matrix(np.concatenate([aux_max, aux_min], 0))
        
        # Budget constraint: sum(active positions) = 0
        A = matrix(np.ones(nr).astype('double')).T
        b = matrix(0.0, tc='d')
        
        # Quadratic programming: min 0.5*x'Px - q'x
        P = matrix(self.exp_cov.astype('double'))
        q = matrix(-self.expected_ret.astype('double'))
        
        # Solve
        sol = solvers.qp(P, q, G, h, A, b)
        active_positions = np.array(sol['x']).flatten()
        
        # Portfolio weight: Strategic weight + Active position
        wts_opt = wts_bmk + active_positions
        
        # Calculate ex-ante tracking error
        port_te = self.tracking_error_ex_ante(wts_bmk, wts_opt)
        
        return wts_opt, port_te
    
    def tracking_error_ex_ante(self, wts_bmk, wts_port):
        """
        Calculate ex-ante tracking error.
        
        TE = sqrt((w_p - w_b)' Σ (w_p - w_b))
        
        Args:
            wts_bmk (array-like): Benchmark weights
            wts_port (array-like): Portfolio weights
        
        Returns:
            float: Annualized tracking error
        """
        active_wts = np.array(wts_port) - np.array(wts_bmk)
        variance = np.dot(np.dot(active_wts, self.exp_cov), active_wts)
        return math.sqrt(variance)
    
    def efficient_frontier(self, x_min, x_max, num_points=50):
        """
        Generate the efficient frontier.
        
        Args:
            x_min (array-like): Minimum weights
            x_max (array-like): Maximum weights
            num_points (int): Number of points on the frontier
        
        Returns:
            tuple: (returns, volatilities, weights)
        """
        portfolios = self.mean_variance_opt(x_min, x_max, num_points)
        
        returns = []
        volatilities = []
        
        for wts in portfolios:
            ret = np.dot(wts, self.expected_ret)
            vol = math.sqrt(np.dot(np.dot(wts, self.exp_cov), wts))
            returns.append(ret)
            volatilities.append(vol)
        
        return np.array(returns), np.array(volatilities), portfolios


###############################################################################
# Black-Litterman Model
###############################################################################
class CBlack_litterman(object):
    """
    Implementation of the Black-Litterman asset allocation model.
    
    Combines market equilibrium with investor views to generate
    expected returns that incorporate both sources of information.
    
    Attributes:
        risk_free_rate (float): Risk-free rate
        tau (float): Scaling parameter (typically 0.025 to 0.05)
        sigma (np.array): Covariance matrix
        p_matrix (np.array): Pick matrix defining views
        lambda_param (float): Risk aversion parameter
        port_wts_eq (np.array): Market equilibrium weights
        conf_level (array-like): Confidence level for each view
        c_coef (float): Scaling coefficient for omega
        reco_vector (list): List of recommendations ('strong_buy', etc.)
    """
    
    def __init__(self, risk_free_rate, tau, sigma, p_matrix, lambda_param,
                 port_wts_eq, conf_level, c_coef, reco_vector):
        """
        Initialize Black-Litterman model.
        
        Args:
            risk_free_rate (float): Risk-free rate
            tau (float): Uncertainty scaling parameter
            sigma (np.array): Covariance matrix of returns
            p_matrix (np.array): Pick matrix (views x assets)
            lambda_param (float): Market risk aversion coefficient
            port_wts_eq (array-like): Market equilibrium weights
            conf_level (array-like): Confidence in each view (0-1)
            c_coef (float): Scaling coefficient for uncertainty
            reco_vector (list): Qualitative recommendations for each view
        """
        self.tau = tau
        self.risk_free_rate = risk_free_rate
        self.sigma = np.array(sigma)
        self.p_matrix = np.array(p_matrix)
        self.lambda_param = lambda_param
        self.port_wts_eq = np.array(port_wts_eq)
        self.conf_level = np.array(conf_level)
        self.c_coef = c_coef
        self.reco_vector = reco_vector
        self.num_recos = len(reco_vector)
        
        # Mapping from qualitative recommendations to quantitative adjustments
        self.reco_dict = {
            'strong_sell': -0.5,
            'sell': -0.25,
            'neutral': 0.0,
            'buy': 0.25,
            'strong_buy': 0.5
        }
        
        self.nu_vector = []
        self.view_returns = None
        self.eq_returns = None
        self.eq_risk_premium = None
        self.omega = None
        self.bl_returns = None
    
    def get_bl_returns(self):
        """
        Calculate Black-Litterman expected returns.
        
        Combines equilibrium returns with investor views using Bayesian updating.
        
        Formula:
        E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) [(τΣ)^(-1)Π + P'Ω^(-1)Q]
        
        Returns:
            np.array: Black-Litterman expected returns
        """
        self.get_eq_risk_premium()
        self.get_eq_returns()
        self.get_omega()
        self.get_view_returns()
        
        inv_tau_sigma = np.linalg.inv(self.tau * self.sigma)
        inv_omega = np.linalg.inv(self.omega)
        
        # Posterior precision
        bl_returns_1 = np.linalg.inv(
            inv_tau_sigma + np.dot(np.dot(self.p_matrix.T, inv_omega), self.p_matrix)
        )
        
        # Weighted average of prior and views
        bl_returns_2 = (np.dot(inv_tau_sigma, self.eq_returns) + 
                       np.dot(np.dot(self.p_matrix.T, inv_omega), self.view_returns))
        
        self.bl_returns = np.dot(bl_returns_1, bl_returns_2)
        return self.bl_returns
    
    def get_eq_risk_premium(self):
        """
        Calculate implied equilibrium risk premium.
        
        Π = λ * Σ * w_eq
        
        Returns:
            np.array: Equilibrium risk premium
        """
        self.eq_risk_premium = (np.dot(self.sigma, self.port_wts_eq.T) * 
                               self.lambda_param)
        return self.eq_risk_premium
    
    def get_eq_returns(self):
        """
        Calculate equilibrium expected returns.
        
        Returns:
            np.array: Equilibrium returns (risk premium + risk-free rate)
        """
        self.eq_returns = self.get_eq_risk_premium() + self.risk_free_rate
        return self.eq_returns
    
    def get_omega(self):
        """
        Calculate uncertainty matrix Omega for the views.
        
        Omega represents the uncertainty in the investor's views.
        
        Returns:
            np.array: Omega matrix (diagonal)
        """
        p_sigma_p = np.dot(np.dot(self.p_matrix, self.sigma), self.p_matrix.T)
        u = np.diag(self.conf_level) / self.c_coef
        self.omega = np.diag(np.diag(np.dot(u, np.dot(p_sigma_p, u))))
        return self.omega
    
    def get_view_returns(self):
        """
        Translate qualitative recommendations into quantitative view returns.
        
        Views are expressed as deviations from equilibrium returns,
        scaled by the volatility of the view portfolio.
        
        Returns:
            np.array: Expected returns according to views
        """
        p_sigma_p = np.dot(np.dot(self.p_matrix, self.sigma), self.p_matrix.T)
        
        # Convert qualitative recommendations to numeric adjustments
        nu_vector = np.array([self.reco_dict[reco] for reco in self.reco_vector])
        self.nu_vector = nu_vector
        
        # View mean = equilibrium view + scaled deviation
        view_mean = np.dot(self.p_matrix, self.eq_returns)
        self.view_returns = (view_mean + 
                            np.multiply(np.sqrt(np.diag(p_sigma_p)), nu_vector))
        
        return self.view_returns
    
    def get_posterior_covariance(self):
        """
        Calculate posterior covariance matrix.
        
        Returns:
            np.array: Posterior covariance matrix
        """
        inv_tau_sigma = np.linalg.inv(self.tau * self.sigma)
        inv_omega = np.linalg.inv(self.omega)
        
        posterior_cov = np.linalg.inv(
            inv_tau_sigma + np.dot(np.dot(self.p_matrix.T, inv_omega), self.p_matrix)
        )
        
        return posterior_cov + self.sigma


###############################################################################
# Market Data
###############################################################################
class CMarket(object):
    """
    Market data container and analysis tools.
    
    Loads market data, calculates statistics, and provides methods for
    multivariate analysis including Mahalanobis distance.
    
    Attributes:
        path (str): Directory path
        file_name (str): Data file name
        market_df (pd.DataFrame): Market data
        correlation_matrix (pd.DataFrame): Correlation matrix
        covariance_matrix (pd.DataFrame): Covariance matrix
        inv_covariance_matrix (np.array): Inverse covariance matrix
        historical_mean (pd.Series): Historical means
        data_freq (str): Data frequency
        asset_class (list): List of asset class names
        num_asset (int): Number of assets
        nobs (int): Number of observations
    """
    
    def __init__(self, path, file_name, data_freq):
        """
        Initialize market data object.
        
        Args:
            path (str): Directory path to data file
            file_name (str): Name of Excel file
            data_freq (str): Data frequency ('monthly', 'daily', 'weekly')
        """
        self.path = path
        self.file_name = file_name
        os.chdir(path)
        
        self.market_df = pd.read_excel(os.path.join(path, file_name))
        self.correlation_matrix = self.market_df.corr(method='pearson')
        self.covariance_matrix = self.market_df.cov()
        self.inv_covariance_matrix = np.linalg.inv(self.covariance_matrix)
        self.historical_mean = self.market_df.mean()
        self.data_freq = data_freq
        self.asset_class = list(self.market_df.columns)[1:]
        self.num_asset = len(self.asset_class)
        self.nobs = len(self.market_df.index)
        self.vector_mahalanobis = None
    
    def get_mahalanobis(self):
        """
        Calculate Mahalanobis distance for each observation.
        
        The Mahalanobis distance measures how far an observation is from
        the mean in terms of the covariance structure.
        
        Returns:
            list: Mahalanobis distances for each observation
        """
        v = self.historical_mean.values
        VI = self.inv_covariance_matrix
        vector_mahalanobis = []
        
        for index in range(self.nobs):
            u = self.market_df.iloc[index][1:].values
            M = spatial.distance.mahalanobis(u, v, VI)
            vector_mahalanobis.append(M)
        
        self.vector_mahalanobis = vector_mahalanobis
        return self.vector_mahalanobis
    
    def get_cdf(self):
        """
        Calculate empirical cumulative distribution function for each asset.
        
        Returns:
            dict: Dictionary with sorted values and empirical CDFs for each asset
        """
        cdf_dict = {}
        
        for asset in self.asset_class:
            sorted_values = np.sort(self.market_df[asset].values)
            cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            cdf_dict[asset] = {'values': sorted_values, 'cdf': cdf}
        
        return cdf_dict
    
    def get_summary_statistics(self):
        """
        Calculate summary statistics for all assets.
        
        Returns:
            pd.DataFrame: Summary statistics (mean, std, skew, kurtosis, etc.)
        """
        summary = pd.DataFrame({
            'mean': self.market_df[self.asset_class].mean(),
            'std': self.market_df[self.asset_class].std(),
            'skewness': self.market_df[self.asset_class].skew(),
            'kurtosis': self.market_df[self.asset_class].kurtosis(),
            'min': self.market_df[self.asset_class].min(),
            'max': self.market_df[self.asset_class].max()
        })
        
        return summary


###############################################################################
# Utility Functions
###############################################################################
class CUtility(object):
    """
    Utility function for portfolio evaluation.
    
    Implements various utility functions (quadratic, exponential, power)
    for evaluating portfolio performance from an investor's perspective.
    
    Attributes:
        utility_type (str): Type of utility function
        risk_aversion (float): Risk aversion parameter
    """
    
    def __init__(self, utility_type='quadratic', risk_aversion=2.0):
        """
        Initialize utility function.
        
        Args:
            utility_type (str): Type ('quadratic', 'exponential', 'power')
            risk_aversion (float): Risk aversion coefficient (γ > 0)
        """
        self.utility_type = utility_type
        self.risk_aversion = risk_aversion
    
    def calculate_utility(self, returns):
        """
        Calculate utility for given returns.
        
        Args:
            returns (array-like): Portfolio returns
        
        Returns:
            float: Utility value
        """
        returns = np.array(returns)
        
        if self.utility_type == 'quadratic':
            # U(W) = E(R) - 0.5 * γ * Var(R)
            mean_return = np.mean(returns)
            variance = np.var(returns)
            utility = mean_return - 0.5 * self.risk_aversion * variance
        
        elif self.utility_type == 'exponential':
            # U(W) = -E[exp(-γ * R)]
            utility = -np.mean(np.exp(-self.risk_aversion * returns))
        
        elif self.utility_type == 'power':
            # U(W) = E[R^(1-γ) / (1-γ)]
            if self.risk_aversion == 1:
                utility = np.mean(np.log(1 + returns))
            else:
                utility = np.mean(
                    ((1 + returns)**(1 - self.risk_aversion) - 1) / 
                    (1 - self.risk_aversion)
                )
        
        else:
            raise ValueError(f'Unknown utility type: {self.utility_type}')
        
        return utility
    
    def certainty_equivalent(self, returns):
        """
        Calculate certainty equivalent return.
        
        The certainty equivalent is the risk-free return that provides
        the same utility as the risky portfolio.
        
        Args:
            returns (array-like): Portfolio returns
        
        Returns:
            float: Certainty equivalent return
        """
        returns = np.array(returns)
        
        if self.utility_type == 'quadratic':
            mean_return = np.mean(returns)
            variance = np.var(returns)
            ce = mean_return - 0.5 * self.risk_aversion * variance
        
        elif self.utility_type == 'exponential':
            utility = self.calculate_utility(returns)
            ce = -np.log(-utility) / self.risk_aversion
        
        elif self.utility_type == 'power':
            utility = self.calculate_utility(returns)
            if self.risk_aversion == 1:
                ce = np.exp(utility) - 1
            else:
                ce = ((1 - self.risk_aversion) * utility + 1)**(1/(1 - self.risk_aversion)) - 1
        
        else:
            raise ValueError(f'Unknown utility type: {self.utility_type}')
        
        return ce


###############################################################################
# Copula Opinion Pooling (Placeholder for future implementation)
###############################################################################
class CCopula_opinion_pooling(object):
    """
    Copula Opinion Pooling for combining expert views.
    
    This is a placeholder for future implementation of advanced
    view aggregation using copula methods.
    
    Attributes:
        market (CMarket): Market data object
        views (list): List of expert views
        nsim (int): Number of simulations
    """
    
    def __init__(self, market, views):
        """
        Initialize Copula Opinion Pooling.
        
        Args:
            market (CMarket): Market data object
            views (list): List of expert views to aggregate
        """
        self.market = market
        self.views = views
        self.nsim = 5000
        self.simulator = None
        self.normal_sample = []
        self.chi_sample = []
        self.chi_df = 5
    
    def aggregate_views(self):
        """
        Aggregate multiple expert views using copula methods.
        
        Returns:
            dict: Aggregated view parameters
        """
        # Placeholder for future implementation
        raise NotImplementedError("Copula opinion pooling not yet implemented")


###############################################################################
# Helper Functions
###############################################################################

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe ratio for a return series.
    
    Args:
        returns (array-like): Return series
        risk_free_rate (float): Annual risk-free rate
    
    Returns:
        float: Sharpe ratio
    """
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe


def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sortino ratio (focuses on downside risk).
    
    Args:
        returns (array-like): Return series
        risk_free_rate (float): Annual risk-free rate
    
    Returns:
        float: Sortino ratio
    """
    mean_return = np.mean(returns)
    downside_returns = returns[returns < risk_free_rate]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return np.inf
    
    sortino = (mean_return - risk_free_rate) / downside_std
    return sortino


def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown from a return series.
    
    Args:
        returns (array-like): Return series
    
    Returns:
        float: Maximum drawdown (positive number)
    """
    cumulative = np.cumprod(1 + np.array(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = np.min(drawdown)
    
    return abs(max_dd)


def calculate_value_at_risk(returns, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns (array-like): Return series
        confidence_level (float): Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        float: VaR at specified confidence level
    """
    return -np.percentile(returns, (1 - confidence_level) * 100)


def calculate_conditional_var(returns, confidence_level=0.95):
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    Args:
        returns (array-like): Return series
        confidence_level (float): Confidence level
    
    Returns:
        float: CVaR at specified confidence level
    """
    var = calculate_value_at_risk(returns, confidence_level)
    returns = np.array(returns)
    tail_returns = returns[returns <= -var]
    
    if len(tail_returns) == 0:
        return var
    
    return -np.mean(tail_returns)


###############################################################################
# End of Module
###############################################################################