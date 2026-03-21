"""
Asset Allocation and Portfolio Optimization Framework

This module provides a comprehensive set of tools for portfolio management,
including asset modeling, copula-based simulations, portfolio optimization,
and the Black-Litterman model.

Author: Fernando Ruiz
Created: 02/08/2017
Updated: 21/03/2026
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
        self.data = np.array(data)
        self.asset_class = asset_class
        self.sim_model = sim_model
        self.data_freq = data_freq
        self.calibrated_param = {}

    def param_calibration(self):
        """
        Calibrate model parameters based on historical data.

        Returns:
            dict: Dictionary containing calibrated parameters
        """
        if self.data_freq == 'monthly':
            dt = 12
        elif self.data_freq == 'daily':
            dt = 250
        elif self.data_freq == 'weekly':
            dt = 52
        else:
            raise ValueError(f'Unknown data frequency: {self.data_freq}')

        if self.sim_model == 'vasicek':
            x = self.data[:-1]
            y = self.data[1:]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            mean_rev = intercept / (1 - slope)
            speed_rev = -math.log(slope) * dt
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
            dt = 1.0

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

    Implements Gaussian and t-Student copulas for modelling dependence
    structures between assets.
    """

    def __init__(self, correlation_matrix, nsim):
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
        self.random_noise = np.random.randn(self.nsim, self.num_assets)
        self.random_noise_corr = np.dot(self.random_noise, self.chol_matrix.T)
        return self.random_noise, self.random_noise_corr

    def get_tstudent_copula(self, chi_df=5):
        normal_sample, _ = self.get_correlated_normal()
        chi_sample = np.random.chisquare(chi_df, size=self.nsim)

        mc_sim_nor_corr = []
        mc_sim_uniform_corr = []

        for sim in range(self.nsim):
            aux = math.sqrt(chi_df) * normal_sample[sim, :] / math.sqrt(chi_sample[sim])
            mc_sim_nor_corr.append(aux)
            u = stats.t.cdf(aux, df=chi_df)
            mc_sim_uniform_corr.append(u)

        self.mc_sim_nor_corr = np.array(mc_sim_nor_corr)
        self.mc_sim_uniform_corr = np.array(mc_sim_uniform_corr)

        return self.mc_sim_nor_corr, self.mc_sim_uniform_corr

    def get_correlated_returns(self, marginal_params):
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
    """

    def __init__(self, port_wts, correlation_matrix, expected_ret, vol):
        self.port_wts = np.array(port_wts)

        if isinstance(correlation_matrix, pd.DataFrame):
            self.correlation_matrix = correlation_matrix.values
        else:
            self.correlation_matrix = np.array(correlation_matrix)

        self.expected_ret = np.array(expected_ret)
        self.vol = np.array(vol)

        vol_matrix = np.diag(self.vol)
        self.exp_cov = np.dot(np.dot(vol_matrix, self.correlation_matrix), vol_matrix)

    # ------------------------------------------------------------------ #
    # Core optimisation methods (unchanged)                                #
    # ------------------------------------------------------------------ #

    def mean_variance_opt(self, x_min, x_max, num_port=20):
        """
        Mean-variance portfolio optimisation for different risk aversion levels.

        Returns:
            list: List of optimal portfolio weights for different risk levels
        """
        lambda_vector = np.linspace(0.5, 2, num_port)

        x_min = np.array(x_min)
        x_max = np.array(x_max)

        h = matrix(np.concatenate([x_max, -x_min], 0))
        nr = len(self.expected_ret)
        G = matrix(np.concatenate([np.eye(nr), -np.eye(nr)], 0))

        A = matrix(np.ones(nr).astype('double')).T
        b = matrix(1.0, tc='d')

        PortWts = []
        q = matrix(-self.expected_ret.astype('double'))

        for lam in lambda_vector:
            P = matrix(lam * self.exp_cov.astype('double'))
            sol = solvers.qp(P, q, G, h, A, b)
            PortWts.append(np.array(sol['x']).flatten())

        return PortWts

    def asset_allocation_TEop(self, wts_bmk, tactical_range):
        """
        Portfolio optimisation minimising tracking error.

        Args:
            wts_bmk (array-like): Benchmark weights
            tactical_range (array-like): Maximum deviation from benchmark

        Returns:
            tuple: (optimal_weights, tracking_error)
        """
        wts_bmk = np.array(wts_bmk)
        tactical_range = np.array(tactical_range)

        x_min = -tactical_range
        x_max = tactical_range
        h = matrix(np.concatenate([x_max, -x_min], 0))

        nr = len(self.expected_ret)
        G = matrix(np.concatenate([np.eye(nr), -np.eye(nr)], 0))

        A = matrix(np.ones(nr).astype('double')).T
        b = matrix(0.0, tc='d')

        P = matrix(self.exp_cov.astype('double'))
        q = matrix(-self.expected_ret.astype('double'))

        sol = solvers.qp(P, q, G, h, A, b)
        active_positions = np.array(sol['x']).flatten()

        wts_opt = wts_bmk + active_positions
        port_te = self.tracking_error_ex_ante(wts_bmk, wts_opt)

        return wts_opt, port_te

    def tracking_error_ex_ante(self, wts_bmk, wts_port):
        """
        Calculate ex-ante tracking error.

        TE = sqrt((w_p - w_b)' Σ (w_p - w_b))

        Returns:
            float: Annualised tracking error
        """
        active_wts = np.array(wts_port) - np.array(wts_bmk)
        variance = np.dot(np.dot(active_wts, self.exp_cov), active_wts)
        return math.sqrt(variance)

    def efficient_frontier(self, x_min, x_max, num_points=50):
        """
        Generate the efficient frontier.

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

    # ------------------------------------------------------------------ #
    # NEW — portfolio risk & return metrics (used by Streamlit app)        #
    # ------------------------------------------------------------------ #

    def get_portfolio_metrics(self, weights, risk_free_rate=0.02, n_sims=50_000,
                              seed=42):
        """
        Compute comprehensive risk and return metrics for a given weight vector.

        Uses Monte Carlo simulation (multivariate normal, annual horizon) to
        derive distributional statistics that are not analytically available
        (probability of loss, Expected Shortfall).

        Args:
            weights (array-like): Portfolio weights (must sum to 1).
            risk_free_rate (float): Annual risk-free rate for Sharpe calculation.
            n_sims (int): Number of Monte Carlo scenarios.
            seed (int): Random seed for reproducibility.

        Returns:
            dict with keys:
                expected_return  – annual portfolio expected return
                volatility       – annual portfolio volatility
                sharpe           – Sharpe ratio
                prob_loss        – probability of negative 1-year return
                var_95           – 5th-percentile 1-year return (VaR 95 %)
                es_95            – Expected Shortfall at 95 % confidence
                sim_returns      – full array of simulated 1-year returns
                marginal_risk    – per-asset fractional variance contribution
        """
        w = np.array(weights, dtype=float)

        # ── Analytical moments ────────────────────────────────────────────
        mu    = float(w @ self.expected_ret)
        sigma = float(np.sqrt(w @ self.exp_cov @ w))

        # ── Monte Carlo ───────────────────────────────────────────────────
        rng = np.random.default_rng(seed)
        sim_returns = rng.normal(mu, sigma, n_sims)

        prob_loss = float(np.mean(sim_returns < 0))
        var_95    = float(np.percentile(sim_returns, 5))
        es_95     = float(sim_returns[sim_returns <= var_95].mean())
        sharpe    = (mu - risk_free_rate) / sigma if sigma > 0 else 0.0

        # ── Marginal risk contribution ────────────────────────────────────
        port_var      = float(w @ self.exp_cov @ w)
        marginal_risk = (w * (self.exp_cov @ w)) / port_var  # sums to 1

        return {
            'expected_return': mu,
            'volatility':      sigma,
            'sharpe':          sharpe,
            'prob_loss':       prob_loss,
            'var_95':          var_95,
            'es_95':           es_95,
            'sim_returns':     sim_returns,
            'marginal_risk':   marginal_risk,
        }


###############################################################################
# Black-Litterman Model
###############################################################################
class CBlack_litterman(object):
    """
    Implementation of the Black-Litterman asset allocation model.

    Combines market equilibrium with investor views to generate
    expected returns that incorporate both sources of information.
    """

    # Qualitative recommendation → numeric view adjustment
    RECO_DICT = {
        'strong_buy':  0.50,
        'buy':         0.25,
        'neutral':     0.00,
        'sell':       -0.25,
        'strong_sell': -0.50,
    }

    def __init__(self, risk_free_rate, tau, sigma, p_matrix, lambda_param,
                 port_wts_eq, conf_level, c_coef, reco_vector):
        self.tau            = tau
        self.risk_free_rate = risk_free_rate
        self.sigma          = np.array(sigma, dtype=float)
        self.p_matrix       = np.array(p_matrix, dtype=float)
        self.lambda_param   = lambda_param
        self.port_wts_eq    = np.array(port_wts_eq, dtype=float)
        self.conf_level     = np.array(conf_level, dtype=float)
        self.c_coef         = c_coef
        self.reco_vector    = reco_vector
        self.num_recos      = len(reco_vector)

        # Computed quantities (populated by get_bl_returns)
        self.nu_vector    = []
        self.view_returns = None
        self.eq_returns   = None
        self.eq_risk_premium = None
        self.omega        = None
        self.bl_returns   = None

    # ------------------------------------------------------------------ #
    # Core BL steps                                                        #
    # ------------------------------------------------------------------ #

    def get_eq_risk_premium(self):
        """Implied equilibrium risk premium: Π = λ Σ w_eq"""
        self.eq_risk_premium = self.lambda_param * (self.sigma @ self.port_wts_eq)
        return self.eq_risk_premium

    def get_eq_returns(self):
        """Equilibrium returns = risk premium + risk-free rate."""
        self.eq_returns = self.get_eq_risk_premium() + self.risk_free_rate
        return self.eq_returns

    def get_omega(self):
        """
        Uncertainty matrix Ω for the views.

        Ω = diag(u · P Σ P' · u)   where u = diag(conf) / c_coef
        """
        p_sigma_p = self.p_matrix @ self.sigma @ self.p_matrix.T
        u = np.diag(self.conf_level) / self.c_coef
        self.omega = np.diag(np.diag(u @ p_sigma_p @ u))
        return self.omega

    def get_view_returns(self):
        """
        Translate qualitative recommendations into quantitative view returns.

        Q = P Π + ν · sqrt(diag(P Σ P'))
        """
        p_sigma_p = self.p_matrix @ self.sigma @ self.p_matrix.T
        nu_vector = np.array([self.RECO_DICT[r] for r in self.reco_vector])
        self.nu_vector = nu_vector

        view_mean = self.p_matrix @ self.eq_returns
        self.view_returns = view_mean + nu_vector * np.sqrt(np.diag(p_sigma_p))
        return self.view_returns

    def get_bl_returns(self):
        """
        Black-Litterman posterior expected returns.

        E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]

        Returns:
            np.array: BL posterior expected returns (one per asset)
        """
        self.get_eq_returns()   # sets self.eq_returns
        self.get_omega()        # sets self.omega
        self.get_view_returns() # sets self.view_returns

        inv_tau_sigma = np.linalg.inv(self.tau * self.sigma)
        inv_omega     = np.linalg.inv(self.omega)

        precision  = inv_tau_sigma + self.p_matrix.T @ inv_omega @ self.p_matrix
        rhs        = inv_tau_sigma @ self.eq_returns + self.p_matrix.T @ inv_omega @ self.view_returns

        self.bl_returns = np.linalg.inv(precision) @ rhs
        return self.bl_returns

    def get_posterior_covariance(self):
        """Posterior covariance matrix (M⁻¹ + Σ)."""
        inv_tau_sigma = np.linalg.inv(self.tau * self.sigma)
        inv_omega     = np.linalg.inv(self.omega)
        M_inv = np.linalg.inv(
            inv_tau_sigma + self.p_matrix.T @ inv_omega @ self.p_matrix
        )
        return M_inv + self.sigma

    # ------------------------------------------------------------------ #
    # NEW — convenience constructor called by the Streamlit app            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_views(cls, eq_returns, cov_matrix, weights_eq, p_matrix,
                   recommendations, confidence_levels,
                   tau=0.025, lambda_param=2.5, risk_free_rate=0.02, c_coef=1.0):
        """
        Simplified factory that accepts pre-built arrays.

        Unlike the main constructor (which derives eq_returns internally from
        weights and lambda), here the caller supplies already-computed
        equilibrium returns—useful when the app stores them in session state.

        Returns:
            tuple: (pi_equilibrium, bl_posterior)  — both as np.array
        """
        instance = cls(
            risk_free_rate=risk_free_rate,
            tau=tau,
            sigma=np.array(cov_matrix),
            p_matrix=np.array(p_matrix),
            lambda_param=lambda_param,
            port_wts_eq=np.array(weights_eq),
            conf_level=np.array(confidence_levels),
            c_coef=c_coef,
            reco_vector=recommendations,
        )
        bl_post = instance.get_bl_returns()
        pi_eq   = instance.eq_returns          # set as side-effect of get_bl_returns
        return pi_eq, bl_post


###############################################################################
# Market Data
###############################################################################
class CMarket(object):
    """
    Market data container and analysis tools.
    """

    def __init__(self, path, file_name, data_freq):
        self.path      = path
        self.file_name = file_name
        os.chdir(path)

        self.market_df           = pd.read_excel(os.path.join(path, file_name))
        self.correlation_matrix  = self.market_df.corr(method='pearson')
        self.covariance_matrix   = self.market_df.cov()
        self.inv_covariance_matrix = np.linalg.inv(self.covariance_matrix)
        self.historical_mean     = self.market_df.mean()
        self.data_freq           = data_freq
        self.asset_class         = list(self.market_df.columns)[1:]
        self.num_asset           = len(self.asset_class)
        self.nobs                = len(self.market_df.index)
        self.vector_mahalanobis  = None

    def get_mahalanobis(self):
        v  = self.historical_mean.values
        VI = self.inv_covariance_matrix
        vector_mahalanobis = []

        for index in range(self.nobs):
            u = self.market_df.iloc[index][1:].values
            M = spatial.distance.mahalanobis(u, v, VI)
            vector_mahalanobis.append(M)

        self.vector_mahalanobis = vector_mahalanobis
        return self.vector_mahalanobis

    def get_cdf(self):
        cdf_dict = {}
        for asset in self.asset_class:
            sorted_values = np.sort(self.market_df[asset].values)
            cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            cdf_dict[asset] = {'values': sorted_values, 'cdf': cdf}
        return cdf_dict

    def get_summary_statistics(self):
        summary = pd.DataFrame({
            'mean':     self.market_df[self.asset_class].mean(),
            'std':      self.market_df[self.asset_class].std(),
            'skewness': self.market_df[self.asset_class].skew(),
            'kurtosis': self.market_df[self.asset_class].kurtosis(),
            'min':      self.market_df[self.asset_class].min(),
            'max':      self.market_df[self.asset_class].max()
        })
        return summary


###############################################################################
# Utility Functions
###############################################################################
class CUtility(object):
    """
    Utility function for portfolio evaluation.
    """

    def __init__(self, utility_type='quadratic', risk_aversion=2.0):
        self.utility_type  = utility_type
        self.risk_aversion = risk_aversion

    def calculate_utility(self, returns):
        returns = np.array(returns)

        if self.utility_type == 'quadratic':
            utility = np.mean(returns) - 0.5 * self.risk_aversion * np.var(returns)

        elif self.utility_type == 'exponential':
            utility = -np.mean(np.exp(-self.risk_aversion * returns))

        elif self.utility_type == 'power':
            if self.risk_aversion == 1:
                utility = np.mean(np.log(1 + returns))
            else:
                utility = np.mean(
                    ((1 + returns) ** (1 - self.risk_aversion) - 1) /
                    (1 - self.risk_aversion)
                )
        else:
            raise ValueError(f'Unknown utility type: {self.utility_type}')

        return utility

    def certainty_equivalent(self, returns):
        returns = np.array(returns)

        if self.utility_type == 'quadratic':
            ce = np.mean(returns) - 0.5 * self.risk_aversion * np.var(returns)

        elif self.utility_type == 'exponential':
            utility = self.calculate_utility(returns)
            ce = -np.log(-utility) / self.risk_aversion

        elif self.utility_type == 'power':
            utility = self.calculate_utility(returns)
            if self.risk_aversion == 1:
                ce = np.exp(utility) - 1
            else:
                ce = ((1 - self.risk_aversion) * utility + 1) ** (1 / (1 - self.risk_aversion)) - 1
        else:
            raise ValueError(f'Unknown utility type: {self.utility_type}')

        return ce


###############################################################################
# Copula Opinion Pooling (placeholder)
###############################################################################
class CCopula_opinion_pooling(object):
    def __init__(self, market, views):
        self.market  = market
        self.views   = views
        self.nsim    = 5000
        self.chi_df  = 5

    def aggregate_views(self):
        raise NotImplementedError("Copula opinion pooling not yet implemented")


###############################################################################
# Module-level helper functions
###############################################################################

def build_cov_matrix(volatilities, correlation_matrix):
    """
    Build an annualised covariance matrix from volatilities and correlations.

    Args:
        volatilities (array-like): Annual volatilities for each asset.
        correlation_matrix (array-like): Symmetric correlation matrix.

    Returns:
        np.array: Covariance matrix.
    """
    v = np.array(volatilities, dtype=float)
    c = np.array(correlation_matrix, dtype=float)
    return np.diag(v) @ c @ np.diag(v)


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    mean_return = np.mean(returns)
    std_return  = np.std(returns)
    if std_return == 0:
        return 0
    return (mean_return - risk_free_rate) / std_return


def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    mean_return     = np.mean(returns)
    downside        = returns[returns < risk_free_rate]
    if len(downside) == 0:
        return np.inf
    downside_std = np.std(downside)
    if downside_std == 0:
        return np.inf
    return (mean_return - risk_free_rate) / downside_std


def calculate_max_drawdown(returns):
    cumulative  = np.cumprod(1 + np.array(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdown    = (cumulative - running_max) / running_max
    return abs(drawdown.min())


def calculate_value_at_risk(returns, confidence_level=0.95):
    return -np.percentile(returns, (1 - confidence_level) * 100)


def calculate_conditional_var(returns, confidence_level=0.95):
    var        = calculate_value_at_risk(returns, confidence_level)
    returns    = np.array(returns)
    tail       = returns[returns <= -var]
    if len(tail) == 0:
        return var
    return -np.mean(tail)


###############################################################################
# End of Module
###############################################################################
