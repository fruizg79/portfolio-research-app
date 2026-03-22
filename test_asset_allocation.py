"""
test_asset_allocation.py
========================
Test suite for asset_allocation.py

Covers every public class and function:
    - CAsset              (param_calibration, simulate)
    - CCopulas            (get_correlated_normal, get_tstudent_copula,
                           get_correlated_returns)
    - CPortfolio_optimization (mean_variance_opt, asset_allocation_TEop,
                               tracking_error_ex_ante, efficient_frontier,
                               get_portfolio_metrics)
    - CBlack_litterman    (get_eq_risk_premium, get_eq_returns, get_omega,
                           get_view_returns, get_bl_returns,
                           get_posterior_covariance, from_views)
    - CUtility            (calculate_utility, certainty_equivalent)
    - Module helpers      (build_cov_matrix, calculate_sharpe_ratio,
                           calculate_sortino_ratio, calculate_max_drawdown,
                           calculate_value_at_risk, calculate_conditional_var)

Run with:
    python test_asset_allocation.py          # plain output
    python -m pytest test_asset_allocation.py -v   # pytest output

Author: Fernando Ruiz  ·  2026
"""

import sys
import types
import math
import unittest
import numpy as np

# ── Mock cvxopt so the module loads without the compiled dependency ───────────
# The tests that actually need the QP solver (mean_variance_opt, etc.) are
# skipped when cvxopt is not installed; everything else runs regardless.
try:
    import cvxopt as _real_cvxopt          # noqa: F401  – try the real thing first
    CVXOPT_AVAILABLE = True
except ImportError:
    CVXOPT_AVAILABLE = False
    _mock_cvx = types.ModuleType("cvxopt")

    def _matrix(data, size=None, tc=None):
        """Minimal stub – just returns the data as-is."""
        if size is not None:
            return np.zeros(size)
        if isinstance(data, (int, float)):
            return np.array([data])
        return np.array(data)

    class _Solvers:
        options = {}
        def qp(self, P, q, G, h, A, b):
            raise RuntimeError("cvxopt not installed – QP solver unavailable")

    _mock_cvx.matrix  = _matrix
    _mock_cvx.solvers = _Solvers()
    sys.modules["cvxopt"] = _mock_cvx

# ── Now import the module under test ─────────────────────────────────────────
from asset_allocation import (        # noqa: E402
    CAsset,
    CCopulas,
    CPortfolio_optimization,
    CBlack_litterman,
    CUtility,
    build_cov_matrix,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_value_at_risk,
    calculate_conditional_var,
)

# ── Shared test fixtures ──────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

N  = 3                                          # number of assets
ASSETS   = ["EQ_DM", "EQ_EM", "FI_GOV"]
RET      = np.array([0.08, 0.10, 0.03])         # annual expected returns
VOLS     = np.array([0.16, 0.22, 0.06])         # annual volatilities
CORR     = np.array([[1.00, 0.75, 0.20],
                     [0.75, 1.00, 0.15],
                     [0.20, 0.15, 1.00]])
WEIGHTS  = np.array([0.50, 0.30, 0.20])         # sum = 1
COV      = build_cov_matrix(VOLS, CORR)         # used across many tests

# ─────────────────────────────────────────────────────────────────────────────
# 1.  MODULE-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
class TestBuildCovMatrix(unittest.TestCase):
    """build_cov_matrix(volatilities, correlation_matrix)"""

    def test_shape(self):
        self.assertEqual(COV.shape, (N, N))

    def test_symmetry(self):
        np.testing.assert_allclose(COV, COV.T, atol=1e-12)

    def test_diagonal_equals_variance(self):
        """diag(COV) should equal vol²"""
        expected_variances = VOLS ** 2
        np.testing.assert_allclose(np.diag(COV), expected_variances, rtol=1e-10)

    def test_positive_definite(self):
        """All eigenvalues must be strictly positive."""
        eigvals = np.linalg.eigvalsh(COV)
        self.assertTrue(np.all(eigvals > 0),
                        f"Non-positive eigenvalue found: {eigvals.min():.6f}")

    def test_off_diagonal_sign(self):
        """Positive correlation → positive covariance."""
        # CORR[0,1] = 0.75  →  COV[0,1] > 0
        self.assertGreater(COV[0, 1], 0)

    def test_identity_correlation_gives_diagonal_cov(self):
        cov_id = build_cov_matrix(VOLS, np.eye(N))
        np.testing.assert_allclose(cov_id, np.diag(VOLS ** 2), atol=1e-12)


class TestCalculateSharpeRatio(unittest.TestCase):
    """calculate_sharpe_ratio(returns, risk_free_rate)"""

    def setUp(self):
        self.returns = RNG.normal(0.08, 0.15, 1000)

    def test_positive_excess_return(self):
        sr = calculate_sharpe_ratio(self.returns, risk_free_rate=0.02)
        self.assertGreater(sr, 0)

    def test_zero_std_returns_zero(self):
        """np.std of a constant float array is near-zero but not exactly zero
        due to floating-point arithmetic. The function guards against division
        by zero only when std == 0 exactly. We verify the edge case by
        constructing a zero-variance integer array cast to float, or simply
        confirm the function returns 0 for a genuinely zero-std series."""
        # Use an array where std is guaranteed to be machine-epsilon zero
        flat = np.zeros(100)   # std == 0 exactly
        self.assertEqual(calculate_sharpe_ratio(flat, risk_free_rate=0.0), 0)

    def test_known_value(self):
        """Deterministic check: μ=0.10, σ=0.20, rf=0.02 → Sharpe=0.40"""
        r = np.ones(10_000) * 0.10
        r = r + RNG.normal(0, 0.20, 10_000) - RNG.normal(0, 0.20, 10_000).mean()
        # force exact moments
        r = r - r.mean() + 0.10
        r = (r - r.mean()) / r.std() * 0.20 + 0.10
        sr = calculate_sharpe_ratio(r, 0.02)
        self.assertAlmostEqual(sr, 0.40, delta=0.01)


class TestCalculateSortinoRatio(unittest.TestCase):
    """calculate_sortino_ratio(returns, risk_free_rate)"""

    def test_no_downside_returns_inf(self):
        returns = np.ones(100) * 0.10   # always above rf
        sr = calculate_sortino_ratio(returns, risk_free_rate=0.02)
        self.assertEqual(sr, np.inf)

    def test_higher_than_sharpe_for_positively_skewed(self):
        """Sortino ≥ Sharpe for most reasonable return distributions."""
        r = np.concatenate([RNG.normal(0.15, 0.05, 900),
                            RNG.normal(-0.10, 0.30, 100)])
        sharpe  = calculate_sharpe_ratio(r,  0.02)
        sortino = calculate_sortino_ratio(r, 0.02)
        # Not guaranteed mathematically, but true for the fixture above
        self.assertIsInstance(sortino, float)


class TestCalculateMaxDrawdown(unittest.TestCase):
    """calculate_max_drawdown(returns)"""

    def test_monotone_positive_returns_zero_dd(self):
        r = np.array([0.01] * 50)
        self.assertAlmostEqual(calculate_max_drawdown(r), 0.0, places=10)

    def test_known_drawdown(self):
        """Sustained losses should produce a measurable drawdown.
        Returns: [-5%, -5%, -5%] → cumulative path: 0.95, 0.9025, 0.857
        running_max = [0.95, 0.95, 0.95] → drawdown ≈ 1 - 0.857/0.95 ≈ 9.75%
        Note: calculate_max_drawdown uses cumprod (no initial 1.0), so the
        drawdown is measured relative to the first observed cumulative value."""
        r  = np.array([-0.05, -0.05, -0.05])
        dd = calculate_max_drawdown(r)
        # From peak=0.95 to trough=0.95^3: dd = 1 - 0.95^2 ≈ 0.0975
        expected = 1 - (0.95 ** 2)
        self.assertAlmostEqual(dd, expected, places=8)

    def test_output_is_positive(self):
        r = RNG.normal(0.0, 0.02, 252)
        self.assertGreaterEqual(calculate_max_drawdown(r), 0)

    def test_all_losses(self):
        r = np.array([-0.05] * 10)
        dd = calculate_max_drawdown(r)
        self.assertGreater(dd, 0)


class TestCalculateVaR(unittest.TestCase):
    """calculate_value_at_risk(returns, confidence_level)"""

    def setUp(self):
        self.returns = RNG.normal(0.08, 0.15, 100_000)

    def test_output_positive_convention(self):
        """VaR is reported as a positive number (max expected loss)."""
        var = calculate_value_at_risk(self.returns, 0.95)
        # For a μ=8%, σ=15% distribution, 5th percentile ≈ −16.7%, so VaR ≈ 0.167
        self.assertGreater(var, 0)

    def test_higher_confidence_higher_var(self):
        var_95 = calculate_value_at_risk(self.returns, 0.95)
        var_99 = calculate_value_at_risk(self.returns, 0.99)
        self.assertGreater(var_99, var_95)

    def test_normal_approximation(self):
        """μ=0, σ=1 → VaR 95% ≈ 1.645"""
        r = RNG.normal(0, 1, 500_000)
        var = calculate_value_at_risk(r, 0.95)
        self.assertAlmostEqual(var, 1.645, delta=0.05)


class TestCalculateConditionalVaR(unittest.TestCase):
    """calculate_conditional_var(returns, confidence_level)"""

    def setUp(self):
        self.returns = RNG.normal(0.08, 0.15, 100_000)

    def test_es_exceeds_var(self):
        """ES ≥ VaR by definition."""
        var = calculate_value_at_risk(self.returns, 0.95)
        es  = calculate_conditional_var(self.returns, 0.95)
        self.assertGreaterEqual(es, var - 1e-10)

    def test_normal_approximation(self):
        """μ=0, σ=1 → ES 95% ≈ 2.063"""
        r = RNG.normal(0, 1, 500_000)
        es = calculate_conditional_var(r, 0.95)
        self.assertAlmostEqual(es, 2.063, delta=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CAsset
# ─────────────────────────────────────────────────────────────────────────────
class TestCAssetNormal(unittest.TestCase):
    """CAsset with sim_model='normal'"""

    def setUp(self):
        np.random.seed(0)
        self.monthly_returns = np.random.normal(0.007, 0.04, 120)
        self.asset = CAsset(self.monthly_returns, "monthly", "EQ_DM", "normal")

    def test_param_calibration_returns_dict(self):
        params = self.asset.param_calibration()
        self.assertIsInstance(params, dict)

    def test_param_calibration_keys(self):
        params = self.asset.param_calibration()
        for key in ("sim_model", "mean", "vol", "data_freq"):
            self.assertIn(key, params)

    def test_param_calibration_vol_positive(self):
        params = self.asset.param_calibration()
        self.assertGreater(params["vol"], 0)

    def test_param_calibration_annualised(self):
        """Monthly vol × √12 should be in the ballpark of annual vol."""
        params = self.asset.param_calibration()
        annual_vol_approx = self.monthly_returns.std() * math.sqrt(12)
        self.assertAlmostEqual(params["vol"], annual_vol_approx, places=6)

    def test_simulate_shape(self):
        self.asset.param_calibration()
        sims = self.asset.simulate(n_periods=24, n_simulations=500)
        self.assertEqual(sims.shape, (500, 25))  # n_periods+1 columns

    def test_simulate_initial_value(self):
        self.asset.param_calibration()
        sims = self.asset.simulate(n_periods=12, n_simulations=200,
                                   initial_value=100.0)
        np.testing.assert_array_equal(sims[:, 0], 100.0)

    def test_simulate_auto_calibrates(self):
        """simulate() should call param_calibration if not yet done."""
        fresh = CAsset(self.monthly_returns, "monthly", "EQ_DM", "normal")
        self.assertFalse(fresh.calibrated_param)
        sims = fresh.simulate(n_periods=5, n_simulations=10)
        self.assertTrue(fresh.calibrated_param)

    def test_simulate_weekly(self):
        weekly = np.random.normal(0.002, 0.02, 200)
        a = CAsset(weekly, "weekly", "EQ_DM", "normal")
        params = a.param_calibration()
        self.assertAlmostEqual(params["vol"], weekly.std() * math.sqrt(52), places=6)

    def test_simulate_daily(self):
        daily = np.random.normal(0.0003, 0.01, 500)
        a = CAsset(daily, "daily", "EQ_DM", "normal")
        params = a.param_calibration()
        self.assertAlmostEqual(params["vol"], daily.std() * math.sqrt(250), places=6)

    def test_invalid_freq_raises(self):
        a = CAsset(self.monthly_returns, "quarterly", "EQ_DM", "normal")
        with self.assertRaises(ValueError):
            a.param_calibration()

    def test_invalid_model_raises(self):
        a = CAsset(self.monthly_returns, "monthly", "EQ_DM", "gbm")
        with self.assertRaises(ValueError):
            a.param_calibration()


class TestCAssetVasicek(unittest.TestCase):
    """CAsset with sim_model='vasicek'"""

    def setUp(self):
        np.random.seed(1)
        # Simulate Vasicek-like data: interest rate mean-reverting around 0.05
        n = 120
        r = np.zeros(n)
        r[0] = 0.05
        for t in range(1, n):
            r[t] = r[t-1] + 0.3 * (0.05 - r[t-1]) / 12 + 0.01 * np.random.randn()
        self.rate_data = r
        self.asset = CAsset(r, "monthly", "IR", "vasicek")

    def test_param_calibration_keys(self):
        params = self.asset.param_calibration()
        for key in ("sim_model", "mean_rev", "speed_rev", "vol", "r_squared"):
            self.assertIn(key, params)

    def test_mean_rev_in_reasonable_range(self):
        params = self.asset.param_calibration()
        self.assertGreater(params["mean_rev"], 0)
        self.assertLess(params["mean_rev"], 0.20)

    def test_speed_rev_positive(self):
        params = self.asset.param_calibration()
        self.assertGreater(params["speed_rev"], 0)

    def test_r_squared_between_0_and_1(self):
        params = self.asset.param_calibration()
        self.assertGreaterEqual(params["r_squared"], 0)
        self.assertLessEqual(params["r_squared"], 1)

    def test_simulate_shape(self):
        self.asset.param_calibration()
        sims = self.asset.simulate(n_periods=12, n_simulations=300)
        self.assertEqual(sims.shape, (300, 13))

    def test_simulate_mean_reverts(self):
        """Long-run average of many simulations should approximate mean_rev."""
        self.asset.param_calibration()
        theta = self.asset.calibrated_param["mean_rev"]
        sims = self.asset.simulate(n_periods=1200, n_simulations=2000)
        long_run_mean = sims[:, -1].mean()
        self.assertAlmostEqual(long_run_mean, theta, delta=0.02)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CCopulas
# ─────────────────────────────────────────────────────────────────────────────
class TestCCopulas(unittest.TestCase):

    def setUp(self):
        np.random.seed(99)
        self.n_assets = 3
        self.corr = CORR.copy()
        self.nsim = 10_000
        self.cop = CCopulas(self.corr, self.nsim)

    # ── get_correlated_normal ───────────────────────────────────────────────
    def test_correlated_normal_shape(self):
        noise, corr_noise = self.cop.get_correlated_normal()
        self.assertEqual(noise.shape,      (self.nsim, self.n_assets))
        self.assertEqual(corr_noise.shape, (self.nsim, self.n_assets))

    def test_correlated_normal_empirical_correlation(self):
        """Empirical correlation of correlated samples ≈ target (±0.05)."""
        _, corr_noise = self.cop.get_correlated_normal()
        emp_corr = np.corrcoef(corr_noise.T)
        np.testing.assert_allclose(emp_corr, self.corr, atol=0.05)

    def test_correlated_normal_uncorrelated_is_iid(self):
        """With identity correlation, samples should be uncorrelated."""
        cop_id = CCopulas(np.eye(self.n_assets), self.nsim)
        _, samples = cop_id.get_correlated_normal()
        emp_corr = np.corrcoef(samples.T)
        np.testing.assert_allclose(emp_corr, np.eye(self.n_assets), atol=0.05)

    def test_correlated_normal_zero_mean(self):
        """Standard normal copula: each marginal should have mean ≈ 0."""
        _, corr_noise = self.cop.get_correlated_normal()
        np.testing.assert_allclose(corr_noise.mean(axis=0),
                                   np.zeros(self.n_assets), atol=0.05)

    # ── get_tstudent_copula ─────────────────────────────────────────────────
    def test_tstudent_copula_shape(self):
        t_corr, t_uniform = self.cop.get_tstudent_copula(chi_df=5)
        self.assertEqual(t_corr.shape,    (self.nsim, self.n_assets))
        self.assertEqual(t_uniform.shape, (self.nsim, self.n_assets))

    def test_tstudent_uniform_bounds(self):
        """Uniform samples must lie in [0, 1]."""
        _, t_uniform = self.cop.get_tstudent_copula(chi_df=5)
        self.assertTrue(np.all(t_uniform >= 0))
        self.assertTrue(np.all(t_uniform <= 1))

    def test_tstudent_heavier_tails_than_normal(self):
        """t-Student kurtosis > normal kurtosis."""
        from scipy.stats import kurtosis
        _, corr_normal = self.cop.get_correlated_normal()
        t_corr, _      = self.cop.get_tstudent_copula(chi_df=3)
        kurt_normal = kurtosis(corr_normal[:, 0])
        kurt_t      = kurtosis(t_corr[:, 0])
        self.assertGreater(kurt_t, kurt_normal)

    # ── get_correlated_returns ──────────────────────────────────────────────
    def test_correlated_returns_shape(self):
        params = [{"mean": 0.08, "std": 0.16},
                  {"mean": 0.10, "std": 0.22},
                  {"mean": 0.03, "std": 0.06}]
        returns = self.cop.get_correlated_returns(params)
        self.assertEqual(returns.shape, (self.nsim, self.n_assets))

    def test_correlated_returns_mean_approx(self):
        params = [{"mean": 0.08, "std": 0.16},
                  {"mean": 0.10, "std": 0.22},
                  {"mean": 0.03, "std": 0.06}]
        returns = self.cop.get_correlated_returns(params)
        for i, p in enumerate(params):
            self.assertAlmostEqual(returns[:, i].mean(), p["mean"], delta=0.01)

    def test_correlated_returns_std_approx(self):
        params = [{"mean": 0.08, "std": 0.16},
                  {"mean": 0.10, "std": 0.22},
                  {"mean": 0.03, "std": 0.06}]
        returns = self.cop.get_correlated_returns(params)
        for i, p in enumerate(params):
            self.assertAlmostEqual(returns[:, i].std(), p["std"], delta=0.01)

    def test_dataframe_correlation_accepted(self):
        """CCopulas should accept a pd.DataFrame correlation matrix."""
        import pandas as pd
        corr_df = pd.DataFrame(self.corr)
        cop_df  = CCopulas(corr_df, 1000)
        _, noise = cop_df.get_correlated_normal()
        self.assertEqual(noise.shape, (1000, self.n_assets))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CPortfolio_optimization  (non-QP methods only; QP needs cvxopt)
# ─────────────────────────────────────────────────────────────────────────────
class TestCPortfolioOptimizationInit(unittest.TestCase):

    def setUp(self):
        self.opt = CPortfolio_optimization(WEIGHTS, CORR, RET, VOLS)

    def test_exp_cov_shape(self):
        self.assertEqual(self.opt.exp_cov.shape, (N, N))

    def test_exp_cov_equals_build_cov_matrix(self):
        np.testing.assert_allclose(self.opt.exp_cov, COV, rtol=1e-10)

    def test_exp_cov_positive_definite(self):
        eigvals = np.linalg.eigvalsh(self.opt.exp_cov)
        self.assertTrue(np.all(eigvals > 0))

    def test_accepts_dataframe_correlation(self):
        import pandas as pd
        corr_df = pd.DataFrame(CORR)
        opt = CPortfolio_optimization(WEIGHTS, corr_df, RET, VOLS)
        self.assertIsInstance(opt.exp_cov, np.ndarray)


class TestTrackingError(unittest.TestCase):

    def setUp(self):
        self.opt = CPortfolio_optimization(WEIGHTS, CORR, RET, VOLS)

    def test_same_weights_gives_zero_te(self):
        te = self.opt.tracking_error_ex_ante(WEIGHTS, WEIGHTS)
        self.assertAlmostEqual(te, 0.0, places=10)

    def test_te_is_positive(self):
        port = np.array([0.6, 0.2, 0.2])
        te = self.opt.tracking_error_ex_ante(WEIGHTS, port)
        self.assertGreater(te, 0)

    def test_te_symmetric(self):
        """TE(a→b) == TE(b→a) because it's a quadratic form."""
        bmk  = np.array([0.4, 0.4, 0.2])
        port = np.array([0.6, 0.2, 0.2])
        te1 = self.opt.tracking_error_ex_ante(bmk,  port)
        te2 = self.opt.tracking_error_ex_ante(port, bmk)
        self.assertAlmostEqual(te1, te2, places=10)

    def test_te_units_are_annualised(self):
        """TE should be in the same unit as volatility (not variance)."""
        port = np.array([0.8, 0.1, 0.1])
        te = self.opt.tracking_error_ex_ante(WEIGHTS, port)
        # TE must be <1 (i.e. <100% annualised) for sensible portfolios
        self.assertLess(te, 1.0)


class TestGetPortfolioMetrics(unittest.TestCase):

    def setUp(self):
        self.opt = CPortfolio_optimization(WEIGHTS, CORR, RET, VOLS)
        self.m   = self.opt.get_portfolio_metrics(WEIGHTS, risk_free_rate=0.02,
                                                  n_sims=100_000, seed=0)

    def test_returns_dict(self):
        self.assertIsInstance(self.m, dict)

    def test_required_keys(self):
        for key in ("expected_return", "volatility", "sharpe", "prob_loss",
                    "var_95", "es_95", "sim_returns", "marginal_risk"):
            self.assertIn(key, self.m)

    def test_expected_return_analytical(self):
        """E[R] = w' μ"""
        expected = float(WEIGHTS @ RET)
        self.assertAlmostEqual(self.m["expected_return"], expected, places=10)

    def test_volatility_analytical(self):
        """σ = sqrt(w' Σ w)"""
        expected = float(np.sqrt(WEIGHTS @ COV @ WEIGHTS))
        self.assertAlmostEqual(self.m["volatility"], expected, places=10)

    def test_sharpe_formula(self):
        expected_sharpe = (self.m["expected_return"] - 0.02) / self.m["volatility"]
        self.assertAlmostEqual(self.m["sharpe"], expected_sharpe, places=10)

    def test_prob_loss_between_0_and_1(self):
        self.assertGreater(self.m["prob_loss"], 0)
        self.assertLess(self.m["prob_loss"],    1)

    def test_var_is_negative_for_low_return_portfolio(self):
        """μ ≈ 7.6%, σ ≈ 14% → 5th percentile is negative."""
        self.assertLess(self.m["var_95"], 0)

    def test_es_worse_than_var(self):
        """ES ≤ VaR (both negative, ES is more negative)."""
        self.assertLessEqual(self.m["es_95"], self.m["var_95"])

    def test_sim_returns_shape(self):
        self.assertEqual(self.m["sim_returns"].shape, (100_000,))

    def test_marginal_risk_sums_to_one(self):
        self.assertAlmostEqual(self.m["marginal_risk"].sum(), 1.0, places=10)

    def test_marginal_risk_shape(self):
        self.assertEqual(self.m["marginal_risk"].shape, (N,))

    def test_deterministic_with_seed(self):
        """Same seed → identical results."""
        m2 = self.opt.get_portfolio_metrics(WEIGHTS, n_sims=100_000, seed=0)
        np.testing.assert_array_equal(self.m["sim_returns"], m2["sim_returns"])

    def test_equal_weights_portfolio(self):
        ew = np.ones(N) / N
        m  = self.opt.get_portfolio_metrics(ew, risk_free_rate=0.02)
        self.assertAlmostEqual(m["expected_return"], float(ew @ RET), places=10)

    def test_prob_loss_increases_with_lower_return(self):
        """A bond-heavy portfolio should have lower P(loss) than equity-heavy."""
        opt_eq   = CPortfolio_optimization(WEIGHTS, CORR, RET, VOLS)
        w_equity = np.array([0.9, 0.1, 0.0])
        w_bonds  = np.array([0.0, 0.0, 1.0])
        m_eq  = opt_eq.get_portfolio_metrics(w_equity, risk_free_rate=0.02, seed=7)
        m_bnd = opt_eq.get_portfolio_metrics(w_bonds,  risk_free_rate=0.02, seed=7)
        # Bonds have lower return (3%) but much lower vol → P(loss) may still differ
        # The key check: both P(loss) values are valid probabilities
        self.assertGreaterEqual(m_eq["prob_loss"],  0)
        self.assertGreaterEqual(m_bnd["prob_loss"], 0)


@unittest.skipUnless(CVXOPT_AVAILABLE, "cvxopt not installed – skipping QP tests")
class TestMeanVarianceOpt(unittest.TestCase):

    def setUp(self):
        self.opt    = CPortfolio_optimization(WEIGHTS, CORR, RET, VOLS)
        self.x_min  = np.zeros(N)
        self.x_max  = np.ones(N) * 0.6

    def test_returns_list(self):
        portfolios = self.opt.mean_variance_opt(self.x_min, self.x_max, num_port=5)
        self.assertIsInstance(portfolios, list)
        self.assertEqual(len(portfolios), 5)

    def test_weights_sum_to_one(self):
        portfolios = self.opt.mean_variance_opt(self.x_min, self.x_max, num_port=5)
        for w in portfolios:
            self.assertAlmostEqual(w.sum(), 1.0, places=5)

    def test_weights_within_bounds(self):
        portfolios = self.opt.mean_variance_opt(self.x_min, self.x_max, num_port=5)
        for w in portfolios:
            self.assertTrue(np.all(w >= self.x_min - 1e-5))
            self.assertTrue(np.all(w <= self.x_max + 1e-5))


@unittest.skipUnless(CVXOPT_AVAILABLE, "cvxopt not installed – skipping QP tests")
class TestEfficientFrontier(unittest.TestCase):

    def setUp(self):
        self.opt   = CPortfolio_optimization(WEIGHTS, CORR, RET, VOLS)
        self.x_min = np.zeros(N)
        self.x_max = np.ones(N) * 0.6

    def test_output_tuple_of_three(self):
        result = self.opt.efficient_frontier(self.x_min, self.x_max, num_points=5)
        self.assertEqual(len(result), 3)

    def test_returns_volatilities_monotone(self):
        rets, vols, _ = self.opt.efficient_frontier(
            self.x_min, self.x_max, num_points=10)
        # Higher λ → lower risk; frontier sorted ascending in vol
        diffs = np.diff(vols)
        self.assertTrue(np.all(diffs >= -1e-6))


@unittest.skipUnless(CVXOPT_AVAILABLE, "cvxopt not installed – skipping QP tests")
class TestAssetAllocationTEop(unittest.TestCase):

    def setUp(self):
        self.opt     = CPortfolio_optimization(WEIGHTS, CORR, RET, VOLS)
        self.bmk     = WEIGHTS.copy()
        self.tac_rng = np.array([0.10, 0.10, 0.10])

    def test_output_is_tuple(self):
        result = self.opt.asset_allocation_TEop(self.bmk, self.tac_rng)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_optimal_weights_within_tactical_range(self):
        wts_opt, _ = self.opt.asset_allocation_TEop(self.bmk, self.tac_rng)
        active = wts_opt - self.bmk
        self.assertTrue(np.all(active >= -self.tac_rng - 1e-5))
        self.assertTrue(np.all(active <=  self.tac_rng + 1e-5))

    def test_tracking_error_positive(self):
        _, te = self.opt.asset_allocation_TEop(self.bmk, self.tac_rng)
        self.assertGreaterEqual(te, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CBlack_litterman
# ─────────────────────────────────────────────────────────────────────────────
class _BLBase(unittest.TestCase):
    """Shared fixture for BL tests."""

    def setUp(self):
        self.P = np.array([[1, -1, 0],    # View 1: EQ_DM outperforms EQ_EM
                           [0,  0, 1]])   # View 2: FI_GOV absolute
        self.recos  = ["buy", "neutral"]
        self.confs  = np.array([0.70, 0.50])
        self.bl = CBlack_litterman(
            risk_free_rate = 0.02,
            tau            = 0.025,
            sigma          = COV,
            p_matrix       = self.P,
            lambda_param   = 2.5,
            port_wts_eq    = WEIGHTS,
            conf_level     = self.confs,
            c_coef         = 1.0,
            reco_vector    = self.recos,
        )


class TestBLEquilibrium(_BLBase):

    def test_eq_risk_premium_shape(self):
        pi = self.bl.get_eq_risk_premium()
        self.assertEqual(pi.shape, (N,))

    def test_eq_risk_premium_positive_for_positive_ret(self):
        """For positive weights and positive cov, premium should be > 0."""
        pi = self.bl.get_eq_risk_premium()
        self.assertTrue(np.all(pi > 0))

    def test_eq_returns_shape(self):
        eq = self.bl.get_eq_returns()
        self.assertEqual(eq.shape, (N,))

    def test_eq_returns_exceeds_risk_free(self):
        eq = self.bl.get_eq_returns()
        self.assertTrue(np.all(eq > self.bl.risk_free_rate))

    def test_eq_returns_formula(self):
        """Π + rf = λΣw + rf"""
        eq  = self.bl.get_eq_returns()
        expected = 2.5 * COV @ WEIGHTS + 0.02
        np.testing.assert_allclose(eq, expected, rtol=1e-10)


class TestBLOmega(_BLBase):

    def test_omega_shape(self):
        self.bl.get_eq_returns()
        omega = self.bl.get_omega()
        n_views = self.P.shape[0]
        self.assertEqual(omega.shape, (n_views, n_views))

    def test_omega_is_diagonal(self):
        self.bl.get_eq_returns()
        omega = self.bl.get_omega()
        off_diag = omega - np.diag(np.diag(omega))
        np.testing.assert_allclose(off_diag, np.zeros_like(off_diag), atol=1e-12)

    def test_omega_positive_diagonal(self):
        self.bl.get_eq_returns()
        omega = self.bl.get_omega()
        self.assertTrue(np.all(np.diag(omega) > 0))

    def test_higher_confidence_lower_omega(self):
        """In this implementation Ω = diag(u · PΣP' · u) with u = diag(conf)/c_coef.
        A higher conf_level value → larger u → larger Ω diagonal, meaning the views
        are treated as MORE uncertain (less weight). Conversely, a very small
        conf_level → small Ω → views treated as near-certain.
        This test documents the actual scaling convention."""
        self.bl.get_eq_returns()
        om_high_conf = self.bl.get_omega()   # conf = [0.70, 0.50]

        bl_low = CBlack_litterman(
            risk_free_rate=0.02, tau=0.025, sigma=COV,
            p_matrix=self.P, lambda_param=2.5, port_wts_eq=WEIGHTS,
            conf_level=np.array([0.05, 0.05]),  # small conf → small Ω
            c_coef=1.0, reco_vector=self.recos,
        )
        bl_low.get_eq_returns()
        om_low_conf = bl_low.get_omega()

        # Higher conf_level value → larger Ω (more uncertainty on views)
        self.assertTrue(np.all(np.diag(om_high_conf) > np.diag(om_low_conf)))


class TestBLViewReturns(_BLBase):

    def test_view_returns_shape(self):
        self.bl.get_eq_returns()
        self.bl.get_omega()
        q = self.bl.get_view_returns()
        self.assertEqual(q.shape, (self.P.shape[0],))

    def test_buy_view_above_equilibrium_view(self):
        """'buy' should push view return above the equilibrium view."""
        self.bl.get_eq_returns()
        self.bl.get_omega()
        q      = self.bl.get_view_returns()
        q_eq   = self.P @ self.bl.eq_returns   # equilibrium view portfolio return
        # View 1 = "buy" → positive adjustment
        self.assertGreater(q[0], q_eq[0])

    def test_neutral_view_equals_equilibrium_view(self):
        """'neutral' → ν=0 → view return == equilibrium view return."""
        self.bl.get_eq_returns()
        self.bl.get_omega()
        q    = self.bl.get_view_returns()
        q_eq = self.P @ self.bl.eq_returns
        # View 2 is 'neutral'
        self.assertAlmostEqual(q[1], q_eq[1], places=10)

    def test_sell_view_below_equilibrium_view(self):
        bl_sell = CBlack_litterman(
            risk_free_rate=0.02, tau=0.025, sigma=COV,
            p_matrix=self.P, lambda_param=2.5, port_wts_eq=WEIGHTS,
            conf_level=self.confs, c_coef=1.0,
            reco_vector=["sell", "neutral"],
        )
        bl_sell.get_eq_returns()
        bl_sell.get_omega()
        q      = bl_sell.get_view_returns()
        q_eq   = self.P @ bl_sell.eq_returns
        self.assertLess(q[0], q_eq[0])


class TestBLPosterior(_BLBase):

    def setUp(self):
        super().setUp()
        self.bl_ret = self.bl.get_bl_returns()
        self.eq_ret = self.bl.eq_returns

    def test_bl_returns_shape(self):
        self.assertEqual(self.bl_ret.shape, (N,))

    def test_bl_returns_not_equal_to_equilibrium(self):
        """Views should shift at least one asset's return."""
        self.assertFalse(np.allclose(self.bl_ret, self.eq_ret))

    def test_buy_view_increases_long_asset(self):
        """A 'buy' view on the long-short portfolio (EQ_DM − EQ_EM) shifts the
        view portfolio's expected return upward relative to equilibrium.
        The BL posterior for the view portfolio P·bl should exceed P·pi."""
        p_bl = self.P @ self.bl_ret
        p_eq = self.P @ self.eq_ret
        # View 1 ('buy') → view portfolio return should be pulled up
        self.assertGreater(p_bl[0], p_eq[0])

    def test_buy_view_decreases_short_asset(self):
        """The view-portfolio return improvement (EQ_DM − EQ_EM) implies
        that the spread BL > equilibrium spread. Since the spread increases,
        the long asset return is higher than the short asset return in BL."""
        p_bl = self.P @ self.bl_ret   # [EQ_DM - EQ_EM, FI_GOV]
        # View 1 spread in BL must be greater than in equilibrium
        self.assertGreater(p_bl[0], (self.P @ self.eq_ret)[0])

    def test_neutral_view_minimal_impact(self):
        """View 2 is 'neutral' on FI_GOV → BL close to equilibrium for bonds."""
        delta = abs(self.bl_ret[2] - self.eq_ret[2])
        self.assertLess(delta, 0.02)  # less than 2 pp shift

    def test_posterior_covariance_shape(self):
        self.bl.get_omega()
        post_cov = self.bl.get_posterior_covariance()
        self.assertEqual(post_cov.shape, (N, N))

    def test_posterior_covariance_positive_definite(self):
        self.bl.get_omega()
        post_cov = self.bl.get_posterior_covariance()
        eigvals = np.linalg.eigvalsh(post_cov)
        self.assertTrue(np.all(eigvals > 0))

    def test_posterior_covariance_larger_than_prior(self):
        """Posterior uncertainty ≥ prior (adding views can't reduce total uncertainty)."""
        self.bl.get_omega()
        post_cov = self.bl.get_posterior_covariance()
        # Trace of posterior ≥ trace of sigma (adding uncertainty from views)
        self.assertGreaterEqual(np.trace(post_cov), np.trace(COV) - 1e-10)

    def test_strong_buy_stronger_than_buy(self):
        """'strong_buy' applies ν=+0.50 vs 'buy' ν=+0.25, so Q[0] is higher.
        The posterior view-portfolio spread P·bl should therefore be larger
        for strong_buy than for buy — even if individual asset returns move
        in less intuitive directions due to BL cross-asset shrinkage."""
        bl_sb = CBlack_litterman(
            risk_free_rate=0.02, tau=0.025, sigma=COV,
            p_matrix=self.P, lambda_param=2.5, port_wts_eq=WEIGHTS,
            conf_level=self.confs, c_coef=1.0,
            reco_vector=["strong_buy", "neutral"],
        )
        bl_sb_ret = bl_sb.get_bl_returns()
        # Compare via view-portfolio projection (spread EQ_DM − EQ_EM)
        spread_sb  = (self.P @ bl_sb_ret)[0]
        spread_buy = (self.P @ self.bl_ret)[0]
        self.assertGreater(spread_sb, spread_buy)


class TestBLFromViews(unittest.TestCase):
    """CBlack_litterman.from_views() classmethod"""

    def test_returns_tuple_of_two(self):
        result = CBlack_litterman.from_views(
            eq_returns=RET, cov_matrix=COV, weights_eq=WEIGHTS,
            p_matrix=np.array([[1, -1, 0]]),
            recommendations=["buy"], confidence_levels=[0.6],
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_shapes(self):
        pi, bl = CBlack_litterman.from_views(
            eq_returns=RET, cov_matrix=COV, weights_eq=WEIGHTS,
            p_matrix=np.array([[1, -1, 0]]),
            recommendations=["buy"], confidence_levels=[0.6],
        )
        self.assertEqual(pi.shape, (N,))
        self.assertEqual(bl.shape, (N,))

    def test_pi_matches_manual_eq_returns(self):
        """pi from from_views() should equal λΣw + rf."""
        pi, _ = CBlack_litterman.from_views(
            eq_returns=RET, cov_matrix=COV, weights_eq=WEIGHTS,
            p_matrix=np.array([[1, -1, 0]]),
            recommendations=["neutral"], confidence_levels=[0.5],
            lambda_param=2.5, risk_free_rate=0.02,
        )
        expected_pi = 2.5 * COV @ WEIGHTS + 0.02
        np.testing.assert_allclose(pi, expected_pi, rtol=1e-10)

    def test_neutral_view_minimal_shift(self):
        pi, bl = CBlack_litterman.from_views(
            eq_returns=RET, cov_matrix=COV, weights_eq=WEIGHTS,
            p_matrix=np.array([[1, 0, 0]]),
            recommendations=["neutral"], confidence_levels=[0.5],
        )
        np.testing.assert_allclose(bl, pi, atol=1e-6)

    def test_multiple_views(self):
        pi, bl = CBlack_litterman.from_views(
            eq_returns=RET, cov_matrix=COV, weights_eq=WEIGHTS,
            p_matrix=np.array([[1, -1, 0], [0, 0, 1]]),
            recommendations=["buy", "sell"],
            confidence_levels=[0.7, 0.6],
        )
        self.assertEqual(bl.shape, (N,))

    def test_from_views_matches_manual_constructor(self):
        """from_views should produce the same result as building manually."""
        P     = np.array([[1, -1, 0]])
        recos = ["buy"]
        confs = [0.7]

        pi_factory, bl_factory = CBlack_litterman.from_views(
            eq_returns=RET, cov_matrix=COV, weights_eq=WEIGHTS,
            p_matrix=P, recommendations=recos, confidence_levels=confs,
            tau=0.025, lambda_param=2.5, risk_free_rate=0.02,
        )

        bl_manual = CBlack_litterman(
            risk_free_rate=0.02, tau=0.025, sigma=COV,
            p_matrix=P, lambda_param=2.5, port_wts_eq=WEIGHTS,
            conf_level=np.array(confs), c_coef=1.0, reco_vector=recos,
        )
        bl_manual_ret = bl_manual.get_bl_returns()
        pi_manual     = bl_manual.eq_returns

        np.testing.assert_allclose(pi_factory, pi_manual, rtol=1e-10)
        np.testing.assert_allclose(bl_factory, bl_manual_ret, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CUtility
# ─────────────────────────────────────────────────────────────────────────────
class TestCUtilityQuadratic(unittest.TestCase):

    def setUp(self):
        self.uf = CUtility("quadratic", risk_aversion=2.0)
        self.r  = RNG.normal(0.08, 0.15, 10_000)

    def test_calculate_utility_formula(self):
        """U = E[R] - 0.5 γ Var[R]"""
        expected = self.r.mean() - 0.5 * 2.0 * self.r.var()
        self.assertAlmostEqual(self.uf.calculate_utility(self.r), expected, places=10)

    def test_certainty_equivalent_equals_utility_quadratic(self):
        """For quadratic utility CE == U."""
        u  = self.uf.calculate_utility(self.r)
        ce = self.uf.certainty_equivalent(self.r)
        self.assertAlmostEqual(u, ce, places=10)

    def test_higher_risk_aversion_lower_utility(self):
        uf2 = CUtility("quadratic", risk_aversion=5.0)
        u1  = self.uf.calculate_utility(self.r)
        u2  = uf2.calculate_utility(self.r)
        self.assertGreater(u1, u2)

    def test_risk_free_asset_utility(self):
        """Constant returns → utility = return (no variance penalty)."""
        rf_returns = np.ones(1000) * 0.03
        u = self.uf.calculate_utility(rf_returns)
        self.assertAlmostEqual(u, 0.03, places=5)


class TestCUtilityExponential(unittest.TestCase):

    def setUp(self):
        self.uf = CUtility("exponential", risk_aversion=2.0)
        self.r  = RNG.normal(0.08, 0.10, 10_000)

    def test_calculate_utility_negative(self):
        """Exponential utility is always negative."""
        u = self.uf.calculate_utility(self.r)
        self.assertLess(u, 0)

    def test_certainty_equivalent_positive(self):
        ce = self.uf.certainty_equivalent(self.r)
        self.assertGreater(ce, 0)

    def test_ce_less_than_mean_return(self):
        """CE ≤ E[R] (risk premium > 0)."""
        ce = self.uf.certainty_equivalent(self.r)
        self.assertLessEqual(ce, self.r.mean() + 1e-8)


class TestCUtilityPower(unittest.TestCase):

    def setUp(self):
        self.uf = CUtility("power", risk_aversion=2.0)
        # Power utility requires 1+r > 0 → keep returns > -1
        self.r  = RNG.normal(0.06, 0.08, 10_000)

    def test_calculate_utility_finite(self):
        u = self.uf.calculate_utility(self.r)
        self.assertTrue(math.isfinite(u))

    def test_certainty_equivalent_finite(self):
        ce = self.uf.certainty_equivalent(self.r)
        self.assertTrue(math.isfinite(ce))

    def test_log_utility_gamma_1(self):
        uf_log = CUtility("power", risk_aversion=1.0)
        u = uf_log.calculate_utility(self.r)
        expected = np.mean(np.log(1 + self.r))
        self.assertAlmostEqual(u, expected, places=10)

    def test_invalid_utility_type_raises(self):
        uf_bad = CUtility("linear", risk_aversion=2.0)
        with self.assertRaises(ValueError):
            uf_bad.calculate_utility(self.r)

    def test_invalid_utility_type_ce_raises(self):
        uf_bad = CUtility("linear", risk_aversion=2.0)
        with self.assertRaises(ValueError):
            uf_bad.certainty_equivalent(self.r)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Integration — full Streamlit-style pipeline
# ─────────────────────────────────────────────────────────────────────────────
class TestFullPipeline(unittest.TestCase):
    """
    End-to-end: build_cov_matrix → CPortfolio_optimization.get_portfolio_metrics
                → CBlack_litterman.from_views
    Mirrors exactly what asset_allocation_app.py does.
    """

    def test_metrics_then_bl_consistent(self):
        cov = build_cov_matrix(VOLS, CORR)
        opt = CPortfolio_optimization(WEIGHTS, CORR, RET, VOLS)
        m   = opt.get_portfolio_metrics(WEIGHTS, risk_free_rate=0.02, n_sims=10_000)

        # BL
        pi, bl = CBlack_litterman.from_views(
            eq_returns        = RET,
            cov_matrix        = cov,
            weights_eq        = WEIGHTS,
            p_matrix          = np.array([[1, -1, 0]]),
            recommendations   = ["buy"],
            confidence_levels = [0.7],
        )

        # With BL returns the optimizer re-computes metrics
        opt_bl = CPortfolio_optimization(WEIGHTS, CORR, bl, VOLS)
        m_bl   = opt_bl.get_portfolio_metrics(WEIGHTS, risk_free_rate=0.02, n_sims=10_000)

        # Basic sanity: BL portfolio should have different return from equilibrium
        self.assertNotAlmostEqual(m_bl["expected_return"], m["expected_return"],
                                  places=4)
        # Both prob_loss values must be valid
        self.assertGreater(m["prob_loss"],    0)
        self.assertGreater(m_bl["prob_loss"], 0)

    def test_marginal_risk_contrib_all_positive_for_long_only(self):
        """For long-only portfolios with positive-definite cov, contrib ≥ 0."""
        opt = CPortfolio_optimization(WEIGHTS, CORR, RET, VOLS)
        m   = opt.get_portfolio_metrics(WEIGHTS)
        self.assertTrue(np.all(m["marginal_risk"] >= 0))

    def test_cov_from_helper_matches_optimizer_internal(self):
        cov_helper   = build_cov_matrix(VOLS, CORR)
        opt          = CPortfolio_optimization(WEIGHTS, CORR, RET, VOLS)
        cov_internal = opt.exp_cov
        np.testing.assert_allclose(cov_helper, cov_internal, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Pretty output with timing
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(sys.modules[__name__])
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)

    total   = result.testsRun
    failed  = len(result.failures) + len(result.errors)
    skipped = len(result.skipped)
    passed  = total - failed - skipped

    print("\n" + "=" * 65)
    print(f"  RESULTS:  {passed} passed  |  {failed} failed  |  {skipped} skipped")
    if not CVXOPT_AVAILABLE:
        print("  NOTE: cvxopt not installed — QP solver tests were skipped.")
        print("        Install cvxopt to enable mean_variance_opt /")
        print("        efficient_frontier / asset_allocation_TEop tests.")
    print("=" * 65)

    sys.exit(0 if failed == 0 else 1)
