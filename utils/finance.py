"""
utils/finance.py
================
Thin, cached wrappers around asset_allocation.py for use in Streamlit pages.

WHY this layer exists
---------------------
1. @st.cache_data prevents re-running expensive calculations (Monte Carlo,
   matrix inversions) on every widget interaction.
2. All type conversion from session-state to numpy happens here, not in pages.
3. When you add a new computation (e.g. Ledoit-Wolf covariance, efficient
   frontier), add a wrapper function here — pages stay thin.

ADDING A NEW COMPUTATION
------------------------
1. Write the pure logic in asset_allocation.py (or ml_portfolio_enhancement.py).
2. Add a cached wrapper below.
3. Import and call it from the relevant page.
"""

import numpy as np
import streamlit as st

from asset_allocation import (
    CPortfolio_optimization,
    CBlack_litterman,
    build_cov_matrix,
    simulate_portfolio_mc,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_value_at_risk,
    calculate_conditional_var,
)

# ── Re-export helpers that pages use directly ─────────────────────────────────
__all__ = [
    "get_cov_matrix",
    "get_portfolio_metrics",
    "run_stochastic_mc",
    "run_black_litterman",
    "get_optimizer",
    "to_cache_args",
    # scalar helpers
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_value_at_risk",
    "calculate_conditional_var",
]


# ── Covariance matrix ─────────────────────────────────────────────────────────
@st.cache_data
def get_cov_matrix(
    volatilities: tuple[float, ...],
    corr_flat: tuple[float, ...],
    n: int,
) -> np.ndarray:
    """
    Build and cache the covariance matrix.

    st.cache_data requires hashable arguments, so arrays are passed as tuples.

    Args:
        volatilities: Annual vols as a flat tuple.
        corr_flat:    Correlation matrix flattened row-major.
        n:            Number of assets (needed to reshape corr_flat).

    Returns:
        np.ndarray of shape (n, n).
    """
    vols = np.array(volatilities)
    corr = np.array(corr_flat).reshape(n, n)
    return build_cov_matrix(vols, corr)


# ── Portfolio metrics ─────────────────────────────────────────────────────────
@st.cache_data
def get_portfolio_metrics(
    weights: tuple[float, ...],
    expected_ret: tuple[float, ...],
    volatilities: tuple[float, ...],
    corr_flat: tuple[float, ...],
    n: int,
    risk_free_rate: float = 0.02,
    n_sims: int = 50_000,
    seed: int = 42,
) -> dict:
    """
    Compute and cache portfolio risk/return metrics.

    Cached so that changing an unrelated widget (e.g. a BL slider)
    does not re-run the 50k-scenario Monte Carlo.

    Returns:
        dict with keys: expected_return, volatility, sharpe, prob_loss,
                        var_95, es_95, sim_returns, marginal_risk.
    """
    vols = np.array(volatilities)
    corr = np.array(corr_flat).reshape(n, n)
    w    = np.array(weights)
    ret  = np.array(expected_ret)

    optimizer = CPortfolio_optimization(
        port_wts           = w,
        correlation_matrix = corr,
        expected_ret       = ret,
        vol                = vols,
    )
    return optimizer.get_portfolio_metrics(
        weights        = w,
        risk_free_rate = risk_free_rate,
        n_sims         = n_sims,
        seed           = seed,
    )


# ── Stochastic-process Monte Carlo ───────────────────────────────────────────
@st.cache_data
def run_stochastic_mc(
    weights: tuple[float, ...],
    asset_classes: tuple[str, ...],
    sim_models_items: tuple[tuple, ...],      # tuple(sorted(sim_models.items()))
    sim_model_params_items: tuple[tuple, ...], # tuple((ac, tuple(sorted(p.items()))) …)
    corr_flat: tuple[float, ...],
    n: int,
    risk_free_rate: float = 0.02,
    n_sims: int = 50_000,
    seed: int | None = None,
) -> dict:
    """
    Cached wrapper around simulate_portfolio_mc().

    All mutable arguments are converted to hashable tuples so that
    @st.cache_data can serialise the cache key.

    Args:
        weights:               Portfolio weights as a flat tuple.
        asset_classes:         Asset class labels as a tuple (same order as weights).
        sim_models_items:      tuple(sorted(sim_models.items())) — dict serialised.
        sim_model_params_items: Serialised params — see _pack_params() helper below.
        corr_flat:             Correlation matrix flattened row-major.
        n:                     Number of assets.
        risk_free_rate:        Annual risk-free rate.
        n_sims:                MC draws.
        seed:                  RNG seed.

    Returns:
        dict identical to get_portfolio_metrics().
    """
    # Reconstruct dicts from hashable tuples
    sim_models = dict(sim_models_items)
    sim_model_params = {
        ac: dict(param_items)
        for ac, param_items in sim_model_params_items
    }
    corr = np.array(corr_flat).reshape(n, n)

    return simulate_portfolio_mc(
        weights           = np.array(weights),
        sim_models        = sim_models,
        sim_model_params  = sim_model_params,
        asset_classes     = list(asset_classes),
        correlation_matrix= corr,
        risk_free_rate    = risk_free_rate,
        n_sims            = n_sims,
        seed              = seed,
    )


def pack_sim_params(
    sim_models: dict,
    sim_model_params: dict,
) -> tuple[tuple, tuple]:
    """
    Convert sim_models and sim_model_params dicts to hashable tuples
    suitable for passing to run_stochastic_mc() (which is @st.cache_data).

    Returns:
        (sim_models_items, sim_model_params_items)
    """
    models_items = tuple(sorted(sim_models.items()))
    params_items = tuple(
        (ac, tuple(sorted(p.items())))
        for ac, p in sorted(sim_model_params.items())
    )
    return models_items, params_items


# ── Black-Litterman ───────────────────────────────────────────────────────────
@st.cache_data
def run_black_litterman(
    eq_returns: tuple[float, ...],
    volatilities: tuple[float, ...],
    corr_flat: tuple[float, ...],
    n: int,
    weights_eq: tuple[float, ...],
    p_matrix_flat: tuple[float, ...],
    n_views: int,
    recommendations: tuple[str, ...],
    confidence_levels: tuple[float, ...],
    tau: float = 0.025,
    lambda_param: float = 2.5,
    risk_free_rate: float = 0.02,
    c_coef: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run and cache the Black-Litterman model.

    Returns:
        (pi_equilibrium, bl_posterior) — both np.ndarray of shape (n,).
    """
    cov    = build_cov_matrix(np.array(volatilities), np.array(corr_flat).reshape(n, n))
    p_mat  = np.array(p_matrix_flat).reshape(n_views, n)

    return CBlack_litterman.from_views(
        eq_returns        = np.array(eq_returns),
        cov_matrix        = cov,
        weights_eq        = np.array(weights_eq),
        p_matrix          = p_mat,
        recommendations   = list(recommendations),
        confidence_levels = list(confidence_levels),
        tau               = tau,
        lambda_param      = lambda_param,
        risk_free_rate    = risk_free_rate,
        c_coef            = c_coef,
    )


# ── Convenience: build optimizer from session state arrays ────────────────────
def get_optimizer(
    weights: np.ndarray,
    expected_ret: np.ndarray,
    volatilities: np.ndarray,
    corr_matrix: np.ndarray,
) -> CPortfolio_optimization:
    """
    Instantiate CPortfolio_optimization from numpy arrays.

    Not cached (instantiation is cheap; caching happens at the metric level).
    Use this when you need the full optimizer object (e.g. for efficient frontier).
    """
    return CPortfolio_optimization(
        port_wts           = weights,
        correlation_matrix = corr_matrix,
        expected_ret       = expected_ret,
        vol                = volatilities,
    )


# ── Array ↔ tuple helpers (for cache calls in pages) ─────────────────────────
def to_cache_args(
    volatilities: np.ndarray,
    corr_matrix: np.ndarray,
) -> tuple[tuple, tuple, int]:
    """
    Convert arrays to hashable tuples for @st.cache_data functions.

    Returns:
        (vol_tuple, corr_flat_tuple, n)
    """
    n = len(volatilities)
    return tuple(volatilities), tuple(corr_matrix.flatten()), n
