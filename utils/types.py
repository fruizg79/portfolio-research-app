"""
utils/types.py
==============
Typed data structures shared across the application.

Why TypedDicts and not dataclasses?
------------------------------------
All data that crosses the DB ↔ session_state boundary is already dict-shaped
(JSON from Supabase, session_state, @st.cache_data).  TypedDicts give us static
type-checking and IDE autocompletion without adding serialisation boilerplate.

Usage
-----
    from utils.types import ScenarioData, PortfolioData, PortfolioMetrics

    def load_scenario(scenario_id: int) -> ScenarioData: ...
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

# Re-exported so pages only need one import location.
# The class itself lives in portfolio_loader to keep the dataclass + methods together.
from utils.portfolio_loader import FundPortfolioResult as FundPortfolioResult  # noqa: F401


# ── Database row shapes ────────────────────────────────────────────────────────

class ScenarioData(TypedDict, total=False):
    """
    Fully hydrated scenario as returned by database.load_scenario().
    All numpy arrays are ready to drop into session_state.

    sim_models and sim_model_params are optional (total=False) for backwards
    compatibility with scenarios saved before the stochastic-MC migration.
    """
    id:               int
    name:             str
    description:      str
    asset_classes:    list[str]
    eq_returns:       np.ndarray   # shape (n,)
    volatilities:     np.ndarray   # shape (n,)
    corr_matrix:      np.ndarray   # shape (n, n)
    sim_models:       dict | None  # {asset_class: 'gbm'|'vasicek'|'normal'} or None
    sim_model_params: dict | None  # {asset_class: {mu, sigma, kappa, theta}} or None


class PortfolioData(TypedDict):
    """
    Fully hydrated portfolio as returned by database.load_portfolio().
    """
    id:              int
    name:            str
    description:     str
    scenario_id:     int
    weights:         np.ndarray   # shape (n,), sums to 1
    tactical_ranges: np.ndarray   # shape (n,), decimal (e.g. 0.10 = ±10 pp)


# ── Computed results ───────────────────────────────────────────────────────────

class PortfolioMetrics(TypedDict):
    """
    Output of utils.finance.get_portfolio_metrics().
    Stored in session_state["portfolio_metrics"].
    """
    expected_return: float         # Annual, decimal
    volatility:      float         # Annual, decimal
    sharpe:          float
    prob_loss:       float         # Fraction of MC paths that end negative
    var_95:          float         # 5th percentile of simulated 1-year returns
    es_95:           float         # Expected Shortfall (CVaR) at 95%
    sim_returns:     np.ndarray    # Full distribution, shape (n_sims,)
    marginal_risk:   np.ndarray    # Marginal risk contribution per asset, shape (n,)


# ── Black-Litterman snapshot payload ──────────────────────────────────────────

class BLSnapshotData(TypedDict, total=False):
    """
    Optional BL block saved inside a results snapshot.
    All keys are optional (total=False) because BL may not have been run.
    """
    tau:          float
    lambda_param: float
    views:        list[dict]       # raw view dicts from the BL page
    eq_returns:   np.ndarray       # π equilibrium returns
    post_returns: np.ndarray       # BL posterior returns
