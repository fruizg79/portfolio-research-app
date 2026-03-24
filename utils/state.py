"""
utils/state.py
==============
Single source of truth for Streamlit session state.

ADD NEW FIELDS HERE when a new module needs to persist data across pages.
Every field must have a default value (None means "not yet configured").

Convention
----------
- market inputs    → None until the user saves them in Configuración
- computed results → None until the relevant page calculates them
- ui preferences   → concrete defaults (booleans, strings, numbers)
"""

import streamlit as st

from utils.config import RISK_FREE_RATE, N_SIMS


# ── Master registry ───────────────────────────────────────────────────────────
# Grouped by the module that owns / writes each field.
# Pages that only READ a field don't need to be listed here.

_DEFAULTS: dict = {

    # ── Configuración ─────────────────────────────────────────────────────────
    "asset_classes": [
        "Renta Variable DM",
        "Renta Variable EM",
        "Renta Fija Gov",
        "Renta Fija Corp",
        "Alternativos",
    ],
    "eq_returns":   None,   # np.array, annual decimal  (e.g. 0.08 = 8%)
    "volatilities": None,   # np.array, annual decimal
    "corr_matrix":  None,   # np.ndarray (n × n), symmetric, diag = 1

    # ── Mi Cartera ────────────────────────────────────────────────────────────
    "portfolio_weights": None,   # np.array, sums to 1
    "tactical_ranges":   None,   # np.array, decimal  (e.g. 0.10 = ±10 pp)

    # ── Riesgo & Retorno (cached results) ────────────────────────────────────
    # Stored so other pages can read metrics without re-running Monte Carlo.
    "portfolio_metrics": None,   # dict returned by CPortfolio_optimization.get_portfolio_metrics()

    # ── Black-Litterman (cached results) ─────────────────────────────────────
    "bl_eq_returns":   None,   # np.array  – equilibrium returns (π)
    "bl_post_returns": None,   # np.array  – BL posterior returns

    # ── UI preferences ────────────────────────────────────────────────────────
    "risk_free_rate": RISK_FREE_RATE,  # float, used across Riesgo & BL pages
    "n_sims":         N_SIMS,         # int, Monte Carlo draws

    # ── Base de datos — referencias activas ──────────────────────────────────
    # Se rellenan cuando el usuario carga un escenario o cartera desde BD.
    # None significa "trabajando en memoria, sin guardar todavía".
    "active_scenario_id":    None,   # int  – ID en market_scenarios
    "active_scenario_name":  None,   # str  – nombre para mostrar en la UI
    "active_portfolio_id":   None,   # int  – ID en portfolios
    "active_portfolio_name": None,   # str  – nombre para mostrar en la UI
    # ── Flags internos ────────────────────────────────────────────────────────
    "_autoloaded": False,   # True después del primer autoload de BD
    # ────────────────────────────────────────────────────────────────────────
    # TEMPLATE — copy & adapt when adding a new module:
    #
    #   # ── Nombre del Módulo Nuevo ───────────────────────────────────────
    #   "new_field_1": None,    # description
    #   "new_field_2": False,   # description
    # ────────────────────────────────────────────────────────────────────────
}


def init_state() -> None:
    """
    Initialise every registered key in st.session_state.

    Call this once at the top of app.py (the entry point).
    Keys that already exist are left untouched so that navigation between
    pages does not reset live data.
    """
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ── Declarative dependency graph ──────────────────────────────────────────────
# For each computed field, list its direct upstream dependencies.
# Adding a new computed field only requires declaring its upstream here;
# reset_downstream() will propagate invalidation automatically.
#
# Convention: only list *direct* parents; transitive dependencies are resolved
# by the BFS traversal in reset_downstream().
_UPSTREAM: dict[str, list[str]] = {
    # Market inputs depend on the asset universe
    "eq_returns":        ["asset_classes"],
    "volatilities":      ["asset_classes"],
    "corr_matrix":       ["asset_classes"],
    "portfolio_weights": ["asset_classes"],
    "tactical_ranges":   ["asset_classes"],
    # Computed results depend on market inputs
    "portfolio_metrics": ["eq_returns", "volatilities", "corr_matrix", "portfolio_weights"],
    "bl_eq_returns":     ["eq_returns", "volatilities", "corr_matrix"],
    "bl_post_returns":   ["eq_returns", "volatilities", "corr_matrix", "portfolio_weights"],
}

# Pre-build reverse graph: upstream_key → [dependent_keys]
# This is computed once at import time so reset_downstream() is O(nodes).
_DOWNSTREAM: dict[str, list[str]] = {}
for _dep, _upstreams in _UPSTREAM.items():
    for _up in _upstreams:
        _DOWNSTREAM.setdefault(_up, []).append(_dep)


def reset_downstream(from_key: str) -> None:
    """
    Reset all fields that transitively depend on *from_key*.

    Uses BFS over the pre-built reverse dependency graph so that adding a new
    computed field only requires updating _UPSTREAM — no manual cascade lists.

    Args:
        from_key: the key that just changed ("asset_classes", "eq_returns", …)
    """
    visited: set[str] = set()
    queue: list[str] = list(_DOWNSTREAM.get(from_key, []))
    while queue:
        key = queue.pop(0)
        if key in visited:
            continue
        visited.add(key)
        queue.extend(_DOWNSTREAM.get(key, []))

    for key in visited:
        if key in _DEFAULTS:
            st.session_state[key] = _DEFAULTS[key]


def is_market_configured() -> bool:
    """True when all three market parameter arrays have been saved."""
    return all(
        st.session_state.get(k) is not None
        for k in ("eq_returns", "volatilities", "corr_matrix")
    )


def is_portfolio_configured() -> bool:
    """True when portfolio weights and tactical ranges have been saved."""
    return all(
        st.session_state.get(k) is not None
        for k in ("portfolio_weights", "tactical_ranges")
    )
