"""
utils/config.py
===============
Central registry of application-wide constants.

All magic numbers, default parameters, and thresholds live here.
Import from this module instead of hard-coding values in pages or utils.

Usage
-----
    from utils.config import RISK_FREE_RATE, N_SIMS, BL_TAU
"""

# ── Monte Carlo ────────────────────────────────────────────────────────────────
N_SIMS: int        = 50_000   # Maximum number of Monte Carlo draws
MC_SEED: int | None = None    # None → non-deterministic in production.
                               # Pass an explicit int (e.g. 42) in tests.

# ── Risk & return defaults ─────────────────────────────────────────────────────
RISK_FREE_RATE: float = 0.02  # Annual, decimal (2 %)

# ── Black-Litterman ───────────────────────────────────────────────────────────
BL_TAU: float         = 0.025  # Uncertainty scalar on the prior
BL_LAMBDA: float      = 2.5    # Risk-aversion coefficient
BL_C_COEF: float      = 1.0    # Confidence scaling coefficient

# ── File upload limits ────────────────────────────────────────────────────────
MAX_UPLOAD_MB: float  = 10.0   # Maximum file size for uploaded Excel/CSV files
MAX_UPLOAD_ROWS: int  = 500_000  # Maximum rows after parsing

# ── Database / pagination ─────────────────────────────────────────────────────
PRICE_PAGE_SIZE: int  = 5_000  # Rows per round-trip when fetching historical data
DB_MAX_RETRIES: int   = 3      # Retry attempts for transient DB write errors
