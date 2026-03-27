"""
utils/portfolio_loader.py
=========================
Look-through decomposition of a fund-of-funds portfolio by underlying
asset class.

Core idea
---------
Given a CSV where each row is a fund with:
  - its weight in the overall portfolio  (Peso_Cartera)
  - its own internal composition by asset class

we compute:

    contribution(fund, asset_class) = Peso_Cartera × exposure(fund, asset_class)

and then aggregate across funds to obtain the look-through weights of the
total portfolio in each asset class.

Expected CSV layout
-------------------
| ISIN | Fund name | Fund weight | Asset class 1 | Asset class 2 | ... |

- The first three columns are always ISIN, name, and weight.
- All remaining columns are asset-class exposures.
- Weights and exposures can be either decimal (0.40) or percentage (40);
  ``normalizar_porcentajes`` handles both automatically.

Public API
----------
    result = parse_fund_portfolio(df)
    result.asset_classes   # list[str]
    result.weights         # np.ndarray, sums ≈ 1
    result.composition     # pd.Series sorted descending
    result.detail          # pd.DataFrame with fund-level contributions
    result.warnings        # list[str] of validation messages
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Normalisation helper ──────────────────────────────────────────────────────

def normalizar_porcentajes(serie: pd.Series) -> pd.Series:
    """
    Convert a series to decimal fractions.

    If any value has absolute magnitude > 1 the series is assumed to be
    in percentage format and divided by 100; otherwise it is returned as-is.

    Non-numeric values are coerced to 0.
    """
    serie = pd.to_numeric(serie, errors="coerce").fillna(0.0)
    if (serie.abs() > 1).any():
        return serie / 100.0
    return serie


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class FundPortfolioResult:
    """
    Output of :func:`parse_fund_portfolio`.

    Attributes
    ----------
    detail:
        DataFrame with one row per fund and one column per asset class,
        showing each fund's *contribution* to the total portfolio
        (= fund weight × fund exposure).  Prefixed with ISIN, Nombre_Fondo,
        Peso_Cartera for easy display.
    composition:
        Series indexed by asset-class name with the aggregated portfolio
        weight in that asset class.  Sorted descending.  Ready to drop into
        ``session_state["portfolio_weights"]``.
    warnings:
        List of human-readable validation messages (e.g. funds whose
        exposures don't sum to ≈ 100 %).  Empty when everything looks clean.
    """
    detail:      pd.DataFrame
    composition: pd.Series
    warnings:    list[str] = field(default_factory=list)

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def asset_classes(self) -> list[str]:
        """Ordered list of asset-class names (same order as composition)."""
        return list(self.composition.index)

    @property
    def weights(self) -> np.ndarray:
        """Aggregated weights as a float numpy array (shape n,)."""
        return self.composition.to_numpy(dtype=float)

    @property
    def weights_sum(self) -> float:
        """Sum of aggregated weights (should be ≈ 1.0)."""
        return float(self.composition.sum())


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_fund_portfolio(
    df: pd.DataFrame,
    tolerance: float = 0.05,
) -> FundPortfolioResult:
    """
    Parse a fund-portfolio DataFrame and compute look-through composition.

    Parameters
    ----------
    df:
        Raw DataFrame read from CSV (or Excel).  Column layout must be:
        [ISIN, Fund name, Fund weight, Asset class 1, Asset class 2, ...]
    tolerance:
        Maximum allowed deviation from 1.0 when validating that fund
        exposures sum to 100 %.  Default 5 pp.

    Returns
    -------
    FundPortfolioResult

    Raises
    ------
    ValueError
        If the DataFrame has fewer than 4 columns.
    """
    if df.shape[1] < 4:
        raise ValueError(
            "El CSV debe tener al menos 4 columnas: "
            "ISIN, Nombre, Peso_Cartera y al menos una clase de activo."
        )

    df = df.copy()

    # Standardise first three column names
    cols = list(df.columns)
    cols[0] = "ISIN"
    cols[1] = "Nombre_Fondo"
    cols[2] = "Peso_Cartera"
    df.columns = cols

    asset_cols = df.columns[3:]

    # Normalise weights and exposures
    df["Peso_Cartera"] = normalizar_porcentajes(df["Peso_Cartera"])
    for col in asset_cols:
        df[col] = normalizar_porcentajes(df[col])

    # ── Validation ────────────────────────────────────────────────────────────
    warnings: list[str] = []

    # Per-fund: exposures should sum to ≈ 100 %
    df["_row_sum"] = df[asset_cols].sum(axis=1)
    for _, row in df.iterrows():
        s = row["_row_sum"]
        if abs(s - 1.0) > tolerance:
            warnings.append(
                f"El fondo '{row['Nombre_Fondo']}' (ISIN {row['ISIN']}) "
                f"suma {s:.2%} en sus clases de activo."
            )
    df = df.drop(columns=["_row_sum"])

    # Total portfolio weight should sum to ≈ 100 %
    total_peso = df["Peso_Cartera"].sum()
    if abs(total_peso - 1.0) > tolerance:
        warnings.append(
            f"Los pesos de los fondos suman {total_peso:.2%} "
            f"(se esperaba 100 %). La composición agregada puede ser incorrecta."
        )

    # ── Contributions ─────────────────────────────────────────────────────────
    # contribution(fund, class) = fund_weight × class_exposure
    contrib = df[asset_cols].multiply(df["Peso_Cartera"], axis=0)

    detail = pd.concat(
        [df[["ISIN", "Nombre_Fondo", "Peso_Cartera"]], contrib],
        axis=1,
    )

    composition = contrib.sum(axis=0).sort_values(ascending=False)

    return FundPortfolioResult(
        detail=detail,
        composition=composition,
        warnings=warnings,
    )
