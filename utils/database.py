"""
utils/database.py
=================
Capa de acceso a datos sobre Supabase (PostgreSQL).

Todas las funciones siguen el mismo patrón:
  - Lectura  → devuelve list[dict] | dict | pd.DataFrame | None
  - Escritura → devuelve el registro insertado/actualizado (dict)
  - Los np.ndarray se convierten a/desde list automáticamente
  - Los errores se propagan como excepciones para que la página los muestre

CONFIGURACIÓN
-------------
Añade en Streamlit Secrets (o .streamlit/secrets.toml en local):

    SUPABASE_URL = "https://xxxx.supabase.co"
    SUPABASE_KEY = "tu-anon-key"
"""

import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

from supabase import create_client, Client

from utils.config import PRICE_PAGE_SIZE
from utils.types import ScenarioData, PortfolioData


# ── Conexión singleton ────────────────────────────────────────────────────────
@st.cache_resource
def get_client() -> Client:
    """Una sola conexión por sesión de Streamlit."""
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# ACTIVOS
# ══════════════════════════════════════════════════════════════════════════════

def get_assets(active_only: bool = True) -> list[dict]:
    """Lista de activos registrados."""
    q = get_client().table("assets").select("*")
    if active_only:
        q = q.eq("active", True)
    return q.order("name").execute().data


def upsert_asset(name: str, asset_class: str = None,
                 currency: str = "EUR") -> dict:
    """Crea o actualiza un activo por nombre."""
    db = get_client()
    existing = db.table("assets").select("id").eq("name", name).execute().data
    data = {"name": name, "asset_class": asset_class, "currency": currency}
    if existing:
        return db.table("assets").update(data).eq("id", existing[0]["id"]).execute().data[0]
    return db.table("assets").insert(data).execute().data[0]


def deactivate_asset(asset_id: int) -> None:
    """Marca un activo como inactivo (no lo borra)."""
    get_client().table("assets").update({"active": False}).eq("id", asset_id).execute()


# ══════════════════════════════════════════════════════════════════════════════
# PRECIOS HISTÓRICOS
# ══════════════════════════════════════════════════════════════════════════════

def save_prices(asset_id: int, df: pd.DataFrame) -> int:
    """
    Inserta o actualiza precios históricos para un activo.

    Args:
        asset_id: ID del activo en la tabla assets.
        df: DataFrame con columnas ['date', 'price'].
            'date' puede ser str (YYYY-MM-DD) o datetime.

    Returns:
        Número de filas insertadas/actualizadas.
    """
    df = df.copy().sort_values("date")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["return_1d"] = df["price"].pct_change().round(8)
    df["asset_id"] = asset_id

    rows = (
        df[["asset_id", "date", "price", "return_1d"]]
        .dropna(subset=["price"])
        .to_dict("records")
    )
    if not rows:
        return 0

    # Paginate to avoid hitting Supabase request-size limits on large uploads.
    db = get_client()
    for i in range(0, len(rows), PRICE_PAGE_SIZE):
        db.table("price_history").upsert(
            rows[i : i + PRICE_PAGE_SIZE], on_conflict="asset_id,date"
        ).execute()
    return len(rows)


def get_prices(asset_id: int,
               date_from: date = None, date_to: date = None) -> pd.DataFrame:
    """Precios históricos de un activo como DataFrame [date, price, return_1d]."""
    q = (get_client().table("price_history")
         .select("date, price, return_1d")
         .eq("asset_id", asset_id)
         .order("date"))
    if date_from:
        q = q.gte("date", str(date_from))
    if date_to:
        q = q.lte("date", str(date_to))
    data = q.execute().data
    if not data:
        return pd.DataFrame(columns=["date", "price", "return_1d"])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_returns_matrix(asset_ids: list[int],
                       date_from: date = None,
                       date_to: date = None) -> pd.DataFrame:
    """
    Retornos diarios de varios activos como DataFrame (fecha × asset_id).
    Útil para calibrar volatilidades y correlaciones.
    """
    q = (get_client().table("price_history")
         .select("asset_id, date, return_1d")
         .in_("asset_id", asset_ids)
         .order("date"))
    if date_from:
        q = q.gte("date", str(date_from))
    if date_to:
        q = q.lte("date", str(date_to))
    data = q.execute().data
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="asset_id", values="return_1d")


# ══════════════════════════════════════════════════════════════════════════════
# DATOS MACROECONÓMICOS
# ══════════════════════════════════════════════════════════════════════════════

def get_macro_series() -> list[dict]:
    """Lista de series macroeconómicas registradas."""
    return (get_client().table("macro_series")
            .select("*").order("name").execute().data)


def upsert_macro_series(series_id: str, name: str,
                        frequency: str = "monthly", unit: str = "") -> dict:
    """Registra o actualiza la definición de una serie macro."""
    db = get_client()
    existing = db.table("macro_series").select("series_id").eq("series_id", series_id).execute().data
    data = {"series_id": series_id, "name": name, "frequency": frequency, "unit": unit}
    if existing:
        return db.table("macro_series").update(data).eq("series_id", series_id).execute().data[0]
    return db.table("macro_series").insert(data).execute().data[0]


def save_macro_data(series_id: str, df: pd.DataFrame) -> int:
    """
    Inserta o actualiza datos de una serie macroeconómica.

    Args:
        series_id: ID de la serie (debe existir en macro_series).
        df: DataFrame con columnas ['date', 'value'].

    Returns:
        Número de filas insertadas/actualizadas.
    """
    df = df.copy().sort_values("date")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["series_id"] = series_id

    rows = (
        df[["series_id", "date", "value"]]
        .dropna(subset=["value"])
        .to_dict("records")
    )
    if not rows:
        return 0

    get_client().table("macro_data").upsert(
        rows, on_conflict="series_id,date"
    ).execute()
    return len(rows)


def get_macro_data(series_id: str,
                   date_from: date = None, date_to: date = None) -> pd.DataFrame:
    """Datos de una serie macroeconómica como DataFrame [date, value]."""
    q = (get_client().table("macro_data")
         .select("date, value")
         .eq("series_id", series_id)
         .order("date"))
    if date_from:
        q = q.gte("date", str(date_from))
    if date_to:
        q = q.lte("date", str(date_to))
    data = q.execute().data
    if not data:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ESCENARIOS DE MERCADO
# ══════════════════════════════════════════════════════════════════════════════

def save_scenario(name: str,
                  asset_classes: list[str],
                  eq_returns: np.ndarray,
                  volatilities: np.ndarray,
                  corr_matrix: np.ndarray,
                  description: str = "",
                  history_from: date = None,
                  history_to: date = None) -> dict:
    """Guarda un nuevo escenario de mercado. Devuelve el registro creado."""
    return get_client().table("market_scenarios").insert({
        "name":          name,
        "description":   description,
        "asset_classes": asset_classes,
        "eq_returns":    eq_returns.tolist(),
        "volatilities":  volatilities.tolist(),
        "corr_matrix":   corr_matrix.tolist(),
        "history_from":  str(history_from) if history_from else None,
        "history_to":    str(history_to)   if history_to   else None,
    }).execute().data[0]


def update_scenario(scenario_id: int,
                    name: str,
                    asset_classes: list[str],
                    eq_returns: np.ndarray,
                    volatilities: np.ndarray,
                    corr_matrix: np.ndarray,
                    description: str = "") -> dict:
    """Actualiza un escenario existente."""
    return get_client().table("market_scenarios").update({
        "name":          name,
        "description":   description,
        "asset_classes": asset_classes,
        "eq_returns":    eq_returns.tolist(),
        "volatilities":  volatilities.tolist(),
        "corr_matrix":   corr_matrix.tolist(),
        "updated_at":    "now()",
    }).eq("id", scenario_id).execute().data[0]


def get_scenarios() -> list[dict]:
    """Lista resumida de escenarios (sin matrices) para el selector."""
    return (get_client().table("market_scenarios")
            .select("id, name, description, created_at, history_from, history_to")
            .order("created_at", desc=True)
            .execute().data)


def load_scenario(scenario_id: int) -> ScenarioData:
    """
    Carga un escenario completo desde BD y devuelve los arrays listos
    para cargar en session_state.
    """
    row = (get_client().table("market_scenarios")
           .select("*")
           .eq("id", scenario_id)
           .single()
           .execute().data)
    return ScenarioData(
        id            = row["id"],
        name          = row["name"],
        description   = row.get("description", ""),
        asset_classes = row["asset_classes"],
        eq_returns    = np.array(row["eq_returns"]),
        volatilities  = np.array(row["volatilities"]),
        corr_matrix   = np.array(row["corr_matrix"]),
    )


def delete_scenario(scenario_id: int) -> None:
    """Borra un escenario. Falla si tiene carteras asociadas (FK)."""
    get_client().table("market_scenarios").delete().eq("id", scenario_id).execute()


# ══════════════════════════════════════════════════════════════════════════════
# CARTERAS
# ══════════════════════════════════════════════════════════════════════════════

def save_portfolio(name: str,
                   weights: np.ndarray,
                   tactical_ranges: np.ndarray,
                   scenario_id: int,
                   description: str = "") -> dict:
    """Guarda una nueva cartera. Devuelve el registro creado."""
    return get_client().table("portfolios").insert({
        "name":            name,
        "description":     description,
        "scenario_id":     scenario_id,
        "weights":         weights.tolist(),
        "tactical_ranges": tactical_ranges.tolist(),
    }).execute().data[0]


def update_portfolio(portfolio_id: int,
                     name: str,
                     weights: np.ndarray,
                     tactical_ranges: np.ndarray,
                     description: str = "") -> dict:
    """Actualiza una cartera existente."""
    return get_client().table("portfolios").update({
        "name":            name,
        "description":     description,
        "weights":         weights.tolist(),
        "tactical_ranges": tactical_ranges.tolist(),
        "updated_at":      "now()",
    }).eq("id", portfolio_id).execute().data[0]


def get_portfolios(scenario_id: int = None) -> list[dict]:
    """Lista resumida de carteras. Filtra por escenario si se indica."""
    q = (get_client().table("portfolios")
         .select("id, name, description, created_at, scenario_id")
         .order("created_at", desc=True))
    if scenario_id:
        q = q.eq("scenario_id", scenario_id)
    return q.execute().data


def load_portfolio(portfolio_id: int) -> PortfolioData:
    """Carga una cartera completa desde BD."""
    row = (get_client().table("portfolios")
           .select("*")
           .eq("id", portfolio_id)
           .single()
           .execute().data)
    return PortfolioData(
        id              = row["id"],
        name            = row["name"],
        description     = row.get("description", ""),
        scenario_id     = row["scenario_id"],
        weights         = np.array(row["weights"]),
        tactical_ranges = np.array(row["tactical_ranges"]),
    )


def delete_portfolio(portfolio_id: int) -> None:
    """Borra una cartera y sus snapshots asociados (CASCADE)."""
    get_client().table("portfolios").delete().eq("id", portfolio_id).execute()


# ══════════════════════════════════════════════════════════════════════════════
# RESULTADOS / SNAPSHOTS
# ══════════════════════════════════════════════════════════════════════════════

def save_snapshot(portfolio_id: int,
                  scenario_id: int,
                  metrics: dict,
                  risk_free_rate: float,
                  n_sims: int,
                  bl_data: dict = None) -> dict:
    """
    Guarda un snapshot de resultados calculados.

    Args:
        metrics: dict devuelto por get_portfolio_metrics()
        bl_data: dict opcional con keys tau, lambda_param, views,
                 eq_returns (np.array), post_returns (np.array)
    """
    row = {
        "portfolio_id":    portfolio_id,
        "scenario_id":     scenario_id,
        "risk_free_rate":  risk_free_rate,
        "n_sims":          n_sims,
        "expected_return": float(metrics["expected_return"]),
        "volatility":      float(metrics["volatility"]),
        "sharpe":          float(metrics["sharpe"]),
        "prob_loss":       float(metrics["prob_loss"]),
        "var_95":          float(metrics["var_95"]),
        "es_95":           float(metrics["es_95"]),
        "marginal_risk":   metrics["marginal_risk"].tolist(),
    }
    if bl_data:
        row.update({
            "bl_tau":           bl_data.get("tau"),
            "bl_lambda":        bl_data.get("lambda_param"),
            "bl_views":         bl_data.get("views"),          # list[dict]
            "bl_eq_returns":    bl_data["eq_returns"].tolist() if "eq_returns" in bl_data else None,
            "bl_post_returns":  bl_data["post_returns"].tolist() if "post_returns" in bl_data else None,
        })
    return get_client().table("results_snapshots").insert(row).execute().data[0]


def get_snapshots(portfolio_id: int, limit: int = 20) -> list[dict]:
    """Historial de snapshots de una cartera (sin los arrays grandes)."""
    return (get_client().table("results_snapshots")
            .select("id, created_at, expected_return, volatility, sharpe, prob_loss, var_95, es_95")
            .eq("portfolio_id", portfolio_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute().data)
