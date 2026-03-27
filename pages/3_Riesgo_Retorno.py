"""
pages/3_Riesgo_Retorno.py
=========================
Métricas de riesgo y retorno de la cartera actual.

Reads from session state:
    eq_returns, volatilities, corr_matrix, portfolio_weights,
    sim_models, sim_model_params  (optional — enables per-asset stochastic MC)
Writes to session state:
    portfolio_metrics  (cached for other pages to read)
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.state   import init_state
from utils.ui      import apply_css, page_header, section, divider, kpi_card, guard, info
from utils.finance import (
    get_portfolio_metrics, run_stochastic_mc,
    pack_sim_params, to_cache_args,
)

init_state()
apply_css()

page_header("📈 Riesgo & Retorno Esperados",
            "Métricas analíticas y Monte Carlo a horizonte 1 año")

guard("eq_returns", "volatilities", "corr_matrix", "portfolio_weights")

assets  = st.session_state["asset_classes"]
eq_ret  = st.session_state["eq_returns"]
vols    = st.session_state["volatilities"]
corr    = st.session_state["corr_matrix"]
weights = st.session_state["portfolio_weights"]
n       = len(assets)

risk_free = st.sidebar.slider("Tasa libre de riesgo (%)", 0.0, 5.0,
                               st.session_state["risk_free_rate"] * 100, 0.1) / 100
st.session_state["risk_free_rate"] = risk_free

n_sims = st.session_state["n_sims"]

# ── Compute (cached) ──────────────────────────────────────────────────────────
vol_t, corr_t, n_assets = to_cache_args(vols, corr)

sim_models       = st.session_state.get("sim_models")
sim_model_params = st.session_state.get("sim_model_params")

if sim_models and sim_model_params:
    # Per-asset stochastic process simulation (GBM / Vasicek / Normal)
    models_t, params_t = pack_sim_params(sim_models, sim_model_params)
    metrics = run_stochastic_mc(
        weights                = tuple(weights),
        asset_classes          = tuple(assets),
        sim_models_items       = models_t,
        sim_model_params_items = params_t,
        corr_flat              = corr_t,
        n                      = n_assets,
        risk_free_rate         = risk_free,
        n_sims                 = n_sims,
    )
    _mc_mode = "stochastic"
else:
    # Fallback: multivariate normal MC (no per-asset model configured)
    metrics = get_portfolio_metrics(
        weights        = tuple(weights),
        expected_ret   = tuple(eq_ret),
        volatilities   = vol_t,
        corr_flat      = corr_t,
        n              = n_assets,
        risk_free_rate = risk_free,
        n_sims         = n_sims,
    )
    _mc_mode = "normal"

# Cache result so other pages (e.g. Escenarios snapshot) can read it
st.session_state["portfolio_metrics"] = metrics

# ── Simulation mode badge ────────────────────────────────────────────────────
if _mc_mode == "stochastic":
    _model_summary = ", ".join(
        f"{ac}: `{sim_models.get(ac, 'normal')}`"
        for ac in assets
    )
    st.success(
        f"🎲 **Monte Carlo estocástico** — procesos por activo: {_model_summary}",
        icon=None,
    )
else:
    st.info(
        "🎲 **Monte Carlo normal** — distribución normal multivariante (todos los activos). "
        "Configura modelos por activo en **Configuración › Sección 3** para activar "
        "GBM / Vasicek.",
        icon=None,
    )

# ── KPI Cards ─────────────────────────────────────────────────────────────────
section("Métricas Principales")
c1, c2, c3, c4 = st.columns(4)
kpi_card(c1, "Rentabilidad Esperada (anual)", metrics["expected_return"], ".2%",
         "positive" if metrics["expected_return"] > 0 else "negative")
kpi_card(c2, "Volatilidad (anual)",  metrics["volatility"],  ".2%")
kpi_card(c3, "Ratio de Sharpe",      metrics["sharpe"],      ".2f",
         "positive" if metrics["sharpe"] > 0.5 else "")
kpi_card(c4, "Prob. Pérdida (1 año)", metrics["prob_loss"],  ".1%",
         "negative" if metrics["prob_loss"] > 0.35 else "")

st.markdown("<br>", unsafe_allow_html=True)
c5, c6, _, _ = st.columns(4)
kpi_card(c5, "VaR 95 % (1 año)",        metrics["var_95"], ".2%",
         "negative" if metrics["var_95"] < 0 else "")
kpi_card(c6, "Expected Shortfall 95 %",  metrics["es_95"],  ".2%",
         "negative" if metrics["es_95"]  < 0 else "")

# ── Return distribution ───────────────────────────────────────────────────────
divider()
section("Distribución de Retornos a 1 Año (Monte Carlo)")

sim = metrics["sim_returns"]
fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=sim, nbinsx=120, histnorm="probability density",
    marker_color="#1e3a5f", opacity=0.6, name="Simulaciones"))
loss_r = sim[sim < 0]
if len(loss_r):
    fig_dist.add_trace(go.Histogram(
        x=loss_r, nbinsx=60, histnorm="probability density",
        marker_color="#9b2226", opacity=0.5, name="Zona de Pérdida"))
fig_dist.add_vline(x=metrics["var_95"], line_dash="dash", line_color="#c9a84c", line_width=2,
                   annotation_text=f"VaR 95%: {metrics['var_95']:.1%}",
                   annotation_font_color="#c9a84c", annotation_position="top right")
fig_dist.add_vline(x=metrics["es_95"], line_dash="dot", line_color="#e05252", line_width=2,
                   annotation_text=f"ES 95%: {metrics['es_95']:.1%}",
                   annotation_font_color="#e05252", annotation_position="top left")
fig_dist.add_vline(x=0, line_color="#888", line_width=1)
fig_dist.update_layout(
    barmode="overlay", xaxis_title="Retorno 1 Año", yaxis_title="Densidad",
    height=370, paper_bgcolor="white", plot_bgcolor="#fafaf8",
    margin=dict(t=20, b=40, l=40, r=20),
    xaxis=dict(tickformat=".0%"),
)
st.plotly_chart(fig_dist, use_container_width=True)

# ── Risk contribution ─────────────────────────────────────────────────────────
divider()
section("Contribución al Riesgo por Activo")

risk_contrib = metrics["marginal_risk"]
fig_rc = go.Figure(go.Bar(
    x=assets, y=risk_contrib * 100,
    marker_color=["#1e3a5f" if v >= 0 else "#9b2226" for v in risk_contrib],
    opacity=0.85,
    text=[f"{v:.1f}%" for v in risk_contrib * 100], textposition="outside",
))
fig_rc.update_layout(
    yaxis_title="% Contribución al Riesgo", height=310,
    paper_bgcolor="white", plot_bgcolor="#fafaf8",
    margin=dict(t=20, b=40, l=40, r=20),
)
fig_rc.update_xaxes(tickangle=-30)
st.plotly_chart(fig_rc, use_container_width=True)

import pandas as pd
df_tbl = {
    "Activo":           assets,
    "Peso":             [f"{w:.1%}" for w in weights],
    "Rentabilidad Eq.": [f"{r:.2%}" for r in eq_ret],
    "Volatilidad":      [f"{v:.2%}" for v in vols],
    "Modelo MC":        [
        (sim_models or {}).get(ac, "normal") for ac in assets
    ],
    "Contrib. Riesgo":  [f"{c:.1f}%" for c in risk_contrib * 100],
}
st.dataframe(pd.DataFrame(df_tbl).set_index("Activo"), use_container_width=True)

# ── Correlation heatmap ───────────────────────────────────────────────────────
divider()
section("Matriz de Correlaciones")
fig_heat = go.Figure(go.Heatmap(
    z=corr, x=assets, y=assets,
    colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
    text=np.round(corr, 2), texttemplate="%{text:.2f}",
    textfont=dict(size=11, family="JetBrains Mono"),
    colorbar=dict(title="ρ"),
))
fig_heat.update_layout(
    height=400, paper_bgcolor="white",
    xaxis=dict(tickangle=-35, tickfont=dict(size=11)),
    yaxis=dict(tickfont=dict(size=11)),
    margin=dict(t=20, b=60, l=100, r=20),
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Navigation ────────────────────────────────────────────────────────────────
divider()
st.markdown("**Guardar o explorar más →**")
col_nav1, col_nav2 = st.columns(2)
with col_nav1:
    st.page_link("pages/6_Escenarios.py", label="Guardar resultados en BD", icon="🗄️")
with col_nav2:
    st.page_link("pages/4_Black_Litterman.py", label="Ajustar con Black-Litterman", icon="🔭")
