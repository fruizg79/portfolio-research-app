"""
pages/4_Black_Litterman.py
==========================
Modelo Black-Litterman: views cualitativos → retornos bayesianos.

Reads from session state:
    eq_returns, volatilities, corr_matrix, portfolio_weights
Writes to session state:
    bl_eq_returns, bl_post_returns
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.state   import init_state
from utils.ui      import apply_css, page_header, section, divider, kpi_card, guard, info
from utils.finance import run_black_litterman, get_portfolio_metrics, to_cache_args

init_state()
apply_css()

page_header("🔭 Black-Litterman",
            "Incorpora tus views y genera retornos esperados bayesianos")

guard("eq_returns", "volatilities", "corr_matrix", "portfolio_weights")

assets     = st.session_state["asset_classes"]
n          = len(assets)
eq_ret     = st.session_state["eq_returns"]
vols       = st.session_state["volatilities"]
corr       = st.session_state["corr_matrix"]
weights_eq = st.session_state["portfolio_weights"]

# ── BL Parameters ─────────────────────────────────────────────────────────────
section("Parámetros del Modelo")
pc1, pc2, pc3 = st.columns(3)
tau          = pc1.number_input("τ (tau)", value=0.025, min_value=0.005,
                                max_value=0.1, step=0.005, format="%.3f",
                                help="Escala la incertidumbre del prior (típicamente 0.025-0.05)")
lambda_param = pc2.number_input("λ (aversión al riesgo)", value=2.5,
                                min_value=0.5, max_value=10.0, step=0.1,
                                help="Coeficiente de aversión al riesgo del mercado")
rf_bl        = pc3.number_input("Tasa libre de riesgo (%)", value=2.0,
                                min_value=0.0, max_value=8.0, step=0.1) / 100

# ── Views ─────────────────────────────────────────────────────────────────────
divider()
section("Views del Gestor")
info("Define views <b>absolutas</b> (un activo) o <b>relativas</b> (activo A vs activo B). "
     "La recomendación cualitativa ajusta el retorno esperado en la dirección indicada, "
     "ponderada por la confianza asignada.")

RECOS = {
    "strong_buy":  "Strong Buy ⬆⬆",
    "buy":         "Buy ⬆",
    "neutral":     "Neutral ➡",
    "sell":        "Sell ⬇",
    "strong_sell": "Strong Sell ⬇⬇",
}

n_views = st.number_input("Número de views", min_value=1,
                           max_value=min(n, 8), value=min(2, n), step=1)

p_rows          = []
recommendations = []
confidence_lvls = []

for v in range(int(n_views)):
    st.markdown(f"**View {v + 1}**")
    vc      = st.columns([2, 2, 2, 2, 1])
    vtype   = vc[0].selectbox("Tipo", ["Absoluta", "Relativa"], key=f"vt_{v}")
    asset_a = vc[1].selectbox("Activo (Long)", assets, key=f"va_{v}")
    asset_b = (vc[2].selectbox("Activo (Short)",
                               [a for a in assets if a != asset_a],
                               key=f"vb_{v}")
               if vtype == "Relativa" else None)
    reco    = vc[3].selectbox("Recomendación", list(RECOS.keys()),
                              format_func=lambda x: RECOS[x],
                              index=1, key=f"vr_{v}")
    conf    = vc[4].slider("Conf.", 0.1, 1.0, 0.6, 0.05, key=f"vc_{v}")

    row = np.zeros(n)
    row[assets.index(asset_a)] = 1.0
    if asset_b:
        row[assets.index(asset_b)] = -1.0
    p_rows.append(row)
    recommendations.append(reco)
    confidence_lvls.append(conf)

# ── Compute ───────────────────────────────────────────────────────────────────
if st.button("🔭  Calcular Black-Litterman", type="primary"):
    try:
        vol_t, corr_t, n_assets = to_cache_args(vols, corr)
        p_flat = tuple(np.array(p_rows).flatten())

        pi_eq, bl_post = run_black_litterman(
            eq_returns        = tuple(eq_ret),
            volatilities      = vol_t,
            corr_flat         = corr_t,
            n                 = n_assets,
            weights_eq        = tuple(weights_eq),
            p_matrix_flat     = p_flat,
            n_views           = int(n_views),
            recommendations   = tuple(recommendations),
            confidence_levels = tuple(confidence_lvls),
            tau               = tau,
            lambda_param      = lambda_param,
            risk_free_rate    = rf_bl,
        )

        # Persist for other pages
        st.session_state["bl_eq_returns"]   = pi_eq
        st.session_state["bl_post_returns"] = bl_post

        # ── Returns comparison chart ──────────────────────────────────────
        divider()
        section("Retornos Esperados")

        fig_bl = go.Figure()
        fig_bl.add_trace(go.Bar(name="Equilibrio",   x=assets, y=pi_eq   * 100,
                                marker_color="#8ca0bb", opacity=0.7))
        fig_bl.add_trace(go.Bar(name="BL Posterior", x=assets, y=bl_post * 100,
                                marker_color="#1e3a5f", opacity=0.85))
        fig_bl.update_layout(
            barmode="group", yaxis_title="Rentabilidad Esperada (%)",
            height=360, paper_bgcolor="white", plot_bgcolor="#fafaf8",
            margin=dict(t=20, b=40, l=40, r=20),
        )
        fig_bl.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_bl, use_container_width=True)

        adj = bl_post - pi_eq
        fig_adj = go.Figure(go.Bar(
            x=assets, y=adj * 100,
            marker_color=["#2d6a4f" if v > 0 else "#9b2226" for v in adj],
            opacity=0.8,
            text=[f"{v*100:+.2f}%" for v in adj], textposition="outside",
        ))
        fig_adj.add_hline(y=0, line_color="#888", line_width=1)
        fig_adj.update_layout(
            title=dict(text="Ajuste BL vs Equilibrio (p.p.)",
                       font=dict(family="DM Serif Display", size=15)),
            yaxis_title="Ajuste (%)", height=300,
            paper_bgcolor="white", plot_bgcolor="#fafaf8",
            margin=dict(t=50, b=40, l=40, r=20),
        )
        fig_adj.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_adj, use_container_width=True)

        df_bl = pd.DataFrame({
            "Equilibrio":   [f"{v:.2%}" for v in pi_eq],
            "BL Posterior": [f"{v:.2%}" for v in bl_post],
            "Ajuste":       [f"{v:+.2%}" for v in adj],
        }, index=assets)
        st.dataframe(df_bl, use_container_width=True)

        # ── Portfolio metrics with BL returns ─────────────────────────────
        divider()
        section("Métricas de Cartera con Retornos BL")

        m_bl = get_portfolio_metrics(
            weights        = tuple(weights_eq),
            expected_ret   = tuple(bl_post),
            volatilities   = vol_t,
            corr_flat      = corr_t,
            n              = n_assets,
            risk_free_rate = rf_bl,
        )
        m_eq = get_portfolio_metrics(
            weights        = tuple(weights_eq),
            expected_ret   = tuple(pi_eq),
            volatilities   = vol_t,
            corr_flat      = corr_t,
            n              = n_assets,
            risk_free_rate = rf_bl,
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        kpi_card(mc1, "Rentabilidad Esperada", m_bl["expected_return"], ".2%",
                 delta=m_bl["expected_return"] - m_eq["expected_return"],
                 delta_label="vs equilibrio")
        kpi_card(mc2, "Volatilidad",           m_bl["volatility"],      ".2%",
                 delta=m_bl["volatility"] - m_eq["volatility"],
                 delta_label="vs equilibrio")
        kpi_card(mc3, "Ratio de Sharpe",       m_bl["sharpe"],          ".2f",
                 delta=m_bl["sharpe"] - m_eq["sharpe"],
                 delta_label="vs equilibrio")
        kpi_card(mc4, "Prob. Pérdida",         m_bl["prob_loss"],       ".1%",
                 delta=m_bl["prob_loss"] - m_eq["prob_loss"],
                 delta_label="vs equilibrio")

        # ── Views summary ─────────────────────────────────────────────────
        divider()
        section("Resumen de Views")
        rows = []
        for v in range(int(n_views)):
            long_a  = [assets[i] for i, val in enumerate(p_rows[v]) if val >  0]
            short_a = [assets[i] for i, val in enumerate(p_rows[v]) if val <  0]
            rows.append({
                "View":          f"View {v+1}",
                "Tipo":          "Relativa" if short_a else "Absoluta",
                "Long":          ", ".join(long_a),
                "Short":         ", ".join(short_a) or "—",
                "Recomendación": RECOS[recommendations[v]],
                "Confianza":     f"{confidence_lvls[v]:.0%}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("View"), use_container_width=True)

    except np.linalg.LinAlgError as e:
        st.error(f"Error de álgebra lineal: {e}. "
                 "Revisa que la matriz de correlaciones sea definida positiva.")
    except Exception as e:
        st.error(f"Error en el cálculo Black-Litterman: {e}")

else:
    info("👆 Define tus views y pulsa <b>Calcular Black-Litterman</b>.")
