"""
pages/2_Mi_Cartera.py
=====================
Pesos de la cartera y rangos tácticos.

Writes to session state:
    portfolio_weights, tactical_ranges
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.state import init_state, reset_downstream
from utils.ui    import apply_css, page_header, section, divider

init_state()
apply_css()

page_header("📋 Mi Cartera", "Pesos actuales y rangos tácticos de inversión")

assets = st.session_state["asset_classes"]
n      = len(assets)

# ── Weights ───────────────────────────────────────────────────────────────────
section("Pesos de la Cartera")
st.caption("Los pesos deben sumar 100 %")

prev_w  = st.session_state["portfolio_weights"]
weights = {}
cols    = st.columns(min(n, 5))

for i, a in enumerate(assets):
    dft = float(prev_w[i] * 100) if prev_w is not None else round(100 / n, 1)
    weights[a] = cols[i % 5].number_input(
        a, value=dft, min_value=0.0, max_value=100.0,
        step=0.5, format="%.1f", key=f"w_{i}")

total_w = sum(weights.values())
if abs(total_w - 100) < 0.01:
    st.success(f"✅  Suma: {total_w:.1f}%")
else:
    st.error(f"❌  Suma: {total_w:.1f}%  (diferencia: {total_w - 100:+.1f}%)")

if abs(total_w - 100) < 0.5:
    palette = px.colors.sequential.Blues_r
    fig_pie = go.Figure(go.Pie(
        labels=list(weights.keys()), values=list(weights.values()),
        hole=0.44,
        marker=dict(colors=[palette[i % len(palette)] for i in range(n)],
                    line=dict(color="white", width=2)),
        textinfo="label+percent",
        textfont=dict(family="DM Sans", size=12),
    ))
    fig_pie.update_layout(
        title=dict(text="Distribución de la Cartera",
                   font=dict(family="DM Serif Display", size=16)),
        height=370, paper_bgcolor="white",
        margin=dict(t=50, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Tactical ranges ───────────────────────────────────────────────────────────
divider()
section("Rangos Tácticos")
st.caption("Desviación máxima permitida respecto al peso estratégico (p.p.)")

prev_tr = st.session_state["tactical_ranges"]
tact    = {}
cols2   = st.columns(min(n, 5))

for i, a in enumerate(assets):
    dft_r = float(prev_tr[i] * 100) if prev_tr is not None else 10.0
    tact[a] = cols2[i % 5].number_input(
        a, value=dft_r, min_value=0.0, max_value=50.0,
        step=0.5, format="%.1f", key=f"tr_{i}") / 100

w_arr  = np.array([weights[a] / 100 for a in assets])
tr_arr = np.array([tact[a] for a in assets])

fig_range = go.Figure()
fig_range.add_trace(go.Bar(name="Peso Actual",     x=assets, y=w_arr * 100,
                            marker_color="#1e3a5f", opacity=0.85))
fig_range.add_trace(go.Bar(name="Límite Superior", x=assets, y=(w_arr + tr_arr) * 100,
                            marker_color="#c9a84c", opacity=0.4))
fig_range.add_trace(go.Bar(name="Límite Inferior", x=assets,
                            y=np.maximum(w_arr - tr_arr, 0) * 100,
                            marker_color="#9b2226", opacity=0.3))
fig_range.update_layout(
    barmode="overlay",
    title=dict(text="Pesos y Rangos Tácticos",
               font=dict(family="DM Serif Display", size=16)),
    yaxis_title="Peso (%)", height=340,
    paper_bgcolor="white", plot_bgcolor="white",
    margin=dict(t=50, b=20, l=20, r=20),
)
fig_range.update_xaxes(tickangle=-30)
st.plotly_chart(fig_range, use_container_width=True)

# ── Save ──────────────────────────────────────────────────────────────────────
if st.button("💾  Guardar Cartera y Rangos", type="primary"):
    if abs(total_w - 100) > 0.5:
        st.error("Los pesos deben sumar 100 % antes de guardar.")
    else:
        st.session_state["portfolio_weights"] = np.array([weights[a] / 100 for a in assets])
        st.session_state["tactical_ranges"]   = np.array([tact[a]          for a in assets])
        reset_downstream("portfolio_weights")
        st.success("✅  Cartera y rangos tácticos guardados.")

if st.session_state.get("portfolio_weights") is not None:
    st.markdown("**Siguiente paso →**")
    st.page_link("pages/3_Riesgo_Retorno.py", label="Calcular Riesgo & Retorno", icon="📈")
