"""
pages/0_Cargar_Fondos.py
========================
Carga una cartera de fondos desde CSV y deriva automáticamente las
clases de activo y los pesos del asset allocation mediante look-through.

Flujo
-----
1. El usuario sube un CSV con la estructura:
       ISIN | Nombre del fondo | Peso en cartera | Clase activo 1 | Clase activo 2 | …
2. La app calcula la contribución de cada fondo a cada clase de activo
   y agrega los pesos al nivel de cartera.
3. El usuario revisa el desglose y los avisos de validación.
4. Al pulsar "Cargar en sesión" se actualizan:
       asset_classes      → nombres de columna del CSV (clases de activo)
       portfolio_weights  → pesos agregados look-through
       tactical_ranges    → rangos editables en esta misma página

Resultado en session state
--------------------------
    asset_classes      list[str]    → reemplaza el universo previo
    portfolio_weights  np.ndarray   → pesos look-through, suma ≈ 1
    tactical_ranges    np.ndarray   → rango táctico por clase (decimal)

Nota: eq_returns, volatilities y corr_matrix NO se tocan; el usuario
debe configurarlos en Configuración (página 1) para el nuevo universo.
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.state        import init_state, reset_downstream
from utils.ui           import (apply_css, page_header, section,
                                divider, info, warn, status_badge)
from utils.portfolio_loader import parse_fund_portfolio

init_state()
apply_css()

page_header(
    "🗂️ Cargar Cartera de Fondos",
    "Derivación automática de clases de activo y pesos por look-through",
)

# ── Plantilla de descarga ─────────────────────────────────────────────────────
with st.expander("📥 Descargar plantilla CSV"):
    info(
        "La plantilla muestra el formato esperado. "
        "Las columnas de clase de activo pueden tener cualquier nombre y "
        "número — la app las detecta automáticamente a partir de la cuarta columna."
    )
    _tpl = pd.DataFrame({
        "ISIN":                ["LU0001234567", "LU0009876543", "LU0005556667"],
        "Nombre_Fondo":        ["Fondo Global Equity", "Fondo Renta Fija", "Fondo Mixto"],
        "Peso_Cartera":        [40, 35, 25],
        "Renta Variable DM":   [70, 0,  30],
        "Renta Variable EM":   [20, 0,  10],
        "Renta Fija Gov":      [0,  60, 25],
        "Renta Fija Corp":     [5,  35, 20],
        "Alternativos":        [5,  5,  15],
    })
    buf = io.BytesIO()
    _tpl.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Descargar plantilla",
        buf.getvalue(),
        "plantilla_cartera_fondos.csv",
        "text/csv",
    )

divider()

# ── Upload ────────────────────────────────────────────────────────────────────
section("1 · Subir fichero")

col_up, col_sep = st.columns([3, 1])
uploaded = col_up.file_uploader(
    "Fichero CSV con la cartera de fondos",
    type=["csv"],
    help="Columnas: ISIN, Nombre, Peso_Cartera, [clase activo 1], [clase activo 2], …",
)
sep = col_sep.selectbox("Separador", [",", ";", "\\t"], index=0,
                        help="Carácter separador de columnas del CSV")
sep = "\t" if sep == "\\t" else sep

if not uploaded:
    info(
        "Sube un <b>.csv</b> con la estructura:<br>"
        "&nbsp;&nbsp;<code>ISIN, Nombre_Fondo, Peso_Cartera, Clase1, Clase2, …</code><br>"
        "Los pesos pueden estar en tanto por uno (0.40) o tanto por ciento (40)."
    )
    st.stop()

# ── Parse ─────────────────────────────────────────────────────────────────────
try:
    raw_df = pd.read_csv(uploaded, sep=sep)
except Exception as e:
    st.error(f"❌ No se pudo leer el fichero: {e}")
    st.stop()

try:
    result = parse_fund_portfolio(raw_df)
except ValueError as e:
    st.error(f"❌ Formato incorrecto: {e}")
    st.stop()

# ── Validation warnings ───────────────────────────────────────────────────────
if result.warnings:
    for msg in result.warnings:
        warn(f"⚠️ {msg}")

divider()

# ── Section 2 — Detail breakdown ─────────────────────────────────────────────
section("2 · Desglose por fondo y clase de activo")

asset_cols = result.asset_classes

# Format detail table for display
detail_display = result.detail.copy()
detail_display["Peso_Cartera"] = detail_display["Peso_Cartera"].map("{:.1%}".format)
for col in asset_cols:
    detail_display[col] = detail_display[col].map("{:.1%}".format)

st.dataframe(
    detail_display.set_index("ISIN"),
    use_container_width=True,
)

divider()

# ── Section 3 — Aggregated composition ───────────────────────────────────────
section("3 · Composición agregada look-through")

n = len(asset_cols)
cols_kpi = st.columns(min(n, 5))
palette = ["#1e3a5f", "#2d5a8e", "#4a7fbb", "#c9a84c", "#e8c97a",
           "#9b2226", "#2d6a4f", "#4a9e6f", "#8ca0bb", "#506070"]

for i, ac in enumerate(asset_cols):
    w = result.composition[ac]
    cols_kpi[i % 5].metric(ac, f"{w:.1%}")

# Bar chart
fig = go.Figure(go.Bar(
    x=asset_cols,
    y=[result.composition[ac] * 100 for ac in asset_cols],
    marker_color=[palette[i % len(palette)] for i in range(n)],
    text=[f"{result.composition[ac]:.1%}" for ac in asset_cols],
    textposition="outside",
    textfont=dict(family="JetBrains Mono, monospace", size=11),
))
fig.update_layout(
    title=dict(
        text="Composición Look-Through de la Cartera",
        font=dict(family="DM Serif Display, serif", size=16),
    ),
    yaxis=dict(title="Peso (%)", ticksuffix="%"),
    xaxis=dict(tickangle=-20),
    height=360,
    paper_bgcolor="white",
    plot_bgcolor="#fafaf8",
    margin=dict(t=60, b=60, l=40, r=20),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

# Total weight info
total = result.weights_sum
if abs(total - 1.0) < 0.02:
    st.success(f"✅ Suma de pesos look-through: **{total:.2%}**")
else:
    st.warning(f"⚠️ Suma de pesos look-through: **{total:.2%}** (esperado ≈ 100 %)")

divider()

# ── Section 4 — Tactical ranges ──────────────────────────────────────────────
section("4 · Rangos tácticos")
st.caption("Desviación máxima permitida respecto al peso look-through (p.p.)")

prev_tr = st.session_state.get("tactical_ranges")
# Only pre-fill if the existing tactical_ranges match the current asset universe
if (
    prev_tr is not None
    and len(prev_tr) == n
    and st.session_state.get("asset_classes") == asset_cols
):
    defaults_tr = [float(prev_tr[i] * 100) for i in range(n)]
else:
    defaults_tr = [10.0] * n

tact_cols = st.columns(min(n, 5))
tact_vals: dict[str, float] = {}
for i, ac in enumerate(asset_cols):
    tact_vals[ac] = tact_cols[i % 5].number_input(
        ac,
        value=defaults_tr[i],
        min_value=0.0,
        max_value=50.0,
        step=0.5,
        format="%.1f",
        key=f"tr_lf_{i}",
    ) / 100.0

divider()

# ── Section 5 — Load into session ────────────────────────────────────────────
section("5 · Cargar en sesión")

# Warn if existing asset classes differ
existing_ac = st.session_state.get("asset_classes", [])
if existing_ac and existing_ac != asset_cols:
    warn(
        f"⚠️ El universo actual de la sesión "
        f"(<b>{', '.join(existing_ac)}</b>) será reemplazado por el del CSV "
        f"(<b>{', '.join(asset_cols)}</b>). "
        "Los parámetros de mercado (rentabilidades, volatilidades, "
        "correlaciones) se resetearán y deberás reconfigurarlos en "
        "<b>Configuración</b>."
    )

col_btn, col_status = st.columns([1, 2])

if col_btn.button("⬆️  Cargar en sesión", type="primary", use_container_width=True):
    # 1. Update asset universe — cascades to everything downstream
    st.session_state["asset_classes"] = asset_cols
    reset_downstream("asset_classes")

    # 2. Set portfolio weights and tactical ranges (downstream was just reset)
    st.session_state["portfolio_weights"] = result.weights
    st.session_state["tactical_ranges"]   = np.array([tact_vals[ac] for ac in asset_cols])

    st.success(
        f"✅ Cargados **{n}** clases de activo y pesos look-through. "
        "Ve a **Configuración** para introducir rentabilidades, "
        "volatilidades y correlaciones del nuevo universo."
    )
    st.rerun()

# Status badges
with col_status:
    loaded = (
        st.session_state.get("asset_classes") == asset_cols
        and st.session_state.get("portfolio_weights") is not None
    )
    st.markdown(
        status_badge(loaded, "Cargado en sesión"),
        unsafe_allow_html=True,
    )
