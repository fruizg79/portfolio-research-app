"""
pages/0_Cargar_Fondos.py
========================
Carga una cartera de fondos desde Excel/CSV y deriva automáticamente las
clases de activo y los pesos del asset allocation mediante look-through.

Flujo
-----
1. El usuario descarga la plantilla Excel (columnas pre-rellenadas con las
   clases de activo configuradas en sesión) y la cumplimenta.
2. Sube el fichero (.xlsx o .csv) y la app calcula la contribución de cada
   fondo a cada clase de activo y agrega los pesos al nivel de cartera.
3. El usuario revisa el desglose y los avisos de validación.
4. Al pulsar "Cargar en sesión":
   - Si las clases de activo coinciden con las ya configuradas, solo se
     actualizan portfolio_weights y tactical_ranges (sin tocar eq_returns,
     volatilities, corr_matrix).
   - Si difieren, se reemplaza el universo completo y se resetean los
     parámetros de mercado.
5. Opcionalmente, el usuario guarda la cartera en BD con un nombre.

Resultado en session state
--------------------------
    asset_classes      list[str]    → puede reemplazar el universo previo
    portfolio_weights  np.ndarray   → pesos look-through, suma ≈ 1
    tactical_ranges    np.ndarray   → rango táctico por clase (decimal)
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
from utils.database     import save_portfolio

init_state()
apply_css()

page_header(
    "🗂️ Cargar Cartera de Fondos",
    "Derivación automática de clases de activo y pesos por look-through",
)

# ── Plantilla de descarga ─────────────────────────────────────────────────────
with st.expander("📥 Descargar plantilla Excel"):
    info(
        "La plantilla usa las <b>clases de activo configuradas en sesión</b>. "
        "Cada fila es un fondo; las columnas de clase de activo indican el "
        "porcentaje de ese fondo invertido en cada clase (los totales de "
        "<i>Peso_Cartera</i> deben sumar 100)."
    )
    # Use current asset classes if available, else sensible defaults
    _tpl_classes = st.session_state.get("asset_classes") or [
        "Renta Variable DM", "Renta Variable EM",
        "Renta Fija Gov", "Renta Fija Corp", "Alternativos",
    ]
    _tpl_data = {
        "ISIN":         ["LU0001234567", "LU0009876543", "LU0005556667"],
        "Nombre_Fondo": ["Fondo Global Equity", "Fondo Renta Fija", "Fondo Mixto"],
        "Peso_Cartera": [40, 35, 25],
    }
    # Fill asset class columns with example allocations (rows sum to 100)
    _example_rows = [
        [round(100 / len(_tpl_classes))] * len(_tpl_classes),
        [0] * (len(_tpl_classes) - 1) + [100],
        [round(100 / len(_tpl_classes))] * len(_tpl_classes),
    ]
    for j, ac in enumerate(_tpl_classes):
        _tpl_data[ac] = [_example_rows[i][j] for i in range(3)]

    _tpl = pd.DataFrame(_tpl_data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as _w:
        _tpl.to_excel(_w, sheet_name="fondos", index=False)
    st.download_button(
        "⬇️ Descargar plantilla",
        buf.getvalue(),
        "plantilla_cartera_fondos.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

divider()

# ── Upload ────────────────────────────────────────────────────────────────────
section("1 · Subir fichero")

col_up, col_sep = st.columns([3, 1])
uploaded = col_up.file_uploader(
    "Fichero Excel (.xlsx) o CSV (.csv) con la cartera de fondos",
    type=["xlsx", "csv"],
    help="Columnas: ISIN, Nombre_Fondo, Peso_Cartera, [clase activo 1], [clase activo 2], …",
)
sep = col_sep.selectbox("Separador CSV", [",", ";", "\\t"], index=0,
                        help="Solo se usa si el fichero es .csv")
sep = "\t" if sep == "\\t" else sep

if not uploaded:
    info(
        "Sube un <b>.xlsx</b> (recomendado) o <b>.csv</b> con la estructura:<br>"
        "&nbsp;&nbsp;<code>ISIN | Nombre_Fondo | Peso_Cartera | Clase1 | Clase2 | …</code><br>"
        "Los pesos pueden estar en tanto por uno (0.40) o tanto por ciento (40)."
    )
    st.stop()

# ── Parse ─────────────────────────────────────────────────────────────────────
try:
    if uploaded.name.lower().endswith(".xlsx"):
        raw_df = pd.read_excel(uploaded, sheet_name=0)
    else:
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

existing_ac  = st.session_state.get("asset_classes") or []
classes_match = (existing_ac == asset_cols)
market_ok     = (
    st.session_state.get("eq_returns")   is not None
    and st.session_state.get("volatilities") is not None
    and st.session_state.get("corr_matrix")  is not None
)

if not classes_match and existing_ac:
    warn(
        f"Las clases de activo del fichero (<b>{', '.join(asset_cols)}</b>) "
        f"difieren de las que hay en sesión (<b>{', '.join(existing_ac)}</b>). "
        "Al cargar se resetearán las rentabilidades, volatilidades y "
        "correlaciones — deberás reconfigurarlas en <b>Configuración</b>."
    )

if classes_match and market_ok:
    info(
        "✅ Las clases de activo coinciden con las ya configuradas. "
        "Solo se actualizarán los <b>pesos de cartera</b> — los parámetros "
        "de mercado se conservarán."
    )

col_btn, col_status = st.columns([1, 2])

if col_btn.button("⬆️  Cargar en sesión", type="primary", use_container_width=True):
    if classes_match:
        # Same asset universe — only update weights, preserve market params
        st.session_state["portfolio_weights"] = result.weights
        st.session_state["tactical_ranges"]   = np.array([tact_vals[ac] for ac in asset_cols])
        reset_downstream("portfolio_weights")
    else:
        # Different universe — full reset cascade then restore weights
        st.session_state["asset_classes"] = asset_cols
        reset_downstream("asset_classes")
        st.session_state["portfolio_weights"] = result.weights
        st.session_state["tactical_ranges"]   = np.array([tact_vals[ac] for ac in asset_cols])

    if classes_match and market_ok:
        st.success(
            f"✅ Pesos look-through cargados para {n} clases de activo. "
            "Los parámetros de mercado se han conservado."
        )
    else:
        st.success(
            f"✅ Cargadas {n} clases de activo y pesos look-through. "
            "Ve a **Configuración** para introducir rentabilidades, "
            "volatilidades y correlaciones."
        )
    st.rerun()

with col_status:
    loaded = (
        st.session_state.get("asset_classes") == asset_cols
        and st.session_state.get("portfolio_weights") is not None
    )
    st.markdown(status_badge(loaded, "Cargado en sesión"), unsafe_allow_html=True)

# Navigation hint after loading
if loaded:
    st.markdown("**Siguiente paso →**")
    if market_ok and classes_match:
        col_n1, col_n2 = st.columns(2)
        col_n1.page_link("pages/3_Riesgo_Retorno.py",
                         label="Calcular Riesgo & Retorno", icon="📈")
        col_n2.page_link("pages/2_Mi_Cartera.py",
                         label="Ver Mi Cartera", icon="📋")
    else:
        st.page_link("pages/1_Configuracion.py",
                     label="Configurar parámetros de mercado", icon="⚙️")

divider()

# ── Section 6 — Save to database ─────────────────────────────────────────────
section("6 · Guardar cartera en base de datos")

sc_id   = st.session_state.get("active_scenario_id")
sc_name = st.session_state.get("active_scenario_name")

if not loaded:
    info("Carga primero la cartera en sesión (Sección 5) antes de guardarla en BD.")
elif sc_id is None:
    warn(
        "Para guardar en BD la cartera debe estar vinculada a un <b>escenario</b>. "
        "Ve a <b>Escenarios &amp; Carteras</b> y guarda o carga un escenario primero."
    )
    st.page_link("pages/6_Escenarios.py", label="Ir a Escenarios & Carteras", icon="🗄️")
else:
    st.caption(f"Se vinculará al escenario activo: **{sc_name}**")
    col_pn, col_pd = st.columns([2, 3])
    pt_name_input = col_pn.text_input(
        "Nombre de la cartera",
        value=st.session_state.get("active_portfolio_name") or "",
        placeholder="Cartera Look-Through Q2 2026",
    )
    pt_desc_input = col_pd.text_input(
        "Descripción (opcional)",
        placeholder="Pesos derivados por look-through del fichero de fondos",
    )

    col_save1, col_save2 = st.columns(2)
    if col_save1.button("💾  Guardar como nueva", use_container_width=True):
        if not pt_name_input.strip():
            st.error("Introduce un nombre para la cartera.")
        else:
            try:
                rec = save_portfolio(
                    name            = pt_name_input.strip(),
                    weights         = st.session_state["portfolio_weights"],
                    tactical_ranges = st.session_state["tactical_ranges"],
                    scenario_id     = sc_id,
                    description     = pt_desc_input.strip(),
                )
                st.session_state["active_portfolio_id"]   = rec["id"]
                st.session_state["active_portfolio_name"] = rec["name"]
                st.success(
                    f"✅ Cartera **{rec['name']}** guardada en BD (ID {rec['id']})."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Error al guardar: {e}")

    pt_id = st.session_state.get("active_portfolio_id")
    if col_save2.button("🔄  Actualizar existente",
                        disabled=pt_id is None, use_container_width=True,
                        help="Solo disponible si hay una cartera cargada desde BD"):
        from utils.database import update_portfolio
        try:
            update_portfolio(
                portfolio_id    = pt_id,
                name            = pt_name_input.strip() or st.session_state["active_portfolio_name"],
                weights         = st.session_state["portfolio_weights"],
                tactical_ranges = st.session_state["tactical_ranges"],
                description     = pt_desc_input.strip(),
            )
            st.success(f"✅ Cartera **{st.session_state['active_portfolio_name']}** actualizada.")
        except Exception as e:
            st.error(f"Error al actualizar: {e}")
