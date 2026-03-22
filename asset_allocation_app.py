"""
Asset Allocation MVP - Streamlit App
=====================================
Author: Fernando Ruiz

Depends on: asset_allocation.py (must live in the same directory)

Run with:
    streamlit run asset_allocation_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io

# ── Import from asset_allocation.py ──────────────────────────────────────────
try:
    from asset_allocation import (
        CPortfolio_optimization,
        CBlack_litterman,
        build_cov_matrix,
    )
    AA_AVAILABLE = True
except ImportError as e:
    AA_AVAILABLE = False
    _AA_ERROR = str(e)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Asset Allocation · Portfolio Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not AA_AVAILABLE:
    st.error(
        f"❌ No se pudo importar `asset_allocation.py`: {_AA_ERROR}\n\n"
        "Asegúrate de que el fichero está en el mismo directorio que esta app."
    )
    st.stop()

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --navy:      #0a1628;
    --navy-mid:  #132240;
    --navy-light:#1e3a5f;
    --gold:      #c9a84c;
    --gold-light:#e8c97a;
    --cream:     #f5f0e8;
    --success:   #2d6a4f;
    --danger:    #9b2226;
    --border:    #ddd6c8;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background-color: #f8f6f1; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy) 0%, var(--navy-mid) 100%);
    border-right: 1px solid var(--navy-light);
}
section[data-testid="stSidebar"] * { color: var(--cream) !important; }
section[data-testid="stSidebar"] .stRadio label {
    color: #aab8cc !important; font-size:0.9rem; letter-spacing:0.03em;
}

.app-header {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
    padding: 2rem 2.5rem 1.8rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 2px solid var(--gold);
}
.app-header h1 { font-family:'DM Serif Display',serif; color:var(--cream); font-size:2rem; margin:0; }
.app-header p  { color:#8ca0bb; font-size:0.85rem; margin:0.3rem 0 0; letter-spacing:0.05em; text-transform:uppercase; }

.section-title {
    font-family: 'DM Serif Display', serif;
    color: var(--navy);
    font-size: 1.35rem;
    border-bottom: 1.5px solid var(--gold);
    padding-bottom: 0.35rem;
    margin-bottom: 1.1rem;
}

.metric-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.15rem 1.3rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-card .label {
    font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;
    color:#4a5568; margin-bottom:0.3rem; font-weight:500;
}
.metric-card .value {
    font-family:'JetBrains Mono',monospace; font-size:1.65rem; font-weight:500; color:var(--navy);
}
.metric-card .delta { font-size:0.77rem; margin-top:0.2rem; color:#718096; }
.metric-card.positive .value { color: var(--success); }
.metric-card.negative .value { color: var(--danger); }

.info-box {
    background:#eef3fb; border-left:3px solid #4a7fcb;
    padding:0.85rem 1rem; border-radius:0 6px 6px 0;
    font-size:0.87rem; color:#2d4a7a; margin:0.9rem 0;
}
.warn-box {
    background:#fef9ec; border-left:3px solid var(--gold);
    padding:0.85rem 1rem; border-radius:0 6px 6px 0;
    font-size:0.87rem; color:#7a5a1a; margin:0.9rem 0;
}
.gold-divider { height:1px; background:linear-gradient(90deg,var(--gold) 0%,transparent 100%); margin:1.4rem 0; }

.stButton > button {
    background:var(--navy); color:var(--cream); border:none; border-radius:6px;
    font-family:'DM Sans',sans-serif; font-weight:500; letter-spacing:0.03em;
    padding:0.5rem 1.4rem; transition:background 0.2s;
}
.stButton > button:hover { background: var(--navy-light); }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
_DEFAULTS = {
    "asset_classes":     ["Renta Variable DM", "Renta Variable EM",
                          "Renta Fija Gov", "Renta Fija Corp", "Alternativos"],
    "eq_returns":        None,   # np.array, annual
    "volatilities":      None,   # np.array, annual
    "corr_matrix":       None,   # np.array (n×n)
    "portfolio_weights": None,   # np.array, sums to 1
    "tactical_ranges":   None,   # np.array, in decimal (e.g. 0.10 = 10 pp)
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Shared UI helpers ────────────────────────────────────────────────────────
def kpi_card(col, label, value, fmt, css_class="", delta=None, delta_label=""):
    delta_html = (
        f'<div class="delta">{delta_label}: {delta:{fmt}}</div>'
        if delta is not None else ""
    )
    col.markdown(f"""
    <div class="metric-card {css_class}">
        <div class="label">{label}</div>
        <div class="value">{value:{fmt}}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


def status_badge(ok, label):
    icon  = "✅" if ok else "⬜"
    bg    = "#eaf4ef" if ok else "#fce8e8"
    bdr   = "#b7dfc9" if ok else "#f5b5b5"
    color = "#2d6a4f" if ok else "#9b2226"
    return (f'<div style="background:{bg};border:1px solid {bdr};border-radius:6px;'
            f'padding:0.8rem 1rem;text-align:center;">'
            f'<div style="font-size:1.2rem">{icon}</div>'
            f'<div style="font-size:0.78rem;color:{color};font-weight:600;margin-top:4px">{label}</div>'
            f'</div>')


def guard(*fields):
    """Show warning and stop if any session-state field is None."""
    missing = [f for f in fields if st.session_state.get(f) is None]
    if missing:
        labels = {
            "eq_returns":        "Rentabilidades de equilibrio (Configuración)",
            "volatilities":      "Volatilidades (Configuración)",
            "corr_matrix":       "Matriz de correlaciones (Configuración)",
            "portfolio_weights": "Pesos de cartera (Mi Cartera)",
            "tactical_ranges":   "Rangos tácticos (Mi Cartera)",
        }
        items = "".join(f"&nbsp;&nbsp;• {labels.get(m, m)}<br>" for m in missing)
        st.markdown(f'<div class="warn-box">⚠️ Faltan los siguientes datos:<br>{items}</div>',
                    unsafe_allow_html=True)
        st.stop()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 0.5rem;text-align:center;'>
        <div style='font-family:DM Serif Display,serif;font-size:1.3rem;color:#f5f0e8;'>Portfolio Research</div>
        <div style='font-size:0.7rem;color:#8ca0bb;letter-spacing:0.1em;text-transform:uppercase;margin-top:2px;'>Asset Allocation · MVP</div>
    </div>
    <hr style='border-color:#1e3a5f;margin:0.8rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Módulo",
        ["⚙️  Configuración", "📋  Mi Cartera", "📈  Riesgo & Retorno", "🔭  Black-Litterman"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style='border-color:#1e3a5f;margin:1rem 0;'>
    <div style='font-size:0.72rem;color:#506070;padding:0 0.5rem;line-height:1.6;'>
        <b style='color:#8ca0bb;'>Nota:</b> Configure primero el módulo
        <b>Configuración</b> antes de usar los demás módulos.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
if page == "⚙️  Configuración":
    st.markdown('<div class="app-header"><h1>⚙️ Configuración</h1>'
                '<p>Definición del universo de inversión y parámetros de mercado</p></div>',
                unsafe_allow_html=True)

    # ── 1 · Asset classes ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">1 · Clases de Activo</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 1])

    with col_l:
        raw = st.text_area(
            "Introduce las clases de activo (una por línea)",
            value="\n".join(st.session_state["asset_classes"]),
            height=155,
        )
        if st.button("✔  Guardar clases de activo"):
            parsed = [a.strip() for a in raw.strip().splitlines() if a.strip()]
            if len(parsed) < 2:
                st.error("Introduce al menos 2 clases de activo.")
            else:
                st.session_state["asset_classes"] = parsed
                for k in ("eq_returns", "volatilities", "corr_matrix",
                          "portfolio_weights", "tactical_ranges"):
                    st.session_state[k] = None
                st.success(f"✅  {len(parsed)} clases de activo guardadas.")

    with col_r:
        st.markdown('<div class="info-box">Las clases de activo definidas aquí se '
                    'usarán en todos los módulos.</div>', unsafe_allow_html=True)
        for i, a in enumerate(st.session_state["asset_classes"], 1):
            st.markdown(f"&nbsp;&nbsp;`{i}.` {a}")

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── 2 · Market data ──────────────────────────────────────────────────────
    assets = st.session_state["asset_classes"]
    n      = len(assets)

    st.markdown('<div class="section-title">2 · Datos de Mercado</div>', unsafe_allow_html=True)
    method = st.radio("Método de entrada", ["📝  Manual", "📂  Subir Excel"], horizontal=True)

    if method == "📝  Manual":
        tab_ret, tab_vol, tab_corr = st.tabs(
            ["Rentabilidades Equilibrio", "Volatilidades", "Correlaciones"])

        with tab_ret:
            st.caption("Rentabilidad anual de equilibrio por activo (en %)")
            cols = st.columns(min(n, 5))
            ret_vals = {}
            for i, a in enumerate(assets):
                dft = float(st.session_state["eq_returns"][i] * 100) \
                      if st.session_state["eq_returns"] is not None else 6.0
                ret_vals[a] = cols[i % 5].number_input(
                    a, value=dft, step=0.1, format="%.2f", key=f"ret_{i}") / 100
            if st.button("Guardar Rentabilidades"):
                st.session_state["eq_returns"] = np.array([ret_vals[a] for a in assets])
                st.success("✅  Rentabilidades guardadas.")

        with tab_vol:
            st.caption("Volatilidad anual por activo (en %)")
            cols = st.columns(min(n, 5))
            vol_vals = {}
            for i, a in enumerate(assets):
                dft = float(st.session_state["volatilities"][i] * 100) \
                      if st.session_state["volatilities"] is not None else 15.0
                vol_vals[a] = cols[i % 5].number_input(
                    a, value=dft, step=0.1, format="%.2f", key=f"vol_{i}") / 100
            if st.button("Guardar Volatilidades"):
                st.session_state["volatilities"] = np.array([vol_vals[a] for a in assets])
                st.success("✅  Volatilidades guardadas.")

        with tab_corr:
            st.caption("Matriz de correlaciones (triangular superior; diagonal = 1)")
            prev_c = st.session_state["corr_matrix"] if st.session_state["corr_matrix"] is not None \
                     else np.eye(n)
            corr_inp = {(i, j): 1.0 for i in range(n) for j in range(n)}
            for i in range(n):
                cols_c = st.columns(n)
                for j in range(n):
                    if i == j:
                        cols_c[j].number_input(
                            f"{assets[i][:5]}×{assets[j][:5]}", value=1.0,
                            disabled=True, key=f"c_{i}_{j}")
                    elif j > i:
                        v = cols_c[j].number_input(
                            f"{assets[i][:5]}×{assets[j][:5]}",
                            value=float(prev_c[i, j]),
                            min_value=-1.0, max_value=1.0, step=0.05,
                            format="%.2f", key=f"c_{i}_{j}")
                        corr_inp[(i, j)] = corr_inp[(j, i)] = v
                    else:
                        cols_c[j].number_input(
                            f"{assets[i][:5]}×{assets[j][:5]}",
                            value=float(prev_c[i, j]), disabled=True,
                            key=f"c_{i}_{j}_r")
                        corr_inp[(i, j)] = float(prev_c[i, j])

            if st.button("Guardar Correlaciones"):
                corr = np.array([[corr_inp[(i, j)] for j in range(n)] for i in range(n)])
                if np.linalg.eigvalsh(corr).min() <= 0:
                    st.error("⚠️ Matriz no definida positiva. Revisa los valores.")
                else:
                    st.session_state["corr_matrix"] = corr
                    st.success("✅  Correlaciones guardadas.")

    else:  # Upload
        st.markdown('<div class="info-box">Sube un <b>.xlsx</b> con tres hojas: '
                    '<code>eq_returns</code>, <code>volatilities</code>, <code>correlation</code>. '
                    'La primera columna debe contener los nombres de los activos.</div>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("Subir Excel", type=["xlsx"])
        if uploaded:
            try:
                xls = pd.ExcelFile(uploaded)
                df_r = pd.read_excel(xls, sheet_name="eq_returns",  index_col=0)
                df_v = pd.read_excel(xls, sheet_name="volatilities", index_col=0)
                df_c = pd.read_excel(xls, sheet_name="correlation",  index_col=0)
                st.session_state["eq_returns"]  = df_r.values.flatten() / 100
                st.session_state["volatilities"] = df_v.values.flatten() / 100
                st.session_state["corr_matrix"]  = df_c.values
                st.success("✅  Datos cargados correctamente.")
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")

    # ── Status ───────────────────────────────────────────────────────────────
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Estado de Configuración</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(status_badge(bool(assets), "Clases de Activo"),            unsafe_allow_html=True)
    c2.markdown(status_badge(st.session_state["eq_returns"]  is not None, "Rentabilidades"), unsafe_allow_html=True)
    c3.markdown(status_badge(st.session_state["volatilities"] is not None, "Volatilidades"), unsafe_allow_html=True)
    c4.markdown(status_badge(st.session_state["corr_matrix"]  is not None, "Correlaciones"), unsafe_allow_html=True)

    # ── Template download ────────────────────────────────────────────────────
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    with st.expander("📥 Descargar plantilla Excel"):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            pd.DataFrame({"Rentabilidad (%)": [6.0] * n}, index=assets).to_excel(w, sheet_name="eq_returns")
            pd.DataFrame({"Volatilidad (%)":  [15.0] * n}, index=assets).to_excel(w, sheet_name="volatilities")
            pd.DataFrame(np.eye(n), index=assets, columns=assets).to_excel(w, sheet_name="correlation")
        st.download_button(
            "⬇️ Descargar plantilla", buf.getvalue(), "plantilla_parametros.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MI CARTERA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋  Mi Cartera":
    st.markdown('<div class="app-header"><h1>📋 Mi Cartera</h1>'
                '<p>Pesos actuales y rangos tácticos de inversión</p></div>',
                unsafe_allow_html=True)

    assets = st.session_state["asset_classes"]
    n      = len(assets)

    # ── Weights ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Pesos de la Cartera</div>', unsafe_allow_html=True)
    st.caption("Los pesos deben sumar 100 %")

    prev_w = st.session_state["portfolio_weights"]
    weights = {}
    cols = st.columns(min(n, 5))
    for i, a in enumerate(assets):
        dft = float(prev_w[i] * 100) if prev_w is not None else round(100 / n, 1)
        weights[a] = cols[i % 5].number_input(
            a, value=dft, min_value=0.0, max_value=100.0, step=0.5,
            format="%.1f", key=f"w_{i}")

    total_w = sum(weights.values())
    if abs(total_w - 100) < 0.01:
        st.success(f"✅  Suma: {total_w:.1f}%")
    else:
        st.error(f"❌  Suma: {total_w:.1f}%  (diferencia: {total_w - 100:+.1f}%)")

    # Pie chart
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

    # ── Tactical ranges ──────────────────────────────────────────────────────
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Rangos Tácticos</div>', unsafe_allow_html=True)
    st.caption("Desviación máxima permitida respecto al peso estratégico (p.p.)")

    prev_tr = st.session_state["tactical_ranges"]
    tact = {}
    cols2 = st.columns(min(n, 5))
    for i, a in enumerate(assets):
        dft_r = float(prev_tr[i] * 100) if prev_tr is not None else 10.0
        tact[a] = cols2[i % 5].number_input(
            a, value=dft_r, min_value=0.0, max_value=50.0, step=0.5,
            format="%.1f", key=f"tr_{i}") / 100

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

    # ── Save ─────────────────────────────────────────────────────────────────
    if st.button("💾  Guardar Cartera y Rangos", type="primary"):
        if abs(total_w - 100) > 0.5:
            st.error("Los pesos deben sumar 100 % antes de guardar.")
        else:
            st.session_state["portfolio_weights"] = np.array([weights[a] / 100 for a in assets])
            st.session_state["tactical_ranges"]   = np.array([tact[a]          for a in assets])
            st.success("✅  Cartera y rangos tácticos guardados.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RIESGO & RETORNO
# Uses: CPortfolio_optimization.get_portfolio_metrics()
#       build_cov_matrix()
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Riesgo & Retorno":
    st.markdown('<div class="app-header"><h1>📈 Riesgo & Retorno Esperados</h1>'
                '<p>Métricas analíticas y Monte Carlo a horizonte 1 año</p></div>',
                unsafe_allow_html=True)

    guard("eq_returns", "volatilities", "corr_matrix", "portfolio_weights")

    assets  = st.session_state["asset_classes"]
    eq_ret  = st.session_state["eq_returns"]
    vols    = st.session_state["volatilities"]
    corr    = st.session_state["corr_matrix"]
    weights = st.session_state["portfolio_weights"]

    risk_free = st.sidebar.slider("Tasa libre de riesgo (%)", 0.0, 5.0, 2.0, 0.1) / 100

    # ── Build optimizer and get metrics via asset_allocation.py ─────────────
    # CPortfolio_optimization builds cov internally; we also expose build_cov_matrix
    cov      = build_cov_matrix(vols, corr)            # from asset_allocation
    optimizer = CPortfolio_optimization(               # from asset_allocation
        port_wts=weights,
        correlation_matrix=corr,
        expected_ret=eq_ret,
        vol=vols,
    )
    metrics = optimizer.get_portfolio_metrics(         # NEW method in asset_allocation
        weights=weights,
        risk_free_rate=risk_free,
        n_sims=50_000,
    )

    # ── KPI Cards ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Métricas Principales</div>', unsafe_allow_html=True)
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
    kpi_card(c5, "VaR 95 % (1 año)",        metrics["var_95"],  ".2%",
             "negative" if metrics["var_95"] < 0 else "")
    kpi_card(c6, "Expected Shortfall 95 %",  metrics["es_95"],   ".2%",
             "negative" if metrics["es_95"]  < 0 else "")

    # ── Distribution chart ───────────────────────────────────────────────────
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Distribución de Retornos a 1 Año (Monte Carlo)</div>',
                unsafe_allow_html=True)

    sim = metrics["sim_returns"]
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=sim, nbinsx=120, histnorm="probability density",
        marker_color="#1e3a5f", opacity=0.6, name="Simulaciones"))
    loss_returns = sim[sim < 0]
    if len(loss_returns):
        fig_dist.add_trace(go.Histogram(
            x=loss_returns, nbinsx=60, histnorm="probability density",
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

    # ── Risk contribution ────────────────────────────────────────────────────
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Contribución al Riesgo por Activo</div>',
                unsafe_allow_html=True)

    risk_contrib = metrics["marginal_risk"]   # from get_portfolio_metrics()
    colors_rc = ["#1e3a5f" if v >= 0 else "#9b2226" for v in risk_contrib]
    fig_rc = go.Figure(go.Bar(
        x=assets, y=risk_contrib * 100,
        marker_color=colors_rc, opacity=0.85,
        text=[f"{v:.1f}%" for v in risk_contrib * 100],
        textposition="outside",
    ))
    fig_rc.update_layout(
        yaxis_title="% Contribución al Riesgo", height=310,
        paper_bgcolor="white", plot_bgcolor="#fafaf8",
        margin=dict(t=20, b=40, l=40, r=20),
    )
    fig_rc.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_rc, use_container_width=True)

    # Summary table
    df_tbl = pd.DataFrame({
        "Activo":             assets,
        "Peso":               [f"{w:.1%}" for w in weights],
        "Rentabilidad Eq.":   [f"{r:.2%}" for r in eq_ret],
        "Volatilidad":        [f"{v:.2%}" for v in vols],
        "Contrib. Riesgo":    [f"{c:.1f}%" for c in risk_contrib * 100],
    }).set_index("Activo")
    st.dataframe(df_tbl, use_container_width=True)

    # ── Correlation heatmap ──────────────────────────────────────────────────
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Matriz de Correlaciones</div>', unsafe_allow_html=True)
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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — BLACK-LITTERMAN
# Uses: CBlack_litterman.from_views()          (factory classmethod)
#       CPortfolio_optimization.get_portfolio_metrics()
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔭  Black-Litterman":
    st.markdown('<div class="app-header"><h1>🔭 Black-Litterman</h1>'
                '<p>Incorpora tus views y genera retornos esperados bayesianos</p></div>',
                unsafe_allow_html=True)

    guard("eq_returns", "volatilities", "corr_matrix", "portfolio_weights")

    assets      = st.session_state["asset_classes"]
    n           = len(assets)
    eq_ret      = st.session_state["eq_returns"]
    vols        = st.session_state["volatilities"]
    corr        = st.session_state["corr_matrix"]
    weights_eq  = st.session_state["portfolio_weights"]
    cov         = build_cov_matrix(vols, corr)

    # ── BL parameters ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Parámetros del Modelo</div>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    tau          = pc1.number_input("τ (tau)", value=0.025, min_value=0.005, max_value=0.1,
                                    step=0.005, format="%.3f",
                                    help="Escala la incertidumbre del prior (típicamente 0.025-0.05)")
    lambda_param = pc2.number_input("λ (aversión al riesgo)", value=2.5, min_value=0.5,
                                    max_value=10.0, step=0.1,
                                    help="Coeficiente de aversión al riesgo del mercado")
    rf_bl        = pc3.number_input("Tasa libre de riesgo (%)", value=2.0,
                                    min_value=0.0, max_value=8.0, step=0.1) / 100

    # ── Views ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Views del Gestor</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Define views <b>absolutas</b> (un activo) o '
                '<b>relativas</b> (activo A vs activo B). La recomendación cualitativa '
                'ajusta el retorno esperado en la dirección indicada, ponderada por la '
                'confianza asignada.</div>', unsafe_allow_html=True)

    RECOS = {
        "strong_buy":  "Strong Buy ⬆⬆",
        "buy":         "Buy ⬆",
        "neutral":     "Neutral ➡",
        "sell":        "Sell ⬇",
        "strong_sell": "Strong Sell ⬇⬇",
    }

    n_views = st.number_input("Número de views", min_value=1, max_value=min(n, 8),
                               value=min(2, n), step=1)

    p_rows         = []
    recommendations = []
    confidence_lvls = []

    for v in range(int(n_views)):
        st.markdown(f"**View {v + 1}**")
        vc = st.columns([2, 2, 2, 2, 1])
        vtype   = vc[0].selectbox("Tipo",           ["Absoluta", "Relativa"], key=f"vt_{v}")
        asset_a = vc[1].selectbox("Activo (Long)",  assets,                   key=f"va_{v}")
        asset_b = (vc[2].selectbox("Activo (Short)",
                                   [a for a in assets if a != asset_a],
                                   key=f"vb_{v}")
                   if vtype == "Relativa" else None)
        reco = vc[3].selectbox("Recomendación", list(RECOS.keys()),
                               format_func=lambda x: RECOS[x],
                               index=1, key=f"vr_{v}")
        conf = vc[4].slider("Conf.", 0.1, 1.0, 0.6, 0.05, key=f"vc_{v}")

        row = np.zeros(n)
        row[assets.index(asset_a)] = 1.0
        if asset_b:
            row[assets.index(asset_b)] = -1.0
        p_rows.append(row)
        recommendations.append(reco)
        confidence_lvls.append(conf)

    # ── Compute ───────────────────────────────────────────────────────────────
    if st.button("🔭  Calcular Black-Litterman", type="primary"):
        try:
            # ── Call CBlack_litterman.from_views() from asset_allocation.py ──
            pi_eq, bl_post = CBlack_litterman.from_views(
                eq_returns      = eq_ret,
                cov_matrix      = cov,
                weights_eq      = weights_eq,
                p_matrix        = np.array(p_rows),
                recommendations = recommendations,
                confidence_levels = confidence_lvls,
                tau             = tau,
                lambda_param    = lambda_param,
                risk_free_rate  = rf_bl,
                c_coef          = 1.0,
            )

            # ── Returns comparison chart ─────────────────────────────────────
            st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Retornos Esperados</div>',
                        unsafe_allow_html=True)

            fig_bl = go.Figure()
            fig_bl.add_trace(go.Bar(name="Equilibrio",   x=assets, y=pi_eq  * 100,
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

            # Adjustment bar
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

            # Returns table
            df_bl = pd.DataFrame({
                "Equilibrio":   [f"{v:.2%}" for v in pi_eq],
                "BL Posterior": [f"{v:.2%}" for v in bl_post],
                "Ajuste":       [f"{v:+.2%}" for v in adj],
            }, index=assets)
            st.dataframe(df_bl, use_container_width=True)

            # ── Portfolio metrics with BL returns ────────────────────────────
            st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Métricas de Cartera con Retornos BL</div>',
                        unsafe_allow_html=True)

            # Build a new optimizer using BL returns but same cov/weights
            opt_bl = CPortfolio_optimization(
                port_wts          = weights_eq,
                correlation_matrix = corr,
                expected_ret      = bl_post,
                vol               = vols,
            )
            opt_eq = CPortfolio_optimization(
                port_wts          = weights_eq,
                correlation_matrix = corr,
                expected_ret      = pi_eq,
                vol               = vols,
            )
            m_bl = opt_bl.get_portfolio_metrics(weights_eq, risk_free_rate=rf_bl)
            m_eq = opt_eq.get_portfolio_metrics(weights_eq, risk_free_rate=rf_bl)

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

            # ── Views summary ────────────────────────────────────────────────
            st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Resumen de Views</div>',
                        unsafe_allow_html=True)
            rows = []
            for v in range(int(n_views)):
                long_a  = [assets[i] for i, val in enumerate(p_rows[v]) if val >  0]
                short_a = [assets[i] for i, val in enumerate(p_rows[v]) if val < 0]
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
        st.markdown('<div class="info-box">👆 Define tus views y pulsa '
                    '<b>Calcular Black-Litterman</b>.</div>', unsafe_allow_html=True)
