"""
pages/1_Configuracion.py
========================
Módulo de configuración: universo de activos y parámetros de mercado.

Writes to session state:
    asset_classes, eq_returns, volatilities, corr_matrix,
    sim_models, sim_model_params
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

from utils.state import init_state, reset_downstream
from utils.ui    import apply_css, page_header, section, divider, status_badge, info

init_state()
apply_css()

page_header("⚙️ Configuración",
            "Definición del universo de inversión y parámetros de mercado")

# ── 1 · Asset classes ─────────────────────────────────────────────────────────
section("1 · Clases de Activo")
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
            reset_downstream("asset_classes")
            st.success(f"✅  {len(parsed)} clases de activo guardadas.")

with col_r:
    info("Las clases de activo definidas aquí se usarán en todos los módulos.")
    for i, a in enumerate(st.session_state["asset_classes"], 1):
        st.markdown(f"&nbsp;&nbsp;`{i}.` {a}")

divider()

# ── 2 · Market data ───────────────────────────────────────────────────────────
assets = st.session_state["asset_classes"]
n      = len(assets)

section("2 · Datos de Mercado")
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
            reset_downstream("eq_returns")
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
            reset_downstream("volatilities")
            st.success("✅  Volatilidades guardadas.")

    with tab_corr:
        st.caption("Matriz de correlaciones (triangular superior; diagonal = 1)")
        prev_c = st.session_state["corr_matrix"] \
                 if st.session_state["corr_matrix"] is not None else np.eye(n)
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
                reset_downstream("corr_matrix")
                st.success("✅  Correlaciones guardadas.")

else:  # Upload
    info("Sube un <b>.xlsx</b> con tres hojas: <code>eq_returns</code>, "
         "<code>volatilities</code>, <code>correlation</code>. "
         "La primera columna debe contener los nombres de los activos.")
    uploaded = st.file_uploader("Subir Excel", type=["xlsx"])
    if uploaded:
        try:
            xls = pd.ExcelFile(uploaded)
            df_r = pd.read_excel(xls, sheet_name="eq_returns",   index_col=0)
            df_v = pd.read_excel(xls, sheet_name="volatilities",  index_col=0)
            df_c = pd.read_excel(xls, sheet_name="correlation",   index_col=0)
            st.session_state["eq_returns"]  = df_r.values.flatten() / 100
            st.session_state["volatilities"] = df_v.values.flatten() / 100
            st.session_state["corr_matrix"]  = df_c.values
            reset_downstream("eq_returns")
            st.success("✅  Datos cargados correctamente.")
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

# ── Status ────────────────────────────────────────────────────────────────────
divider()
section("Estado de Configuración")
c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(status_badge(bool(assets),                                     "Clases de Activo"), unsafe_allow_html=True)
c2.markdown(status_badge(st.session_state["eq_returns"]   is not None,    "Rentabilidades"),   unsafe_allow_html=True)
c3.markdown(status_badge(st.session_state["volatilities"] is not None,    "Volatilidades"),    unsafe_allow_html=True)
c4.markdown(status_badge(st.session_state["corr_matrix"]  is not None,    "Correlaciones"),    unsafe_allow_html=True)
c5.markdown(status_badge(st.session_state["sim_models"]   is not None,    "Modelos MC"),       unsafe_allow_html=True)

_mkt_ok = (st.session_state["eq_returns"] is not None
           and st.session_state["volatilities"] is not None
           and st.session_state["corr_matrix"] is not None)
if _mkt_ok:
    st.markdown("**Siguiente paso →**")
    st.page_link("pages/2_Mi_Cartera.py", label="Ir a Mi Cartera", icon="📋")

# ── 3 · Simulation models ─────────────────────────────────────────────────────
divider()
section("3 · Modelos de Simulación")
info(
    "Selecciona el <b>proceso estocástico</b> que seguirá cada clase de activo "
    "en la simulación Monte Carlo de <i>Riesgo &amp; Retorno</i>.<br>"
    "<b>GBM</b> — Movimiento Browniano Geométrico (rentabilidades lognormales; "
    "adecuado para renta variable).<br>"
    "<b>Vasicek</b> — Reversión a la media (tipos de interés, spreads de crédito; "
    "requiere velocidad de reversión κ).<br>"
    "<b>Normal</b> — Distribución normal simple (por defecto para cualquier activo)."
)

_market_ready = (
    st.session_state["eq_returns"] is not None
    and st.session_state["volatilities"] is not None
)
if not _market_ready:
    st.warning(
        "⚠️  Configura primero las Rentabilidades y Volatilidades (Sección 2) "
        "para poder definir los modelos de simulación.")
else:
    _assets    = st.session_state["asset_classes"]
    _eq_rets   = st.session_state["eq_returns"]
    _vols      = st.session_state["volatilities"]
    _prev_models = st.session_state.get("sim_models") or {}
    _prev_params  = st.session_state.get("sim_model_params") or {}

    MODEL_OPTIONS = {"GBM (Browniano Geométrico)": "gbm",
                     "Vasicek (reversión media)":  "vasicek",
                     "Normal":                     "normal"}
    MODEL_LABELS  = {v: k for k, v in MODEL_OPTIONS.items()}

    new_models = {}
    new_params = {}

    for i, ac in enumerate(_assets):
        st.markdown(f"**{ac}**")
        cur_model  = _prev_models.get(ac, "normal")
        cur_params = _prev_params.get(ac, {})
        cur_kappa  = cur_params.get("kappa", 1.0)

        col_mod, col_kap = st.columns([2, 1])
        chosen_label = col_mod.selectbox(
            "Modelo",
            list(MODEL_OPTIONS.keys()),
            index=list(MODEL_OPTIONS.values()).index(cur_model),
            key=f"sim_model_{i}",
            label_visibility="collapsed",
        )
        chosen_model = MODEL_OPTIONS[chosen_label]

        # κ input — only meaningful for Vasicek
        kappa_val = cur_kappa
        if chosen_model == "vasicek":
            kappa_val = col_kap.number_input(
                "κ (velocidad reversión)",
                value=float(cur_kappa),
                min_value=0.01, max_value=20.0,
                step=0.1, format="%.2f",
                key=f"sim_kappa_{i}",
                help="Velocidad de reversión a la media. "
                     "Valores típicos: 0.5–5 para tipos de interés.",
            )
        else:
            col_kap.markdown("&nbsp;", unsafe_allow_html=True)

        new_models[ac] = chosen_model
        new_params[ac] = {
            "mu":    float(_eq_rets[i]),   # auto-filled from market config
            "sigma": float(_vols[i]),      # auto-filled from market config
            "kappa": float(kappa_val),
            "theta": float(_eq_rets[i]),   # long-run mean = equilibrium return
        }

    st.markdown("")
    if st.button("💾  Guardar modelos de simulación", type="primary"):
        st.session_state["sim_models"]       = new_models
        st.session_state["sim_model_params"] = new_params
        reset_downstream("sim_models")
        st.success("✅  Modelos de simulación guardados.")

    # Quick summary badge
    if st.session_state.get("sim_models"):
        cols_s = st.columns(min(len(_assets), 5))
        for i, ac in enumerate(_assets):
            m = st.session_state["sim_models"].get(ac, "normal")
            colour = {"gbm": "🟢", "vasicek": "🟡", "normal": "🔵"}.get(m, "⚪")
            cols_s[i % 5].markdown(f"{colour} **{ac[:15]}**  \n`{m}`")

# ── Template download ─────────────────────────────────────────────────────────
divider()
with st.expander("📥 Descargar plantilla Excel"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame({"Rentabilidad (%)": [6.0] * n}, index=assets).to_excel(w, sheet_name="eq_returns")
        pd.DataFrame({"Volatilidad (%)":  [15.0] * n}, index=assets).to_excel(w, sheet_name="volatilities")
        pd.DataFrame(np.eye(n), index=assets, columns=assets).to_excel(w, sheet_name="correlation")
    st.download_button(
        "⬇️ Descargar plantilla", buf.getvalue(), "plantilla_parametros.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
