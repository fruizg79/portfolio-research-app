"""
pages/1_Configuracion.py
========================
Módulo de configuración: universo de activos y parámetros de mercado.

Writes to session state:
    asset_classes, eq_returns, volatilities, corr_matrix
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
c1, c2, c3, c4 = st.columns(4)
c1.markdown(status_badge(bool(assets),                                 "Clases de Activo"), unsafe_allow_html=True)
c2.markdown(status_badge(st.session_state["eq_returns"]  is not None, "Rentabilidades"),   unsafe_allow_html=True)
c3.markdown(status_badge(st.session_state["volatilities"] is not None, "Volatilidades"),   unsafe_allow_html=True)
c4.markdown(status_badge(st.session_state["corr_matrix"]  is not None, "Correlaciones"),   unsafe_allow_html=True)

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
