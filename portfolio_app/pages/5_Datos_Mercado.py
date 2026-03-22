"""
pages/5_Datos_Mercado.py
========================
Gestión de datos históricos: precios de activos y series macroeconómicas.
El equipo sube ficheros Excel o CSV; la página los valida y los guarda en BD.

Reads from session state:  —
Writes to session state:   —  (datos van directamente a BD)
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.state import init_state
from utils.ui    import apply_css, page_header, section, divider, info, warn
from utils.database import (
    get_assets, upsert_asset, deactivate_asset, save_prices, get_prices,
    get_macro_series, upsert_macro_series, save_macro_data, get_macro_data,
)

init_state()
apply_css()

page_header("📂 Datos de Mercado",
            "Carga de precios históricos y series macroeconómicas")

# ── Comprobación de conexión ──────────────────────────────────────────────────
try:
    _ = get_assets()
    _db_ok = True
except Exception as e:
    st.error(f"❌ No se puede conectar a la base de datos: {e}\n\n"
             "Comprueba que `SUPABASE_URL` y `SUPABASE_KEY` están configurados "
             "en Streamlit Secrets.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRECIOS DE ACTIVOS
# TAB 2 — DATOS MACRO
# TAB 3 — GESTIÓN DE ACTIVOS
# ══════════════════════════════════════════════════════════════════════════════
tab_prices, tab_macro, tab_assets = st.tabs(
    ["📈  Precios Históricos", "🌍  Datos Macro", "⚙️  Gestión de Activos"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRECIOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_prices:
    section("Subir precios históricos")
    info(
        "Sube un <b>Excel (.xlsx)</b> o <b>CSV (.csv)</b> con las columnas "
        "<code>date</code> y <code>price</code>.<br>"
        "Formatos de fecha aceptados: <code>YYYY-MM-DD</code>, "
        "<code>DD/MM/YYYY</code>, <code>MM/DD/YYYY</code>."
    )

    assets = get_assets()
    if not assets:
        warn("No hay activos registrados. Crea al menos uno en la pestaña "
             "<b>Gestión de Activos</b> antes de subir precios.")
    else:
        col_sel, col_up = st.columns([1, 2])

        with col_sel:
            asset_options = {a["name"]: a["id"] for a in assets}
            selected_asset = st.selectbox(
                "Activo", list(asset_options.keys()), key="price_asset_sel")
            asset_id = asset_options[selected_asset]

        with col_up:
            uploaded = st.file_uploader(
                "Fichero de precios", type=["xlsx", "csv"], key="price_upload")

        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded)

                # Normalizar nombres de columna
                df_raw.columns = [c.strip().lower() for c in df_raw.columns]

                if "date" not in df_raw.columns or "price" not in df_raw.columns:
                    st.error("El fichero debe tener columnas 'date' y 'price'.")
                else:
                    df_raw["date"]  = pd.to_datetime(df_raw["date"], dayfirst=True)
                    df_raw["price"] = pd.to_numeric(df_raw["price"], errors="coerce")
                    df_clean = df_raw[["date", "price"]].dropna()

                    # Preview
                    st.markdown(f"**{len(df_clean)} filas válidas** "
                                f"({df_clean['date'].min().date()} → "
                                f"{df_clean['date'].max().date()})")

                    fig = go.Figure(go.Scatter(
                        x=df_clean["date"], y=df_clean["price"],
                        mode="lines", line=dict(color="#1e3a5f", width=1.5)
                    ))
                    fig.update_layout(
                        height=250, margin=dict(t=10, b=30, l=40, r=10),
                        paper_bgcolor="white", plot_bgcolor="#fafaf8",
                        yaxis_title="Precio",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    if st.button("💾  Guardar precios en BD", type="primary"):
                        n = save_prices(asset_id, df_clean)
                        st.success(f"✅  {n} registros guardados para **{selected_asset}**.")

            except Exception as e:
                st.error(f"Error al leer el fichero: {e}")

    # ── Explorar precios existentes ───────────────────────────────────────────
    divider()
    section("Explorar precios guardados")

    if assets:
        col_exp1, col_exp2, col_exp3 = st.columns([2, 1, 1])
        with col_exp1:
            exp_asset = st.selectbox(
                "Activo", list(asset_options.keys()), key="price_explore_sel")
        with col_exp2:
            date_from = st.date_input("Desde", value=None, key="price_from")
        with col_exp3:
            date_to = st.date_input("Hasta", value=None, key="price_to")

        df_stored = get_prices(
            asset_options[exp_asset], date_from or None, date_to or None)

        if df_stored.empty:
            info("No hay precios guardados para este activo en el periodo seleccionado.")
        else:
            st.markdown(f"**{len(df_stored)} registros** · "
                        f"{df_stored['date'].min().date()} → "
                        f"{df_stored['date'].max().date()}")

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_stored["date"], y=df_stored["price"],
                name="Precio", mode="lines",
                line=dict(color="#1e3a5f", width=1.5)
            ))
            fig2.update_layout(
                height=280, margin=dict(t=10, b=30, l=40, r=10),
                paper_bgcolor="white", plot_bgcolor="#fafaf8",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Descargar
            csv_buf = io.StringIO()
            df_stored.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️  Descargar CSV", csv_buf.getvalue(),
                f"{exp_asset}_prices.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MACRO
# ══════════════════════════════════════════════════════════════════════════════
with tab_macro:
    section("Subir datos macroeconómicos")
    info(
        "Sube un <b>Excel</b> o <b>CSV</b> con columnas <code>date</code> "
        "y <code>value</code>.<br>"
        "Ejemplos de series: PIB, IPC, VIX, tipos de interés, spreads de crédito."
    )

    series_list = get_macro_series()
    series_options = {s["series_id"]: s for s in series_list}

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        macro_mode = st.radio(
            "Serie", ["Usar existente", "Crear nueva"], horizontal=True)

    if macro_mode == "Crear nueva":
        col_n1, col_n2, col_n3, col_n4 = st.columns(4)
        new_sid   = col_n1.text_input("ID (sin espacios)", placeholder="CPI_EU")
        new_sname = col_n2.text_input("Nombre", placeholder="IPC Eurozona")
        new_freq  = col_n3.selectbox("Frecuencia", ["monthly", "quarterly", "daily", "annual"])
        new_unit  = col_n4.text_input("Unidad", placeholder="%")

        if st.button("Registrar serie"):
            if not new_sid or not new_sname:
                st.error("ID y Nombre son obligatorios.")
            else:
                upsert_macro_series(new_sid, new_sname, new_freq, new_unit)
                st.success(f"✅  Serie **{new_sid}** registrada.")
                st.rerun()
        selected_series_id = new_sid
    else:
        if not series_list:
            warn("No hay series registradas. Crea una nueva primero.")
            selected_series_id = None
        else:
            selected_series_id = st.selectbox(
                "Serie existente",
                list(series_options.keys()),
                format_func=lambda x: f"{x} — {series_options[x]['name']}")

    # Upload de datos
    if selected_series_id:
        macro_file = st.file_uploader(
            "Fichero de datos", type=["xlsx", "csv"], key="macro_upload")

        if macro_file:
            try:
                if macro_file.name.endswith(".csv"):
                    df_macro = pd.read_csv(macro_file)
                else:
                    df_macro = pd.read_excel(macro_file)

                df_macro.columns = [c.strip().lower() for c in df_macro.columns]

                if "date" not in df_macro.columns or "value" not in df_macro.columns:
                    st.error("El fichero debe tener columnas 'date' y 'value'.")
                else:
                    df_macro["date"]  = pd.to_datetime(df_macro["date"], dayfirst=True)
                    df_macro["value"] = pd.to_numeric(df_macro["value"], errors="coerce")
                    df_macro = df_macro[["date", "value"]].dropna()

                    st.markdown(f"**{len(df_macro)} filas** · "
                                f"{df_macro['date'].min().date()} → "
                                f"{df_macro['date'].max().date()}")

                    fig_m = go.Figure(go.Scatter(
                        x=df_macro["date"], y=df_macro["value"],
                        mode="lines", line=dict(color="#c9a84c", width=1.5)
                    ))
                    fig_m.update_layout(
                        height=220, margin=dict(t=10, b=30, l=40, r=10),
                        paper_bgcolor="white", plot_bgcolor="#fafaf8",
                    )
                    st.plotly_chart(fig_m, use_container_width=True)

                    if st.button("💾  Guardar datos macro en BD", type="primary"):
                        n = save_macro_data(selected_series_id, df_macro)
                        st.success(f"✅  {n} registros guardados para **{selected_series_id}**.")

            except Exception as e:
                st.error(f"Error al leer el fichero: {e}")

    # ── Explorar series ───────────────────────────────────────────────────────
    if series_list:
        divider()
        section("Explorar series guardadas")
        col_e1, col_e2, col_e3 = st.columns([2, 1, 1])
        with col_e1:
            exp_series = col_e1.selectbox(
                "Serie", list(series_options.keys()),
                format_func=lambda x: f"{x} — {series_options[x]['name']}",
                key="macro_explore")
        with col_e2:
            m_from = st.date_input("Desde", value=None, key="macro_from")
        with col_e3:
            m_to   = st.date_input("Hasta", value=None, key="macro_to")

        df_m_stored = get_macro_data(exp_series, m_from or None, m_to or None)

        if df_m_stored.empty:
            info("No hay datos para esta serie en el periodo seleccionado.")
        else:
            fig_ms = go.Figure(go.Scatter(
                x=df_m_stored["date"], y=df_m_stored["value"],
                mode="lines+markers",
                marker=dict(size=3),
                line=dict(color="#c9a84c", width=1.5)
            ))
            unit = series_options[exp_series].get("unit", "")
            fig_ms.update_layout(
                height=260, margin=dict(t=10, b=30, l=50, r=10),
                paper_bgcolor="white", plot_bgcolor="#fafaf8",
                yaxis_title=unit,
            )
            st.plotly_chart(fig_ms, use_container_width=True)

            csv_m = io.StringIO()
            df_m_stored.to_csv(csv_m, index=False)
            st.download_button(
                "⬇️  Descargar CSV", csv_m.getvalue(),
                f"{exp_series}.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GESTIÓN DE ACTIVOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_assets:
    section("Activos registrados")

    assets_all = get_assets(active_only=False)

    if assets_all:
        df_assets = pd.DataFrame(assets_all)[
            ["id", "name", "asset_class", "currency", "active", "created_at"]]
        df_assets["created_at"] = pd.to_datetime(
            df_assets["created_at"]).dt.strftime("%Y-%m-%d")
        st.dataframe(df_assets.set_index("id"), use_container_width=True)
    else:
        info("No hay activos registrados todavía.")

    divider()
    section("Añadir activo")
    col_a1, col_a2, col_a3 = st.columns(3)
    new_name  = col_a1.text_input("Nombre", placeholder="Renta Variable DM")
    new_class = col_a2.selectbox(
        "Clase de activo",
        ["equity", "fixed_income", "alternatives", "real_estate",
         "commodities", "cash", "other"])
    new_curr  = col_a3.selectbox("Divisa", ["EUR", "USD", "GBP", "JPY", "CHF"])

    if st.button("➕  Añadir activo"):
        if not new_name.strip():
            st.error("El nombre es obligatorio.")
        else:
            try:
                upsert_asset(new_name.strip(), new_class, new_curr)
                st.success(f"✅  Activo **{new_name}** registrado.")
                st.rerun()
            except Exception as e:
                st.error(f"Error al guardar: {e}")

    divider()
    section("Desactivar activo")
    active_assets = [a for a in assets_all if a["active"]]
    if active_assets:
        col_d1, col_d2 = st.columns([3, 1])
        deact_opts = {a["name"]: a["id"] for a in active_assets}
        to_deact = col_d1.selectbox("Activo a desactivar", list(deact_opts.keys()))
        if col_d2.button("Desactivar", type="primary"):
            deactivate_asset(deact_opts[to_deact])
            st.success(f"✅  **{to_deact}** desactivado.")
            st.rerun()
    else:
        info("No hay activos activos para desactivar.")
