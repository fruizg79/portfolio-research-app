"""
pages/6_Escenarios.py
=====================
Gestión de escenarios de mercado, carteras y snapshots de resultados en BD.

Permite:
  - Guardar el escenario y cartera activos en BD
  - Cargar un escenario/cartera guardado a session state
  - Consultar el historial de resultados calculados
  - Borrar escenarios y carteras

Reads from session state:
    eq_returns, volatilities, corr_matrix, asset_classes,
    sim_models, sim_model_params,
    portfolio_weights, tactical_ranges,
    portfolio_metrics, bl_eq_returns, bl_post_returns,
    active_scenario_id, active_scenario_name,
    active_portfolio_id, active_portfolio_name

Writes to session state:
    eq_returns, volatilities, corr_matrix, asset_classes,
    sim_models, sim_model_params,
    portfolio_weights, tactical_ranges,
    active_scenario_id, active_scenario_name,
    active_portfolio_id, active_portfolio_name
"""

import pandas as pd
import streamlit as st

from utils.state import (init_state, reset_downstream,
                         load_scenario_to_state, load_portfolio_to_state)
from utils.ui    import (apply_css, page_header, section, divider,
                         kpi_card, info, warn, status_badge)
from utils.database import (
    # escenarios
    save_scenario, update_scenario, get_scenarios, load_scenario, delete_scenario,
    # carteras
    save_portfolio, update_portfolio, get_portfolios, load_portfolio, delete_portfolio,
    # snapshots
    save_snapshot, get_snapshots,
)

init_state()
apply_css()

page_header("🗄️ Escenarios & Carteras",
            "Guardar, cargar y gestionar escenarios de mercado y carteras")

# ── Comprobación de conexión + carga única de escenarios ─────────────────────
# scenarios is fetched ONCE here and reused throughout the page to avoid
# redundant round-trips (previously called 3 times per render).
try:
    scenarios = get_scenarios()
except Exception as e:
    st.error(f"❌ No se puede conectar a la base de datos: {e}")
    st.stop()

# ── Estado activo en sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("**En memoria ahora:**")
    sc_name = st.session_state.get("active_scenario_name") or "*(sin guardar)*"
    pt_name = st.session_state.get("active_portfolio_name") or "*(sin guardar)*"
    st.markdown(f"📊 Escenario: `{sc_name}`")
    st.markdown(f"📋 Cartera: `{pt_name}`")

# ══════════════════════════════════════════════════════════════════════════════
tab_sc, tab_pt, tab_hist = st.tabs(
    ["📊  Escenarios", "📋  Carteras", "📜  Historial de Resultados"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ESCENARIOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_sc:

    # ── Guardar escenario actual ──────────────────────────────────────────────
    section("Guardar escenario actual")

    market_ok = all(
        st.session_state.get(k) is not None
        for k in ("eq_returns", "volatilities", "corr_matrix"))

    if not market_ok:
        warn("No hay datos de mercado en memoria. "
             "Ve a <b>Configuración</b> primero.")
    else:
        col_sn, col_sd = st.columns([2, 3])
        sc_name_input = col_sn.text_input(
            "Nombre del escenario",
            value=st.session_state.get("active_scenario_name") or "",
            placeholder="Base Q1 2026")
        sc_desc_input = col_sd.text_input(
            "Descripción (opcional)",
            placeholder="Escenario central de consenso")

        col_b1, col_b2 = st.columns(2)

        # Guardar como nuevo
        if col_b1.button("💾  Guardar como nuevo", use_container_width=True):
            if not sc_name_input.strip():
                st.error("Introduce un nombre para el escenario.")
            else:
                try:
                    rec = save_scenario(
                        name             = sc_name_input.strip(),
                        asset_classes    = st.session_state["asset_classes"],
                        eq_returns       = st.session_state["eq_returns"],
                        volatilities     = st.session_state["volatilities"],
                        corr_matrix      = st.session_state["corr_matrix"],
                        description      = sc_desc_input.strip(),
                        sim_models       = st.session_state.get("sim_models"),
                        sim_model_params = st.session_state.get("sim_model_params"),
                    )
                    st.session_state["active_scenario_id"]   = rec["id"]
                    st.session_state["active_scenario_name"] = rec["name"]
                    st.success(f"✅  Escenario **{rec['name']}** guardado (ID {rec['id']}).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al guardar: {e}")

        # Actualizar existente
        sc_id = st.session_state.get("active_scenario_id")
        if col_b2.button("🔄  Actualizar actual", disabled=sc_id is None,
                         use_container_width=True,
                         help="Solo disponible si hay un escenario cargado desde BD"):
            try:
                update_scenario(
                    scenario_id      = sc_id,
                    name             = sc_name_input.strip() or st.session_state["active_scenario_name"],
                    asset_classes    = st.session_state["asset_classes"],
                    eq_returns       = st.session_state["eq_returns"],
                    volatilities     = st.session_state["volatilities"],
                    corr_matrix      = st.session_state["corr_matrix"],
                    description      = sc_desc_input.strip(),
                    sim_models       = st.session_state.get("sim_models"),
                    sim_model_params = st.session_state.get("sim_model_params"),
                )
                st.success(f"✅  Escenario **{st.session_state['active_scenario_name']}** actualizado.")
            except Exception as e:
                st.error(f"Error al actualizar: {e}")

    # ── Lista y carga de escenarios ───────────────────────────────────────────
    divider()
    section("Escenarios guardados")

    if not scenarios:
        info("No hay escenarios guardados todavía.")
    else:
        for sc in scenarios:
            with st.expander(
                f"**{sc['name']}**  ·  "
                f"{pd.to_datetime(sc['created_at']).strftime('%d/%m/%Y %H:%M')}"
                + (f"  ·  _{sc['description']}_" if sc.get("description") else "")
            ):
                col_l, col_d = st.columns([3, 1])

                with col_l:
                    hist_str = ""
                    if sc.get("history_from") and sc.get("history_to"):
                        hist_str = (f"Periodo de referencia: "
                                    f"{sc['history_from']} → {sc['history_to']}")
                    if hist_str:
                        st.caption(hist_str)

                    is_active = st.session_state.get("active_scenario_id") == sc["id"]
                    st.markdown(
                        status_badge(is_active, "Cargado en memoria"),
                        unsafe_allow_html=True)

                with col_d:
                    if st.button("⬇️  Cargar", key=f"load_sc_{sc['id']}",
                                 use_container_width=True):
                        try:
                            data = load_scenario(sc["id"])
                            load_scenario_to_state(data)
                            st.success(f"✅  Escenario **{data['name']}** cargado.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al cargar: {e}")

                    if st.button("🗑️  Borrar", key=f"del_sc_{sc['id']}",
                                 use_container_width=True,
                                 type="primary"):
                        try:
                            delete_scenario(sc["id"])
                            if st.session_state.get("active_scenario_id") == sc["id"]:
                                st.session_state["active_scenario_id"]   = None
                                st.session_state["active_scenario_name"] = None
                            st.success("✅  Escenario borrado.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al borrar (¿tiene carteras asociadas?): {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CARTERAS
# ══════════════════════════════════════════════════════════════════════════════
with tab_pt:

    # ── Guardar cartera actual ────────────────────────────────────────────────
    section("Guardar cartera actual")

    portfolio_ok = all(
        st.session_state.get(k) is not None
        for k in ("portfolio_weights", "tactical_ranges"))
    sc_id = st.session_state.get("active_scenario_id")

    if not portfolio_ok:
        warn("No hay cartera en memoria. Ve a <b>Mi Cartera</b> primero.")
    elif sc_id is None:
        warn("La cartera necesita estar vinculada a un escenario guardado. "
             "Carga o guarda un escenario en la pestaña <b>Escenarios</b> primero.")
    else:
        col_pn, col_pd = st.columns([2, 3])
        pt_name_input = col_pn.text_input(
            "Nombre de la cartera",
            value=st.session_state.get("active_portfolio_name") or "",
            placeholder="Cartera Conservadora")
        pt_desc_input = col_pd.text_input(
            "Descripción (opcional)",
            placeholder="Perfil moderado, 5 activos")

        st.caption(f"Se vinculará al escenario: "
                   f"**{st.session_state.get('active_scenario_name', '—')}**")

        col_pb1, col_pb2 = st.columns(2)

        if col_pb1.button("💾  Guardar como nueva", use_container_width=True):
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
                    st.success(f"✅  Cartera **{rec['name']}** guardada (ID {rec['id']}).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al guardar: {e}")

        pt_id = st.session_state.get("active_portfolio_id")
        if col_pb2.button("🔄  Actualizar actual", disabled=pt_id is None,
                          use_container_width=True):
            try:
                update_portfolio(
                    portfolio_id    = pt_id,
                    name            = pt_name_input.strip() or st.session_state["active_portfolio_name"],
                    weights         = st.session_state["portfolio_weights"],
                    tactical_ranges = st.session_state["tactical_ranges"],
                    description     = pt_desc_input.strip(),
                )
                st.success(f"✅  Cartera **{st.session_state['active_portfolio_name']}** actualizada.")
            except Exception as e:
                st.error(f"Error al actualizar: {e}")

    # ── Lista y carga de carteras ─────────────────────────────────────────────
    divider()
    section("Carteras guardadas")

    # Filtro por escenario
    filter_sc = st.checkbox("Mostrar solo carteras del escenario activo",
                             value=False, disabled=sc_id is None)
    portfolios = get_portfolios(scenario_id=sc_id if filter_sc else None)

    if not portfolios:
        info("No hay carteras guardadas todavía.")
    else:
        # Mapear scenario_id → nombre para mostrarlo (reuse scenarios fetched at top)
        sc_names = {s["id"]: s["name"] for s in scenarios}

        for pt in portfolios:
            sc_label = sc_names.get(pt.get("scenario_id"), "—")
            with st.expander(
                f"**{pt['name']}**  ·  "
                f"{pd.to_datetime(pt['created_at']).strftime('%d/%m/%Y %H:%M')}"
                f"  ·  escenario: _{sc_label}_"
            ):
                col_pl, col_pd2 = st.columns([3, 1])
                with col_pl:
                    if pt.get("description"):
                        st.caption(pt["description"])
                    is_active_pt = st.session_state.get("active_portfolio_id") == pt["id"]
                    st.markdown(
                        status_badge(is_active_pt, "Cargada en memoria"),
                        unsafe_allow_html=True)

                with col_pd2:
                    if st.button("⬇️  Cargar", key=f"load_pt_{pt['id']}",
                                 use_container_width=True):
                        try:
                            data = load_portfolio(pt["id"])
                            load_portfolio_to_state(data)
                            st.success(f"✅  Cartera **{data['name']}** cargada.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al cargar: {e}")

                    if st.button("🗑️  Borrar", key=f"del_pt_{pt['id']}",
                                 use_container_width=True, type="primary"):
                        try:
                            delete_portfolio(pt["id"])
                            if st.session_state.get("active_portfolio_id") == pt["id"]:
                                st.session_state["active_portfolio_id"]   = None
                                st.session_state["active_portfolio_name"] = None
                            st.success("✅  Cartera borrada.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al borrar: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORIAL DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    section("Guardar resultados actuales")

    metrics   = st.session_state.get("portfolio_metrics")
    pt_id     = st.session_state.get("active_portfolio_id")
    sc_id_cur = st.session_state.get("active_scenario_id")

    if not metrics:
        warn("No hay métricas calculadas. Ve a <b>Riesgo & Retorno</b> primero.")
    elif pt_id is None or sc_id_cur is None:
        warn("Necesitas tener una cartera y un escenario guardados en BD "
             "antes de registrar un snapshot.")
    else:
        bl_available = (st.session_state.get("bl_post_returns") is not None)

        include_bl = False
        if bl_available:
            include_bl = st.checkbox(
                "Incluir resultados Black-Litterman en el snapshot", value=True)

        if st.button("💾  Guardar snapshot de resultados", type="primary"):
            bl_payload = None
            if include_bl and bl_available:
                bl_payload = {
                    "tau":          st.session_state.get("_bl_tau"),
                    "lambda_param": st.session_state.get("_bl_lambda"),
                    "views":        st.session_state.get("_bl_views"),
                    "eq_returns":   st.session_state["bl_eq_returns"],
                    "post_returns": st.session_state["bl_post_returns"],
                }
            try:
                rec = save_snapshot(
                    portfolio_id   = pt_id,
                    scenario_id    = sc_id_cur,
                    metrics        = metrics,
                    risk_free_rate = st.session_state["risk_free_rate"],
                    n_sims         = st.session_state["n_sims"],
                    bl_data        = bl_payload,
                )
                st.success(f"✅  Snapshot guardado (ID {rec['id']}).")
            except Exception as e:
                st.error(f"Error al guardar: {e}")

    # ── Historial ─────────────────────────────────────────────────────────────
    divider()
    section("Historial de snapshots")

    portfolios_all = get_portfolios()
    if not portfolios_all:
        info("No hay carteras guardadas.")
    else:
        hist_pt_opts = {p["name"]: p["id"] for p in portfolios_all}
        hist_pt_sel  = st.selectbox("Cartera", list(hist_pt_opts.keys()))
        snaps        = get_snapshots(hist_pt_opts[hist_pt_sel], limit=20)

        if not snaps:
            info("No hay snapshots para esta cartera.")
        else:
            df_snaps = pd.DataFrame(snaps)
            df_snaps["created_at"] = pd.to_datetime(
                df_snaps["created_at"]).dt.strftime("%d/%m/%Y %H:%M")

            # Formatear columnas numéricas
            for col in ["expected_return", "volatility", "prob_loss", "var_95", "es_95"]:
                df_snaps[col] = df_snaps[col].map(
                    lambda x: f"{x:.2%}" if x is not None else "—")
            df_snaps["sharpe"] = df_snaps["sharpe"].map(
                lambda x: f"{x:.2f}" if x is not None else "—")

            df_snaps = df_snaps.rename(columns={
                "created_at":      "Fecha",
                "expected_return": "E[R]",
                "volatility":      "Vol",
                "sharpe":          "Sharpe",
                "prob_loss":       "P(pérd.)",
                "var_95":          "VaR 95%",
                "es_95":           "ES 95%",
            }).drop(columns=["id", "portfolio_id"], errors="ignore")

            st.dataframe(df_snaps.set_index("Fecha"), use_container_width=True)

            # Gráfico evolución Sharpe
            if len(snaps) > 1:
                import plotly.graph_objects as go
                df_chart = pd.DataFrame(snaps)
                df_chart["created_at"] = pd.to_datetime(df_chart["created_at"])
                df_chart = df_chart.sort_values("created_at")

                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(
                    x=df_chart["created_at"], y=df_chart["sharpe"],
                    mode="lines+markers", name="Sharpe",
                    line=dict(color="#1e3a5f", width=2),
                    marker=dict(size=6)
                ))
                fig_h.add_trace(go.Scatter(
                    x=df_chart["created_at"],
                    y=df_chart["expected_return"].apply(lambda x: x * 100 if x else None),
                    mode="lines+markers", name="E[R] (%)",
                    line=dict(color="#c9a84c", width=2),
                    marker=dict(size=6),
                    yaxis="y2"
                ))
                fig_h.update_layout(
                    height=300,
                    paper_bgcolor="white", plot_bgcolor="#fafaf8",
                    margin=dict(t=20, b=40, l=40, r=60),
                    legend=dict(font=dict(size=11)),
                    yaxis=dict(title="Sharpe"),
                    yaxis2=dict(title="E[R] (%)", overlaying="y", side="right"),
                )
                st.plotly_chart(fig_h, use_container_width=True)
