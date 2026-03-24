"""
app.py  —  Entry point
======================
Keep this file SHORT. Its only jobs are:
    1. st.set_page_config  (must be first Streamlit call)
    2. Initialise session state
    3. Inject shared CSS
    4. Render the sidebar header and the home/welcome screen

All business logic and page content live in pages/*.py
"""

import logging
import streamlit as st

logger = logging.getLogger(__name__)

# ── Must be the very first Streamlit call ─────────────────────────────────────
st.set_page_config(
    page_title="Asset Allocation · Portfolio Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Check core dependency ─────────────────────────────────────────────────────
try:
    import asset_allocation  # noqa: F401
except ImportError:
    st.error(
        "❌ No se encontró `asset_allocation.py`.\n\n"
        "Asegúrate de que el fichero está en la raíz del proyecto "
        "(junto a este `app.py`)."
    )
    st.stop()

# ── Session state & CSS ───────────────────────────────────────────────────────
from utils.state import init_state
from utils.ui    import apply_css, page_header, section, divider, status_badge
from utils.database import get_scenarios, load_scenario, get_portfolios, load_portfolio

init_state()

def autoload_last_session():
    """
    Si session_state está vacío y hay datos en BD,
    carga automáticamente el escenario y cartera más recientes.
    Solo se ejecuta una vez por sesión (guarda un flag en state).
    """
    if st.session_state.get("_autoloaded"):
        return

    st.session_state["_autoloaded"] = True  # no volver a ejecutar

    try:
        # Cargar último escenario si no hay nada en memoria
        if st.session_state.get("eq_returns") is None:
            scenarios = get_scenarios()
            if scenarios:
                data = load_scenario(scenarios[0]["id"])  # el más reciente
                st.session_state["asset_classes"]        = data["asset_classes"]
                st.session_state["eq_returns"]           = data["eq_returns"]
                st.session_state["volatilities"]         = data["volatilities"]
                st.session_state["corr_matrix"]          = data["corr_matrix"]
                st.session_state["active_scenario_id"]   = data["id"]
                st.session_state["active_scenario_name"] = data["name"]

        # Cargar última cartera vinculada al escenario activo
        if st.session_state.get("portfolio_weights") is None:
            sc_id = st.session_state.get("active_scenario_id")
            portfolios = get_portfolios(scenario_id=sc_id)
            if portfolios:
                data = load_portfolio(portfolios[0]["id"])
                st.session_state["portfolio_weights"]     = data["weights"]
                st.session_state["tactical_ranges"]       = data["tactical_ranges"]
                st.session_state["active_portfolio_id"]   = data["id"]
                st.session_state["active_portfolio_name"] = data["name"]

    except Exception as e:
        # The app starts normally even without DB access.
        # Show a non-blocking warning so the user knows auto-load was skipped.
        logger.warning("Auto-load omitted: %s — %s", type(e).__name__, e, exc_info=True)
        st.sidebar.warning(
            f"⚠️ Auto-carga omitida ({type(e).__name__}). "
            "Puedes cargar un escenario manualmente desde la página Escenarios."
        )


autoload_last_session()


apply_css()

# ── Sidebar branding ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 0.5rem; text-align:center;'>
        <div style='font-family:DM Serif Display,serif; font-size:1.3rem; color:#f5f0e8;'>
            Portfolio Research
        </div>
        <div style='font-size:0.7rem; color:#8ca0bb; letter-spacing:0.1em;
                    text-transform:uppercase; margin-top:2px;'>
            Asset Allocation · v0.1
        </div>
    </div>
    <hr style='border-color:#1e3a5f; margin:0.8rem 0;'>
    <div style='font-size:0.72rem; color:#506070; padding:0 0.5rem; line-height:1.6;'>
        Navega por los módulos usando el menú de arriba.<br><br>
        <b style='color:#8ca0bb;'>Orden recomendado:</b><br>
        1 · Configuración<br>
        2 · Mi Cartera<br>
        3 · Riesgo & Retorno<br>
        4 · Black-Litterman
    </div>
    """, unsafe_allow_html=True)

# ── Home / welcome screen ─────────────────────────────────────────────────────
page_header("📊 Asset Allocation", "Portfolio Research · Herramienta de gestión de carteras")

from utils.state import is_market_configured, is_portfolio_configured

col1, col2, col3, col4 = st.columns(4)
col1.markdown(status_badge(bool(st.session_state["asset_classes"]),    "Clases de Activo"),  unsafe_allow_html=True)
col2.markdown(status_badge(is_market_configured(),                     "Datos de Mercado"),  unsafe_allow_html=True)
col3.markdown(status_badge(is_portfolio_configured(),                  "Mi Cartera"),        unsafe_allow_html=True)
col4.markdown(status_badge(st.session_state["portfolio_metrics"] is not None, "Métricas"),   unsafe_allow_html=True)

divider()
section("Módulos disponibles")

m1, m2, m3, m4 = st.columns(4)

def _module_card(col, icon, title, desc, page_hint):
    col.markdown(f"""
    <div style='background:white; border:1px solid #ddd6c8; border-radius:8px;
                padding:1.4rem; height:180px; box-shadow:0 1px 4px rgba(0,0,0,0.06);'>
        <div style='font-size:1.8rem; margin-bottom:0.5rem;'>{icon}</div>
        <div style='font-family:DM Serif Display,serif; font-size:1.05rem;
                    color:#0a1628; margin-bottom:0.4rem;'>{title}</div>
        <div style='font-size:0.82rem; color:#4a5568; line-height:1.5;'>{desc}</div>
        <div style='font-size:0.72rem; color:#8ca0bb; margin-top:0.6rem;
                    text-transform:uppercase; letter-spacing:0.05em;'>{page_hint}</div>
    </div>""", unsafe_allow_html=True)

_module_card(m1, "⚙️", "Configuración",
             "Define el universo de inversión y sube los parámetros de mercado.",
             "→ pages/1_Configuracion.py")
_module_card(m2, "📋", "Mi Cartera",
             "Introduce pesos actuales y rangos tácticos por clase de activo.",
             "→ pages/2_Mi_Cartera.py")
_module_card(m3, "📈", "Riesgo & Retorno",
             "Rentabilidad esperada, volatilidad, ES y probabilidad de pérdida.",
             "→ pages/3_Riesgo_Retorno.py")
_module_card(m4, "🔭", "Black-Litterman",
             "Incorpora views cualitativos y genera retornos bayesianos.",
             "→ pages/4_Black_Litterman.py")

divider()
st.caption("Selecciona un módulo en la barra lateral para comenzar.")
