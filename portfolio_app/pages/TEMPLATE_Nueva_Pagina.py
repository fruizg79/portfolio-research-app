"""
pages/TEMPLATE_Nueva_Pagina.py
==============================
PLANTILLA para añadir un nuevo módulo.

INSTRUCCIONES
-------------
1. Copia este fichero a pages/5_Nombre_Modulo.py
   (el número controla el orden en la barra lateral de Streamlit)

2. Rellena las secciones marcadas con TODO.

3. Si el módulo necesita guardar datos nuevos:
   a. Añade los campos en utils/state.py  (_DEFAULTS)
   b. Añade las dependencias en reset_downstream()
   c. Añade la etiqueta legible en utils/ui.py  (_FIELD_LABELS)

4. Si necesitas un cálculo pesado nuevo:
   a. Escribe la lógica pura en asset_allocation.py
   b. Añade un wrapper cacheado en utils/finance.py
   c. Importa y llama el wrapper aquí

5. Elimina este bloque de comentarios cuando el módulo esté listo.
"""

import streamlit as st

# ── Siempre estas tres líneas al principio de cada página ─────────────────────
from utils.state import init_state
from utils.ui    import apply_css, page_header, section, divider, kpi_card, guard, info, warn

init_state()
apply_css()

# ── TODO: cambia título y subtítulo ───────────────────────────────────────────
page_header("🆕 Nombre del Módulo", "Descripción breve en una línea")

# ── TODO: declara qué campos del session state necesita este módulo ───────────
# Descomenta y ajusta según lo que necesite la página:
#
# guard("eq_returns", "volatilities", "corr_matrix")          # solo mercado
# guard("eq_returns", "volatilities", "corr_matrix",
#       "portfolio_weights")                                   # mercado + cartera
# guard("bl_post_returns")                                     # necesita BL previo

# ── TODO: lee las variables que vayas a usar ──────────────────────────────────
assets  = st.session_state["asset_classes"]
# eq_ret  = st.session_state["eq_returns"]
# vols    = st.session_state["volatilities"]
# corr    = st.session_state["corr_matrix"]
# weights = st.session_state["portfolio_weights"]
n       = len(assets)

# ── TODO: parámetros del módulo en sidebar o en la página ────────────────────
# Ejemplo:
# param = st.sidebar.slider("Mi parámetro", 0.0, 1.0, 0.5, 0.01)

# ── TODO: lógica y visualización ─────────────────────────────────────────────
section("Sección 1")
st.write("Contenido de la sección 1")

divider()
section("Sección 2")
st.write("Contenido de la sección 2")

# ── TODO: si guardas resultados en session_state, hazlo aquí ─────────────────
# st.session_state["mi_campo_nuevo"] = resultado
