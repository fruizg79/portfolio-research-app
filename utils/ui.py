"""
utils/ui.py
===========
Shared UI helpers: CSS injection, reusable components, and the page guard.

Import in any page with:
    from utils.ui import apply_css, kpi_card, status_badge, guard, section, divider
"""

import streamlit as st


# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --navy:       #0a1628;
    --navy-mid:   #132240;
    --navy-light: #1e3a5f;
    --gold:       #c9a84c;
    --gold-light: #e8c97a;
    --cream:      #f5f0e8;
    --success:    #2d6a4f;
    --danger:     #9b2226;
    --border:     #ddd6c8;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background-color: #f8f6f1; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy) 0%, var(--navy-mid) 100%);
    border-right: 1px solid var(--navy-light);
}
section[data-testid="stSidebar"] * { color: var(--cream) !important; }
section[data-testid="stSidebar"] .stRadio label {
    color: #aab8cc !important; font-size: 0.9rem; letter-spacing: 0.03em;
}

/* ── Page header ── */
.app-header {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
    padding: 2rem 2.5rem 1.8rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 2px solid var(--gold);
}
.app-header h1 {
    font-family: 'DM Serif Display', serif;
    color: var(--cream); font-size: 2rem; margin: 0;
}
.app-header p {
    color: #8ca0bb; font-size: 0.85rem; margin: 0.3rem 0 0;
    letter-spacing: 0.05em; text-transform: uppercase;
}

/* ── Section title ── */
.section-title {
    font-family: 'DM Serif Display', serif;
    color: var(--navy); font-size: 1.35rem;
    border-bottom: 1.5px solid var(--gold);
    padding-bottom: 0.35rem; margin-bottom: 1.1rem;
}

/* ── KPI cards ── */
.metric-card {
    background: white; border: 1px solid var(--border);
    border-radius: 8px; padding: 1.15rem 1.3rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-card .label {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: #4a5568; margin-bottom: 0.3rem; font-weight: 500;
}
.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.65rem; font-weight: 500; color: var(--navy);
}
.metric-card .delta { font-size: 0.77rem; margin-top: 0.2rem; color: #718096; }
.metric-card.positive .value { color: var(--success); }
.metric-card.negative .value { color: var(--danger); }

/* ── Info / warning boxes ── */
.info-box {
    background: #eef3fb; border-left: 3px solid #4a7fcb;
    padding: 0.85rem 1rem; border-radius: 0 6px 6px 0;
    font-size: 0.87rem; color: #2d4a7a; margin: 0.9rem 0;
}
.warn-box {
    background: #fef9ec; border-left: 3px solid var(--gold);
    padding: 0.85rem 1rem; border-radius: 0 6px 6px 0;
    font-size: 0.87rem; color: #7a5a1a; margin: 0.9rem 0;
}

/* ── Decorative divider ── */
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, var(--gold) 0%, transparent 100%);
    margin: 1.4rem 0;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--navy); color: var(--cream);
    border: none; border-radius: 6px;
    font-family: 'DM Sans', sans-serif; font-weight: 500;
    letter-spacing: 0.03em; padding: 0.5rem 1.4rem;
    transition: background 0.2s;
}
.stButton > button:hover { background: var(--navy-light); }
</style>
"""


def apply_css() -> None:
    """Inject shared CSS. Call once per page, before any other st.* call."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Page header ───────────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str = "") -> None:
    """Render the dark gradient header banner at the top of each page."""
    st.markdown(
        f'<div class="app-header"><h1>{title}</h1>'
        f'<p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


# ── Section title ─────────────────────────────────────────────────────────────
def section(title: str) -> None:
    """Gold-underlined section heading."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ── Divider ───────────────────────────────────────────────────────────────────
def divider() -> None:
    """Thin gold-to-transparent horizontal rule."""
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)


# ── Info / warning boxes ──────────────────────────────────────────────────────
def info(text: str) -> None:
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def warn(text: str) -> None:
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)


# ── KPI card ──────────────────────────────────────────────────────────────────
def kpi_card(
    col,
    label: str,
    value: float,
    fmt: str,
    css_class: str = "",
    delta: float | None = None,
    delta_label: str = "",
) -> None:
    """
    Render a single KPI metric card inside *col*.

    Args:
        col:         Streamlit column object.
        label:       Short uppercase label shown above the value.
        value:       Numeric value to display.
        fmt:         Python format spec, e.g. ".2%", ".2f".
        css_class:   Optional extra CSS class ("positive" / "negative").
        delta:       Optional comparison value (shown as small annotation).
        delta_label: Text prefix for the delta line (e.g. "vs equilibrio").
    """
    delta_html = (
        f'<div class="delta">{delta_label}: {delta:{fmt}}</div>'
        if delta is not None else ""
    )
    col.markdown(
        f"""<div class="metric-card {css_class}">
            <div class="label">{label}</div>
            <div class="value">{value:{fmt}}</div>
            {delta_html}
        </div>""",
        unsafe_allow_html=True,
    )


# ── Status badge ──────────────────────────────────────────────────────────────
def status_badge(ok: bool, label: str) -> str:
    """Return HTML string for a green/red status indicator tile."""
    icon  = "✅" if ok else "⬜"
    bg    = "#eaf4ef" if ok else "#fce8e8"
    bdr   = "#b7dfc9" if ok else "#f5b5b5"
    color = "#2d6a4f" if ok else "#9b2226"
    return (
        f'<div style="background:{bg};border:1px solid {bdr};border-radius:6px;'
        f'padding:0.8rem 1rem;text-align:center;">'
        f'<div style="font-size:1.2rem">{icon}</div>'
        f'<div style="font-size:0.78rem;color:{color};font-weight:600;'
        f'margin-top:4px">{label}</div>'
        f'</div>'
    )


# ── Guard ─────────────────────────────────────────────────────────────────────
# Human-readable labels for every registered session-state key.
_FIELD_LABELS: dict[str, str] = {
    "eq_returns":        "Rentabilidades de equilibrio (Configuración)",
    "volatilities":      "Volatilidades (Configuración)",
    "corr_matrix":       "Matriz de correlaciones (Configuración)",
    "portfolio_weights": "Pesos de cartera (Mi Cartera)",
    "tactical_ranges":   "Rangos tácticos (Mi Cartera)",
    "portfolio_metrics": "Métricas de cartera (Riesgo & Retorno)",
    "bl_post_returns":   "Retornos Black-Litterman (módulo BL)",
    # ── ADD new fields here when you register them in state.py ──
}


def guard(*fields: str) -> None:
    """
    Stop page rendering if any required session-state field is None.

    Usage:
        guard("eq_returns", "volatilities", "corr_matrix")

    Shows a styled warning listing every missing field, then calls st.stop().
    """
    missing = [f for f in fields if st.session_state.get(f) is None]
    if missing:
        items = "".join(
            f"&nbsp;&nbsp;• {_FIELD_LABELS.get(m, m)}<br>" for m in missing
        )
        st.markdown(
            f'<div class="warn-box">⚠️ Faltan los siguientes datos:<br>{items}</div>',
            unsafe_allow_html=True,
        )
        st.stop()
