# Asset Allocation App

Herramienta de gestión de carteras basada en Streamlit.
Depende de `asset_allocation.py` y `ml_portfolio_enhancement.py`.

## Estructura del proyecto

```
portfolio_app/
│
├── app.py                          ← Punto de entrada (streamlit run app.py)
├── asset_allocation.py              ← Librería de negocio (copia aquí)
├── ml_portfolio_enhancement.py     ← Módulo ML       (copia aquí)
├── requirements.txt
│
├── pages/                          ← Streamlit detecta estas páginas automáticamente
│   ├── 1_Configuracion.py
│   ├── 2_Mi_Cartera.py
│   ├── 3_Riesgo_Retorno.py
│   ├── 4_Black_Litterman.py
│   └── TEMPLATE_Nueva_Pagina.py   ← Copia esto para añadir un módulo nuevo
│
└── utils/
    ├── state.py                    ← Session state: campos y dependencias
    ├── ui.py                      ← CSS, componentes, guard()
    └── finance.py                  ← Wrappers cacheados sobre asset_allocation.py
```

## Arrancar la app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Añadir un módulo nuevo

1. **Copia** `pages/TEMPLATE_Nueva_Pagina.py` → `pages/5_Nombre.py`
2. **Rellena** los TODO del fichero
3. Si necesitas **nuevos campos** en session state:
   - Añádelos en `utils/state.py` → `_DEFAULTS`
   - Añade sus dependencias en `reset_downstream()`
   - Añade su etiqueta en `utils/ui.py` → `_FIELD_LABELS`
4. Si necesitas un **nuevo cálculo pesado**:
   - Escribe la lógica en `asset_allocation.py`
   - Añade un wrapper `@st.cache_data` en `utils/finance.py`
   - Importa y llama desde la página

## Módulos actuales

| # | Página | Escribe en session state |
|---|--------|--------------------------|
| 1 | Configuración | `asset_classes`, `eq_returns`, `volatilities`, `corr_matrix` |
| 2 | Mi Cartera | `portfolio_weights`, `tactical_ranges` |
| 3 | Riesgo & Retorno | `portfolio_metrics` |
| 4 | Black-Litterman | `bl_eq_returns`, `bl_post_returns` |

## Módulos planificados (ejemplos)

| # | Página | Depende de |
|---|--------|------------|
| 5 | Optimización (Frontera Eficiente) | mercado + cartera |
| 6 | Backtesting | mercado + cartera |
| 7 | ML Enhancement | mercado |
| 8 | Informe PDF | todos los anteriores |
