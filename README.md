# Asset Allocation App

Herramienta profesional de gestión de carteras basada en Streamlit.  
Implementa Modern Portfolio Theory, el modelo Black-Litterman y métricas de riesgo vía Monte Carlo.

---

## Ficheros del repositorio

```
portfolio-optimization/
│
├── setup_project.py                ← Instalación automática (ejecutar primero)
│
├── asset_allocation.py             ← Librería de negocio principal
├── ml_portfolio_enhancement.py     ← Módulo ML (opcional)
├── test_asset_allocation.py        ← Suite de tests (121 tests)
│
└── portfolio_app/                  ← Aplicación Streamlit
    ├── app.py                      ← Punto de entrada
    ├── asset_allocation.py         ← Copia de la librería (generada por setup)
    ├── requirements.txt
    │
    ├── pages/
    │   ├── 1_Configuracion.py
    │   ├── 2_Mi_Cartera.py
    │   ├── 3_Riesgo_Retorno.py
    │   ├── 4_Black_Litterman.py
    │   └── TEMPLATE_Nueva_Pagina.py
    │
    └── utils/
        ├── state.py                ← Session state centralizado
        ├── ui.py                   ← CSS y componentes compartidos
        └── finance.py              ← Wrappers cacheados sobre asset_allocation.py
```

---

## Instalación y arranque

### 1 · Generar la estructura del proyecto

```bash
python setup_project.py             # crea ./portfolio_app/
python setup_project.py mi_nombre  # o con nombre personalizado
```

El script copia `asset_allocation.py` automáticamente al proyecto y avisa si
falta `ml_portfolio_enhancement.py`.

### 2 · Instalar dependencias

```bash
cd portfolio_app
pip install -r requirements.txt
```

### 3 · Arrancar la app

```bash
streamlit run app.py
```

---

## Tests

```bash
python test_asset_allocation.py          # salida detallada
python -m pytest test_asset_allocation.py -v  # con pytest
```

La suite cubre 121 tests sobre todas las clases y funciones de `asset_allocation.py`.
Los 8 tests del solver QP (`mean_variance_opt`, `efficient_frontier`, `asset_allocation_TEop`)
se saltan si `cvxopt` no está instalado y se activan automáticamente cuando lo está.

| Clase / función | Tests |
|---|---|
| `build_cov_matrix` y helpers estadísticos | 21 |
| `CAsset` (Normal + Vasicek) | 17 |
| `CCopulas` (Gaussiana + t-Student) | 12 |
| `CPortfolio_optimization` | 22 |
| `CBlack_litterman` + `from_views()` | 28 |
| `CUtility` (quadratic, exponential, power) | 12 |
| Pipeline de integración end-to-end | 3 |

---

## Módulos de la app

| # | Página | Lee de session state | Escribe en session state |
|---|--------|----------------------|--------------------------|
| 1 | Configuración | — | `asset_classes`, `eq_returns`, `volatilities`, `corr_matrix` |
| 2 | Mi Cartera | `asset_classes` | `portfolio_weights`, `tactical_ranges` |
| 3 | Riesgo & Retorno | mercado + cartera | `portfolio_metrics` |
| 4 | Black-Litterman | mercado + cartera | `bl_eq_returns`, `bl_post_returns` |

---

## Librería `asset_allocation.py`

### Clases principales

| Clase | Descripción |
|---|---|
| `CAsset` | Calibración (Vasicek / Normal) y simulación Monte Carlo de activos |
| `CCopulas` | Generación de retornos correlacionados (Gaussiana y t-Student) |
| `CPortfolio_optimization` | Optimización media-varianza y tracking error (QP via cvxopt) |
| `CBlack_litterman` | Modelo Black-Litterman con views cualitativos |
| `CUtility` | Funciones de utilidad: cuadrática, exponencial, power |
| `CMarket` | Carga de datos de mercado desde Excel |

### Métodos añadidos para la app

**`CPortfolio_optimization.get_portfolio_metrics(weights, risk_free_rate, n_sims)`**  
Calcula retorno esperado, volatilidad, Sharpe, VaR 95%, Expected Shortfall, probabilidad de pérdida
y contribución marginal al riesgo — todo en una sola llamada. Usa Monte Carlo internamente.

**`CBlack_litterman.from_views(eq_returns, cov_matrix, weights_eq, p_matrix, ...)`**  
Constructor alternativo (classmethod) que acepta arrays ya preparados y devuelve
`(pi_equilibrio, bl_posterior)` directamente. Es la interfaz que usa la app.

### Funciones de módulo

```python
build_cov_matrix(volatilities, correlation_matrix)  # Σ = V · C · V
calculate_sharpe_ratio(returns, risk_free_rate)
calculate_sortino_ratio(returns, risk_free_rate)
calculate_max_drawdown(returns)
calculate_value_at_risk(returns, confidence_level)
calculate_conditional_var(returns, confidence_level)
```

---

## Arquitectura de la app (`utils/`)

### `utils/state.py` — Session state centralizado

Registro único de todos los campos de `st.session_state`.  
**Cuando añadas un módulo nuevo, añade sus campos aquí primero.**

```python
_DEFAULTS = {
    # Configuración
    "asset_classes":     [...],
    "eq_returns":        None,
    "volatilities":      None,
    "corr_matrix":       None,
    # Mi Cartera
    "portfolio_weights": None,
    "tactical_ranges":   None,
    # Resultados cacheados
    "portfolio_metrics": None,
    "bl_eq_returns":     None,
    "bl_post_returns":   None,
    # UI
    "risk_free_rate":    0.02,
    "n_sims":            50_000,
}
```

`reset_downstream(from_key)` invalida automáticamente los resultados calculados
cuando cambia un input upstream (p.ej. cambiar las volatilidades borra `portfolio_metrics`).

### `utils/ui.py` — Componentes compartidos

```python
apply_css()                        # inyecta el CSS en la página
page_header(title, subtitle)       # banner oscuro superior
section(title)                     # título de sección con subrayado dorado
divider()                          # separador decorativo
info(text) / warn(text)            # cajas de información / aviso
kpi_card(col, label, value, fmt)   # tarjeta de métrica
status_badge(ok, label)            # indicador verde/rojo
guard(*fields)                     # para la página si faltan datos
```

### `utils/finance.py` — Wrappers cacheados

Todas las funciones están decoradas con `@st.cache_data` para evitar
recalcular Monte Carlo o inversiones de matrices en cada interacción del usuario.

```python
get_cov_matrix(volatilities, corr_flat, n)
get_portfolio_metrics(weights, expected_ret, volatilities, corr_flat, n, ...)
run_black_litterman(eq_returns, volatilities, ..., recommendations, confidence_levels, ...)
get_optimizer(weights, expected_ret, volatilities, corr_matrix)  # sin caché
to_cache_args(volatilities, corr_matrix)  # convierte arrays a tuplas hasheables
```

---

## Añadir un módulo nuevo

1. **Copia** la plantilla:
   ```bash
   cp pages/TEMPLATE_Nueva_Pagina.py pages/5_Mi_Modulo.py
   ```

2. **Rellena** los `TODO` del fichero (título, `guard()`, variables, lógica).

3. Si necesitas **nuevos campos** en session state:
   - Añádelos en `utils/state.py` → `_DEFAULTS`
   - Añade sus dependencias en `reset_downstream()`
   - Añade su etiqueta en `utils/ui.py` → `_FIELD_LABELS`

4. Si necesitas un **nuevo cálculo pesado**:
   - Escribe la lógica pura en `asset_allocation.py`
   - Añade un wrapper `@st.cache_data` en `utils/finance.py`
   - Importa y llama desde la página

Streamlit detecta las páginas automáticamente por el nombre del fichero.
El número al principio (`5_`) controla el orden en la barra lateral.

---

## Módulos planificados

| # | Módulo | Depende de |
|---|--------|------------|
| 5 | Optimización — Frontera Eficiente | mercado + cartera |
| 6 | Backtesting | mercado + cartera |
| 7 | ML Enhancement (`ml_portfolio_enhancement.py`) | mercado |
| 8 | Informe / Export PDF | todos los anteriores |

---

## Dependencias

| Paquete | Uso |
|---|---|
| `streamlit >= 1.35` | Framework de la app |
| `numpy >= 1.26` | Álgebra lineal y simulación |
| `pandas >= 2.1` | Manipulación de datos |
| `scipy >= 1.11` | Estadística (regresión, distribuciones) |
| `plotly >= 5.20` | Visualizaciones interactivas |
| `cvxopt >= 1.3.2` | Programación cuadrática (optimización) |
| `scikit-learn >= 1.4` | ML (usado por `ml_portfolio_enhancement.py`) |
| `openpyxl >= 3.1` | Lectura/escritura de Excel |

---

## Autor

**Fernando Ruiz** · [portfolio-research.com](https://portfolio-research.com)  
Licencia MIT · © 2026
