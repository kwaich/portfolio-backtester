# App.py Refactoring Analysis

**Date**: 2025-11-15
**Task**: Phase 2, Task 2.1 - Refactor app.py into Modules
**Current Size**: 874 lines (monolithic)
**Target**: Modular structure with files <200 lines each

---

## Current Structure Analysis

### File Statistics
- **Total Lines**: 874
- **Functions**: 0 (all code in main script)
- **Classes**: 0
- **Imports**: 13 (4 standard lib, 4 third-party, 5 from backtest)

### Code Sections Identified

| Section | Lines | % of File | Description |
|---------|-------|-----------|-------------|
| **Imports & Checks** | 1-102 | 12% | Dependency imports and error handling |
| **Page Config** | 104-137 | 4% | Streamlit config, CSS styling |
| **Presets** | 142-281 | 16% | Portfolio and date range presets |
| **Session State** | Scattered | 5% | Session state initialization |
| **Input Controls** | 139-330 | 22% | Sidebar inputs (tickers, weights, dates) |
| **Backtest Logic** | 332-420 | 10% | Download, compute, validate |
| **Results Display** | 420-558 | 16% | Metrics, tables, summaries |
| **Charts** | 559-874 | 36% | Plotly chart generation |

---

## Refactoring Strategy

### New Module Structure

```
app/
├── __init__.py          # Package initialization
├── main.py              # Entry point (~150 lines)
├── config.py            # Constants and configuration (~80 lines)
├── presets.py           # Portfolio and date presets (~100 lines)
├── ui_components.py     # Reusable UI elements (~150 lines)
├── charts.py            # Chart generation (~250 lines)
└── validation.py        # Input validation and session state (~120 lines)
```

**Total Estimated Lines**: ~850 (similar to original, but organized)
**Longest Module**: charts.py (~250 lines, acceptable for complex visualizations)

---

## Module Breakdown

### 1. `app/config.py` (~80 lines)

**Purpose**: Centralize all configuration constants

**Contents**:
- Page configuration (title, icon, layout)
- UI limits (max tickers, max benchmarks)
- Default values (capital, start date)
- Chart colors and styling
- Metric labels mapping
- CSS styling

**Example**:
```python
# Page Configuration
PAGE_TITLE = "ETF Backtester"
PAGE_ICON = "📈"
LAYOUT = "wide"

# Limits
MAX_TICKERS = 10
MIN_TICKERS = 1
MAX_BENCHMARKS = 3

# Chart Colors
PORTFOLIO_COLOR = "#1f77b4"
BENCHMARK_COLORS = ['#9467bd', '#e377c2', '#bcbd22']
```

**Extracted From**:
- Lines 105-109 (page config)
- Lines 113-133 (CSS)
- Lines 439-448 (metric labels)
- Lines 597 (benchmark colors)
- Hardcoded values throughout

---

### 2. `app/presets.py` (~100 lines)

**Purpose**: Define portfolio and date range presets

**Contents**:
- `get_portfolio_presets()` function
- `get_date_presets()` function
- Portfolio configurations (6 presets)
- Date range calculations

**Example**:
```python
def get_portfolio_presets() -> dict:
    return {
        "Custom (Manual Entry)": {...},
        "Default UK ETFs": {...},
        # ... 4 more
    }

def get_date_presets() -> dict:
    today = datetime.today()
    return {
        "1Y": today - timedelta(days=365),
        "3Y": today - timedelta(days=365*3),
        # ...
    }
```

**Extracted From**:
- Lines 143-150 (portfolio presets)
- Lines 274-281 (date presets)

---

### 3. `app/validation.py` (~120 lines)

**Purpose**: Input validation and session state management

**Contents**:
- `initialize_session_state()` function
- `get_session_defaults()` function
- `update_portfolio_preset()` function
- `validate_backtest_inputs()` function
- `normalize_weights()` function

**Example**:
```python
def initialize_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        'selected_portfolio': "Custom (Manual Entry)",
        'num_tickers': 2,
        'start_date': datetime(2018, 1, 1),
        # ...
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def validate_backtest_inputs(tickers, benchmarks, start_date, end_date):
    """Validate all inputs before running backtest."""
    if not all(tickers):
        return False, "Please enter all ticker symbols"
    # ...
    return True, ""
```

**Extracted From**:
- Lines 160-171 (session state init)
- Lines 284-287 (date state)
- Lines 334-354 (input validation)
- Lines 356-360 (weight normalization)

---

### 4. `app/ui_components.py` (~150 lines)

**Purpose**: Reusable UI components

**Contents**:
- `render_metric()` function
- `render_metrics_column()` function
- `render_portfolio_composition()` function
- `render_relative_metrics()` function
- `format_metric_value()` function

**Example**:
```python
def format_metric_value(key: str, value: float) -> str:
    """Format metric based on type."""
    if key == "ending_value":
        return f"${value:,.2f}"
    elif key in ["total_return", "cagr", "volatility", "max_drawdown"]:
        return f"{value:.2%}"
    elif key in ["sharpe_ratio", "sortino_ratio"]:
        return f"{value:.3f}"
    return f"{value:.2f}"

def render_metric(key: str, value: float, label: str = None) -> None:
    """Render single metric with proper formatting."""
    from .config import METRIC_LABELS
    if label is None:
        label = METRIC_LABELS.get(key, key)
    formatted = format_metric_value(key, value)
    st.metric(label, formatted)
```

**Extracted From**:
- Lines 439-498 (metric rendering - duplicated 3 times)
- Lines 549-558 (portfolio composition table)

---

### 5. `app/charts.py` (~250 lines)

**Purpose**: Chart generation with Plotly

**Contents**:
- `create_main_dashboard()` function
- `create_rolling_returns_chart()` function
- `calculate_drawdown()` function
- Chart configuration helpers

**Example**:
```python
def create_main_dashboard(
    results: pd.DataFrame,
    all_benchmark_results: dict,
    benchmarks: list
) -> go.Figure:
    """Create 2x2 dashboard with all main charts."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Portfolio vs Benchmark", "Cumulative Returns", ...)
    )

    # Add traces for each chart
    # ...

    return fig

def calculate_drawdown(series: pd.Series) -> pd.Series:
    """Calculate drawdown from value series."""
    cumulative = series / series.expanding().max()
    return (cumulative - 1)
```

**Extracted From**:
- Lines 564-700 (main dashboard charts)
- Lines 702-803 (rolling returns chart)
- Lines 567-574 (drawdown calculation - duplicated)

---

### 6. `app/main.py` (~150 lines)

**Purpose**: Main entry point, orchestrates the app

**Contents**:
- Import all modules
- Page configuration
- Session state initialization
- Sidebar input collection
- Backtest execution workflow
- Results display coordination

**Example**:
```python
import streamlit as st
from .config import PAGE_TITLE, PAGE_ICON, LAYOUT
from .presets import get_portfolio_presets, get_date_presets
from .validation import initialize_session_state, validate_backtest_inputs
from .ui_components import render_metrics_column, render_portfolio_composition
from .charts import create_main_dashboard, create_rolling_returns_chart

# Page configuration
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

# Initialize
initialize_session_state()

# Sidebar inputs
# ... (collect all inputs)

# Run backtest
if st.sidebar.button("Run Backtest"):
    # Validate
    is_valid, error = validate_backtest_inputs(...)

    # Download and compute
    # ...

    # Display results
    render_metrics_column(portfolio_summary, "Portfolio")
    # ...
```

**Extracted From**:
- Main orchestration logic from lines 104-874

---

## Code Duplication Analysis

### Metric Rendering (Duplicated 3x)
**Locations**: Lines 451-463, 464-476, 477-498
**Lines**: ~48 lines x 3 = 144 lines total
**Solution**: Extract to `render_metric()` in `ui_components.py`
**Savings**: ~120 lines

### Drawdown Calculation (Duplicated 2x)
**Locations**: Lines 567-570, 571-574
**Lines**: 4 lines x 2 = 8 lines
**Solution**: Extract to `calculate_drawdown()` in `charts.py`
**Savings**: ~4 lines

### Session State Init (Scattered)
**Locations**: Lines 160-161, 284-287, plus inline checks
**Lines**: ~15 lines scattered
**Solution**: Centralize in `validation.py`
**Savings**: ~10 lines through consolidation

**Total Potential Savings**: ~134 lines

---

## Magic Numbers to Extract

| Value | Current Location | Move To | Constant Name |
|-------|------------------|---------|---------------|
| `10` | Line 177 | config.py | `MAX_TICKERS` |
| `3` | Line 239 | config.py | `MAX_BENCHMARKS` |
| `100_000` | Line 313 | config.py | `DEFAULT_CAPITAL` |
| `1_000` | Line 314 | config.py | `MIN_CAPITAL` |
| `10_000_000` | Line 315 | config.py | `MAX_CAPITAL` |
| `2018-01-01` | Line 285 | config.py | `DEFAULT_START_DATE` |
| `2010-01-01` | Line 280 | config.py | `MAX_START_DATE` |
| `"#1f77b4"` | Inline | config.py | `PORTFOLIO_COLOR` |
| `['#9467bd', ...]` | Line 597 | config.py | `BENCHMARK_COLORS` |
| `['dash', 'dot', ...]` | Line 598 | config.py | `BENCHMARK_DASH_STYLES` |
| `[30, 90, 180]` | Line 709 | config.py | `ROLLING_WINDOWS` |

---

## Backward Compatibility Strategy

### 1. Keep `app.py` as Wrapper
```python
# app.py (backward compatibility wrapper)
"""
Backward compatibility wrapper for app.py.
The main application has been refactored into the app/ package.
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run main app
if __name__ == "__main__":
    from app.main import main
    main()
```

### 2. Maintain API Compatibility
- All functionality remains accessible
- `streamlit run app.py` still works
- New way: `streamlit run app/main.py`

### 3. Import Path Handling
- Relative imports within app/ package
- Absolute imports for backtest module

---

## Testing Strategy

### 1. Functional Testing
- Run refactored app: `streamlit run app/main.py`
- Test all presets
- Test custom inputs
- Verify charts render
- Test multiple benchmarks
- Test validation errors

### 2. Backward Compatibility Testing
- Run old entry point: `streamlit run app.py`
- Verify same functionality
- Check no import errors

### 3. Unit Testing
- Test each new module independently
- Mock Streamlit functions
- Test validation functions
- Test metric formatting
- Test chart data preparation

---

## Implementation Order

1. **Create Directory Structure** ✅
   - `mkdir app`
   - `touch app/__init__.py`

2. **Extract Constants** (Task 2.3 integration)
   - Create `app/config.py`
   - Move all constants
   - Test imports

3. **Extract Presets**
   - Create `app/presets.py`
   - Move portfolio and date presets
   - Add getter functions

4. **Extract Validation** (Task 2.2 integration)
   - Create `app/validation.py`
   - Move session state logic
   - Move validation functions

5. **Extract UI Components** (Task 2.4 integration)
   - Create `app/ui_components.py`
   - Extract metric rendering
   - Extract table rendering

6. **Extract Charts**
   - Create `app/charts.py`
   - Move Plotly chart code
   - Extract helper functions

7. **Create Main Entry Point**
   - Create `app/main.py`
   - Import all modules
   - Orchestrate workflow

8. **Create Backward Compatibility Wrapper**
   - Update `app.py`
   - Add path handling
   - Test both entry points

9. **Testing & Validation**
   - Run app, test all features
   - Write unit tests
   - Update documentation

---

## Risk Mitigation

### Risks
1. **Breaking changes** - App doesn't run
2. **Import errors** - Circular dependencies
3. **Session state issues** - State not preserved
4. **Streamlit compatibility** - Module imports don't work

### Mitigations
1. **Incremental refactoring** - One module at a time
2. **Keep original app.py** - Can revert easily
3. **Test after each module** - Verify app still works
4. **Use relative imports** - Avoid circular dependencies
5. **Git commits per module** - Easy rollback

---

## Success Criteria

✅ All functionality preserved
✅ Each module < 200 lines (except charts.py ~250)
✅ No code duplication
✅ Clear module boundaries
✅ Backward compatibility maintained
✅ All tests pass
✅ App runs with both entry points

---

## Estimated Effort

| Step | Estimated Time |
|------|----------------|
| Directory setup | 15 min |
| config.py | 30 min |
| presets.py | 30 min |
| validation.py | 45 min |
| ui_components.py | 60 min |
| charts.py | 90 min |
| main.py | 60 min |
| Backward compat | 15 min |
| Testing | 60 min |
| Documentation | 30 min |
| **Total** | **6 hours** |

Matches plan estimate of 6 hours ✅

---

**Analysis Complete** - Ready to begin implementation.
