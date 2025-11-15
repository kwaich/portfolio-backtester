# Implementation Plan - Code Review Fixes

**Created**: 2025-11-15
**Based on**: Code Review Report
**Target Completion**: 2-3 weeks

---

## Overview

This plan addresses the issues identified in the comprehensive code review, organized by priority and estimated effort.

**Total Tasks**: 35
**Estimated Effort**: 40-50 hours
**Success Criteria**: All tests passing, 85%+ coverage maintained

---

## Phase 1: Critical & High-Priority Fixes (Week 1)

**Goal**: Address issues that could impact functionality or user experience
**Effort**: 12-16 hours

### Task 1.1: Implement Cache Expiration System ⚡ HIGH
**Priority**: High
**Effort**: 3 hours
**Files**: `backtest.py`
**Issue**: 2.1 - Cache Has No Expiration

#### Implementation Steps:
1. Add cache metadata structure
   ```python
   # Cache format: {"data": DataFrame, "timestamp": float, "version": str}
   ```

2. Modify `save_cached_prices()` (backtest.py:121-128)
   ```python
   def save_cached_prices(cache_path: Path, prices: pd.DataFrame) -> None:
       """Save price data to cache with metadata."""
       cache_data = {
           "data": prices,
           "timestamp": time.time(),
           "version": "1.0"
       }
       try:
           with open(cache_path, "wb") as f:
               pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
           logger.info(f"Saved data to cache: {cache_path}")
       except Exception as e:
           logger.warning(f"Failed to save cache: {e}")
   ```

3. Modify `load_cached_prices()` (backtest.py:107-118)
   ```python
   def load_cached_prices(cache_path: Path, max_age_hours: int = 24) -> pd.DataFrame | None:
       """Load cached price data if it exists and is not stale."""
       if not cache_path.exists():
           return None

       try:
           with open(cache_path, "rb") as f:
               cache_data = pickle.load(f)

           # Handle old cache format (migration)
           if isinstance(cache_data, pd.DataFrame):
               logger.warning("Old cache format detected, will re-download")
               return None

           # Check cache age
           cache_age_hours = (time.time() - cache_data["timestamp"]) / 3600
           if cache_age_hours > max_age_hours:
               logger.info(f"Cache expired (age: {cache_age_hours:.1f}h, max: {max_age_hours}h)")
               cache_path.unlink()  # Delete stale cache
               return None

           logger.info(f"Loaded cached data (age: {cache_age_hours:.1f}h)")
           return cache_data["data"]

       except Exception as e:
           logger.warning(f"Failed to load cache: {e}")
           return None
   ```

4. Add CLI argument for cache TTL (backtest.py:85-89)
   ```python
   parser.add_argument(
       "--cache-ttl",
       type=int,
       default=24,
       help="Cache time-to-live in hours (default: 24)"
   )
   ```

5. Update `download_prices()` signature (backtest.py:131)
   ```python
   def download_prices(
       tickers: List[str],
       start: str,
       end: str,
       use_cache: bool = True,
       cache_ttl_hours: int = 24
   ) -> pd.DataFrame:
   ```

#### Testing Requirements:
- [ ] Test with fresh cache (< 24h old)
- [ ] Test with stale cache (> 24h old)
- [ ] Test with corrupted cache file
- [ ] Test with old cache format (DataFrame only)
- [ ] Test cache migration path
- [ ] Test custom TTL values (1h, 168h)

**Acceptance Criteria**:
- ✅ Stale cache is automatically re-downloaded
- ✅ Cache age is logged
- ✅ Old cache format migrates gracefully
- ✅ All existing tests pass

---

### Task 1.2: Add Rate Limiting & Retry Logic ⚡ HIGH
**Priority**: High
**Effort**: 4 hours
**Files**: `backtest.py`, `app.py`
**Issue**: 2.2 - No Rate Limiting for API Calls

#### Implementation Steps:

1. Create retry decorator (backtest.py, after imports)
   ```python
   import time
   from functools import wraps
   from typing import Callable, Any

   def retry_with_backoff(
       max_retries: int = 3,
       base_delay: float = 1.0,
       max_delay: float = 60.0,
       exceptions: tuple = (Exception,)
   ) -> Callable:
       """Decorator to retry function with exponential backoff."""
       def decorator(func: Callable) -> Callable:
           @wraps(func)
           def wrapper(*args, **kwargs) -> Any:
               for attempt in range(max_retries):
                   try:
                       return func(*args, **kwargs)
                   except exceptions as e:
                       if attempt == max_retries - 1:
                           raise

                       delay = min(base_delay * (2 ** attempt), max_delay)
                       logger.warning(
                           f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                           f"Retrying in {delay:.1f}s..."
                       )
                       time.sleep(delay)
               return None  # Should never reach here
           return wrapper
       return decorator
   ```

2. Apply decorator to download function (backtest.py:143)
   ```python
   @retry_with_backoff(max_retries=3, base_delay=2.0)
   def _download_from_yfinance(tickers: List[str], start: str, end: str) -> Any:
       """Internal function to download from yfinance with retry logic."""
       logger.info(f"Downloading data for {len(tickers)} ticker(s) from {start} to {end}")

       return yf.download(
           tickers=tickers,
           start=start,
           end=end,
           interval="1d",
           auto_adjust=True,
           progress=False,
       )
   ```

3. Update `download_prices()` to use retry wrapper
   ```python
   def download_prices(...):
       # Cache check logic...

       # Use retry wrapper for actual download
       data = _download_from_yfinance(tickers, start, end)

       # Rest of existing logic...
   ```

4. Add rate limiting for multiple tickers (backtest.py)
   ```python
   def download_prices_batch(
       ticker_batches: List[List[str]],
       start: str,
       end: str,
       delay_between_batches: float = 0.5
   ) -> Dict[str, pd.DataFrame]:
       """Download multiple batches with delays to avoid rate limiting."""
       results = {}

       for i, batch in enumerate(ticker_batches):
           if i > 0:
               time.sleep(delay_between_batches)

           batch_data = download_prices(batch, start, end)
           for ticker in batch:
               results[ticker] = batch_data[ticker]

       return results
   ```

5. Update app.py benchmark downloads (app.py:282-290)
   ```python
   # Download all benchmarks with delays
   all_benchmark_prices = {}
   for idx, bench in enumerate(benchmarks):
       if idx > 0:
           time.sleep(0.5)  # 500ms delay between downloads

       bench_data = download_prices(
           [bench],
           start_date.strftime("%Y-%m-%d"),
           end_date.strftime("%Y-%m-%d"),
           use_cache=use_cache
       )[bench]
       all_benchmark_prices[bench] = bench_data
   ```

#### Testing Requirements:
- [ ] Test successful download on first attempt
- [ ] Test retry on transient failure (mock)
- [ ] Test failure after max retries
- [ ] Test exponential backoff delays
- [ ] Test batch downloading with delays
- [ ] Test concurrent download requests

**Acceptance Criteria**:
- ✅ Transient errors are retried automatically
- ✅ Exponential backoff prevents API hammering
- ✅ Failed downloads raise clear errors
- ✅ Rate limiting prevents rapid successive calls

---

### Task 1.3: Add Import Error Handling in app.py ⚡ HIGH
**Priority**: High
**Effort**: 1 hour
**Files**: `app.py`
**Issue**: 2.3 - Missing Import Error Handling

#### Implementation Steps:

1. Wrap imports with try/except (app.py:18-19)
   ```python
   # Standard imports
   from __future__ import annotations
   import streamlit as st
   import pandas as pd
   import numpy as np
   from datetime import datetime, timedelta
   from pathlib import Path
   import io
   import plotly.graph_objects as go
   from plotly.subplots import make_subplots

   # Import backtest module with error handling
   try:
       from backtest import download_prices, compute_metrics, summarize
   except ImportError as e:
       st.error(
           f"❌ Failed to import backtest module.\n\n"
           f"Error: {str(e)}\n\n"
           f"Please ensure:\n"
           f"1. backtest.py is in the same directory as app.py\n"
           f"2. All dependencies are installed: pip install -r requirements.txt\n"
           f"3. Python version is 3.8+"
       )
       st.stop()
   except Exception as e:
       st.error(
           f"❌ Unexpected error loading backtest module: {str(e)}\n\n"
           f"Please check backtest.py for syntax errors."
       )
       st.stop()
   ```

2. Add dependency check function (app.py, after imports)
   ```python
   def check_dependencies() -> bool:
       """Verify all required dependencies are available."""
       required_modules = {
           'numpy': 'numpy>=1.24.0',
           'pandas': 'pandas>=2.0.0',
           'plotly': 'plotly>=5.14.0',
           'streamlit': 'streamlit>=1.28.0',
       }

       missing = []
       for module, requirement in required_modules.items():
           try:
               __import__(module)
           except ImportError:
               missing.append(requirement)

       if missing:
           st.error(
               f"❌ Missing required dependencies:\n\n"
               f"{chr(10).join(f'  - {req}' for req in missing)}\n\n"
               f"Install with: pip install {' '.join(missing)}"
           )
           return False

       return True

   # Check dependencies before proceeding
   if not check_dependencies():
       st.stop()
   ```

#### Testing Requirements:
- [ ] Test with missing backtest.py
- [ ] Test with syntax error in backtest.py
- [ ] Test with missing numpy
- [ ] Test with missing pandas
- [ ] Test with all dependencies present

**Acceptance Criteria**:
- ✅ Clear error message when backtest.py is missing
- ✅ App doesn't crash with confusing traceback
- ✅ Helpful suggestions for fixing the issue

---

### Task 1.4: Add Ticker Validation 🟡 MEDIUM
**Priority**: Medium
**Effort**: 2 hours
**Files**: `backtest.py`, `app.py`
**Issue**: 3.3 - No Ticker Format Validation

#### Implementation Steps:

1. Create validation function (backtest.py, after imports)
   ```python
   import re

   def validate_ticker(ticker: str) -> tuple[bool, str]:
       """Validate ticker symbol format.

       Returns:
           (is_valid, error_message)
       """
       if not ticker:
           return False, "Ticker cannot be empty"

       if len(ticker) > 10:
           return False, f"Ticker too long: {ticker} (max 10 characters)"

       # Allow: letters, numbers, dots (for UK), hyphens, carets (for indices)
       if not re.match(r'^[A-Z0-9\.\-\^]+$', ticker.upper()):
           return False, f"Invalid ticker format: {ticker} (use letters, numbers, ., -, ^)"

       # Check for common invalid patterns
       if ticker.isdigit():
           return False, f"Ticker cannot be all numbers: {ticker}"

       return True, ""

   def validate_tickers(tickers: List[str]) -> None:
       """Validate list of tickers, raise ValueError if any invalid."""
       if not tickers:
           raise ValueError("No tickers provided")

       errors = []
       for ticker in tickers:
           is_valid, error_msg = validate_ticker(ticker)
           if not is_valid:
               errors.append(f"  • {error_msg}")

       if errors:
           raise ValueError(
               f"Invalid ticker(s) detected:\n" + "\n".join(errors) + "\n\n"
               f"Valid ticker examples: AAPL, MSFT, VWRA.L, ^GSPC"
           )
   ```

2. Add validation to `download_prices()` (backtest.py:131)
   ```python
   def download_prices(...) -> pd.DataFrame:
       """Fetch adjusted closes for the requested tickers."""

       # Validate tickers before attempting download
       validate_tickers(tickers)

       # Rest of existing code...
   ```

3. Add validation to CLI (backtest.py:324)
   ```python
   def main(argv: List[str]) -> None:
       args = parse_args(argv)

       tickers = args.tickers

       # Validate tickers early
       try:
           validate_tickers(tickers)
           validate_tickers([args.benchmark])
       except ValueError as e:
           raise SystemExit(f"Ticker validation failed:\n{e}")

       # Rest of existing code...
   ```

4. Add validation to app.py (app.py:251-254)
   ```python
   if run_backtest:
       # Validate inputs
       if not all(tickers):
           st.error("❌ Please enter all ticker symbols")
           st.stop()

       # Validate ticker format
       from backtest import validate_tickers
       try:
           validate_tickers(tickers)
           validate_tickers(benchmarks)
       except ValueError as e:
           st.error(f"❌ {str(e)}")
           st.stop()

       # Rest of existing code...
   ```

#### Testing Requirements:
- [ ] Test valid tickers: AAPL, MSFT, VWRA.L, ^GSPC
- [ ] Test invalid tickers: "", "123", "TOOLONGticker", "A@B"
- [ ] Test mixed valid/invalid lists
- [ ] Test case insensitivity (aapl → AAPL)
- [ ] Test special formats (.L suffix, ^ prefix)

**Acceptance Criteria**:
- ✅ Invalid tickers rejected before API call
- ✅ Clear error messages for each validation failure
- ✅ Common ticker formats supported

---

### Task 1.5: Date Format Validation 🟡 MEDIUM
**Priority**: Medium
**Effort**: 2 hours
**Files**: `backtest.py`
**Issue**: 3.4 - Hardcoded Date Format

#### Implementation Steps:

1. Create date validation function (backtest.py, after imports)
   ```python
   def validate_date_string(date_str: str) -> str:
       """Validate and normalize date string to YYYY-MM-DD format.

       Args:
           date_str: Date string in various formats

       Returns:
           Normalized date string in YYYY-MM-DD format

       Raises:
           argparse.ArgumentTypeError: If date format is invalid
       """
       try:
           # Try parsing with pandas (accepts many formats)
           dt = pd.Timestamp(date_str)

           # Check if date is not too far in the past
           if dt.year < 1970:
               raise argparse.ArgumentTypeError(
                   f"Date too far in the past: {date_str} (minimum: 1970-01-01)"
               )

           # Check if date is not in the future
           if dt > pd.Timestamp.today():
               raise argparse.ArgumentTypeError(
                   f"Date is in the future: {date_str}"
               )

           # Return normalized format
           return dt.strftime("%Y-%m-%d")

       except (ValueError, TypeError) as e:
           raise argparse.ArgumentTypeError(
               f"Invalid date format: '{date_str}'\n"
               f"Expected format: YYYY-MM-DD (e.g., 2020-01-01)\n"
               f"Error: {str(e)}"
           )
   ```

2. Update argparse (backtest.py:64-72)
   ```python
   parser.add_argument(
       "--start",
       type=validate_date_string,
       default="2018-01-01",
       help="Backtest start date (YYYY-MM-DD)",
   )
   parser.add_argument(
       "--end",
       type=validate_date_string,
       default=pd.Timestamp.today().strftime("%Y-%m-%d"),
       help="Backtest end date (YYYY-MM-DD)",
   )
   ```

3. Add date range validation (backtest.py:324)
   ```python
   def main(argv: List[str]) -> None:
       args = parse_args(argv)

       # Validate date range
       start_dt = pd.Timestamp(args.start)
       end_dt = pd.Timestamp(args.end)

       if start_dt >= end_dt:
           raise SystemExit(
               f"Invalid date range: start ({args.start}) must be before end ({args.end})"
           )

       if (end_dt - start_dt).days < 30:
           logger.warning(
               f"Short backtest period: {(end_dt - start_dt).days} days. "
               f"Results may be unreliable for periods < 30 days."
           )

       # Rest of existing code...
   ```

#### Testing Requirements:
- [ ] Test valid formats: "2020-01-01", "2020/01/01", "Jan 1, 2020"
- [ ] Test invalid formats: "2020-13-01", "abc", "01-01-2020"
- [ ] Test future dates (should fail)
- [ ] Test dates before 1970 (should fail)
- [ ] Test start >= end (should fail)
- [ ] Test short date ranges (< 30 days warning)

**Acceptance Criteria**:
- ✅ Multiple date formats accepted and normalized
- ✅ Future dates rejected
- ✅ Invalid date ranges rejected
- ✅ Warning for short backtests

---

## Phase 2: Code Quality & Organization (Week 2)

**Goal**: Improve code maintainability and organization
**Effort**: 16-20 hours

### Task 2.1: Refactor app.py into Modules 🟡 MEDIUM
**Priority**: Medium
**Effort**: 6 hours
**Files**: `app.py` → `app/` directory
**Issue**: 3.5 - Large app.py File

#### Implementation Steps:

1. Create module structure
   ```
   app/
   ├── __init__.py           # Package init
   ├── main.py               # Streamlit entry point (100 lines)
   ├── config.py             # Constants and configuration (50 lines)
   ├── presets.py            # Portfolio and date presets (100 lines)
   ├── ui_components.py      # Reusable UI elements (150 lines)
   ├── charts.py             # Chart generation (200 lines)
   └── validation.py         # Input validation (100 lines)
   ```

2. Create `app/config.py`
   ```python
   """Configuration constants for the backtester UI."""

   from datetime import datetime

   # UI Configuration
   PAGE_TITLE = "ETF Backtester"
   PAGE_ICON = "📈"
   LAYOUT = "wide"

   # Limits
   MAX_TICKERS = 10
   MIN_TICKERS = 1
   MAX_BENCHMARKS = 3
   MIN_BENCHMARKS = 1

   # Defaults
   DEFAULT_CAPITAL = 100_000
   DEFAULT_CAPITAL_MIN = 1_000
   DEFAULT_CAPITAL_MAX = 10_000_000
   DEFAULT_START_DATE = datetime(2018, 1, 1)

   # Chart Configuration
   PORTFOLIO_COLOR = "#1f77b4"
   BENCHMARK_COLORS = ['#9467bd', '#e377c2', '#bcbd22']
   BENCHMARK_DASH_STYLES = ['dash', 'dot', 'dashdot']

   # Rolling Windows
   ROLLING_WINDOWS = [30, 90, 180]

   # Metric Labels
   METRIC_LABELS = {
       "ending_value": "Ending Value",
       "total_return": "Total Return",
       "cagr": "CAGR",
       "volatility": "Volatility",
       "sharpe_ratio": "Sharpe Ratio",
       "sortino_ratio": "Sortino Ratio",
       "max_drawdown": "Max Drawdown"
   }
   ```

3. Create `app/presets.py`
   ```python
   """Portfolio and date range presets."""

   from datetime import datetime, timedelta
   from typing import Dict, List

   def get_portfolio_presets() -> Dict[str, Dict[str, any]]:
       """Return dictionary of predefined portfolio configurations."""
       return {
           "Custom (Manual Entry)": {
               "tickers": [],
               "weights": [],
               "benchmark": "VWRA.L"
           },
           "Default UK ETFs": {
               "tickers": ["VDCP.L", "VHYD.L"],
               "weights": [0.5, 0.5],
               "benchmark": "VWRA.L"
           },
           # ... rest of presets
       }

   def get_date_presets() -> Dict[str, datetime]:
       """Return dictionary of predefined date ranges."""
       today = datetime.today()
       return {
           "1Y": today - timedelta(days=365),
           "3Y": today - timedelta(days=365*3),
           "5Y": today - timedelta(days=365*5),
           "10Y": today - timedelta(days=365*10),
           "YTD": datetime(today.year, 1, 1),
           "Max": datetime(2010, 1, 1)
       }
   ```

4. Create `app/ui_components.py`
   ```python
   """Reusable UI components."""

   import streamlit as st
   import pandas as pd
   from typing import Dict, List
   from .config import METRIC_LABELS

   def render_metrics_column(summary: Dict[str, float], title: str) -> None:
       """Render a column of metrics."""
       st.markdown(f"### {title}")
       for key, value in summary.items():
           label = METRIC_LABELS.get(key, key)
           if key == "ending_value":
               st.metric(label, f"${value:,.2f}")
           elif key in ["total_return", "cagr", "volatility", "max_drawdown"]:
               st.metric(label, f"{value:.2%}")
           elif key in ["sharpe_ratio", "sortino_ratio"]:
               st.metric(label, f"{value:.3f}")
           else:
               st.metric(label, f"{value:.2f}")

   def render_portfolio_composition(tickers: List[str], weights: np.ndarray) -> None:
       """Render portfolio composition table."""
       composition_data = {
           "Ticker": tickers,
           "Weight": [f"{w:.1%}" for w in weights],
           "Normalized Weight": [f"{w:.3%}" for w in weights]
       }
       st.table(pd.DataFrame(composition_data))

   # ... more UI components
   ```

5. Create `app/charts.py`
   ```python
   """Chart generation functions."""

   import plotly.graph_objects as go
   from plotly.subplots import make_subplots
   import pandas as pd
   from typing import Dict, List
   from .config import PORTFOLIO_COLOR, BENCHMARK_COLORS, BENCHMARK_DASH_STYLES

   def create_main_dashboard(
       results: pd.DataFrame,
       all_benchmark_results: Dict[str, pd.DataFrame],
       benchmarks: List[str]
   ) -> go.Figure:
       """Create 2x2 dashboard with all main charts."""
       # Move chart creation logic here
       # ... (lines 485-633 from current app.py)
       pass

   def create_rolling_returns_chart(
       results: pd.DataFrame,
       all_benchmark_results: Dict[str, pd.DataFrame],
       benchmarks: List[str],
       windows: List[int]
   ) -> go.Figure:
       """Create rolling returns analysis chart."""
       # Move rolling returns logic here
       # ... (lines 645-700 from current app.py)
       pass
   ```

6. Create `app/main.py` (new entry point)
   ```python
   """Main Streamlit application entry point."""

   import streamlit as st
   from .config import PAGE_TITLE, PAGE_ICON, LAYOUT
   from .presets import get_portfolio_presets, get_date_presets
   from .ui_components import render_metrics_column, render_portfolio_composition
   from .charts import create_main_dashboard, create_rolling_returns_chart
   from .validation import validate_inputs, initialize_session_state

   # Page configuration
   st.set_page_config(
       page_title=PAGE_TITLE,
       page_icon=PAGE_ICON,
       layout=LAYOUT,
       initial_sidebar_state="expanded"
   )

   # Initialize session state
   initialize_session_state()

   # ... rest of main app logic using imported functions
   ```

7. Update root `app.py` (backward compatibility)
   ```python
   """
   Backward compatibility wrapper for app.py.
   The main application has been refactored into the app/ package.
   """

   import sys
   from pathlib import Path

   # Add app directory to path
   sys.path.insert(0, str(Path(__file__).parent))

   # Import and run main app
   from app.main import main

   if __name__ == "__main__":
       main()
   ```

8. Update `streamlit run` command in docs
   ```bash
   # Old way (still works)
   streamlit run app.py

   # New way
   streamlit run app/main.py
   ```

#### Testing Requirements:
- [ ] Test backward compatibility (streamlit run app.py)
- [ ] Test new structure (streamlit run app/main.py)
- [ ] Test all imports work correctly
- [ ] Test session state initialization
- [ ] Test all UI components render
- [ ] Test chart generation
- [ ] Run all existing tests (should pass unchanged)

**Acceptance Criteria**:
- ✅ App functionality unchanged
- ✅ Code is more maintainable
- ✅ Each module < 200 lines
- ✅ Backward compatible with old app.py

---

### Task 2.2: Centralize Session State Management 🟡 MEDIUM
**Priority**: Medium
**Effort**: 2 hours
**Files**: `app/validation.py` (new), `app.py`
**Issue**: 3.6 - Session State Scattered

#### Implementation Steps:

1. Create `app/validation.py`
   ```python
   """Input validation and session state management."""

   import streamlit as st
   from datetime import datetime
   from typing import Dict, Any
   from .config import DEFAULT_START_DATE, DEFAULT_CAPITAL

   def get_session_defaults() -> Dict[str, Any]:
       """Get default values for session state."""
       return {
           'selected_portfolio': "Custom (Manual Entry)",
           'num_tickers': 2,
           'start_date': DEFAULT_START_DATE,
           'end_date': datetime.today(),
           'capital': DEFAULT_CAPITAL,
           'use_cache': True,
           'num_benchmarks': 1,
           'preset_tickers': [],
           'preset_weights': [],
           'preset_benchmark': "VWRA.L",
       }

   def initialize_session_state() -> None:
       """Initialize all session state variables with defaults."""
       defaults = get_session_defaults()

       for key, value in defaults.items():
           if key not in st.session_state:
               st.session_state[key] = value

   def update_portfolio_preset(preset_name: str, preset_config: Dict) -> None:
       """Update session state when portfolio preset changes."""
       if preset_name == "Custom (Manual Entry)":
           # Don't override for custom
           return

       st.session_state.num_tickers = len(preset_config["tickers"])
       st.session_state.preset_tickers = preset_config["tickers"]
       st.session_state.preset_weights = preset_config["weights"]
       st.session_state.preset_benchmark = preset_config["benchmark"]

   def validate_backtest_inputs(
       tickers: List[str],
       benchmarks: List[str],
       start_date: datetime,
       end_date: datetime
   ) -> tuple[bool, str]:
       """Validate all backtest inputs.

       Returns:
           (is_valid, error_message)
       """
       # Check tickers
       if not all(tickers):
           return False, "Please enter all ticker symbols"

       # Check benchmarks
       if not benchmarks or not all(benchmarks):
           return False, "Please enter at least one valid benchmark ticker"

       # Check date range
       if start_date >= end_date:
           return False, "Start date must be before end date"

       # Check minimum date range
       if (end_date - start_date).days < 7:
           return False, "Date range must be at least 7 days"

       return True, ""
   ```

2. Replace scattered session state code in app.py
   ```python
   # Old code (lines 76-88, 200-204)
   if 'selected_portfolio' not in st.session_state:
       st.session_state.selected_portfolio = selected_portfolio
   # ... more scattered initializations

   # New code (single call at top of main.py)
   from .validation import initialize_session_state
   initialize_session_state()
   ```

3. Replace validation code (app.py:251-262)
   ```python
   # Old code
   if not all(tickers):
       st.error("❌ Please enter all ticker symbols")
       st.stop()
   # ... more validation

   # New code
   from .validation import validate_backtest_inputs

   is_valid, error_msg = validate_backtest_inputs(
       tickers, benchmarks, start_date, end_date
   )
   if not is_valid:
       st.error(f"❌ {error_msg}")
       st.stop()
   ```

#### Testing Requirements:
- [ ] Test session state initialization on first load
- [ ] Test session state persistence across reruns
- [ ] Test validation with valid inputs
- [ ] Test validation with each invalid input type
- [ ] Test preset updates to session state

**Acceptance Criteria**:
- ✅ All session state in one place
- ✅ All validation in one place
- ✅ Easier to test and maintain

---

### Task 2.3: Extract Magic Numbers to Constants 🟢 LOW
**Priority**: Low
**Effort**: 1 hour
**Files**: `backtest.py`, `app.py`, `plot_backtest.py`
**Issue**: 4.2 - Magic Numbers

#### Implementation Steps:

1. Create constants in backtest.py (after imports)
   ```python
   # Financial constants
   TRADING_DAYS_PER_YEAR = 252
   HOURS_PER_DAY = 24
   SECONDS_PER_HOUR = 3600

   # Cache configuration
   DEFAULT_CACHE_TTL_HOURS = 24
   CACHE_VERSION = "1.0"

   # Validation limits
   MIN_YEAR = 1970
   MIN_BACKTEST_DAYS = 7
   RECOMMENDED_MIN_DAYS = 30
   MAX_TICKER_LENGTH = 10
   ```

2. Replace magic numbers in calculations (backtest.py:299-311)
   ```python
   # Old
   volatility = daily_returns.std() * np.sqrt(252)
   downside_std = downside_returns.std() * np.sqrt(252)

   # New
   volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
   downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
   ```

3. Add constants to app/config.py
   ```python
   # Already created in Task 2.1
   # Add any app-specific constants
   ```

4. Update plot_backtest.py
   ```python
   # Chart configuration
   DEFAULT_DPI = 150
   DEFAULT_STYLE = "seaborn-v0_8"
   MIN_PLOT_POINTS = 30

   # Color scheme (already defined, just formalize)
   PORTFOLIO_COLOR = "#2E86AB"
   BENCHMARK_COLOR = "#A23B72"
   POSITIVE_COLOR = "#06A77D"
   NEGATIVE_COLOR = "#D62246"
   ACTIVE_COLOR = "#8B4789"
   ```

#### Testing Requirements:
- [ ] Verify all calculations unchanged
- [ ] Test with different constant values
- [ ] Check imports work correctly

**Acceptance Criteria**:
- ✅ No hardcoded numbers in calculations
- ✅ All constants documented
- ✅ Easier to modify behavior

---

### Task 2.4: Remove Duplicate Code 🟢 LOW
**Priority**: Low
**Effort**: 2 hours
**Files**: `app.py`
**Issue**: 4.2 - Duplicate Code

#### Implementation Steps:

1. Extract metric formatting (app/ui_components.py)
   ```python
   def format_metric_value(key: str, value: float) -> str:
       """Format metric value based on metric type."""
       if key == "ending_value":
           return f"${value:,.2f}"
       elif key in ["total_return", "cagr", "volatility", "max_drawdown"]:
           return f"{value:.2%}"
       elif key in ["sharpe_ratio", "sortino_ratio"]:
           return f"{value:.3f}"
       else:
           return f"{value:.2f}"

   def render_metric(key: str, value: float, label: str = None) -> None:
       """Render a single metric with appropriate formatting."""
       if label is None:
           from .config import METRIC_LABELS
           label = METRIC_LABELS.get(key, key)

       formatted_value = format_metric_value(key, value)
       st.metric(label, formatted_value)
   ```

2. Replace duplicate metric rendering code
   ```python
   # Old code (repeated 3 times in app.py)
   for key, value in portfolio_summary.items():
       label = metric_labels.get(key, key)
       if key == "ending_value":
           st.metric(label, f"${value:,.2f}")
       elif key in ["total_return", "cagr", ...]:
           # ... lots of duplicate code

   # New code
   from .ui_components import render_metric

   for key, value in portfolio_summary.items():
       render_metric(key, value)
   ```

3. Extract delta calculation logic
   ```python
   def calculate_relative_metrics(
       portfolio: Dict[str, float],
       benchmark: Dict[str, float]
   ) -> Dict[str, float]:
       """Calculate relative performance metrics."""
       return {
           'excess_return': portfolio['total_return'] - benchmark['total_return'],
           'excess_cagr': portfolio['cagr'] - benchmark['cagr'],
           'excess_sharpe': portfolio['sharpe_ratio'] - benchmark['sharpe_ratio'],
           'excess_volatility': portfolio['volatility'] - benchmark['volatility'],
           'excess_sortino': portfolio['sortino_ratio'] - benchmark['sortino_ratio'],
       }

   def render_relative_metrics(relative: Dict[str, float]) -> None:
       """Render relative performance metrics with delta indicators."""
       st.metric("Excess Return", f"{relative['excess_return']:.2%}",
                delta=f"{relative['excess_return']:.2%}",
                delta_color="normal")
       st.metric("Excess CAGR", f"{relative['excess_cagr']:.2%}",
                delta=f"{relative['excess_cagr']:.2%}",
                delta_color="normal")
       st.metric("Volatility Diff", f"{relative['excess_volatility']:.2%}",
                delta=f"{relative['excess_volatility']:.2%}",
                delta_color="inverse")
       st.metric("Sharpe Difference", f"{relative['excess_sharpe']:.3f}",
                delta=f"{relative['excess_sharpe']:.3f}",
                delta_color="normal")
       st.metric("Sortino Difference", f"{relative['excess_sortino']:.3f}",
                delta=f"{relative['excess_sortino']:.3f}",
                delta_color="normal")
   ```

#### Testing Requirements:
- [ ] Test metric formatting for all types
- [ ] Test delta calculations
- [ ] Verify UI unchanged
- [ ] Test with edge cases (0, negative, very large)

**Acceptance Criteria**:
- ✅ No duplicate formatting logic
- ✅ Consistent behavior across all metric displays
- ✅ Easier to update formatting in one place

---

### Task 2.5: Add Logging to plot_backtest.py 🟢 LOW
**Priority**: Low
**Effort**: 1 hour
**Files**: `plot_backtest.py`
**Issue**: 4.1 - Logging Consistency

#### Implementation Steps:

1. Add logging setup (plot_backtest.py:12-28)
   ```python
   from __future__ import annotations

   import argparse
   import logging
   from pathlib import Path

   import matplotlib.pyplot as plt
   import matplotlib.ticker as mticker
   import pandas as pd

   # Configure logging
   logging.basicConfig(
       level=logging.INFO,
       format="%(asctime)s - %(levelname)s - %(message)s",
       datefmt="%Y-%m-%d %H:%M:%S",
   )
   logger = logging.getLogger(__name__)
   ```

2. Replace print statements with logger
   ```python
   # Old (line 313)
   print(f"Saved dashboard to {args.output}_dashboard.png")

   # New
   logger.info(f"Saved dashboard to {args.output}_dashboard.png")

   # Old (lines 345-347)
   print(f"Saved {len(saved_files)} plots:")
   for filepath in saved_files:
       print(f"  - {filepath}")

   # New
   logger.info(f"Saved {len(saved_files)} plots:")
   for filepath in saved_files:
       logger.info(f"  - {filepath}")
   ```

3. Add logging for key operations
   ```python
   def main() -> None:
       args = parse_args()

       logger.info(f"Loading data from {args.csv}")
       df = pd.read_csv(args.csv)

       logger.info(f"Data loaded: {len(df)} rows, date range: {df.index[0]} to {df.index[-1]}")

       # ... rest of code
   ```

#### Testing Requirements:
- [ ] Test logging output format
- [ ] Test different log levels
- [ ] Verify no print statements remain

**Acceptance Criteria**:
- ✅ Consistent logging across all modules
- ✅ Structured log messages
- ✅ Easy to redirect output

---

## Phase 3: Performance & Advanced Features (Week 3)

**Goal**: Optimize performance and add advanced functionality
**Effort**: 12-14 hours

### Task 3.1: Batch Benchmark Downloads 🟡 MEDIUM
**Priority**: Medium
**Effort**: 3 hours
**Files**: `app.py`, `backtest.py`
**Issue**: 3.1 - Inefficient Multiple Benchmark Downloads

#### Implementation Steps:

1. Modify `download_prices()` to return dict for multiple tickers
   ```python
   def download_prices(
       tickers: List[str],
       start: str,
       end: str,
       use_cache: bool = True,
       cache_ttl_hours: int = 24,
       return_dict: bool = False
   ) -> pd.DataFrame | Dict[str, pd.Series]:
       """
       Fetch adjusted closes for the requested tickers.

       Args:
           tickers: List of ticker symbols
           start: Start date (YYYY-MM-DD)
           end: End date (YYYY-MM-DD)
           use_cache: Whether to use cached data
           cache_ttl_hours: Cache time-to-live in hours
           return_dict: If True, return dict of {ticker: Series}, else DataFrame

       Returns:
           DataFrame with all tickers or dict of Series per ticker
       """
       # Existing download logic...

       if return_dict:
           return {ticker: prices[ticker] for ticker in prices.columns}
       return prices
   ```

2. Update app.py to batch download (app.py:273-293)
   ```python
   # Old code - downloads individually
   for bench in benchmarks:
       bench_data = download_prices([bench], ...)

   # New code - batch download
   with st.spinner("Downloading price data..."):
       try:
           # Download portfolio tickers
           portfolio_prices = download_prices(
               tickers,
               start_date.strftime("%Y-%m-%d"),
               end_date.strftime("%Y-%m-%d"),
               use_cache=use_cache
           )

           # Download all benchmarks in one call
           all_benchmark_prices = download_prices(
               benchmarks,
               start_date.strftime("%Y-%m-%d"),
               end_date.strftime("%Y-%m-%d"),
               use_cache=use_cache,
               return_dict=True  # Get dict of Series
           )

           # Primary benchmark for backward compatibility
           benchmark_prices = all_benchmark_prices[benchmarks[0]]

           st.success(
               f"✅ Downloaded data for {len(tickers)} portfolio ticker(s) "
               f"and {len(benchmarks)} benchmark(s)"
           )

       except Exception as e:
           st.error(f"❌ Error downloading data: {str(e)}")
           st.stop()
   ```

3. Update cache logic to handle batches
   ```python
   def download_prices(...):
       # For multiple tickers, try individual cache first
       if use_cache and len(tickers) > 1:
           cached_results = {}
           uncached_tickers = []

           for ticker in tickers:
               cache_path = get_cache_path([ticker], start, end)
               cached_data = load_cached_prices(cache_path, cache_ttl_hours)

               if cached_data is not None:
                   cached_results[ticker] = cached_data[ticker]
               else:
                   uncached_tickers.append(ticker)

           # Download only uncached tickers
           if uncached_tickers:
               logger.info(f"Downloading {len(uncached_tickers)} uncached ticker(s)")
               new_data = _download_from_yfinance(uncached_tickers, start, end)

               # Cache individually
               for ticker in uncached_tickers:
                   cache_path = get_cache_path([ticker], start, end)
                   save_cached_prices(cache_path, new_data[[ticker]])
                   cached_results[ticker] = new_data[ticker]

           # Combine cached and new data
           return pd.DataFrame(cached_results)

       # Original single-ticker or no-cache logic...
   ```

#### Testing Requirements:
- [ ] Test batch download of 2-3 benchmarks
- [ ] Test mixed cache hits/misses
- [ ] Test performance improvement
- [ ] Test backward compatibility
- [ ] Verify cache works correctly for batches

**Acceptance Criteria**:
- ✅ Single API call for multiple benchmarks
- ✅ Faster than sequential downloads
- ✅ Cache still works efficiently
- ✅ No regression in functionality

---

### Task 3.2: Add Minimum Data Validation 🟡 MEDIUM
**Priority**: Medium
**Effort**: 2 hours
**Files**: `plot_backtest.py`, `backtest.py`
**Issue**: 3.8 - No Minimum Data Validation

#### Implementation Steps:

1. Add validation to plot_backtest.py (plot_backtest.py:287-305)
   ```python
   def main() -> None:
       args = parse_args()

       logger.info(f"Loading data from {args.csv}")
       df = pd.read_csv(args.csv)

       if "date" not in df.columns:
           raise SystemExit("CSV missing 'date' column; run backtest.py with --output")

       df = df.set_index(pd.to_datetime(df["date"]))

       # Validate data quantity
       if len(df) < 2:
           raise SystemExit(
               f"Insufficient data: only {len(df)} row(s) found.\n"
               f"Need at least 2 data points for plotting."
           )

       if len(df) < 30:
           logger.warning(
               f"Limited data: only {len(df)} rows. "
               f"Charts may not be meaningful with < 30 data points."
           )

       # Verify required columns
       required_cols = ["portfolio_value", "benchmark_value", "portfolio_return",
                       "benchmark_return", "active_return"]
       missing_cols = [col for col in required_cols if col not in df.columns]
       if missing_cols:
           raise SystemExit(
               f"CSV missing required columns: {', '.join(missing_cols)}\n"
               f"Make sure the CSV was generated by backtest.py"
           )

       # Validate data quality
       if df[required_cols].isna().all().any():
           cols_all_na = df[required_cols].columns[df[required_cols].isna().all()].tolist()
           raise SystemExit(
               f"Columns contain no valid data: {', '.join(cols_all_na)}\n"
               f"Check the backtest configuration."
           )

       logger.info(f"Data validated: {len(df)} rows, date range: {df.index[0].date()} to {df.index[-1].date()}")

       # Rest of existing code...
   ```

2. Add validation to compute_metrics (backtest.py:216-284)
   ```python
   def compute_metrics(...) -> pd.DataFrame:
       """Builds the backtest table and summary columns."""

       # Existing alignment logic...

       # Validate sufficient data after alignment
       if len(aligned) < 2:
           raise ValueError(
               f"Insufficient overlapping data: only {len(aligned)} trading day(s).\n"
               f"Need at least 2 days for meaningful backtest.\n"
               f"Try using a longer date range or different tickers."
           )

       if len(aligned) < 30:
           logger.warning(
               f"Limited data: only {len(aligned)} trading days. "
               f"Statistics may be unreliable for periods < 30 days."
           )

       # Rest of existing code...
   ```

3. Add data quality checks
   ```python
   def validate_price_data(df: pd.DataFrame, tickers: List[str]) -> None:
       """Validate price data quality."""
       issues = []

       for ticker in tickers:
           series = df[ticker]

           # Check for all NaN
           if series.isna().all():
               issues.append(f"{ticker}: all values are NaN")
               continue

           # Check for excessive NaN (>50%)
           nan_pct = series.isna().sum() / len(series)
           if nan_pct > 0.5:
               issues.append(f"{ticker}: {nan_pct:.1%} missing values")

           # Check for zero/negative prices
           valid_prices = series.dropna()
           if (valid_prices <= 0).any():
               issues.append(f"{ticker}: contains zero or negative prices")

           # Check for extreme values (likely data errors)
           price_changes = valid_prices.pct_change().dropna()
           if (price_changes.abs() > 0.5).any():
               issues.append(f"{ticker}: contains extreme price changes (>50%/day)")

       if issues:
           raise ValueError(
               "Price data quality issues detected:\n" +
               "\n".join(f"  • {issue}" for issue in issues)
           )
   ```

#### Testing Requirements:
- [ ] Test with 1 row (should fail)
- [ ] Test with 10 rows (warning)
- [ ] Test with 100 rows (normal)
- [ ] Test with all NaN column
- [ ] Test with >50% NaN values
- [ ] Test with negative prices
- [ ] Test with extreme price changes

**Acceptance Criteria**:
- ✅ Clear error for insufficient data
- ✅ Warning for limited data
- ✅ Data quality issues detected
- ✅ Helpful error messages

---

### Task 3.3: Implement Parallel Downloads (Optional) 🟢 LOW
**Priority**: Low
**Effort**: 4 hours
**Files**: `backtest.py`
**Issue**: 5.2 - Potential Optimizations

#### Implementation Steps:

1. Add concurrent download function
   ```python
   from concurrent.futures import ThreadPoolExecutor, as_completed
   import threading

   # Thread-safe cache lock
   _cache_lock = threading.Lock()

   def download_prices_parallel(
       ticker_groups: List[List[str]],
       start: str,
       end: str,
       max_workers: int = 3
   ) -> pd.DataFrame:
       """Download multiple ticker groups in parallel.

       Args:
           ticker_groups: List of ticker lists to download
           start: Start date
           end: End date
           max_workers: Maximum parallel downloads

       Returns:
           Combined DataFrame with all tickers
       """
       results = {}
       errors = {}

       def download_group(group: List[str]) -> tuple[List[str], pd.DataFrame]:
           """Download single group of tickers."""
           try:
               data = download_prices(group, start, end, use_cache=True)
               return (group, data)
           except Exception as e:
               return (group, e)

       with ThreadPoolExecutor(max_workers=max_workers) as executor:
           # Submit all download tasks
           future_to_group = {
               executor.submit(download_group, group): group
               for group in ticker_groups
           }

           # Collect results as they complete
           for future in as_completed(future_to_group):
               group = future_to_group[future]
               try:
                   group_tickers, data = future.result()

                   if isinstance(data, Exception):
                       errors[tuple(group_tickers)] = data
                   else:
                       for ticker in group_tickers:
                           results[ticker] = data[ticker]

               except Exception as e:
                   errors[tuple(group)] = e

       # Check for errors
       if errors:
           error_msg = "\n".join(
               f"  • {', '.join(tickers)}: {str(err)}"
               for tickers, err in errors.items()
           )
           raise ValueError(f"Failed to download some tickers:\n{error_msg}")

       return pd.DataFrame(results)
   ```

2. Make cache operations thread-safe
   ```python
   def save_cached_prices(cache_path: Path, prices: pd.DataFrame) -> None:
       """Save price data to cache (thread-safe)."""
       with _cache_lock:
           try:
               cache_data = {
                   "data": prices,
                   "timestamp": time.time(),
                   "version": CACHE_VERSION
               }
               with open(cache_path, "wb") as f:
                   pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
               logger.info(f"Saved data to cache: {cache_path}")
           except Exception as e:
               logger.warning(f"Failed to save cache: {e}")
   ```

3. Add option to use parallel downloads
   ```python
   parser.add_argument(
       "--parallel-downloads",
       action="store_true",
       help="Use parallel downloads for faster data fetching (experimental)"
   )
   ```

#### Testing Requirements:
- [ ] Test parallel download of 3 ticker groups
- [ ] Test thread safety of cache
- [ ] Test error handling in parallel mode
- [ ] Benchmark performance improvement
- [ ] Test with high worker count

**Acceptance Criteria**:
- ✅ Faster downloads for multiple tickers
- ✅ Thread-safe cache operations
- ✅ Graceful error handling
- ✅ Optional feature (not breaking)

---

### Task 3.4: Add Comprehensive Integration Tests 🟡 MEDIUM
**Priority**: Medium
**Effort**: 5 hours
**Files**: `test_integration.py` (new)
**Issue**: 6.1 - Testing Gaps

#### Implementation Steps:

1. Create `test_integration.py`
   ```python
   """Integration tests for the backtester system."""

   import pytest
   import tempfile
   from pathlib import Path
   from datetime import datetime, timedelta
   import pandas as pd
   import numpy as np

   import backtest
   from backtest import (
       download_prices,
       compute_metrics,
       summarize,
       validate_ticker,
       validate_tickers
   )

   class TestEndToEndWorkflow:
       """Test complete workflows from start to finish."""

       def test_cli_to_csv_workflow(self, tmp_path):
           """Test CLI workflow: args → download → compute → CSV output."""
           output_file = tmp_path / "test_output.csv"

           # Simulate CLI arguments
           args = [
               "--tickers", "AAPL",
               "--weights", "1.0",
               "--benchmark", "SPY",
               "--start", "2023-01-01",
               "--end", "2023-12-31",
               "--output", str(output_file),
               "--no-cache"
           ]

           # Run backtest
           with patch('backtest.yf.download') as mock_download:
               # Mock return data
               dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
               mock_download.return_value = pd.DataFrame({
                   "AAPL": np.linspace(150, 180, len(dates)),
                   "SPY": np.linspace(400, 450, len(dates))
               }, index=dates)

               # Execute
               backtest.main(args)

           # Verify output
           assert output_file.exists()
           df = pd.read_csv(output_file)
           assert len(df) > 0
           assert "portfolio_value" in df.columns
           assert "benchmark_value" in df.columns

       def test_ui_workflow_simulation(self):
           """Simulate UI workflow: inputs → download → compute → display."""
           # This would test the app.py workflow
           # Mock streamlit components
           pass

       def test_cache_workflow(self, tmp_path):
           """Test caching workflow: download → cache → reload."""
           pass

   class TestEdgeCases:
       """Test edge cases and boundary conditions."""

       def test_single_day_backtest(self):
           """Test backtest with only 1 trading day."""
           dates = pd.date_range("2023-01-01", periods=1, freq="D")
           prices = pd.DataFrame({"AAPL": [150]}, index=dates)
           benchmark = pd.Series([400], index=dates)
           weights = np.array([1.0])

           # Should handle gracefully or raise clear error
           with pytest.raises(ValueError, match="Insufficient.*data"):
               backtest.compute_metrics(prices, weights, benchmark, 100000)

       def test_leap_year_dates(self):
           """Test date calculations across leap years."""
           # 2020 was a leap year
           dates = pd.date_range("2019-12-01", "2020-03-01", freq="B")
           values = pd.Series(np.linspace(100, 110, len(dates)), index=dates)

           stats = backtest.summarize(values, 100000)

           # Should handle Feb 29, 2020 correctly
           assert stats['cagr'] > 0
           assert not np.isnan(stats['cagr'])

       def test_extreme_drawdown(self):
           """Test with >90% drawdown."""
           dates = pd.date_range("2020-01-01", periods=100, freq="D")
           # Drop from 100 to 5 (95% drawdown)
           values = pd.Series(
               np.concatenate([
                   np.linspace(100, 5, 50),
                   np.linspace(5, 10, 50)
               ]),
               index=dates
           )

           stats = backtest.summarize(values, 100)

           assert stats['max_drawdown'] < -0.9
           assert stats['max_drawdown'] > -1.0

       def test_zero_volatility_period(self):
           """Test period with no price movement."""
           dates = pd.date_range("2020-01-01", periods=100, freq="D")
           values = pd.Series([100000] * 100, index=dates)

           stats = backtest.summarize(values, 100000)

           assert stats['volatility'] == 0.0
           assert stats['sharpe_ratio'] == 0.0
           assert stats['max_drawdown'] == 0.0

   class TestConcurrency:
       """Test concurrent operations."""

       def test_concurrent_cache_access(self, tmp_path):
           """Test multiple processes accessing cache simultaneously."""
           # Use threading to simulate concurrent access
           pass

       def test_corrupted_cache_recovery(self, tmp_path):
           """Test recovery from corrupted cache file."""
           cache_file = tmp_path / "corrupted.pkl"

           # Write corrupted data
           with open(cache_file, "wb") as f:
               f.write(b"corrupted data")

           # Should return None and handle gracefully
           result = backtest.load_cached_prices(cache_file)
           assert result is None

   class TestDataQuality:
       """Test data validation and quality checks."""

       def test_missing_data_handling(self):
           """Test handling of tickers with missing data."""
           dates = pd.date_range("2020-01-01", periods=100, freq="D")

           # 50% missing data
           prices = pd.DataFrame({
               "AAPL": [np.nan if i % 2 else 100 + i for i in range(100)]
           }, index=dates)

           # Should forward-fill or raise clear error
           pass

       def test_negative_prices(self):
           """Test rejection of negative prices."""
           dates = pd.date_range("2020-01-01", periods=10, freq="D")
           prices = pd.DataFrame({
               "BAD": [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35]
           }, index=dates)

           # Should detect and raise error
           pass
   ```

2. Add fixture for mock data
   ```python
   @pytest.fixture
   def sample_price_data():
       """Generate sample price data for testing."""
       dates = pd.date_range("2020-01-01", periods=252, freq="B")

       # Realistic price movement
       np.random.seed(42)
       returns = np.random.normal(0.0005, 0.01, 252)
       prices = 100 * (1 + returns).cumprod()

       return pd.DataFrame({
           "AAPL": prices,
           "MSFT": prices * 1.5,
           "SPY": prices * 3
       }, index=dates)
   ```

3. Run integration tests
   ```bash
   pytest test_integration.py -v --tb=short
   ```

#### Testing Requirements:
- [ ] All integration tests pass
- [ ] Edge cases handled gracefully
- [ ] Concurrency tests pass
- [ ] Data quality checks work
- [ ] Tests run in < 10 seconds

**Acceptance Criteria**:
- ✅ Comprehensive integration test suite
- ✅ Edge cases covered
- ✅ Concurrent operations tested
- ✅ Data quality validated

---

## Phase 4: Documentation & Polish (Week 3)

**Goal**: Complete documentation and final improvements
**Effort**: 6-8 hours

### Task 4.1: Update All Documentation 📚
**Priority**: Medium
**Effort**: 3 hours
**Files**: `README.md`, `CLAUDE.md`, `PROJECT_SUMMARY.md`

#### Implementation Steps:

1. Update README.md
   - Document new cache TTL feature
   - Document ticker validation
   - Update troubleshooting section
   - Add section on parallel downloads (if implemented)

2. Update CLAUDE.md
   - Document new module structure (app/)
   - Update dependency list (add plotly explicitly)
   - Document new constants
   - Update code conventions section

3. Update PROJECT_SUMMARY.md
   - Document refactoring changes
   - Update feature list
   - Add performance improvements section

4. Create CHANGELOG.md
   ```markdown
   # Changelog

   ## [Unreleased]

   ### Added
   - Cache expiration system with configurable TTL
   - Retry logic with exponential backoff for API calls
   - Ticker format validation
   - Date format validation
   - Minimum data validation for plotting
   - Parallel download option (experimental)
   - Comprehensive integration tests

   ### Changed
   - Refactored app.py into modular structure (app/ directory)
   - Centralized session state management
   - Improved error messages with actionable suggestions
   - Batch benchmark downloads for better performance

   ### Fixed
   - Stale cache data issue
   - Missing import error handling in UI
   - Duplicate code in metric rendering
   - Magic numbers replaced with constants

   ### Security
   - Documented pickle usage in cache
   - Added input validation to prevent bad data
   ```

#### Acceptance Criteria:
- ✅ All documentation updated and accurate
- ✅ New features documented
- ✅ Examples updated
- ✅ Changelog maintained

---

### Task 4.2: Add Deployment Guide 📚
**Priority**: Low
**Effort**: 2 hours
**Files**: `DEPLOYMENT.md` (new)

#### Create deployment documentation:

```markdown
# Deployment Guide

## Local Development

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run CLI
python backtest.py --help

# Run UI
streamlit run app/main.py
```

## Production Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository at share.streamlit.io
3. Configure:
   - Main file: `app/main.py`
   - Python version: 3.11
4. Deploy

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.address", "0.0.0.0"]
```

### Environment Variables

- `CACHE_TTL_HOURS`: Cache expiration time (default: 24)
- `MAX_WORKERS`: Parallel download workers (default: 3)
```

#### Acceptance Criteria:
- ✅ Deployment guide complete
- ✅ Multiple deployment options documented
- ✅ Configuration documented

---

### Task 4.3: Create GitHub Issues Template 🔧
**Priority**: Low
**Effort**: 1 hour
**Files**: `.github/ISSUE_TEMPLATE/` (new directory)

#### Create issue templates:

1. Bug report template
2. Feature request template
3. Performance issue template

#### Acceptance Criteria:
- ✅ Issue templates created
- ✅ Clear structure for reporting

---

## Testing Strategy

### Test Execution Plan

**Before each commit**:
```bash
# Run all tests
pytest -v

# Check coverage
pytest --cov=backtest --cov=app --cov-report=term-missing

# Ensure > 85% coverage
pytest --cov=backtest --cov=app --cov-report=html
```

**Integration testing**:
```bash
# Run integration tests
pytest test_integration.py -v

# Run end-to-end test
python backtest.py --tickers AAPL --weights 1.0 --benchmark SPY --start 2023-01-01 --end 2023-12-31 --output /tmp/test.csv
streamlit run app/main.py
```

**Performance testing**:
```bash
# Benchmark downloads
time python backtest.py --tickers AAPL MSFT GOOGL --benchmark SPY --no-cache
time python backtest.py --tickers AAPL MSFT GOOGL --benchmark SPY  # With cache
```

---

## Success Criteria

### Overall Goals

- [ ] All high-priority issues fixed
- [ ] Test coverage ≥ 85%
- [ ] All tests passing
- [ ] Documentation complete
- [ ] No regressions in functionality
- [ ] Performance improved (downloads 2x faster)
- [ ] Code more maintainable

### Quality Metrics

**Before**:
- Lines of code: 2,773
- Test coverage: 86.1%
- Tests: 86
- Modules: 5

**After (Target)**:
- Lines of code: ~3,200 (with new features)
- Test coverage: ≥85%
- Tests: ~110+
- Modules: 10+

---

## Risk Management

### Potential Risks

1. **Breaking Changes**
   - Mitigation: Comprehensive test suite, backward compatibility

2. **Performance Regression**
   - Mitigation: Benchmark before/after, load testing

3. **Cache Corruption**
   - Mitigation: Version checking, graceful fallback

4. **API Rate Limiting**
   - Mitigation: Retry logic, delays, batching

---

## Timeline

### Week 1 (Days 1-7)
- Day 1-2: Tasks 1.1, 1.2 (Cache + Rate Limiting)
- Day 3: Task 1.3 (Import Errors)
- Day 4-5: Tasks 1.4, 1.5 (Validation)
- Day 6-7: Testing Phase 1

### Week 2 (Days 8-14)
- Day 8-10: Task 2.1 (Refactor app.py)
- Day 11: Task 2.2 (Session State)
- Day 12-13: Tasks 2.3, 2.4, 2.5 (Cleanup)
- Day 14: Testing Phase 2

### Week 3 (Days 15-21)
- Day 15-16: Task 3.1 (Batch Downloads)
- Day 17: Task 3.2 (Data Validation)
- Day 18-19: Task 3.4 (Integration Tests)
- Day 20: Task 3.3 (Parallel - Optional)
- Day 21: Phase 4 (Documentation)

---

## Next Steps

1. Review and approve this plan
2. Create GitHub project board with all tasks
3. Set up branch: `feature/code-review-fixes`
4. Start with Phase 1, Task 1.1
5. Commit after each task completion
6. Regular testing throughout

---

**Questions?**
- Which tasks should be prioritized first?
- Should we implement parallel downloads (Task 3.3)?
- Any additional requirements or constraints?
