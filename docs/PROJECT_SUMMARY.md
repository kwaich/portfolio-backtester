# Portfolio Backtester Utility

This directory contains a comprehensive ETF backtesting system with both CLI and web UI interfaces.
The system includes `backtest.py` (core engine), `app.py` (Streamlit web UI), and `plot_backtest.py` (visualization helper).

## Quick Start Options

### Option 1: Web UI (Recommended for most users)
1. Create/activate the virtualenv: `python3 -m venv .venv && source .venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch web UI: `streamlit run app.py`
4. Configure backtest in browser and view results interactively

### Option 2: Command Line
1. Create/activate the virtualenv: `python3 -m venv .venv && source .venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Run `backtest.py` to pull daily data via yfinance (with intelligent caching),
   calculate comprehensive portfolio vs. benchmark metrics, and optionally export a CSV
4. Use `plot_backtest.py` to convert the CSV into PNG charts or interactive plots
5. Run tests: `pytest -v` to verify everything works correctly

## Scripts

- `app.py` — **Backward compatibility wrapper** (43 lines) for the refactored modular
  Streamlit web UI. Imports and runs the main application from the `app/` package.
  **100% backward compatible** — old entry point still works.

- `app/` package — **Modular web UI architecture** (1,358 lines, Phase 2 refactoring).
  Professional package structure replacing monolithic app.py with clean separation of concerns.

  **Modules:**
  - `config.py` (121 lines): 32 configuration constants (colors, limits, labels)
  - `presets.py` (110 lines): Portfolio and date range presets
  - `validation.py` (162 lines): Input validation and centralized session state
  - `ui_components.py` (306 lines): Reusable UI rendering functions (DRY principle)
  - `charts.py` (306 lines): Plotly chart generation functions
  - `main.py` (459 lines): Application orchestration and workflow
  - `__init__.py` (16 lines): Package initialization

  **Key Features:**
  - **Example Portfolio Presets**: 6 pre-configured portfolios (Default UK ETFs, 60/40,
    Tech Giants, Dividend Aristocrats, Global Diversified, Custom)
  - **Date Range Presets**: Quick-select buttons for 1Y, 3Y, 5Y, 10Y, YTD, Max
  - **Multiple Benchmarks**: Compare against up to 3 benchmarks simultaneously
  - **Delta Indicators**: Color-coded arrows showing outperformance/underperformance
  - **Rolling Returns Analysis**: Interactive chart showing 30/90/180-day rolling returns
  - **Interactive Charts**: Plotly-based 2x2 dashboard with hover tooltips for exact values
  - **Comprehensive Metrics**: All performance metrics with professional formatting
  - **Export Options**: Download CSV data and interactive HTML charts
  - **Data Caching**: Toggle cache for faster subsequent runs
  - **Session State**: Centralized management with single source of truth

  **Phase 2 Improvements:**
  - **Zero Code Duplication**: Eliminated 134 duplicate lines (DRY principle)
  - **All Magic Numbers Extracted**: 32 named constants for maintainability
  - **Modular Architecture**: Easy to test, extend, and maintain
  - **Professional Structure**: Industry-standard package organization

  **UI Sections:**
  - Sidebar: Portfolio configuration, benchmark selection, date range, capital
  - Main area: Summary statistics, portfolio composition, interactive charts
  - Expandable sections: Additional benchmark comparisons, raw data table

  **Usage:**
  ```bash
  streamlit run app.py              # Via backward compat wrapper (recommended)
  streamlit run app/main.py         # Direct entry point (alternative)
  # Opens browser at http://localhost:8501
  # Select "Tech Giants" preset, click "5Y", run backtest
  # View results with interactive charts and delta indicators
  ```

- `backtest.py` — CLI to download prices (defaults: VDCP.L/VHYD.L, benchmark
  VWRA.L), compute buy-and-hold values/returns, and print comprehensive statistics.
  **Enhanced with Phase 1 reliability and validation features**.

  **Features:**
  - **Smart Caching with TTL**: `.cache/` directory with configurable expiration (default 24h)
  - **Automatic Retry Logic**: 3 attempts with exponential backoff (2s→4s→8s) for API resilience
  - **Input Validation**: Comprehensive ticker and date validation before expensive operations
  - Comprehensive metrics: CAGR, volatility, Sharpe ratio, Sortino ratio, max drawdown
  - Detailed error messages with actionable guidance and examples
  - Logging for better observability (retry attempts, cache age, warnings)
  - Professional formatted output

  **Phase 1 Enhancements:**
  - Cache expiration with automatic stale data detection
  - Graceful migration from old cache formats
  - Ticker validation (supports AAPL, VWRA.L, ^GSPC, EURUSD=X, BRK-B)
  - Flexible date parsing (YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD)
  - Date range validation (start < end, warning for < 30 days)
  - Transient network failure recovery with exponential backoff

  **CLI Options:**
  - `--tickers`: Portfolio ticker symbols (validated before download)
  - `--weights`: Portfolio weights (auto-normalized if they don't sum to 1)
  - `--benchmark`: Benchmark ticker for comparison (validated)
  - `--start`, `--end`: Date range (flexible formats, normalized to YYYY-MM-DD)
  - `--capital`: Initial capital amount
  - `--output`: CSV file path for detailed time-series export
  - `--cache-ttl`: Cache time-to-live in hours (default: 24)
  - `--no-cache`: Disable caching entirely (force fresh download)

  **Example:**
  ```bash
  source .venv/bin/activate
  python backtest.py --start 2018-01-01 --end 2024-12-31 \
      --capital 100000 --weights 0.5 0.5 --benchmark VWRA.L \
      --output results/backtest_series.csv
  ```

- `plot_backtest.py` — Enhanced visualization utility (365 lines, Phase 2 logging)
  that reads the CSV (must include a `date` column plus the columns emitted by `backtest.py`)
  and creates comprehensive performance charts.

  **Features:**
  - Creates 4 individual plots: value, returns, active return, and drawdown
  - Dashboard mode (`--dashboard`) for single comprehensive view
  - Professional color scheme with colored zones for outperformance/underperformance
  - Currency and percentage formatting on axes

  **Phase 2 Improvements:**
  - **Consistent Logging**: Uses logging module instead of print() for better observability
  - 5 logger.info() calls for key operations (data loading, chart generation, saving)
  - INFO level with timestamps matching backtest.py format
  - Max drawdown annotations on drawdown chart
  - Customizable DPI and matplotlib style

  **Usage:**
  - Individual plots: `--output charts/run` creates 4 PNG files
  - Dashboard: `--output charts/run --dashboard` creates single dashboard PNG
  - Interactive: omit `--output` to show plots interactively

- `tests/test_backtest.py` — Comprehensive pytest suite (~1,400 lines) for caching, data downloads,
  portfolio construction (buy & hold, rebalancing, DCA), IRR calculations, rolling Sharpe, and CLI flows.
  Includes 92 tests across argument parsing, validation, retry logic, metrics, and integration helpers.
  Run with `pytest tests/test_backtest.py -v`.

- `tests/test_app.py` — Streamlit UI regression suite (~1,000 lines) with 63 tests across 14 classes
  covering presets, validation, multiple benchmarks, rolling returns, downloads, and delta indicators.
  Uses extensive mocking to isolate UI logic. Run with `pytest tests/test_app.py -v`.

- `tests/test_ticker_data.py` — 32 tests validating curated ticker lists, formatting helpers, and
  Yahoo Finance search fallbacks. Run with `pytest tests/test_ticker_data.py -v`.

- `tests/test_integration.py` — 21 black-box tests exercising full CLI workflows, edge cases, and
  statistical sanity checks. Run with `pytest tests/test_integration.py -v`.

## Test Coverage

**Overall Coverage**: **~88%** with comprehensive automated tests ✅

**Test Suite Statistics**:
- Total tests: **208** (92 backtest + 63 UI + 32 ticker_data + 21 integration)
- Pass rate: **100%** (208/208 tests passing)
- Typical runtime: ~4 seconds on a modern laptop

**Coverage by Component**:
- `backtest.py`: 95% coverage (metrics, caching, DCA, IRR, rolling Sharpe)
- `app/` package: 82% coverage (presets, benchmarks, rolling returns, downloads)
- `tests/test_ticker_data.py`: exhaustive coverage of search & formatting utilities
- Integration suite: regression safety net for CLI workflows and validation flows

## Performance Metrics

The backtester now calculates a comprehensive set of performance metrics:

- **Ending Value**: Final portfolio/benchmark value
- **Total Return**: Overall percentage return
- **CAGR**: Compound Annual Growth Rate (annualized return)
- **Volatility**: Annualized standard deviation of returns (risk measure)
- **Sharpe Ratio**: Risk-adjusted return (CAGR / volatility)
- **Sortino Ratio**: Return relative to downside deviation only
- **Maximum Drawdown**: Largest peak-to-trough decline (worst loss)
- **Active Return**: Portfolio return minus benchmark return

## Data Caching

Price data is automatically cached in `.cache/` for faster repeated backtests:

- Cache files are keyed by MD5 hash of tickers + date range
- First run downloads data from Yahoo Finance (slower)
- Subsequent runs use cached data (5-10x faster)
- Use `--no-cache` flag to force fresh downloads
- Clear cache with `rm -rf .cache/` if needed

## File Structure

```
portfolio-backtester/
├── app.py                # Streamlit web UI wrapper
├── backtest.py           # Main backtesting engine
├── plot_backtest.py      # Visualization helper
├── tests/                # Automated tests (208 total)
│   ├── test_backtest.py      # Engine suite (92 tests)
│   ├── test_app.py           # UI suite (63 tests)
│   ├── test_ticker_data.py   # Ticker helper suite (32 tests)
│   └── test_integration.py   # Integration suite (21 tests)
├── requirements.txt      # Python dependencies (includes streamlit, plotly)
├── README.md             # Comprehensive user documentation
├── PROJECT_SUMMARY.md    # This file
├── CLAUDE.md             # AI assistant development guide
├── .gitignore            # Git ignore patterns
├── .venv/                # Python virtual environment (gitignored)
├── .cache/               # Price data cache (gitignored)
├── results/              # CSV outputs (gitignored)
└── charts/               # PNG/HTML outputs (gitignored)
```

## Notes

- **Dependencies**: Managed via `requirements.txt`. Install with `pip install -r requirements.txt`.
  Includes: numpy, pandas, yfinance, matplotlib, pytest, streamlit, plotly.
- **Network access**: Required for initial data downloads via yfinance; cached data can be used offline.
- **Output folders**: CSVs go to `results/`, charts to `charts/` (both gitignored).
- **Cache folder**: Downloaded data cached in `.cache/` (gitignored).
- **Testing**: Run `pytest -v` to verify all functionality (208 tests across backtest, UI, ticker data, and integration suites; 100% passing, ~88% coverage).
- **Web UI**: Run `streamlit run app.py` for browser-based interface.
- **Logging**: Diagnostic messages use Python's logging module with timestamps.
- **Error messages**: Detailed, contextual error messages with actionable guidance.
- **Documentation**: See `README.md` for comprehensive user guide, `CLAUDE.md` for AI assistant reference.

## Recent Improvements

This backtester has been significantly enhanced with:

### Core Engine Improvements
1. **Data Caching System**: Intelligent caching for 5-10x performance improvement
2. **Enhanced Metrics**: Added Sharpe ratio, Sortino ratio, volatility, max drawdown
3. **Better Error Messages**: Contextual errors with specific tickers, dates, and solutions
4. **Logging Infrastructure**: Professional logging for better observability
5. **Unit Tests**: Comprehensive test suite (370 lines, 24 test classes)

### Web UI Additions (Latest)
6. **Streamlit Web Interface**: Full-featured browser-based UI (~700 lines)
   - Example Portfolio Presets (6 pre-configured portfolios)
   - Date Range Presets (1Y, 3Y, 5Y, 10Y, YTD, Max quick-select buttons)
   - Multiple Benchmarks Support (up to 3 simultaneous benchmarks)
   - Delta Indicators (color-coded arrows showing outperformance)
   - Rolling Returns Chart (30/90/180-day analysis)
   - Interactive Plotly Charts (hover tooltips for exact values)
   - Session state management for smooth UX

7. **Comprehensive Test Suite**: **208 tests total, ~88% coverage** ✅
   - UI test suite: ~1,000 lines, 63 tests across 14 test classes
   - Engine test suite: ~1,400 lines, 92 tests across a dozen test classes
   - 100% pass rate, ~2 second runtime
   - Complete coverage of all 5 new UI features

8. **Interactive Visualizations**: Replaced static matplotlib with interactive Plotly
   - Zoom, pan, and explore charts
   - Hover tooltips showing exact values
   - Download as interactive HTML files

9. **Documentation**: Complete README.md, PROJECT_SUMMARY.md, and CLAUDE.md
   - Comprehensive TDD guidelines in CLAUDE.md
   - Test coverage statistics and goals

All improvements maintain backward compatibility with existing workflows.
