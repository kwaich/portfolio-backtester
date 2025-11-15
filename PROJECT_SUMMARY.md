# Backtest Utility

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

- `app.py` — Streamlit web UI for interactive backtesting (~700 lines). Provides a
  user-friendly browser-based interface without requiring command-line knowledge.

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
  - **Session State**: Smooth UX with persistent selections

  **UI Sections:**
  - Sidebar: Portfolio configuration, benchmark selection, date range, capital
  - Main area: Summary statistics, portfolio composition, interactive charts
  - Expandable sections: Additional benchmark comparisons, raw data table

  **Usage:**
  ```bash
  streamlit run app.py
  # Opens browser at http://localhost:8501
  # Select "Tech Giants" preset, click "5Y", run backtest
  # View results with interactive charts and delta indicators
  ```

- `backtest.py` — CLI to download prices (defaults: VDCP.L/VHYD.L, benchmark
  VWRA.L), compute buy-and-hold values/returns, and print comprehensive statistics.

  **Features:**
  - Intelligent data caching (`.cache/` directory) for 5-10x faster repeated runs
  - Comprehensive metrics: CAGR, volatility, Sharpe ratio, Sortino ratio, max drawdown
  - Detailed error messages with actionable guidance
  - Logging for better observability
  - Professional formatted output

  **CLI Options:**
  - `--tickers`: Portfolio ticker symbols (space-separated)
  - `--weights`: Portfolio weights (auto-normalized if they don't sum to 1)
  - `--benchmark`: Benchmark ticker for comparison
  - `--start`, `--end`: Date range (YYYY-MM-DD format)
  - `--capital`: Initial capital amount
  - `--output`: CSV file path for detailed time-series export
  - `--no-cache`: Disable caching (force fresh download)

  **Example:**
  ```bash
  source .venv/bin/activate
  python backtest.py --start 2018-01-01 --end 2024-12-31 \
      --capital 100000 --weights 0.5 0.5 --benchmark VWRA.L \
      --output results/backtest_series.csv
  ```

- `plot_backtest.py` — Enhanced visualization utility that reads the CSV (must include
  a `date` column plus the columns emitted by `backtest.py`) and creates comprehensive
  performance charts.

  **New Features:**
  - Creates 4 individual plots: value, returns, active return, and drawdown
  - Dashboard mode (`--dashboard`) for single comprehensive view
  - Professional color scheme with colored zones for outperformance/underperformance
  - Currency and percentage formatting on axes
  - Max drawdown annotations on drawdown chart
  - Customizable DPI and matplotlib style

  **Usage:**
  - Individual plots: `--output charts/run` creates 4 PNG files
  - Dashboard: `--output charts/run --dashboard` creates single dashboard PNG
  - Interactive: omit `--output` to show plots interactively

- `test_backtest.py` — Comprehensive unit test suite for backtest.py using pytest (~370 lines).
  Tests all major functions including caching, error handling, calculations, and CLI parsing.
  Mocks external dependencies (yfinance) for reliable testing. 24 test classes covering all
  functionality. Run with `pytest test_backtest.py -v`.

- `test_app.py` — Comprehensive unit test suite for Streamlit web UI (~426 lines). 23 tests
  across 9 test classes covering UI workflow integration, metric formatting, error handling,
  portfolio composition, chart data, download functionality, cache toggle, and input validation.
  Mocks Streamlit components for isolated testing. Run with `pytest test_app.py -v`.

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
backtester/
├── app.py                # Streamlit web UI (~700 lines, NEW)
├── backtest.py           # Main backtesting engine (~377 lines)
├── plot_backtest.py      # Visualization helper (~354 lines, ENHANCED)
├── test_app.py           # UI test suite (~426 lines, NEW)
├── test_backtest.py      # Engine test suite (~370 lines)
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
- **Testing**: Run `pytest -v` to verify all functionality (47 tests: 24 backtest + 23 UI).
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

7. **UI Test Suite**: 426 lines, 23 tests across 9 test classes

8. **Interactive Visualizations**: Replaced static matplotlib with interactive Plotly
   - Zoom, pan, and explore charts
   - Hover tooltips showing exact values
   - Download as interactive HTML files

9. **Documentation**: Complete README.md, PROJECT_SUMMARY.md, and CLAUDE.md

All improvements maintain backward compatibility with existing workflows.
