# File Reference - Detailed Documentation

**Purpose**: Comprehensive documentation of all project files and their functions.

This document provides detailed information about each file in the portfolio-backtester project. For a quick overview, see [CLAUDE.md](../CLAUDE.md).

---

## Web UI (`app/` package)

### app.py (43 lines)
**Backward compatibility wrapper** for refactored modular architecture
- Imports and runs `app.main.main()` from new package structure
- Maintains old entry point: `streamlit run app.py` still works
- Provides helpful error messages if imports fail
- Allows seamless transition to modular architecture without breaking existing workflows

### app/__init__.py (16 lines)
**Package initialization**
- Version information (v1.0.0)
- Exports main entry point for direct imports

### app/config.py (121 lines, 32 constants)
**Centralized configuration management**
- Page settings (title, icon, layout)
- UI limits (MAX_TICKERS=10, MAX_BENCHMARKS=3, DEFAULT_CAPITAL=100k)
- Chart colors (PORTFOLIO_COLOR, BENCHMARK_COLORS, etc.)
- Chart configuration (ROLLING_WINDOWS=[30,90,180], BENCHMARK_DASH_STYLES)
- Metric labels (METRIC_LABELS dictionary for display names)
- Default values (DEFAULT_NUM_TICKERS, DEFAULT_START_DATE, etc.)

### app/presets.py (110 lines)
**Portfolio and date presets**
- Portfolio presets: 6 pre-configured portfolios
  - Custom (Manual Entry)
  - Default UK ETFs: VDCP.L + VHYD.L vs VWRA.L
  - 60/40 US Stocks/Bonds: VOO + BND vs SPY
  - Tech Giants: AAPL + MSFT + GOOGL + AMZN vs QQQ
  - Dividend Aristocrats: JNJ + PG + KO + PEP vs SPY
  - Global Diversified: VTI + VXUS + BND vs VT
- Date presets: 6 quick-select date ranges (1Y, 3Y, 5Y, 10Y, YTD, Max)
- Functions: `get_portfolio_presets()`, `get_date_presets()`

### app/validation.py (162 lines)
**Centralized session state management** (single source of truth)
- `get_session_defaults()`: Returns dictionary of all default session state values
- `initialize_session_state()`: Initializes all session state variables at startup
- `validate_backtest_inputs()`: Validates tickers, benchmarks, dates before execution
- `normalize_weights()`: Auto-normalizes portfolio weights to sum to 1.0
- Input validation for all user inputs (tickers, dates, capital)

### app/ui_components.py (184 lines)
**Reusable UI rendering functions** (DRY principle)
- `format_metric_value()`: Format values based on metric type (currency, percentage, ratio)
- `render_metric()`: Render single metric with proper formatting
- `render_metrics_column()`: Render all metrics for portfolio/benchmark in column layout
- `render_delta_metric()`: Render delta indicator with color-coded arrow (↑/↓)
- `create_portfolio_table()`: Generate portfolio composition table
- Eliminates metric rendering duplication across the codebase

### app/charts.py (306 lines)
**Plotly chart generation functions** (interactive visualizations)
- `calculate_drawdown()`: Calculate drawdown from value series
- `calculate_active_return()`: Calculate portfolio - benchmark returns
- `calculate_rolling_returns()`: Calculate rolling returns for specified window
- `create_main_dashboard()`: Generate 2x2 dashboard with all main charts
  - Portfolio vs Benchmark Value (currency-formatted, multiple benchmarks)
  - Cumulative Returns (percentage-formatted, multiple benchmarks)
  - Active Return with colored zones (green=outperformance, red=underperformance)
  - Drawdown Over Time with max drawdown annotations
- `create_rolling_returns_chart()`: Generate rolling returns analysis (30/90/180-day)
- Consistent color scheme and formatting across all charts

### app/main.py (459 lines)
**Main application orchestration** (replaces original app.py logic)
- `main()`: Entry point called by app.py wrapper or directly
- Streamlit page configuration and layout setup
- Session state initialization via validation.py
- Sidebar input collection (tickers, weights, benchmarks, dates, capital)
- Portfolio preset handling with auto-population
- Date preset buttons for quick date selection
- Multiple benchmark support (1-3 benchmarks)
- Backtest execution workflow:
  1. Validate all inputs
  2. Download prices with caching
  3. Compute metrics for portfolio and all benchmarks
  4. Display results with delta indicators
  5. Generate and display interactive charts
  6. Provide download options (CSV, HTML)
- Expandable sections for additional benchmark comparisons
- Portfolio composition table display
- Error handling with user-friendly messages
- Run with: `streamlit run app.py` (via wrapper) or `streamlit run app/main.py` (direct)

---

## Core Engine

### backtest.py (830 lines)
**Core backtesting logic with CLI interface**
- Downloads price data via yfinance with intelligent caching (TTL-based)
- Automatic retry logic with exponential backoff for API resilience
- Comprehensive input and data quality validation
- Computes comprehensive portfolio metrics
- Uses logging for better observability
- Exports time-series data to CSV

**Phase 1 Enhancements**: Cache expiration, retry logic, input validation  
**Phase 3 Enhancements**: Batch downloads, data quality validation, minimum data checks

**Key Functions**:
- `parse_args()`: CLI argument parsing (includes --no-cache, --cache-ttl)
- `get_cache_key()`, `get_cache_path()`: Cache management with MD5 hashing
- `load_cached_prices()`: Cache I/O with TTL checking and auto-migration
- `save_cached_prices()`: Cache I/O with metadata (timestamp, version)
- `retry_with_backoff()`: Decorator for exponential backoff retry (2s→4s→8s)
- `validate_ticker()`: Individual ticker validation (returns tuple[bool, str])
- `validate_tickers()`: Batch ticker validation with aggregated errors
- `validate_date_string()`: Flexible date parsing and normalization
- **Phase 3: `validate_price_data()`**: Data quality validation (NaN%, zero/negative prices, extreme changes)
- **Phase 3: `_process_yfinance_data()`**: Helper to process and validate yfinance data
- `download_prices()`: Fetches prices with optimized batch caching (per-ticker cache checks)
- `compute_metrics()`: Calculates metrics with minimum data validation (≥2 days required)
- `summarize()`: Generates comprehensive statistics (Sharpe, Sortino, drawdown, etc.)
- `main()`: Orchestrates the backtest workflow with early validation

**File Location**: `backtest.py:199-270` (compute_metrics), `backtest.py:272-307` (summarize)

### plot_backtest.py (395 lines)
**Comprehensive visualization utility for backtest results**
- **Consistent logging** (Phase 2.5): Uses logging module instead of print()
- **Data validation** (Phase 3): Minimum data checks, quality validation
- Reads CSV output from backtest.py
- Generates four professional plots:
  1. Portfolio vs benchmark value (currency-formatted axes)
  2. Cumulative returns comparison (percentage-formatted)
  3. Active return with colored zones (outperformance/underperformance)
  4. Drawdown over time with max drawdown annotations
- Dashboard mode: single 2x2 grid with all metrics
- Professional color scheme (blue/purple palette with green/red zones)
- Customizable: --style, --dpi, --dashboard options
- Supports both interactive display and PNG export

**Phase 3 Validation**:
- Minimum 2 rows required for plotting
- Warning for < 30 data points
- All-NaN column detection
- Excessive missing data warnings (>50%)
- Logging configuration: INFO level with timestamps
- 5 logger.info() calls for key operations (data loading, chart generation, file saving)

---

## Test Suite

### tests/test_backtest.py (72 tests)
**Comprehensive unit test suite using pytest**
- Tests all major functions and edge cases
- Mocks external dependencies (yfinance) for isolation
- Covers caching, error handling, calculations, validation, and CLI

**Phase 1 Tests** (28 tests):
- Cache expiration and TTL (6 tests)
- Retry logic with exponential backoff (4 tests)
- Ticker validation for multiple formats (11 tests)
- Date validation and normalization (7 tests)

**Phase 3 Tests** (15 tests):
- Batch download optimization (5 tests)
- Data quality validation (10 tests)

**Rolling Sharpe Tests** (5 new tests):
- Rolling 12-month Sharpe ratio column creation
- NaN handling for first 252 days
- Calculation accuracy with realistic volatility
- Zero volatility edge case
- Insufficient data handling

**Test Classes** (11 total):
- `TestParseArgs`: CLI argument parsing (7 tests)
- `TestCacheFunctions`: Cache with TTL and migration (6 tests)
- `TestSummarize`: Statistics calculation (4 tests)
- `TestComputeMetrics`: Backtest computation (9 tests, +5 for rolling Sharpe)
- `TestRetryLogic`: Exponential backoff decorator (4 tests)
- `TestTickerValidation`: Ticker format validation (11 tests)
- `TestDateValidation`: Date parsing and ranges (7 tests)
- `TestDownloadPrices`: Price fetching with batch caching (9 tests)
- `TestDataValidation`: Data quality checks (10 tests)
- `TestMain`: Integration tests (2 tests)

**Run with**: `pytest tests/test_backtest.py -v`

### tests/test_app.py (64 tests)
**Comprehensive test suite for Streamlit UI**
- Tests UI workflow integration with backtest module
- Validates metric formatting and display logic
- Tests error handling and input validation
- Covers portfolio composition, chart data, and export functionality
- Complete coverage of all UI enhancements
- Mocks Streamlit components for isolated testing

**Test Classes** (14 total):
- `TestMetricLabels`: Metric display formatting
- `TestBacktestIntegration`: UI workflow with backtest.py
- `TestMetricFormatting`: Currency, percentage, ratio formatting
- `TestErrorHandling`: Invalid input scenarios
- `TestPortfolioComposition`: Table generation
- `TestChartData`: Drawdown and active return calculations
- `TestDownloadFunctionality`: CSV and PNG export
- `TestCacheToggle`: Cache enable/disable behavior
- `TestInputValidation`: Form input validation
- `TestPortfolioPresets`: Portfolio preset validation (8 tests)
- `TestDateRangePresets`: Date preset calculations (7 tests)
- `TestMultipleBenchmarks`: Multi-benchmark logic (9 tests)
- `TestDeltaIndicators`: Delta calculation and formatting (7 tests)
- `TestRollingReturns`: Rolling returns windows (8 tests)

**Run with**: `pytest tests/test_app.py -v`

### tests/test_integration.py (16 tests)
**Comprehensive integration test suite**
- Tests system integration and real-world scenarios
- Mocks yfinance for isolated, reproducible tests
- Run with: `pytest tests/test_integration.py -v`

**Test Classes** (6 total):

1. **TestEndToEndWorkflow** (1 test):
   - Cache workflow (download → cache → reload)

2. **TestEdgeCases** (6 tests):
   - Single day backtest (should fail)
   - Leap year date handling
   - Extreme drawdowns (>90%)
   - Zero volatility periods
   - Very short date ranges
   - Missing ticker data

3. **TestDataQuality** (5 tests):
   - All-NaN data rejection
   - Excessive missing data
   - Negative prices detection
   - Zero prices detection
   - Extreme price changes

4. **TestValidation** (4 tests):
   - Ticker format validation (valid/invalid)
   - Date format validation (valid/future/too old)

5. **TestStatisticalEdgeCases** (4 tests):
   - Sharpe ratio with zero volatility
   - Sortino ratio with no downside
   - CAGR calculation precision
   - Max drawdown recovery

---

## Configuration & Documentation

### requirements.txt
- Pin all Python dependencies with minimum versions
- Easy setup: `pip install -r requirements.txt`
- Includes pytest for testing

### README.md
- Comprehensive user documentation
- Quick start guide and examples
- Command-line reference
- Troubleshooting section
- Development guidelines

### CLAUDE.md
- AI assistant development guide
- Architecture overview
- Key rules and conventions
- Quick reference

### PROJECT_SUMMARY.md
- Additional project documentation

### IMPLEMENTATION_PLAN.md
- Code improvement roadmap

### IMPLEMENTATION_CHECKLIST.md
- Progress tracking (87.5% complete)

### TEST_REPORT.md
- Phase 2 validation report

### CHANGELOG.md
- Version history
- Release notes (v2.1.0 latest)

---

## Quick Reference

### File Locations by Function

**Core Logic**: `backtest.py:199-307` (compute_metrics, summarize)  
**Data Download**: `backtest.py:364-538` (download_prices, validation)  
**CLI Interface**: `backtest.py:206-260` (parse_args)  
**Caching**: `backtest.py:263-338` (cache functions)  
**Validation**: `backtest.py:93-204` (ticker, date, price validation)  
**Plotting**: `plot_backtest.py:35-293` (chart functions)  
**Web UI**: `app/main.py:75-459` (main UI orchestration)  
**Charts**: `app/charts.py` (Plotly visualizations)  
**Configuration**: `app/config.py` (all constants)  

### Testing

**Total Tests**: 184 (72 backtest + 64 UI + 32 ticker_data + 16 integration)
**Coverage**: ~88%
**Pass Rate**: 100%

**Run All Tests**: `pytest -v`
**Run Backtest Tests**: `pytest tests/test_backtest.py -v`
**Run UI Tests**: `pytest tests/test_app.py -v`
**Run Ticker Data Tests**: `pytest tests/test_ticker_data.py -v`
**Run Integration Tests**: `pytest tests/test_integration.py -v`
**Check Coverage**: `pytest --cov=backtest --cov=app --cov-report=term-missing`

---

**Last Updated**: 2025-11-15  
**For**: Portfolio Backtester v2.1.0  
**See Also**: [CLAUDE.md](../CLAUDE.md), [TESTING_GUIDE.md](TESTING_GUIDE.md), [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
