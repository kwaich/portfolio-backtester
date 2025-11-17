# Portfolio Backtester

A lightweight, flexible Python utility for backtesting ETF portfolio strategies against benchmarks. Download historical price data, calculate performance metrics, and visualize results with ease.

## Features

- **Portfolio Backtesting**: Test buy-and-hold strategies with customizable weights
- **Dollar-Cost Averaging (DCA)**: Simulate regular contributions at configurable intervals (daily, weekly, monthly, quarterly, yearly)
- **Rebalancing Strategies**: Test periodic rebalancing at various frequencies
- **Comprehensive Metrics**: Returns, CAGR, IRR (for DCA), Sharpe ratio, Sortino ratio, volatility, and maximum drawdown
- **Smart Data Caching**: Automatic caching with configurable TTL (time-to-live) for fresh data
- **Resilient API Calls**: Automatic retry logic with exponential backoff for network reliability
- **Input Validation**: Comprehensive validation for tickers, dates, and parameters before execution
- **Searchable Ticker Selection**: Built-in search for 50+ popular ETFs and stocks with optional Yahoo Finance integration
- **Flexible Visualization**: Generate publication-ready charts or interactive plots
- **Colorblind-Accessible Charts**: Uses Wong's colorblind-safe palette with line style differentiation for universal accessibility
- **Easy CLI**: Simple command-line interface with sensible defaults
- **Data Quality Validation**: Automatic detection of data issues (missing values, invalid prices, extreme changes)
- **Optimized Performance**: Batch downloads with per-ticker caching for faster multi-ticker operations
- **Well-Tested**: Comprehensive test coverage with 208 tests (100% pass rate)

## Quick Start

### Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

Run a backtest with default settings (VDCP.L/VHYD.L vs VWRA.L):

```bash
python backtest.py
```

### Custom Portfolio

Test a custom portfolio allocation:

```bash
python backtest.py \
  --tickers AAPL MSFT GOOGL \
  --weights 0.5 0.3 0.2 \
  --benchmark SPY \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --capital 100000 \
  --output results/my_backtest.csv
```

### Dollar-Cost Averaging (DCA)

Test a DCA strategy with regular monthly contributions:

```bash
python backtest.py \
  --tickers AAPL MSFT \
  --weights 0.6 0.4 \
  --benchmark SPY \
  --start 2018-01-01 \
  --capital 10000 \
  --dca-amount 1000 \
  --dca-freq M \
  --output results/dca_backtest.csv
```

This simulates starting with $10,000 and contributing $1,000 every month, with purchases split according to the target weights (60% AAPL, 40% MSFT).

**Note**: DCA and rebalancing are mutually exclusive. If both are specified, DCA takes precedence.

### Visualization

Generate charts from backtest results:

```bash
python plot_backtest.py --csv results/my_backtest.csv --output charts/my_backtest
```

This creates **four** PNG files with comprehensive visualizations:
- `my_backtest_value.png` - Portfolio vs benchmark value over time (currency-formatted)
- `my_backtest_returns.png` - Cumulative returns comparison (percentage-formatted)
- `my_backtest_active.png` - Active return with colored zones (outperformance/underperformance)
- `my_backtest_drawdown.png` - Drawdown over time with max drawdown annotations

**Dashboard Mode**: Create a single comprehensive dashboard with all plots:

```bash
python plot_backtest.py --csv results/my_backtest.csv --output charts/my_backtest --dashboard
```

This creates a single `my_backtest_dashboard.png` with a 2x2 grid of all metrics.

**Interactive Mode**: For interactive plots, omit the `--output` parameter:

```bash
python plot_backtest.py --csv results/my_backtest.csv
```

## Web UI (Interactive Dashboard)

For a more user-friendly experience, launch the Streamlit web interface:

```bash
streamlit run app.py
```

This opens an interactive web application in your browser with:

- **Interactive Forms**: Configure tickers, weights, benchmarks, and date ranges
- **Real-time Results**: View comprehensive metrics and charts instantly
- **Hover Tooltips**: See exact values by hovering over any point on the charts
- **Interactive Charts**: Zoom, pan, and explore data visually
- **Download Options**: Export both CSV data and interactive HTML charts
- **Data Caching**: Toggle cache for faster subsequent runs
- **No Command-Line Required**: Perfect for non-technical users

### Web UI Features

1. **Sidebar Configuration**:
   - Example portfolio presets (Default UK ETFs, 60/40, Tech Giants, Dividend Aristocrats, Global Diversified)
   - Dynamic number of portfolio tickers (1-10)
   - **Searchable Ticker Inputs**: Search and select from 50+ popular ETFs and stocks, or enter any ticker manually
   - **Yahoo Finance Integration**: Optional live search to find any ticker symbol by name or symbol
   - Auto-normalized weights
   - Multiple benchmark support (compare up to 3 benchmarks)
   - Date range presets (1Y, 3Y, 5Y, 10Y, YTD, Max) with custom date picker
   - Capital input with validation
   - Cache toggle

2. **Results Display**:
   - Side-by-side comparison of Portfolio vs Benchmark vs Relative Performance
   - Delta indicators showing outperformance/underperformance with color-coded arrows
   - All metrics: Total Return, CAGR, Volatility, Sharpe, Sortino, Max Drawdown
   - Expandable sections for additional benchmark comparisons
   - Portfolio composition table with ticker symbols, full company/fund names (fetched from Yahoo Finance), and weights

3. **Interactive Visualizations**:
   - 2x2 Dashboard: Portfolio vs Benchmark Value, Cumulative Returns, Active Return, Drawdown
   - Rolling Returns Analysis (30/90/180-day periods)
   - Rolling 12-Month Sharpe Ratio: Track risk-adjusted performance over time
   - Multiple benchmarks displayed on all charts with distinct colors and line styles
   - **Colorblind-Accessible Design**: Uses Wong's colorblind-safe palette (blue, orange, teal, pink) avoiding problematic blue-purple and red-green combinations
   - **Multiple Visual Cues**: Line styles (solid, dashed, dotted) provide differentiation beyond color alone
   - Hover tooltips for exact values at any point
   - Zoom, pan, and explore interactively

4. **Export Options**:
   - Download results as CSV
   - Download interactive charts as HTML (with hover functionality preserved)
   - View raw data in expandable table

### Usage Example

1. Run `streamlit run app.py`
2. Browser opens automatically at `http://localhost:8501`
3. Configure your backtest in the sidebar:
   - Select "Tech Giants" from Example Portfolio dropdown (or choose Custom)
   - Tickers auto-populate: AAPL, MSFT, GOOGL, AMZN
   - Weights: 0.25 each (auto-populated)
   - Benchmark: QQQ (auto-populated)
   - Add additional benchmarks if desired (e.g., SPY, VTI)
   - Click "5Y" preset button for 5-year backtest
   - Capital: $100,000
4. Click "Run Backtest"
5. View results with delta indicators showing outperformance
6. Explore interactive charts including rolling returns analysis
7. Expand additional benchmark sections for detailed comparisons
8. Download data and charts

### Searchable Ticker Feature

The web UI includes a powerful ticker search feature:

1. **Direct Entry**: Simply type any ticker symbol (e.g., AAPL, MSFT, VWRA.L) in the text field
2. **Search Button**: Click the 🔍 Search button to:
   - Search from 50+ curated popular ETFs and stocks
   - Optionally search Yahoo Finance for any ticker (if API is available)
   - Browse results by company/fund name or ticker symbol
3. **Click to Select**: Click any result to instantly populate the ticker field

**Curated Tickers Include**:
- Global ETFs: VWRA.L, VT, ACWI
- US ETFs: SPY, VOO, VTI, QQQ
- International ETFs: VXUS, VEA, VWO
- European ETFs: VEUR.L, VGK
- Fixed Income: VDCP.L, VHYD.L, AGG, BND
- Sector ETFs: XLK, XLF, XLE
- Popular Stocks: AAPL, MSFT, GOOGL, AMZN, TSLA, and more

**Notes**:
- Yahoo Finance search may be rate-limited or blocked. The app will automatically fall back to the curated list.
- **Ticker names** are fetched dynamically from Yahoo Finance when displaying results. This ensures you always see accurate, up-to-date company/fund names for any ticker.

## Command-Line Options

### backtest.py

| Option | Default | Description |
|--------|---------|-------------|
| `--tickers` | VDCP.L VHYD.L | Portfolio ticker symbols (validated before download) |
| `--weights` | 0.5 0.5 | Portfolio weights (auto-normalized) |
| `--benchmark` | VWRA.L | Benchmark ticker for comparison (validated) |
| `--start` | 2018-01-01 | Backtest start date (flexible formats: YYYY-MM-DD, YYYY/MM/DD) |
| `--end` | Today | Backtest end date (flexible formats accepted) |
| `--capital` | 100000 | Initial capital |
| `--output` | None | CSV file path for detailed results |
| `--cache-ttl` | 24 | Cache time-to-live in hours (configurable freshness) |
| `--no-cache` | False | Disable data caching entirely |
| `--rebalance` | None | Rebalancing frequency: D/daily, W/weekly, M/monthly, Q/quarterly, Y/yearly |
| `--dca-amount` | None | Dollar-cost averaging: amount to contribute at each interval |
| `--dca-freq` | None | DCA frequency: D/daily, W/weekly, M/monthly, Q/quarterly, Y/yearly (requires --dca-amount) |

### plot_backtest.py

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | (required) | Path to backtest CSV output |
| `--output` | None | Prefix for PNG files (shows interactive plots if omitted) |
| `--style` | seaborn-v0_8 | Matplotlib style for plots |
| `--dpi` | 150 | Output DPI for PNG files |
| `--dashboard` | False | Create single dashboard instead of individual plots |

## Performance Metrics

The backtester calculates the following metrics:

- **Ending Value**: Final portfolio/benchmark value
- **Total Return**: Overall return over the period (calculated as (value - total_contributions) / total_contributions for DCA)
- **CAGR**: Compound Annual Growth Rate (approximation for DCA strategies)
- **IRR**: Internal Rate of Return (calculated for DCA strategies with multiple contributions, using time-weighted cashflows for accuracy)
- **Volatility**: Annualized standard deviation of returns (252 trading days; for DCA, contribution impacts are excluded)
- **Sharpe Ratio**: Risk-adjusted return (assuming 0% risk-free rate; uses IRR for DCA when available)
- **Sortino Ratio**: Return relative to downside deviation only
- **Max Drawdown**: Largest peak-to-trough decline (for DCA, calculated on return percentage from peak)
- **Active Return**: Portfolio return minus benchmark return

### DCA Metrics (Special Handling)

For Dollar-Cost Averaging strategies, metrics are calculated with special considerations:

1. **Returns**: Based on total invested amount, not just initial capital
2. **Volatility**: Excludes the artificial "returns" from new contributions
3. **Sharpe/Sortino**: Uses true market volatility and IRR (when available)
4. **Max Drawdown**: Calculated on return percentage (gains/losses vs. contributions), not absolute value
5. **IRR**: Time-weighted internal rate of return, more accurate than CAGR for irregular cashflows
6. **Weekend/Holiday Handling**: DCA contributions scheduled for non-trading days execute on the next available trading day

## Example Output

```
======================================================================
BACKTEST RESULTS
======================================================================
Capital: $100,000.00
Time Span: 2018-01-02 → 2024-12-31
Portfolio: VDCP.L (50.0%), VHYD.L (50.0%)
Benchmark: VWRA.L
----------------------------------------------------------------------

PORTFOLIO PERFORMANCE:
  Ending Value:    $     142,538.21
  Total Return:            42.54%
  CAGR:                     5.32%
  Volatility:              12.45%
  Sharpe Ratio:             0.43
  Sortino Ratio:            0.61
  Max Drawdown:           -18.32%

BENCHMARK PERFORMANCE:
  Ending Value:    $     138,291.45
  Total Return:            38.29%
  CAGR:                     4.87%
  Volatility:              14.21%
  Sharpe Ratio:             0.34
  Sortino Ratio:            0.48
  Max Drawdown:           -22.15%

RELATIVE PERFORMANCE:
  Active Return:           +4.25%
  Active CAGR:             +0.45%
======================================================================
```

## Data Caching & Reliability

### Smart Caching with TTL

Price data is automatically cached in `.cache/` with configurable expiration:

- **Cache Key**: MD5 hash of tickers and date range for unique identification
- **TTL (Time-to-Live)**: Default 24 hours, configurable via `--cache-ttl`
- **Automatic Expiration**: Stale cache files are deleted and re-downloaded
- **Format Versioning**: Automatically migrates from old cache formats
- **Corruption Handling**: Gracefully handles corrupted cache files

```bash
# Use default 24-hour cache
python backtest.py --tickers AAPL MSFT

# Configure 48-hour cache for less frequent updates
python backtest.py --tickers AAPL MSFT --cache-ttl 48

# Force fresh download (bypass cache)
python backtest.py --tickers AAPL MSFT --no-cache

# Clear all cached data
rm -rf .cache/
```

### Automatic Retry Logic

Network failures are handled gracefully with exponential backoff:

- **Max Retries**: 3 attempts per API call
- **Backoff Strategy**: 2s → 4s → 8s (exponential)
- **Detailed Logging**: Each retry attempt is logged with timing
- **Success Tracking**: Reports successful downloads after retries

This ensures reliable backtests even with intermittent network issues or API rate limits.

### Input Validation

All inputs are validated before expensive operations:

**Ticker Validation**:
- Supports standard tickers (AAPL, MSFT)
- UK tickers with exchange suffix (VWRA.L, VDCP.L)
- Indices with caret prefix (^GSPC, ^DJI)
- Currency pairs with equals suffix (EURUSD=X)
- Hyphenated tickers (BRK-B)
- Rejects invalid characters, empty strings, and all-numeric tickers

**Date Validation**:
- Flexible format parsing (YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD)
- Normalized to standard YYYY-MM-DD format
- Validates date ranges (start < end)
- Warns for short periods (< 30 days)
- Rejects future dates and dates before 1970

**Example Error Messages**:
```bash
# Invalid ticker
python backtest.py --tickers 123 INVALID@TICKER
ERROR: Ticker validation failed:
  • Ticker cannot be all numbers: 123
  • Invalid ticker format: INVALID@TICKER (use only letters, numbers, ., -, ^, =)
Valid ticker examples: AAPL, MSFT, VWRA.L, ^GSPC, EURUSD=X

# Invalid date range
python backtest.py --start 2023-12-31 --end 2023-01-01
ERROR: Invalid date range: start (2023-12-31) must be before end (2023-01-01)
```

### Data Quality Validation

Comprehensive validation ensures reliable results by detecting data issues early:

**Price Data Validation**:
- Detects all-NaN data (no prices available)
- Flags excessive missing data (>50% NaN values)
- Identifies zero or negative prices (data errors)
- Detects extreme price changes (>90% single-day moves - likely errors)
- Validates minimum data requirements (≥2 trading days)
- Warns for limited data (<30 days - statistics may be unreliable)

**Example Data Quality Error**:
```bash
# Data with quality issues
python backtest.py --tickers BADINVALID
ERROR: Price data quality issues detected:
  • BADINVALID: all values are NaN (no price data available)

# Or for data with extreme movements:
ERROR: Price data quality issues detected:
  • TICKER: contains extreme price change (99.0%/day - possible data error)
```

### Performance Optimization

**Batch Downloads with Smart Caching**:

When downloading multiple tickers (e.g., portfolio + benchmarks), the system:
- Checks cache individually for each ticker
- Downloads only uncached tickers in a single API call
- Combines cached and fresh data seamlessly
- Significantly reduces download time for repeat backtests

**Example Performance Benefit**:
```bash
# First run: Downloads AAPL, MSFT, SPY (3 API calls worth)
python backtest.py --tickers AAPL MSFT --benchmark SPY
> Downloaded 3 ticker(s)

# Second run: AAPL and MSFT cached, only downloads new ticker GOOGL
python backtest.py --tickers AAPL MSFT GOOGL --benchmark SPY
> Cache hit for AAPL
> Cache hit for MSFT
> Cache hit for SPY
> Downloading 1 uncached ticker(s): GOOGL
> Batch download complete: 4 ticker(s) (3 cached, 1 downloaded)
```

This optimization is especially beneficial when:
- Testing multiple portfolio variations with the same benchmark
- Comparing different benchmarks against the same portfolio
- Running backtests with overlapping tickers

## Project Structure

```
portfolio-backtester/
├── app.py                  # Streamlit web UI (backward compat wrapper)
├── app/                    # Modular web UI package (NEW)
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration constants (32 constants)
│   ├── presets.py          # Portfolio and date presets
│   ├── validation.py       # Input validation and session state
│   ├── ui_components.py    # Reusable UI rendering functions
│   ├── charts.py           # Plotly chart generation
│   └── main.py             # Main application orchestration
├── backtest.py             # Core backtesting engine
├── plot_backtest.py        # Visualization utility
├── tests/                  # Test suite (208 tests, ~88% coverage)
│   ├── test_backtest.py    # Unit tests for backtest.py (92 tests)
│   ├── test_app.py         # Unit tests for app.py UI (63 tests)
│   ├── test_ticker_data.py # Unit tests for ticker_data.py (32 tests)
│   └── test_integration.py # Integration tests (21 tests)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── CLAUDE.md               # AI assistant guide
├── docs/                   # Documentation directory
│   ├── FILE_REFERENCE.md         # Detailed file documentation
│   ├── TESTING_GUIDE.md          # TDD rules and test patterns
│   ├── DEVELOPER_GUIDE.md        # Development workflows
│   ├── CHANGELOG.md              # Version history
│   └── PROJECT_SUMMARY.md        # Additional documentation
├── .gitignore              # Git ignore rules
├── .venv/                  # Virtual environment (gitignored)
├── .cache/                 # Price data cache (gitignored)
├── results/                # CSV outputs (gitignored)
└── charts/                 # PNG outputs (gitignored)
```

### Modular Architecture (NEW)

The Streamlit web UI has been refactored into a clean, modular architecture:

**Benefits**:
- ✅ **Maintainable**: Each module has a single responsibility
- ✅ **Testable**: Small, focused modules are easy to test
- ✅ **Extensible**: Add new features without touching existing code
- ✅ **Professional**: Industry-standard package structure
- ✅ **DRY**: Zero code duplication (eliminated 134 duplicate lines)
- ✅ **Backward Compatible**: Old `streamlit run app.py` still works

**Module Breakdown**:
- `config.py` (121 lines): All configuration constants, colors, labels
- `presets.py` (110 lines): Portfolio and date range presets
- `validation.py` (162 lines): Input validation, session state management
- `ui_components.py` (306 lines): Reusable metric and table rendering
- `charts.py` (306 lines): Plotly chart generation functions
- `main.py` (459 lines): Application orchestration and workflow
- `app.py` (43 lines): Backward compatibility wrapper

## Development

### Running Tests

The project has comprehensive test coverage with **208 tests** achieving **100% pass rate**.

```bash
# Run all tests (208 tests: 92 backtest + 63 UI + 32 ticker_data + 21 integration)
pytest -v

# Run only backtest tests (92 tests)
pytest tests/test_backtest.py -v

# Run only UI tests (63 tests)
pytest tests/test_app.py -v

# Run only ticker data tests (32 tests)
pytest tests/test_ticker_data.py -v

# Run only integration tests (21 tests)
pytest tests/test_integration.py -v

# Run with coverage report
pytest --cov=backtest --cov=app --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=backtest --cov=app --cov-report=html
open htmlcov/index.html

# Run specific test class
pytest tests/test_backtest.py::TestSummarize -v
pytest tests/test_app.py::TestMetricLabels -v
pytest tests/test_integration.py::TestEndToEndWorkflow -v
```

### Test Coverage

**Overall Coverage**: **100% pass rate** with comprehensive test suite

| Component | Tests | Coverage | Status |
|-----------|-------|----------|--------|
| backtest.py | 92 tests | 95% | ✅ Excellent |
| app package | 63 tests | 82% | ✅ Good |
| ticker_data.py | 32 tests | - | ✅ Covered |
| Integration | 21 tests | - | ✅ Comprehensive |
| **Total** | **208 tests** | **~88%** | **✅ Production-ready** |

**Test Breakdown**:

**Backtest Engine** (92 tests):
- CLI argument parsing
- Cache functions with TTL
- Performance metrics (returns, drawdown, risk measures)
- Portfolio computation (buy & hold, rebalancing, DCA)
- Retry logic and caching
- Ticker/date validation and data downloads
- Rolling Sharpe, drawdown, and IRR edge cases

**Web UI** (63 tests):
- Metric formatting and validation flows
- Portfolio & date presets
- Multiple benchmarks and delta indicators
- Rolling returns widgets and chart data
- File downloads, caching toggle, and error paths

**Ticker Data Utilities** (32 tests):
- Curated ticker lists and formatting helpers
- Search capabilities with/without Yahoo Finance
- Edge cases for duplicate handling and malformed input

**Integration Tests** (21 tests):
- End-to-end CLI workflows
- Data quality validation
- Input validation edge cases
- Statistical sanity checks for Sharpe, Sortino, drawdown, etc.

All tests pass with **100% success rate**.

### Code Quality

The codebase follows professional conventions and best practices:
- **PEP 8 style guidelines**: Consistent formatting throughout
- **Type hints**: All function signatures have type annotations
- **Comprehensive docstrings**: Module and function-level documentation
- **Extensive error handling**: Helpful, actionable error messages
- **Modular architecture**: Clean separation of concerns (app/ package)
- **DRY principle**: Zero code duplication (134 duplicate lines eliminated)
- **Configuration management**: All magic numbers extracted to constants
- **Consistent logging**: Structured logging with timestamps across modules
- **100% backward compatibility**: Wrapper pattern for seamless migration

## Requirements

- Python 3.9+
- numpy >= 1.24.0
- pandas >= 2.0.0
- yfinance >= 0.2.0
- matplotlib >= 3.7.0
- pytest >= 7.0.0 (for testing)
- streamlit >= 1.28.0 (for web UI)
- plotly >= 5.14.0 (for interactive charts)

## Data Source

Historical price data is fetched from Yahoo Finance via the [yfinance](https://github.com/ranaroussi/yfinance) library. An internet connection is required for initial data downloads (cached data can be used offline).

## Limitations

- **Buy-and-hold only**: No rebalancing or dynamic strategies
- **No transaction costs**: Assumes zero trading fees
- **No taxes**: Does not model tax implications
- **Daily granularity**: Uses daily adjusted close prices
- **Yahoo Finance dependency**: Data quality depends on Yahoo Finance
- **No dividends toggle**: Always uses adjusted prices (includes dividends)

## Future Enhancements

Potential improvements (see [PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) for details):

- Periodic rebalancing strategies
- Transaction cost modeling
- Configuration file support (YAML/JSON)
- Multiple time period analysis
- Currency conversion for multi-currency portfolios
- Additional data sources beyond Yahoo Finance

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest -v`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is open source and available for educational and personal use.

## Troubleshooting

### Ticker Validation Errors

**Problem**: `ERROR: Ticker validation failed`

**Solutions**:
- Verify ticker format matches supported patterns
- Use `.L` suffix for London Stock Exchange (e.g., `VWRA.L`)
- Use `^` prefix for indices (e.g., `^GSPC` for S&P 500)
- Use `=X` suffix for currency pairs (e.g., `EURUSD=X`)
- Remove special characters except `.`, `-`, `^`, `=`
- Ensure tickers aren't all numeric (e.g., `123` is invalid)

**Valid Examples**: `AAPL`, `MSFT`, `VWRA.L`, `^GSPC`, `EURUSD=X`, `BRK-B`

### Date Validation Errors

**Problem**: `ERROR: Invalid date format` or `ERROR: Invalid date range`

**Solutions**:
- Use supported formats: `YYYY-MM-DD`, `YYYY/MM/DD`, or `YYYY.MM.DD`
- Ensure start date is before end date
- Use dates after 1970-01-01
- Don't use future dates
- Be aware that periods < 30 days will trigger a warning (metrics may be unreliable)

### "Missing data for tickers"

- Ticker validation passed, but no data available for date range
- Check that tickers were trading during your specified period
- Try a more recent start date (data availability varies by security)
- Verify ticker symbols are correct for your exchange

### "No overlapping data"

- Portfolio tickers and benchmark must have overlapping trading history
- Use `--start` date after all securities began trading
- Check for delisted or recently IPO'd securities
- Ensure all tickers use the correct exchange suffix

### Cache Issues

**Stale cache**:
- Default TTL is 24 hours - cache automatically expires
- Force fresh download: `--no-cache` flag
- Adjust cache lifetime: `--cache-ttl 48` (hours)

**Cache corruption**:
- Automatically detected and cleared
- Manual clear if needed: `rm -rf .cache/`
- Old cache format auto-migrates with warning

**Cache location**: `.cache/` directory in project root (gitignored)

### Network / API Issues

**Problem**: Downloads failing intermittently

**Solutions**:
- Automatic retry logic handles most transient failures (3 attempts)
- Check logs for retry messages showing backoff timing
- If all retries fail, check internet connection
- Verify Yahoo Finance is accessible
- Consider using `--cache-ttl` for longer cache to reduce API dependency

### Import Errors

**Problem**: `ModuleNotFoundError` or import failures

**Solutions**:
- Activate virtual environment: `source .venv/bin/activate`
- Reinstall all dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (requires 3.9+)
- For Streamlit: verify with `streamlit --version`
- If specific package fails, install individually: `pip install <package-name>`

## Support

For issues, questions, or suggestions:
- Open an issue in the repository
- Check existing documentation in [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)
- Review unit tests for usage examples

---

**Happy Backtesting!** 📈
