# Backtest Utility

This directory contains a lightweight ETF backtester (`backtest.py`) plus a
plotting helper (`plot_backtest.py`). The workflow is:

1. Create/activate the virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
2. Install dependencies: `pip install -r requirements.txt` (numpy, pandas, yfinance, matplotlib, pytest).
3. Run `backtest.py` to pull daily data via yfinance (with intelligent caching),
   calculate comprehensive portfolio vs. benchmark metrics, and optionally export a CSV.
4. Use `plot_backtest.py` to convert the CSV into PNG charts or interactive plots.
5. Run tests: `pytest test_backtest.py -v` to verify everything works correctly.

## Scripts

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

- `plot_backtest.py` — Reads the CSV (must include a `date` column plus the columns
  emitted by `backtest.py`) and plots portfolio vs. benchmark value plus active
  return. Use `--output charts/run` to save PNGs (`run_value.png`,
  `run_active.png`); omit `--output` to show the plots interactively.

- `test_backtest.py` — Comprehensive unit test suite using pytest. Tests all major
  functions including caching, error handling, calculations, and CLI parsing. Mocks
  external dependencies for reliable testing. Run with `pytest test_backtest.py -v`.

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
├── backtest.py           # Main backtesting engine (~377 lines)
├── plot_backtest.py      # Visualization helper (66 lines)
├── test_backtest.py      # Unit test suite (~370 lines)
├── requirements.txt      # Python dependencies
├── README.md            # Comprehensive user documentation
├── PROJECT_SUMMARY.md   # This file
├── CLAUDE.md           # AI assistant development guide
├── .gitignore          # Git ignore patterns
├── .venv/              # Python virtual environment (gitignored)
├── .cache/             # Price data cache (gitignored)
├── results/            # CSV outputs (gitignored)
└── charts/             # PNG outputs (gitignored)
```

## Notes

- **Dependencies**: Managed via `requirements.txt`. Install with `pip install -r requirements.txt`.
- **Network access**: Required for initial data downloads via yfinance; cached data can be used offline.
- **Output folders**: CSVs go to `results/`, charts to `charts/` (both gitignored).
- **Cache folder**: Downloaded data cached in `.cache/` (gitignored).
- **Testing**: Run `pytest test_backtest.py -v` to verify functionality.
- **Logging**: Diagnostic messages use Python's logging module with timestamps.
- **Error messages**: Detailed, contextual error messages with actionable guidance.
- **Documentation**: See `README.md` for comprehensive user guide, `CLAUDE.md` for AI assistant reference.

## Recent Improvements

This backtester has been significantly enhanced with:

1. **Data Caching System**: Intelligent caching for 5-10x performance improvement
2. **Enhanced Metrics**: Added Sharpe ratio, Sortino ratio, volatility, max drawdown
3. **Better Error Messages**: Contextual errors with specific tickers, dates, and solutions
4. **Logging Infrastructure**: Professional logging for better observability
5. **Unit Tests**: Comprehensive test suite with 370+ lines covering all major functions
6. **Documentation**: Complete README.md and updated CLAUDE.md for developers
7. **Requirements Management**: Proper `requirements.txt` for easy setup

All improvements maintain backward compatibility with existing workflows.
