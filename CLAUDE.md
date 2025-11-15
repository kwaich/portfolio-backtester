# CLAUDE.md - AI Assistant Development Guide

This document provides comprehensive guidance for AI assistants working on the backtester repository.

## Project Overview

This is a lightweight Python-based ETF backtesting utility that allows users to:
- Compare portfolio performance against benchmarks
- Download historical price data from Yahoo Finance via yfinance
- Calculate buy-and-hold returns with static weights
- Generate visualization charts of performance metrics

**Primary Use Case**: Testing portfolio allocations (default: VDCP.L/VHYD.L vs VWRA.L benchmark)

## Repository Structure

```
backtester/
├── backtest.py           # Main backtesting engine
├── plot_backtest.py      # Visualization helper
├── test_backtest.py      # Unit tests (NEW)
├── requirements.txt      # Python dependencies (NEW)
├── README.md            # Main documentation (NEW)
├── PROJECT_SUMMARY.md    # Additional documentation
├── CLAUDE.md            # This file - AI assistant guide
├── .gitignore           # Git ignore patterns
├── .venv/               # Python virtual environment (gitignored)
├── .cache/              # Price data cache (gitignored, NEW)
├── results/             # CSV outputs (gitignored)
└── charts/              # PNG outputs (gitignored)
```

### File Purposes

**backtest.py** (~377 lines)
- Core backtesting logic with CLI interface
- Downloads price data via yfinance with intelligent caching
- Computes comprehensive portfolio metrics
- Uses logging for better observability
- Exports time-series data to CSV
- Key functions:
  - `parse_args()`: CLI argument parsing (now includes --no-cache)
  - `get_cache_key()`, `get_cache_path()`: Cache management
  - `load_cached_prices()`, `save_cached_prices()`: Cache I/O
  - `download_prices()`: Fetches adjusted close prices with caching
  - `compute_metrics()`: Calculates portfolio vs benchmark metrics
  - `summarize()`: Generates comprehensive statistics (Sharpe, Sortino, drawdown, etc.)
  - `main()`: Orchestrates the backtest workflow

**plot_backtest.py** (66 lines)
- Visualization utility for backtest results
- Reads CSV output from backtest.py
- Generates two plots:
  1. Portfolio vs benchmark value over time
  2. Active return (difference) over time
- Supports both interactive display and PNG export
- Uses matplotlib with seaborn-v0_8 style

**test_backtest.py** (~370 lines, NEW)
- Comprehensive unit test suite using pytest
- Tests all major functions and edge cases
- Mocks external dependencies (yfinance) for isolation
- Covers caching, error handling, calculations, and CLI
- Run with: `pytest test_backtest.py -v`

**requirements.txt** (NEW)
- Pin all Python dependencies with minimum versions
- Easy setup: `pip install -r requirements.txt`
- Includes pytest for testing

**README.md** (NEW)
- Comprehensive user documentation
- Quick start guide and examples
- Command-line reference
- Troubleshooting section
- Development guidelines

## Development Environment Setup

### Initial Setup
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies (RECOMMENDED)
pip install -r requirements.txt

# Or install individually
python -m pip install numpy pandas yfinance matplotlib pytest

# Upgrade pip if needed
./.venv/bin/python -m pip install --upgrade pip
```

### Dependencies
- **numpy** (>=1.24.0): Numerical computations (weights, calculations)
- **pandas** (>=2.0.0): Time-series data handling and manipulation
- **yfinance** (>=0.2.0): Yahoo Finance API wrapper for price data
- **matplotlib** (>=3.7.0): Plotting and visualization
- **pytest** (>=7.0.0): Testing framework

### Testing
```bash
# Run all tests
pytest test_backtest.py -v

# Run with coverage
pytest test_backtest.py --cov=backtest --cov-report=html

# Run specific test class
pytest test_backtest.py::TestSummarize -v
```

## Code Conventions

### Python Style
- Uses modern Python features: `from __future__ import annotations`
- Type hints for function signatures
- Docstrings for module-level and function-level documentation
- PEP 8 compliant code style
- Line length: Generally <100 characters for readability

### Error Handling Patterns
- **Dependency checks**: Guards for missing imports with clear error messages
- **Data validation**: Raises `ValueError` for invalid inputs/data with detailed context
- **System exits**: Uses `SystemExit` for CLI-level errors
- **Empty data checks**: Validates data availability before processing
- **Contextual errors**: All error messages include:
  - Relevant ticker names and date ranges
  - Specific problem description
  - Actionable suggestions for resolution
  - Available vs. missing data breakdown

### Logging
- Uses Python's `logging` module (not print statements)
- INFO level for key operations (downloads, cache hits, computation steps)
- WARNING level for cache failures (non-critical)
- Formatted timestamps for all log messages
- Clean separation of user output (print) vs. diagnostic logging

### Key Patterns Used

**DataFrame Operations**:
```python
# Forward-fill for missing data
aligned = prices.loc[start_date:].ffill().dropna()

# Calculate returns
portfolio_return = portfolio_value / capital - 1
```

**Weight Normalization**:
```python
if not np.isclose(weights.sum(), 1.0):
    weights = weights / weights.sum()
```

**Date Alignment**:
```python
# Find common start date across all series
start_date = max(first_valid_points)
combined_start = max(aligned.index[0], bench_start)
```

**Data Caching**:
```python
# Cache keyed by tickers and date range
cache_key = get_cache_key(tickers, start, end)  # MD5 hash
cache_path = Path(".cache") / f"{cache_key}.pkl"

# Load from cache if available
cached_data = load_cached_prices(cache_path)
if cached_data is not None:
    return cached_data

# Save to cache after download
save_cached_prices(cache_path, prices)
```

## New Features (Added 2025-11-15)

### Data Caching System
- **Location**: `.cache/` directory (gitignored)
- **Cache keys**: MD5 hash of sorted tickers + date range
- **Format**: Pickled pandas DataFrames
- **Behavior**:
  - Automatic cache on first download
  - Reused on subsequent identical requests
  - Bypass with `--no-cache` flag
- **Benefits**: 5-10x faster for repeated backtests
- **Functions**: `get_cache_key()`, `get_cache_path()`, `load_cached_prices()`, `save_cached_prices()`

### Enhanced Performance Metrics
Added to `summarize()` function (backtest.py:272-307):

1. **Volatility**: Annualized standard deviation (252 trading days)
   ```python
   volatility = daily_returns.std() * np.sqrt(252)
   ```

2. **Sharpe Ratio**: Risk-adjusted return (assumes 0% risk-free rate)
   ```python
   sharpe_ratio = (cagr / volatility) if volatility > 0 else 0.0
   ```

3. **Sortino Ratio**: Return vs. downside deviation only
   ```python
   downside_returns = daily_returns[daily_returns < 0]
   downside_std = downside_returns.std() * np.sqrt(252)
   sortino_ratio = (cagr / downside_std) if downside_std > 0 else 0.0
   ```

4. **Maximum Drawdown**: Largest peak-to-trough decline
   ```python
   cumulative = series / series.expanding().max()
   drawdown = (cumulative - 1).min()
   ```

### Improved Output Format
- Professional formatting with section headers
- Aligned numerical columns for readability
- Shows portfolio composition with weights
- Separate sections for portfolio, benchmark, and relative performance
- All metrics displayed with consistent precision

### Unit Test Suite
- **Coverage**: All major functions tested
- **Mocking**: External dependencies (yfinance) mocked for reliability
- **Test classes**:
  - `TestParseArgs`: CLI argument parsing
  - `TestCacheFunctions`: Cache key generation and I/O
  - `TestSummarize`: Statistics calculation
  - `TestComputeMetrics`: Backtest computation
  - `TestDownloadPrices`: Price fetching with caching
  - `TestMain`: Integration tests
- **Run**: `pytest test_backtest.py -v`

## Git Workflow

### Branch Naming Convention
- Feature branches: `claude/claude-md-<session-id>`
- Current branch: `claude/claude-md-mhzru0wxtf2fhp26-01BiHcWsAHGCMM49CgJT4PL2`

### Commit History
Recent commits show incremental development:
- `64160d1`: "Improve backtest robustness"
- `3ac63a9`: "Initial backtest scaffolding"

### Commit Message Style
- Use imperative mood ("Add feature" not "Added feature")
- Be descriptive but concise
- Focus on the "why" when the "what" isn't obvious

### Git Operations
- Always develop on designated Claude branches
- Push with: `git push -u origin <branch-name>`
- Never force push without explicit permission

## Common Development Tasks

### Adding New Metrics
1. Modify `compute_metrics()` in backtest.py:118-166
2. Add new calculated column to the returned DataFrame
3. Update `summarize()` if summary stat needed
4. Consider adding plot to plot_backtest.py

### Supporting New Data Sources
1. Modify `download_prices()` in backtest.py:77-115
2. Ensure compatibility with existing DataFrame structure
3. Handle new data format edge cases
4. Update documentation in PROJECT_SUMMARY.md

### Adding CLI Arguments
1. Update `parse_args()` in backtest.py:33-74
2. Add argument with clear help text and sensible defaults
3. Use argument in `main()` function
4. Test with various input combinations

### Improving Visualizations
1. Modify plot_backtest.py
2. Maintain seaborn-v0_8 style consistency
3. Ensure both interactive and PNG output work
4. Keep DPI at 150 for consistent quality

## Testing & Validation

### Manual Testing Workflow
```bash
# Activate environment
source .venv/bin/activate

# Test basic backtest
python backtest.py --start 2018-01-01 --end 2024-12-31 \
    --capital 100000 --weights 0.5 0.5 --benchmark VWRA.L \
    --output results/test_run.csv

# Verify CSV was created
ls -lh results/test_run.csv

# Test plotting
python plot_backtest.py --csv results/test_run.csv \
    --output charts/test_run

# Verify PNGs were created
ls -lh charts/
```

### What to Verify
- **Data downloads**: All tickers successfully fetch data
- **Caching**: Verify .cache/ directory created and used on subsequent runs
- **Calculations**: All metrics (CAGR, Sharpe, Sortino, drawdown) are reasonable
- **Alignment**: Portfolio and benchmark have matching date ranges
- **Output**: CSV and PNG files are created correctly
- **Logging**: Check log messages appear with timestamps
- **Edge cases**: Empty data, single ticker, mismatched weights
- **Tests**: Run `pytest test_backtest.py -v` to verify all tests pass

### Known Edge Cases
1. **Missing data**: Tickers with limited history may cause alignment issues
2. **Date ranges**: Very short periods may produce unreliable CAGR
3. **Weight mismatch**: Length of weights must match length of tickers
4. **Network dependency**: Requires internet access for yfinance

## Important Notes for AI Assistants

### Critical Constraints
1. **Requirements file**: Use `requirements.txt` for dependency management
2. **Network required**: yfinance needs internet access (except when using cached data)
3. **Data quality**: Yahoo Finance data may have gaps or errors
4. **Gitignored folders**: .venv/, .cache/, results/, charts/ are not committed
5. **Cache directory**: .cache/ created automatically on first run
6. **Testing**: Always run tests before committing significant changes

### When Making Changes

**DO**:
- Preserve existing error handling patterns (detailed, contextual errors)
- Maintain backward compatibility with existing CSV format
- Keep imports at module level with guards for missing deps
- Use logging module for diagnostic output, print() for user results
- Use descriptive variable names matching existing style
- Write unit tests for new functionality
- Test with multiple ticker combinations
- Validate weight normalization works correctly
- Update requirements.txt if adding dependencies
- Update README.md for user-facing changes
- Update CLAUDE.md for AI-relevant implementation details

**DON'T**:
- Remove or modify the CSV output columns without careful consideration
- Change default tickers without good reason (user expectation)
- Add new dependencies without updating requirements.txt, README.md, and CLAUDE.md
- Break the existing CLI interface (add flags, don't change existing behavior)
- Modify plotting style without maintaining consistency
- Commit .venv/, .cache/, results/, or charts/ directories
- Use print() for diagnostic/debug output (use logging instead)
- Skip writing tests for significant new features

### Security Considerations
- **No credential handling**: No API keys or secrets in this codebase
- **Public data only**: All data from public Yahoo Finance
- **No user input sanitization needed**: CLI args are type-checked by argparse
- **Path traversal**: Uses pathlib.Path which handles paths safely

### Performance Notes
- **Network latency**: yfinance downloads can be slow (mitigated by caching)
- **Cache performance**: 5-10x faster for cached data (no network calls)
- **Cache size**: Minimal - pickled DataFrames are compact
- **Memory usage**: Minimal - all data fits in memory for typical use cases
- **CPU usage**: Negligible - numpy operations are efficient
- **Date ranges**: Longer periods = more data but not problematic
- **First run**: Slower (downloads data), subsequent runs are fast (cached)

## Typical AI Assistant Workflows

### Scenario 1: User Wants Different Default Tickers
1. Modify `parse_args()` default values in backtest.py:38-39
2. Update PROJECT_SUMMARY.md example commands
3. Test with new defaults
4. Commit with message: "Change default tickers to X and Y"

### Scenario 2: Add New Performance Metric
1. Add calculation in `compute_metrics()` function
2. Add to DataFrame returned at backtest.py:156-164
3. Optionally add to `summarize()` if it's a summary stat
4. Test the calculation manually
5. Update documentation if user-facing

### Scenario 3: Enhance Plotting
1. Read plot_backtest.py to understand current structure
2. Add new subplot or modify existing plots
3. Ensure both --output and interactive modes work
4. Test with real CSV data
5. Maintain seaborn style consistency

### Scenario 4: Fix Data Alignment Bug
1. Understand the date alignment logic in `compute_metrics()`
2. Review lines 127-149 for benchmark alignment
3. Test with tickers that have different start dates
4. Ensure forward-fill logic is correct
5. Validate no data leakage (look-ahead bias)

### Scenario 5: Improve Error Messages
1. Identify where errors occur (ValueError, SystemExit)
2. Make messages more actionable
3. Include relevant context (ticker, date range, etc.)
4. Test error paths explicitly
5. Ensure errors don't expose sensitive info

## Quick Reference

### File Locations
- **Main logic**: backtest.py:199-270 (`compute_metrics`)
- **CLI parsing**: backtest.py:44-90 (`parse_args`)
- **Caching**: backtest.py:93-128 (cache helper functions)
- **Data fetching**: backtest.py:131-196 (`download_prices`)
- **Metrics**: backtest.py:272-307 (`summarize`)
- **Main flow**: backtest.py:310-373 (`main`)
- **Plotting**: plot_backtest.py:35-61 (`main`)
- **Tests**: test_backtest.py (all test classes)
- **User docs**: README.md
- **AI docs**: CLAUDE.md (this file)

### Key Dependencies
- **yfinance**: `yf.download()` at backtest.py:143-150
- **pandas**: DataFrames, Series, datetime handling
- **numpy**: Arrays, numerical operations, statistics
- **matplotlib**: Plotting infrastructure
- **pytest**: Testing framework
- **pickle**: Cache serialization
- **hashlib**: Cache key generation (MD5)
- **logging**: Diagnostic output

### Default Values
- **Tickers**: VDCP.L, VHYD.L
- **Weights**: 0.5, 0.5 (auto-normalized)
- **Benchmark**: VWRA.L
- **Start**: 2018-01-01
- **End**: Today
- **Capital**: 100,000
- **Cache**: Enabled (use --no-cache to disable)
- **Plot Style**: seaborn-v0_8

### Performance Metrics Available
- Ending Value
- Total Return
- CAGR (Compound Annual Growth Rate)
- Volatility (annualized std dev)
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown

### Common Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest
python backtest.py --tickers A B --weights 0.6 0.4 --benchmark SPY

# Plot results
python plot_backtest.py --csv results/backtest.csv --output charts/test

# Run tests
pytest test_backtest.py -v

# Clear cache
rm -rf .cache/
```

---

**Last Updated**: 2025-11-15 (Major update: caching, metrics, tests, docs)
**Repository State**: Multiple commits, comprehensive improvements
**Current Branch**: claude/claude-md-mhzru0wxtf2fhp26-01BiHcWsAHGCMM49CgJT4PL2
**Key Files**: backtest.py (377 lines), test_backtest.py (370 lines), README.md, requirements.txt
