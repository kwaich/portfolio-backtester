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
├── PROJECT_SUMMARY.md    # User-facing documentation
├── CLAUDE.md            # This file - AI assistant guide
├── .gitignore           # Git ignore patterns
├── .venv/               # Python virtual environment (gitignored)
├── results/             # CSV outputs (gitignored)
└── charts/              # PNG outputs (gitignored)
```

### File Purposes

**backtest.py** (228 lines)
- Core backtesting logic with CLI interface
- Downloads price data via yfinance
- Computes portfolio metrics: value, returns, CAGR
- Exports time-series data to CSV
- Key functions:
  - `parse_args()`: CLI argument parsing
  - `download_prices()`: Fetches adjusted close prices
  - `compute_metrics()`: Calculates portfolio vs benchmark metrics
  - `summarize()`: Generates summary statistics
  - `main()`: Orchestrates the backtest workflow

**plot_backtest.py** (66 lines)
- Visualization utility for backtest results
- Reads CSV output from backtest.py
- Generates two plots:
  1. Portfolio vs benchmark value over time
  2. Active return (difference) over time
- Supports both interactive display and PNG export
- Uses matplotlib with seaborn-v0_8 style

## Development Environment Setup

### Initial Setup
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
python -m pip install numpy pandas yfinance matplotlib

# Upgrade pip if needed
./.venv/bin/python -m pip install --upgrade pip
```

### Dependencies
- **numpy**: Numerical computations (weights, calculations)
- **pandas**: Time-series data handling and manipulation
- **yfinance**: Yahoo Finance API wrapper for price data
- **matplotlib**: Plotting and visualization

## Code Conventions

### Python Style
- Uses modern Python features: `from __future__ import annotations`
- Type hints for function signatures
- Docstrings for module-level and function-level documentation
- PEP 8 compliant code style
- Line length: Generally <100 characters for readability

### Error Handling Patterns
- **Dependency checks**: Guards for missing imports with clear error messages
- **Data validation**: Raises `ValueError` for invalid inputs/data
- **System exits**: Uses `SystemExit` for CLI-level errors
- **Empty data checks**: Validates data availability before processing

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
- **Calculations**: Returns and CAGR are reasonable
- **Alignment**: Portfolio and benchmark have matching date ranges
- **Output**: CSV and PNG files are created correctly
- **Edge cases**: Empty data, single ticker, mismatched weights

### Known Edge Cases
1. **Missing data**: Tickers with limited history may cause alignment issues
2. **Date ranges**: Very short periods may produce unreliable CAGR
3. **Weight mismatch**: Length of weights must match length of tickers
4. **Network dependency**: Requires internet access for yfinance

## Important Notes for AI Assistants

### Critical Constraints
1. **No dependencies on package managers**: No requirements.txt or setup.py exists
2. **Manual pip installs**: Dependencies installed via direct pip commands
3. **Network required**: yfinance needs internet access
4. **Data quality**: Yahoo Finance data may have gaps or errors
5. **Results folder**: Outputs go to gitignored directories

### When Making Changes

**DO**:
- Preserve existing error handling patterns
- Maintain backward compatibility with existing CSV format
- Keep imports at module level with guards for missing deps
- Use descriptive variable names matching existing style
- Test with multiple ticker combinations
- Validate weight normalization works correctly

**DON'T**:
- Remove or modify the CSV output columns without careful consideration
- Change default tickers without good reason (user expectation)
- Add new dependencies without updating PROJECT_SUMMARY.md
- Break the existing CLI interface
- Modify plotting style without maintaining consistency
- Commit .venv/, results/, or charts/ directories

### Security Considerations
- **No credential handling**: No API keys or secrets in this codebase
- **Public data only**: All data from public Yahoo Finance
- **No user input sanitization needed**: CLI args are type-checked by argparse
- **Path traversal**: Uses pathlib.Path which handles paths safely

### Performance Notes
- **Network latency**: yfinance downloads can be slow for many tickers
- **Memory usage**: Minimal - all data fits in memory for typical use cases
- **CPU usage**: Negligible - numpy operations are efficient
- **Date ranges**: Longer periods = more data but not problematic

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
- Main logic: backtest.py:118-166 (`compute_metrics`)
- CLI parsing: backtest.py:33-74 (`parse_args`)
- Data fetching: backtest.py:77-115 (`download_prices`)
- Plotting: plot_backtest.py:35-61 (`main`)
- Documentation: PROJECT_SUMMARY.md

### Key Dependencies
- yfinance: `yf.download()` at backtest.py:80-87
- pandas: DataFrames, Series, datetime handling
- numpy: Arrays, numerical operations
- matplotlib: Plotting infrastructure

### Default Values
- Tickers: VDCP.L, VHYD.L
- Weights: 0.5, 0.5
- Benchmark: VWRA.L
- Start: 2018-01-01
- End: Today
- Capital: 100,000
- Style: seaborn-v0_8

---

**Last Updated**: 2025-11-15
**Repository State**: 2 commits, clean working directory
**Current Branch**: claude/claude-md-mhzru0wxtf2fhp26-01BiHcWsAHGCMM49CgJT4PL2
