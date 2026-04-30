# CLAUDE.md - AI Assistant Development Guide

**Purpose**: Concise guidance for AI assistants working on the portfolio-backtester repository.

**Detailed Documentation**:
- **[FILE_REFERENCE.md](docs/FILE_REFERENCE.md)**: Comprehensive file-by-file documentation
- **[TESTING_GUIDE.md](docs/TESTING_GUIDE.md)**: Test-driven development rules and patterns
- **[DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)**: Development workflows and best practices

---

## Project Overview

This is a lightweight Python-based Portfolio Backtesting utility that allows users to:
- Compare portfolio performance against benchmarks
- Download historical price data from Yahoo Finance via yfinance
- Calculate buy-and-hold returns with static weights
- Generate visualization charts of performance metrics

**Primary Use Case**: Testing portfolio allocations (default: VDCP.L/VHYD.L vs VWRA.L benchmark)

**Current Status**:
- **Version**: v2.5.0-dev (Unreleased - 2025-11-17)
- **Test Coverage**: ~88% (293 tests, 293 passing / 100% ✅)
- **Progress**: Security hardening (Parquet caching), StateManager validation, UI improvements
- **Branch**: claude/review-code-01FMcnN697didaPWUPyzpyT8

---

## Repository Structure (High-Level)

```
portfolio-backtester/
├── .venv/                    # Python virtual environment (gitignored - create with: python -m venv .venv)
├── app.py                    # Streamlit web UI (backward compatibility wrapper - 43 lines)
├── app/                      # Modular web UI package (12 modules, 2,871 lines)
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration constants (32 constants)
│   ├── presets.py            # Portfolio and date presets
│   ├── validation.py         # Input validation (delegates to StateManager)
│   ├── state_manager.py      # Centralized session state management (507 lines)
│   ├── ui_components.py      # Reusable UI rendering with searchable inputs
│   ├── ticker_data.py        # Ticker search & Yahoo Finance integration (w/ caching)
│   ├── charts.py             # Plotly chart generation
│   ├── sidebar.py            # Form-based sidebar rendering (310 lines)
│   ├── results.py            # Results display functions (330 lines)
│   ├── utils.py              # URL params, error handling, progress (269 lines)
│   └── main.py               # Application orchestration (reduced to 310 lines)
├── backtest.py               # Core backtesting engine (830 lines - Phases 1 & 3)
├── plot_backtest.py          # Visualization utility (395 lines - Phases 2 & 3)
├── tests/                    # Test suite (293 tests, ~88% coverage; see docs/TESTING_GUIDE.md)
│   ├── conftest.py           # pytest configuration (with session state reset fixture)
│   ├── test_backtest.py      # Unit tests for backtest.py (93 tests)
│   ├── test_app.py           # Unit tests for app.py UI (76 tests - includes URL params)
│   ├── test_state_manager.py # Unit tests for state_manager.py (67 tests)
│   ├── test_ticker_data.py   # Unit tests for ticker_data.py (32 tests)
│   ├── test_ticker_names.py  # Placeholder for upcoming ticker name scenarios
│   └── test_integration.py   # Integration tests (21 tests)
├── requirements.txt          # Python dependencies (includes requests)
├── README.md                 # Main user documentation
├── CLAUDE.md                 # This file - AI assistant guide
└── docs/                     # Documentation directory
    ├── FILE_REFERENCE.md             # Detailed file documentation
    ├── TESTING_GUIDE.md              # TDD rules and test patterns
    ├── DEVELOPER_GUIDE.md            # Development workflows
    ├── CHANGELOG.md                  # Version history (v2.4.0: Streamlit best practices)
    └── PROJECT_SUMMARY.md            # Additional project documentation
```

**Gitignored Directories** (do not commit):
- `.venv/` - Python virtual environment
- `.cache/` - Price data cache
- `results/` - CSV outputs
- `charts/` - PNG outputs

---

## Overall Architecture

### Key Subsystems

#### 1. Core Backtesting Engine (backtest.py - 830 lines)

**Purpose**: Download prices, compute metrics, calculate statistics

**Key Functions**:
- `parse_args()`: CLI argument parsing with validation (includes DCA arguments)
- `download_prices()`: Fetch prices with batch caching & retry logic (Phase 1 & 3)
- `validate_price_data()`: Data quality validation (Phase 3)
- `compute_metrics()`: Calculate portfolio vs benchmark metrics with minimum data checks (DCA-aware)
- `_calculate_dca_portfolio()`: DCA contribution logic with weekend/holiday handling
- `summarize()`: Generate comprehensive statistics (Sharpe, Sortino, drawdown, IRR for DCA, etc.)
- `_calculate_xirr()`: Time-weighted IRR calculation using Newton-Raphson method
- `main()`: Orchestrate the backtest workflow (DCA-aware)

**Phase Enhancements**:
- **Phase 1**: Cache expiration (TTL), retry logic (exponential backoff), ticker/date validation
- **Phase 3**: Batch download optimization (per-ticker caching), data quality validation
- **DCA Phase**: Dollar-Cost Averaging support, IRR calculation, weekend/holiday handling, contribution-adjusted metrics

**Key Patterns**:
- Weight normalization (always sum to 1.0)
- Date alignment (common start date across all series)
- Forward-fill for missing data
- MD5-based caching with Parquet format (secure, compressed, cross-platform)
- Cache metadata stored as JSON (timestamp, version)
- DCA weekend handling: next available trading day
- Contribution-adjusted returns: `(value_change - contribution_change) / previous_value`

#### 2. Web UI (app/ package - 12 modules, 2,871 lines)

**Purpose**: High-performance interactive Streamlit dashboard with best practices

**Architecture** (Best Practices Refactor):
- **config.py**: Centralized configuration (32 constants)
- **presets.py**: Portfolio & date presets (6 portfolios + 6 date ranges)
- **validation.py**: Session state management & input validation
- **state_manager.py**: Centralized session state management with comprehensive type validation (507 lines)
- **ui_components.py**: Reusable UI rendering with searchable ticker inputs
- **ticker_data.py**: Ticker search with @st.cache_data (1h TTL for names, 30min for search)
- **charts.py**: Plotly chart generation (interactive visualizations)
- **sidebar.py**: Form-based sidebar rendering (310 lines) - 90% fewer reruns
- **results.py**: Results display functions (330 lines)
- **utils.py**: URL params, error handling, progress tracking (269 lines)
- **main.py**: Application orchestration (reduced to 310 lines from 764)
- **app.py**: 43-line backward compatibility wrapper

**Features**:
- **Form-Based Sidebar**: Batched inputs reduce reruns by ~90%
- **Smart Caching**: @st.cache_data for ticker names (1h) and searches (30min) - 80% fewer API calls
- **URL Sharing**: Full state preservation (tickers, weights, benchmarks, capital, dates)
- **Progress Tracking**: Visual progress bars for long operations
- **Better Error Messages**: User-friendly errors with actionable suggestions
- Portfolio presets (6 pre-configured portfolios)
- **Searchable ticker inputs**: Search from 50+ popular ETFs/stocks or use Yahoo Finance API
- **Portfolio composition table**: Displays ticker symbols, full company/fund names (fetched dynamically from Yahoo Finance), and weights
- Date range presets (1Y, 3Y, 5Y, 10Y, YTD, Max)
- Multiple benchmarks (up to 3 simultaneously)
- Delta indicators (color-coded outperformance)
- Rolling returns (30/90/180-day windows)
- Rolling 12-month Sharpe ratio chart (252-day window for risk-adjusted performance tracking)
- Rebalancing strategies (buy-and-hold, daily, weekly, monthly, quarterly, yearly)
- **Dollar-Cost Averaging (DCA)**: Regular contributions at configurable intervals (daily, weekly, monthly, quarterly, yearly)
- Logarithmic scale toggle for portfolio value charts
- CSV & HTML export

**Architecture Patterns**:
- ✅ **Caching**: @st.cache_data with TTL for expensive operations
- ✅ **Forms**: Sidebar wrapped in st.form to prevent unnecessary reruns
- ✅ **Modular Code**: 764-line main.py split into focused modules (60% reduction)
- ✅ **URL Parameters**: Shareable links with full configuration preservation
- ✅ **Error Handling**: Contextual error messages with help text
- ✅ **Progress Tracking**: ProgressTracker context manager for long operations

#### 3. Visualization (plot_backtest.py - 395 lines)

**Purpose**: Generate professional charts from backtest CSV output

**Charts Generated**:
1. Portfolio vs Benchmark Value (currency-formatted)
2. Cumulative Returns (percentage-formatted)
3. Active Return with colored zones (outperformance/underperformance)
4. Drawdown Over Time with max drawdown annotations
5. Rolling 12-Month Sharpe Ratio with reference lines (Sharpe = 1, 2)

**Modes**:
- Dashboard mode: single 2x2 grid
- Interactive display or PNG export (150 DPI)
- Consistent style: seaborn-v0_8

**Phase Enhancements**:
- **Phase 2**: Logging instead of print statements
- **Phase 3**: Data quality validation (min 2 rows, NaN checks)

#### 4. Testing Infrastructure (5 test files, 293 tests)

**Test Coverage**: ~88% overall, 100% pass rate (293/293 passing ✅)

**Test Files** (in `tests/` directory):
- **tests/conftest.py**: pytest configuration with auto session state reset fixture
- **tests/test_backtest.py** (93 tests): Unit tests for backtest engine
  - Cache expiration, retry logic, ticker/date validation
  - Batch downloads, data quality validation
  - Rolling 12-month Sharpe ratio calculation and edge cases
  - **DCA tests**: Monthly/weekly contributions, multi-ticker, price decline, precedence, weekend handling
  - **IRR/XIRR tests**: Basic calculation, multiple contributions, negative/zero returns, convergence
  - 13 test classes covering all major functions

- **tests/test_app.py** (76 tests): Unit tests for web UI
  - Portfolio presets, date presets, multiple benchmarks
  - Delta indicators, rolling returns, metric formatting
  - Portfolio composition with ticker names (fetched from yfinance)
  - **DCA metrics**: IRR and total_contributions display
  - **Colorblind accessibility**: 8 tests for Wong palette validation
  - **URL Parameters**: 5 tests for capital/benchmarks sharing bug fixes
  - 16 test classes with comprehensive coverage

- **tests/test_ticker_data.py** (32 tests): Unit tests for ticker search and name fetching
  - Curated ticker list validation
  - Search functionality (by symbol and name)
  - Yahoo Finance API mocking (search and ticker info)
  - Dynamic ticker name fetching with yfinance
  - Fallback to shortName, error handling
  - Cache clearing and edge cases

- **tests/test_state_manager.py** (67 tests): Unit tests for StateManager
  - State getters and setters (39 tests - preserved)
  - Input validation tests (28 tests - NEW)
  - Type validation for all StateManager setter methods
  - DateTime compatibility tests (accepts both datetime.datetime and datetime.date)
  - 11 test classes with comprehensive validation coverage

- **tests/test_integration.py** (21 tests): Integration tests
  - End-to-end workflows, edge cases, data quality
  - Statistical edge cases, multi-ticker scenarios
  - 6 test classes covering real-world usage

- **tests/conftest.py**: pytest configuration
  - Automatically adds parent directory to Python path
  - Enables tests to import modules from project root

**See**: [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for comprehensive testing rules and patterns.

---

## Key Rules and Conventions

### Testing Rules (CRITICAL)

**MANDATORY**: Always write tests for new functionality. Testing is not optional.

**Coverage Requirements**:
- New functions: 90%+ coverage required
- New features: 80%+ coverage required
- Bug fixes: Must include regression test
- Overall codebase: Maintain 85%+ coverage (currently ~88%)

**Test Before Commit**:
```bash
# CRITICAL: Create and activate virtual environment if not already done
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi
source .venv/bin/activate

# Install dependencies if needed
pip install -q -r requirements.txt

# ALWAYS run tests before committing
pytest -v

# If tests fail, fix them before committing
```

**See**: [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for detailed testing guidelines.

### Code Style

**Python Conventions**:
- Modern Python: `from __future__ import annotations`
- Type hints for all function signatures
- Docstrings for modules and functions
- PEP 8 compliant
- Line length: <100 characters

**Error Handling**:
- Contextual error messages (include ticker names, dates, suggestions)
- `ValueError` for invalid inputs/data
- `SystemExit` for CLI-level errors
- Dependency guards with installation guidance

**Logging**:
- Use `logging` module (NOT print statements for diagnostics)
- INFO level for key operations (downloads, cache hits, computations)
- WARNING level for non-critical issues (cache failures, limited data)
- Clean separation: `logging` for diagnostics, `print()` for user results

### Git Workflow

**Branch Naming**: `claude/<description>-<session-id>`

**Commit Messages**:
- Use imperative mood: "Add feature" not "Added feature"
- Prefixes: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `perf:`
- Be descriptive but concise

**Git Operations**:
```bash
# Always push with -u on first push
git push -u origin claude/<description>-<session-id>

# Never force push without permission
# Always run tests before committing
pytest -v && git commit
```

### When Making Changes

#### DO ✅

- ✅ Write tests for all new functionality (TDD preferred)
- ✅ Preserve existing error handling patterns
- ✅ Maintain backward compatibility with CSV format
- ✅ Use logging for diagnostics, print() for user results
- ✅ Update requirements.txt if adding dependencies
- ✅ Update README.md for user-facing changes
- ✅ Update CLAUDE.md for AI-relevant changes
- ✅ Test with multiple ticker combinations
- ✅ Run `pytest -v` before every commit
- ✅ Normalize weights to sum to 1.0
- ✅ Use descriptive variable names

#### DON'T ❌

- ❌ Skip writing tests (testing is mandatory!)
- ❌ Commit code with failing tests
- ❌ Modify CSV output columns without careful consideration
- ❌ Change default tickers without good reason
- ❌ Add dependencies without updating all docs
- ❌ Break existing CLI interface
- ❌ Commit .venv/, .cache/, results/, or charts/ directories
- ❌ Use print() for diagnostic output (use logging)
- ❌ Remove or modify error handling patterns
- ❌ Ignore test coverage (maintain 85%+ coverage)

---

## Phase History

See [CHANGELOG.md](docs/CHANGELOG.md) for full phase history (Phases 1–5 complete as of v2.5.0-dev).

---

## Quick Reference

### Common Commands

```bash
# CRITICAL: Always use virtual environment for testing and development
# Check if virtual environment exists, create if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run web UI
streamlit run app.py

# Run backtest (CLI)
python backtest.py --tickers AAPL MSFT --weights 0.6 0.4 --benchmark SPY

# Plot results
python plot_backtest.py --csv results/backtest.csv --output charts/test

# Run all tests
pytest -v

# Run with coverage
pytest --cov=backtest --cov=app --cov-report=term-missing

# Clear cache
rm -rf .cache/

# Deactivate virtual environment
deactivate
```

### Key File Locations

**See [FILE_REFERENCE.md](docs/FILE_REFERENCE.md) for comprehensive documentation.**

**Quick Links**:
- Core logic: `backtest.py:199-307` (compute_metrics, summarize)
- Data download: `backtest.py:364-538` (download_prices, validation)
- CLI parsing: `backtest.py:206-260` (parse_args)
- Caching: `backtest.py:263-338` (cache functions)
- Web UI: `app/main.py:75-459` (main orchestration)
- Charts: `app/charts.py` (Plotly visualizations)
- Config: `app/config.py` (32 constants)

### Default Values

- **Tickers**: VDCP.L, VHYD.L
- **Weights**: 0.5, 0.5 (auto-normalized)
- **Benchmark**: VWRA.L
- **Start Date**: 2018-01-01
- **End Date**: Today
- **Capital**: 100,000
- **Cache**: Enabled (use `--no-cache` to disable)
- **Cache TTL**: 24 hours (use `--cache-ttl` to change)

### Performance Metrics

- Ending Value
- Total Return
- CAGR (Compound Annual Growth Rate)
- Volatility (annualized standard deviation)
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown

### Dependencies

- **numpy** (>=1.24.0): Numerical computations
- **pandas** (>=2.0.0): Time-series data handling
- **yfinance** (>=0.2.0): Yahoo Finance API
- **matplotlib** (>=3.7.0): Plotting (plot_backtest.py)
- **pytest** (>=7.0.0): Testing framework
- **streamlit** (>=1.28.0): Web UI framework
- **plotly** (>=5.14.0): Interactive charts
- **requests** (>=2.28.0): HTTP library for Yahoo Finance search API
- **pyarrow** (>=10.0.0): Parquet format for secure cache storage (NEW)

---

## Typical Development Workflows

### Adding New Features

**Standard Process**:
1. **Plan**: Understand requirements, check existing patterns
2. **Write Tests**: Create failing tests first (TDD approach)
3. **Implement**: Write minimal code to pass tests
4. **Refactor**: Clean up while keeping tests green
5. **Document**: Update README.md, CLAUDE.md, FILE_REFERENCE.md as needed
6. **Commit**: `pytest -v && git commit -m "feat: description"`

**See**: [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for detailed workflows and scenarios.

### Example: Adding New Metric

```python
# 1. Write test first
def test_new_metric_calculation(self):
    """Test that new_metric is calculated correctly"""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # ... setup data ...
    results = compute_metrics(prices, weights, benchmark, capital)
    assert 'new_metric' in results.columns
    assert results['new_metric'].iloc[0] > 0

# 2. Implement in compute_metrics()
def compute_metrics(prices, weights, benchmark, capital):
    # ... existing code ...
    results['new_metric'] = calculate_new_metric(results['portfolio_value'])
    return results

# 3. Run tests
pytest tests/test_backtest.py::test_new_metric_calculation -v

# 4. Commit
git add tests/test_backtest.py backtest.py
git commit -m "feat: add new_metric calculation"
```

---

## Known Constraints and Edge Cases

**Critical Constraints**: Network required (yfinance), Yahoo Finance data may have gaps, `.cache/` gitignored, testing required before commits.

**Edge Cases**: Missing data (forward-fill & validate), short date ranges (<30 days warned), weight mismatch (auto-normalize), network failures (retry with backoff), bad data (comprehensive validation). See [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for details.

---

## Detailed Documentation References

**📄 [FILE_REFERENCE.md](docs/FILE_REFERENCE.md)**: File-by-file documentation with line counts, purposes, key functions.

**📄 [TESTING_GUIDE.md](docs/TESTING_GUIDE.md)**: TDD rules, test structure, coverage requirements, mocking patterns.

**📄 [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)**: Environment setup, code conventions, git workflow, common tasks.

**📄 [README.md](README.md)**: User documentation with quick start, usage examples, CLI reference, troubleshooting.

**📄 [CHANGELOG.md](docs/CHANGELOG.md)**: Version history and release notes with rolling Sharpe ratio feature.

---

## Important Notes for AI Assistants

### Development Priorities

1. **Testing First**: Write tests before or alongside code (TDD)
2. **Maintain Coverage**: Keep test coverage at 85%+ (currently ~88%)
3. **Backward Compatibility**: Never break existing CLI or CSV format
4. **Documentation**: Update all relevant docs (README, CLAUDE, FILE_REFERENCE, etc.)
5. **Code Quality**: Follow existing patterns, no duplication, extract constants

### Security Considerations

- **No credentials**: No API keys or secrets in codebase
- **Public data only**: All data from public Yahoo Finance
- **Safe paths**: Uses pathlib.Path (handles paths safely)
- **No sanitization needed**: CLI args type-checked by argparse

### Performance Notes

- **Caching**: 5-10x faster for cached data (first run slow, subsequent fast)
- **Memory**: Minimal - all data fits in memory
- **CPU**: Negligible - numpy operations are efficient
- **Network**: yfinance downloads can be slow (1-5 seconds per ticker)

---

**Last Updated**: 2026-04-30
**Version**: v2.5.0-dev
**Test Coverage**: ~88% (293 tests, 100% passing ✅)
