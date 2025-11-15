# CLAUDE.md - AI Assistant Development Guide

**Purpose**: Concise guidance for AI assistants working on the portfolio-backtester repository.

**Detailed Documentation**:
- **[FILE_REFERENCE.md](docs/FILE_REFERENCE.md)**: Comprehensive file-by-file documentation
- **[TESTING_GUIDE.md](docs/TESTING_GUIDE.md)**: Test-driven development rules and patterns
- **[DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)**: Development workflows and best practices

---

## Project Overview

This is a lightweight Python-based ETF backtesting utility that allows users to:
- Compare portfolio performance against benchmarks
- Download historical price data from Yahoo Finance via yfinance
- Calculate buy-and-hold returns with static weights
- Generate visualization charts of performance metrics

**Primary Use Case**: Testing portfolio allocations (default: VDCP.L/VHYD.L vs VWRA.L benchmark)

**Current Status**:
- **Version**: v2.2.0-dev (Unreleased - 2025-11-15)
- **Test Coverage**: ~88% (184 tests, 100% passing)
- **Progress**: 87.5% complete (14/16 tasks)
- **Branch**: claude/make-ticker-searchable-01Nb4CzjMJBJ9y2PugkUtCW7

---

## Repository Structure (High-Level)

```
portfolio-backtester/
├── app.py                    # Streamlit web UI (backward compatibility wrapper - 43 lines)
├── app/                      # Modular web UI package (Phase 2 - 8 modules, 1,695 lines)
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration constants (32 constants)
│   ├── presets.py            # Portfolio and date presets
│   ├── validation.py         # Input validation & session state
│   ├── ui_components.py      # Reusable UI rendering with searchable inputs
│   ├── ticker_data.py        # Ticker search & Yahoo Finance integration (NEW)
│   ├── charts.py             # Plotly chart generation
│   └── main.py               # Application orchestration
├── backtest.py               # Core backtesting engine (830 lines - Phases 1 & 3)
├── plot_backtest.py          # Visualization utility (395 lines - Phases 2 & 3)
├── test_backtest.py          # Unit tests for backtest.py (858 lines, 67 tests)
├── test_app.py               # Unit tests for app.py UI (933 lines, 64 tests)
├── test_ticker_data.py       # Unit tests for ticker_data.py (NEW - 32 tests)
├── test_integration.py       # Integration tests (420 lines, 16 tests - Phase 3)
├── requirements.txt          # Python dependencies (includes requests)
├── README.md                 # Main user documentation
├── CLAUDE.md                 # This file - AI assistant guide
└── docs/                     # Documentation directory
    ├── FILE_REFERENCE.md         # Detailed file documentation
    ├── TESTING_GUIDE.md          # TDD rules and test patterns
    ├── DEVELOPER_GUIDE.md        # Development workflows
    ├── IMPLEMENTATION_PLAN.md    # Code improvement roadmap
    ├── IMPLEMENTATION_CHECKLIST.md # Progress tracking (87.5% complete)
    ├── CHANGELOG.md              # Version history (v2.1.0)
    ├── TEST_REPORT.md            # Phase 2 validation report
    └── PROJECT_SUMMARY.md        # Additional project documentation
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
- `parse_args()`: CLI argument parsing with validation
- `download_prices()`: Fetch prices with batch caching & retry logic (Phase 1 & 3)
- `validate_price_data()`: Data quality validation (Phase 3)
- `compute_metrics()`: Calculate portfolio vs benchmark metrics with minimum data checks
- `summarize()`: Generate comprehensive statistics (Sharpe, Sortino, drawdown, etc.)
- `main()`: Orchestrate the backtest workflow

**Phase Enhancements**:
- **Phase 1**: Cache expiration (TTL), retry logic (exponential backoff), ticker/date validation
- **Phase 3**: Batch download optimization (per-ticker caching), data quality validation

**Key Patterns**:
- Weight normalization (always sum to 1.0)
- Date alignment (common start date across all series)
- Forward-fill for missing data
- MD5-based caching (tickers + date range)

#### 2. Web UI (app/ package - 8 modules, 1,695 lines)

**Purpose**: Interactive Streamlit dashboard with presets, multiple benchmarks, and charts

**Architecture** (Phase 2 - Modular + Searchable Tickers):
- **config.py**: Centralized configuration (32 constants)
- **presets.py**: Portfolio & date presets (6 portfolios + 6 date ranges)
- **validation.py**: Session state management & input validation
- **ui_components.py**: Reusable UI rendering with searchable ticker inputs
- **ticker_data.py**: Ticker search with 50+ curated tickers & Yahoo Finance integration
- **charts.py**: Plotly chart generation (interactive visualizations)
- **main.py**: Application orchestration & workflow
- **app.py**: 43-line backward compatibility wrapper

**Features**:
- Portfolio presets (6 pre-configured portfolios)
- **Searchable ticker inputs**: Search from 50+ popular ETFs/stocks or use Yahoo Finance API
- **Portfolio composition table**: Displays ticker symbols, full company/fund names (fetched dynamically from Yahoo Finance), and weights
- Date range presets (1Y, 3Y, 5Y, 10Y, YTD, Max)
- Multiple benchmarks (up to 3 simultaneously)
- Delta indicators (color-coded outperformance)
- Rolling returns (30/90/180-day windows)
- Rolling 12-month Sharpe ratio chart (252-day window for risk-adjusted performance tracking)
- Rebalancing strategies (buy-and-hold, daily, weekly, monthly, quarterly, yearly)
- Logarithmic scale toggle for portfolio value charts
- CSV & HTML export

**Phase 2 Improvements**:
- Zero code duplication (eliminated 134 duplicate lines)
- All magic numbers extracted to constants
- Consistent logging throughout
- 100% backward compatibility

**Searchable Ticker Feature** (New):
- **Curated List**: 50+ popular ETFs (Global, US, European, Fixed Income, Sector) and stocks for search
- **Yahoo Finance Integration**: Optional live search (may be rate-limited)
- **Dynamic Ticker Names**: Company/fund names fetched from Yahoo Finance API in real-time (not hardcoded)
- **Graceful Fallback**: Uses curated list if Yahoo Finance search unavailable; returns empty name if ticker info unavailable
- **User-Friendly**: Click-to-select from search results or manual entry
- **LRU Caching**: Ticker names cached (500 entries) to minimize API calls

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

#### 4. Testing Infrastructure (4 test files, 184 tests)

**Test Coverage**: ~88% overall, 100% pass rate

**Test Files**:
- **test_backtest.py** (72 tests): Unit tests for backtest engine
  - Cache expiration, retry logic, ticker/date validation
  - Batch downloads, data quality validation
  - Rolling 12-month Sharpe ratio calculation and edge cases
  - 11 test classes covering all major functions

- **test_app.py** (933 lines, 64 tests): Unit tests for web UI
  - Portfolio presets, date presets, multiple benchmarks
  - Delta indicators, rolling returns, metric formatting
  - Portfolio composition with ticker names (fetched from yfinance)
  - 14 test classes with comprehensive coverage

- **test_ticker_data.py** (32 tests): Unit tests for ticker search and name fetching
  - Curated ticker list validation
  - Search functionality (by symbol and name)
  - Yahoo Finance API mocking (search and ticker info)
  - Dynamic ticker name fetching with yfinance
  - Fallback to shortName, error handling
  - Cache clearing and edge cases

- **test_integration.py** (420 lines, 16 tests - Phase 3): Integration tests
  - End-to-end workflows, edge cases, data quality
  - Statistical edge cases, multi-ticker scenarios
  - 6 test classes covering real-world usage

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

## Phase Completion Summary

### ✅ Phase 1: Reliability & Validation
Cache expiration (TTL), retry logic (exponential backoff), ticker/date validation. **+28 tests** (86→113), ~88% coverage.

### ✅ Phase 2: Code Quality & Organization
Modular architecture (874→1,358 lines across 7 modules), zero duplication (-134 lines), 32 extracted constants, centralized state. **113 tests**, ~88% coverage.

### ✅ Phase 3: Performance & Data Validation
Batch download optimization, data quality validation (NaN/zero/negative/extreme), min data requirements, integration tests. **+42 tests** (113→155), ~88% coverage.

### 🚧 Phase 4: Documentation & Polish (1/3 complete)
**Completed**: README, CLAUDE.md, CHANGELOG, FILE_REFERENCE, TESTING_GUIDE, DEVELOPER_GUIDE.
**Pending** (optional): Deployment guide, GitHub templates.

---

## Quick Reference

### Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run web UI
streamlit run app.py

# Run backtest (CLI)
python backtest.py --tickers AAPL MSFT --weights 0.6 0.4 --benchmark SPY

# Plot results
python plot_backtest.py --csv results/backtest.csv --output charts/test

# Run all tests (184 tests)
pytest -v

# Run with coverage
pytest --cov=backtest --cov=app --cov-report=term-missing

# Clear cache
rm -rf .cache/
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
pytest test_backtest.py::test_new_metric_calculation -v

# 4. Commit
git add test_backtest.py backtest.py
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

**📄 [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)**: Phases 1-4 roadmap with task breakdown and timeline.

**📄 [IMPLEMENTATION_CHECKLIST.md](docs/IMPLEMENTATION_CHECKLIST.md)**: Task tracking (14/16 complete - 87.5%).

**📄 [CHANGELOG.md](docs/CHANGELOG.md)**: Version history and release notes (v1.0.0 → v2.1.0).

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

**Last Updated**: 2025-11-15
**Version**: v2.2.0-dev (Unreleased)
**Current Branch**: claude/make-ticker-searchable-01Nb4CzjMJBJ9y2PugkUtCW7
**Test Coverage**: ~88% (179 tests, 100% passing)
**Progress**: 87.5% complete (14/16 tasks, Phase 4 in progress)
