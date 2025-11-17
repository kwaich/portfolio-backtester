# Developer Guide - Workflows and Best Practices

**Purpose**: Practical development workflows, code conventions, and common tasks for the portfolio-backtester project.

For file-specific documentation, see [FILE_REFERENCE.md](FILE_REFERENCE.md).
For testing guidelines, see [TESTING_GUIDE.md](TESTING_GUIDE.md).
For AI development guidance, see [CLAUDE.md](../CLAUDE.md).

---

## Development Environment Setup

### Initial Setup

```bash
# Clone repository
git clone https://github.com/kwaich/portfolio-backtester.git
cd portfolio-backtester

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies (RECOMMENDED)
pip install -r requirements.txt

# Upgrade pip if needed
./.venv/bin/python -m pip install --upgrade pip
```

### Dependencies

Core dependencies with minimum versions:
- **numpy** (>=1.24.0): Numerical computations
- **pandas** (>=2.0.0): Time-series data handling
- **yfinance** (>=0.2.0): Yahoo Finance API wrapper
- **matplotlib** (>=3.7.0): Plotting and visualization
- **pytest** (>=7.0.0): Testing framework
- **streamlit** (>=1.28.0): Web UI framework
- **plotly** (>=5.14.0): Interactive charts

### Testing Setup

```bash
# Run all tests (208 tests)
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
pytest --cov=backtest --cov=app --cov-report=html

# Run specific test class
pytest tests/test_backtest.py::TestSummarize -v

# Run specific test function
pytest tests/test_app.py::TestPortfolioPresets::test_tech_giants_preset_values -v
```

---

## Code Conventions

### Python Style

**General Guidelines**:
- Modern Python features: `from __future__ import annotations`
- Type hints for function signatures
- Docstrings for module-level and function-level documentation
- PEP 8 compliant code style
- Line length: Generally <100 characters for readability

**Example**:
```python
from __future__ import annotations

def compute_metrics(
    prices: pd.DataFrame,
    weights: np.ndarray,
    benchmark: pd.Series,
    capital: float
) -> pd.DataFrame:
    """Calculate portfolio metrics vs benchmark.

    Args:
        prices: DataFrame with ticker columns and date index
        weights: Array of portfolio weights (must sum to 1.0)
        benchmark: Series of benchmark prices with date index
        capital: Initial capital amount

    Returns:
        DataFrame with portfolio_value, benchmark_value, returns, etc.

    Raises:
        ValueError: If weights don't match tickers or data is insufficient
    """
    # Implementation...
```

### Error Handling Patterns

**Contextual Error Messages**:
- Include relevant ticker names and date ranges
- Specific problem description
- Actionable suggestions for resolution
- Available vs. missing data breakdown

**Error Types**:
- `ValueError`: Invalid inputs or data quality issues
- `SystemExit`: CLI-level errors (user-facing)
- Dependency guards: Clear messages for missing imports

**Examples**:
```python
# Data validation errors
if len(aligned) < 2:
    raise ValueError(
        f"Insufficient overlapping data: only {len(aligned)} trading day(s).\n"
        f"Need at least 2 days for meaningful backtest.\n"
        f"Portfolio start: {portfolio_start}, Benchmark start: {bench_start}"
    )

# Ticker validation errors
if not is_valid:
    raise ValueError(
        f"Invalid ticker format: '{ticker}'\n"
        f"Supported formats:\n"
        f"  - Standard: AAPL, MSFT\n"
        f"  - UK exchange: VWRA.L, VDCP.L\n"
        f"  - Indices: ^GSPC, ^DJI\n"
        f"  - Currency pairs: EURUSD=X"
    )

# Dependency errors
try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance library not found.")
    print("Install with: pip install yfinance")
    sys.exit(1)
```

### Logging

**Guidelines**:
- Use Python's `logging` module (NOT print statements for diagnostics)
- INFO level for key operations (downloads, cache hits, computation)
- WARNING level for non-critical issues (cache failures, limited data)
- Formatted timestamps for all log messages
- Clean separation: `logging` for diagnostics, `print()` for user results

**Configuration**:
```python
import logging

# Configure logging (in main modules)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# Usage examples
logger.info(f"Downloading prices for {len(tickers)} tickers: {tickers}")
logger.info(f"Cache hit for {ticker}: {cache_path}")
logger.warning(f"Cache file corrupted or unreadable: {cache_path}")
logger.warning(f"Limited data: only {len(aligned)} trading days")
```

### Key Code Patterns

#### DataFrame Operations
```python
# Forward-fill for missing data
aligned = prices.loc[start_date:].ffill().dropna()

# Calculate returns
portfolio_return = portfolio_value / capital - 1

# Calculate rolling metrics
rolling_30d = portfolio_value.pct_change(30)
```

#### Weight Normalization
```python
# Always normalize weights to sum to 1.0
if not np.isclose(weights.sum(), 1.0):
    logger.warning(f"Weights sum to {weights.sum():.4f}, normalizing to 1.0")
    weights = weights / weights.sum()
```

#### Date Alignment
```python
# Find common start date across all series
first_valid_points = [s.first_valid_index() for s in [prices[col] for col in prices.columns]]
start_date = max(first_valid_points)

# Align portfolio and benchmark
combined_start = max(aligned.index[0], bench_start)
aligned = aligned.loc[combined_start:]
bench_aligned = bench_aligned.loc[combined_start:]
```

#### Data Caching
```python
# Cache keyed by tickers and date range
cache_key = get_cache_key(tickers, start, end)  # MD5 hash
cache_path = Path(".cache") / f"{cache_key}.pkl"

# Load from cache if available
cached_data = load_cached_prices(cache_path, max_age_hours=cache_ttl_hours)
if cached_data is not None:
    logger.info(f"Cache hit: {cache_path}")
    return cached_data

# Download and save to cache
prices = download_from_yfinance(tickers, start, end)
save_cached_prices(cache_path, prices)
logger.info(f"Cached prices to: {cache_path}")
```

---

## Git Workflow

### Branch Naming Convention

**Pattern**: `claude/<description>-<session-id>`

**Examples**:
- `claude/create-ui-framework-01D656RsUmycaEV3SNmffGrx`
- `claude/review-implementation-plan-01CNKXvBZAn7UQMEcwXn5eGw`
- `claude/read-imple-01QaXd8PwRSeMGMtHvEeqFGf`

### Commit Message Style

**Format**: Imperative mood, descriptive but concise

**Prefixes**:
- `feat:` - New features
- `fix:` - Bug fixes
- `refactor:` - Code restructuring
- `test:` - Adding tests
- `docs:` - Documentation updates
- `perf:` - Performance improvements

**Examples**:
```bash
git commit -m "feat: add cache expiration with configurable TTL"
git commit -m "fix: handle missing ticker data gracefully"
git commit -m "refactor: extract app.py into modular structure"
git commit -m "test: add comprehensive integration test suite"
git commit -m "docs: update README with Phase 3 features"
git commit -m "perf: optimize batch downloads with per-ticker caching"
```

### Git Operations

**Basic Workflow**:
```bash
# Create feature branch
git checkout -b claude/<description>-<session-id>

# Make changes and stage
git add .

# Commit with descriptive message
git commit -m "feat: implement feature X"

# Push to remote (always use -u for first push)
git push -u origin claude/<description>-<session-id>

# Subsequent pushes
git push
```

**Important Rules**:
- Always develop on designated Claude branches
- Push with `-u` flag on first push: `git push -u origin <branch-name>`
- Never force push without explicit permission
- Run tests before committing: `pytest -v && git commit`

### Recent Commit History

Example of incremental development:
```
6c0d2f8 Merge pull request #5 from kwaich/claude/review-implementation-plan-01CNKXvBZAn7UQMEcwXn5eGw
48a91b2 docs: comprehensive documentation update for Phase 2 completion
c81af27 docs: update checklist with Phase 2 completion
63fd3de test: comprehensive validation report for Phase 2 refactoring
786ceff docs: Phase 2 completion summary
```

---

## Common Development Tasks

### Adding New Metrics

**Steps**:
1. Modify `compute_metrics()` in backtest.py
2. Add new calculated column to the returned DataFrame
3. Update `summarize()` if it's a summary statistic
4. Add tests for the new metric
5. Consider adding visualization to plot_backtest.py or app/charts.py
6. Update documentation

**Example**:
```python
# In compute_metrics() function
def compute_metrics(prices, weights, benchmark, capital):
    # ... existing code ...

    # Add new metric
    results['new_metric'] = calculate_new_metric(results['portfolio_value'])

    return results

# Add test
def test_new_metric_calculation(self):
    """Test that new_metric is calculated correctly"""
    # ARRANGE
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # ... setup data ...

    # ACT
    results = compute_metrics(prices, weights, benchmark, capital)

    # ASSERT
    assert 'new_metric' in results.columns
    assert results['new_metric'].iloc[0] > 0
```

### Supporting New Data Sources

**Steps**:
1. Modify `download_prices()` in backtest.py
2. Ensure compatibility with existing DataFrame structure
3. Handle new data format edge cases
4. Add data quality validation
5. Write tests with mocked data source
6. Update documentation in README.md

**Requirements**:
- Must return DataFrame with ticker columns and date index
- Must handle missing data gracefully
- Must integrate with existing cache system

### Adding CLI Arguments

**Steps**:
1. Update `parse_args()` in backtest.py
2. Add argument with clear help text and sensible defaults
3. Use argument in `main()` function
4. Test with various input combinations
5. Update README.md with new argument

**Example**:
```python
# In parse_args() function
parser.add_argument(
    '--new-option',
    type=str,
    default='default_value',
    help='Description of what this option does (default: default_value)'
)

# In main() function
def main(args=None):
    args = parse_args(args)
    new_value = args.new_option
    # Use new_value in logic
```

### Improving Visualizations

**For plot_backtest.py** (matplotlib):
1. Read plot_backtest.py to understand current structure
2. Add new subplot or modify existing plots
3. Maintain seaborn-v0_8 style consistency
4. Ensure both `--output` (PNG) and interactive modes work
5. Test with real CSV data
6. Keep DPI at 150 for consistent quality

**For app/charts.py** (Plotly):
1. Create new chart function following existing patterns
2. Use consistent color scheme (PORTFOLIO_COLOR, BENCHMARK_COLORS)
3. Add hover tooltips with formatted values
4. Test with various data scenarios
5. Ensure responsive layout

### Modifying the Web UI

**Steps**:
1. Edit appropriate module in `app/` package
2. Import necessary functions from backtest.py (don't duplicate logic)
3. Use `st.sidebar` for input controls, main area for results
4. Test with `streamlit run app.py` during development
5. Maintain consistency with CLI functionality
6. Write tests in test_app.py

**Key Streamlit Patterns**:
```python
# Progress indicators
with st.spinner('Running backtest...'):
    results = run_backtest()

# Display metrics
st.metric("Total Return", f"{total_return:.2%}")

# Columns for side-by-side layout
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Portfolio", portfolio_value)

# Error handling
try:
    results = validate_and_run()
except ValueError as e:
    st.error(f"Validation Error: {e}")
    st.stop()

# Download buttons
csv_data = results.to_csv(index=True)
st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="backtest_results.csv",
    mime="text/csv"
)
```

---

## Manual Testing Workflow

### Basic Backtest Testing

```bash
# Activate environment
source .venv/bin/activate

# Test basic backtest
python backtest.py \
    --tickers VDCP.L VHYD.L \
    --weights 0.5 0.5 \
    --benchmark VWRA.L \
    --start 2018-01-01 \
    --end 2024-12-31 \
    --capital 100000 \
    --output results/test_run.csv

# Verify CSV was created
ls -lh results/test_run.csv
cat results/test_run.csv | head -n 10

# Test with caching
python backtest.py --tickers AAPL --benchmark SPY  # First run (slow)
python backtest.py --tickers AAPL --benchmark SPY  # Second run (fast, cached)

# Test cache expiration
python backtest.py --tickers AAPL --benchmark SPY --cache-ttl 0  # Force download

# Test without cache
python backtest.py --tickers AAPL --benchmark SPY --no-cache
```

### Plotting Testing

```bash
# Test plotting
python plot_backtest.py \
    --csv results/test_run.csv \
    --output charts/test_run

# Verify PNGs were created
ls -lh charts/
# Should see: test_run_dashboard.png

# Test interactive mode (no --output)
python plot_backtest.py --csv results/test_run.csv
```

### Web UI Testing

```bash
# Test web UI
streamlit run app.py
# Opens browser at http://localhost:8501

# Manual testing checklist:
# 1. Try different portfolio presets
# 2. Test date range presets (1Y, 3Y, 5Y, etc.)
# 3. Add multiple benchmarks (up to 3)
# 4. Test with invalid tickers (should show error)
# 5. Test with mismatched date ranges
# 6. Download CSV export
# 7. Check all charts render correctly
# 8. Verify delta indicators show correct colors
```

### What to Verify

**Data Downloads**:
- ✅ All tickers successfully fetch data
- ✅ Error messages are clear for invalid tickers
- ✅ Network errors are handled gracefully

**Caching**:
- ✅ `.cache/` directory created on first run
- ✅ Subsequent runs use cached data (faster)
- ✅ Cache expiration works with `--cache-ttl`
- ✅ `--no-cache` bypasses cache

**Calculations**:
- ✅ All metrics (CAGR, Sharpe, Sortino, drawdown) are reasonable
- ✅ Portfolio value starts at capital amount
- ✅ Returns are calculated correctly
- ✅ Metrics match between CLI and UI

**Alignment**:
- ✅ Portfolio and benchmark have matching date ranges
- ✅ Forward-filling handles missing data correctly
- ✅ No look-ahead bias in calculations

**Output**:
- ✅ CSV files created correctly with all required columns
- ✅ PNG files generated with proper formatting
- ✅ Charts display all data clearly

**Logging**:
- ✅ Log messages appear with timestamps
- ✅ Cache hits/misses logged
- ✅ Warnings shown for limited data

**Tests**:
- ✅ Run `pytest -v` to verify all 155 tests pass
- ✅ Coverage remains at ~88%

### Known Edge Cases

1. **Missing data**: Tickers with limited history may cause alignment issues
   - Solution: Handle with forward-fill and validation

2. **Date ranges**: Very short periods may produce unreliable CAGR
   - Solution: Warn for periods < 30 days

3. **Weight mismatch**: Length of weights must match length of tickers
   - Solution: Validate and normalize weights

4. **Network dependency**: Requires internet access for yfinance
   - Solution: Use cache for offline testing

---

## Important Notes and Constraints

### Critical Constraints

1. **Requirements file**: Always use `requirements.txt` for dependency management
2. **Network required**: yfinance needs internet access (except when using cached data)
3. **Data quality**: Yahoo Finance data may have gaps or errors
4. **Gitignored folders**: `.venv/`, `.cache/`, `results/`, `charts/` are NOT committed
5. **Cache directory**: `.cache/` created automatically on first run
6. **Testing**: Always run tests before committing significant changes

### When Making Changes

#### DO ✅

- ✅ Preserve existing error handling patterns (detailed, contextual errors)
- ✅ Maintain backward compatibility with existing CSV format
- ✅ Keep imports at module level with guards for missing deps
- ✅ Use logging module for diagnostic output, print() for user results
- ✅ Use descriptive variable names matching existing style
- ✅ Write unit tests for new functionality (aim for 90%+ coverage)
- ✅ Test with multiple ticker combinations
- ✅ Validate weight normalization works correctly
- ✅ Update `requirements.txt` if adding dependencies
- ✅ Update `README.md` for user-facing changes
- ✅ Update `CLAUDE.md` for AI-relevant implementation details
- ✅ Run `pytest -v` before every commit

#### DON'T ❌

- ❌ Remove or modify CSV output columns without careful consideration
- ❌ Change default tickers without good reason (user expectation)
- ❌ Add dependencies without updating requirements.txt, README.md, CLAUDE.md
- ❌ Break the existing CLI interface (add flags, don't change behavior)
- ❌ Modify plotting style without maintaining consistency
- ❌ Commit `.venv/`, `.cache/`, `results/`, or `charts/` directories
- ❌ Use `print()` for diagnostic/debug output (use logging instead)
- ❌ Skip writing tests for significant new features
- ❌ Commit code with failing tests

### Security Considerations

- **No credential handling**: No API keys or secrets in this codebase
- **Public data only**: All data from public Yahoo Finance
- **No user input sanitization needed**: CLI args are type-checked by argparse
- **Path traversal**: Uses pathlib.Path which handles paths safely
- **No sensitive data**: All data is public market prices

### Performance Notes

- **Network latency**: yfinance downloads can be slow (1-5 seconds per ticker)
  - Mitigated by caching (5-10x faster for cached data)
- **Cache performance**: Pickled DataFrames are compact and fast to load
- **Cache size**: Minimal - typically <1MB per ticker-year
- **Memory usage**: Minimal - all data fits in memory for typical use cases
- **CPU usage**: Negligible - numpy operations are very efficient
- **Date ranges**: Longer periods = more data but not problematic
- **First run**: Slower (downloads data), subsequent runs are fast (cached)

---

## Typical Development Workflows

### Scenario 1: User Wants Different Default Tickers

**Steps**:
1. Modify `parse_args()` default values in backtest.py
   ```python
   parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT'])
   ```
2. Update PROJECT_SUMMARY.md or README.md example commands
3. Test with new defaults: `python backtest.py`
4. Commit with message: `feat: change default tickers to AAPL and MSFT`

### Scenario 2: Add New Performance Metric

**Steps**:
1. Add calculation in `compute_metrics()` function
   ```python
   # Calculate new metric
   results['beta'] = calculate_beta(results['portfolio_return'], results['benchmark_return'])
   ```
2. Add to DataFrame returned at end of function
3. Optionally add to `summarize()` if it's a summary stat
4. Write tests for the new metric
5. Update documentation if user-facing

**Example Test**:
```python
def test_beta_calculation(self):
    """Test beta metric is calculated correctly"""
    # ... setup ...
    results = compute_metrics(prices, weights, benchmark, capital)
    assert 'beta' in results.columns
    assert results['beta'].mean() > 0  # Positive correlation expected
```

### Scenario 3: Enhance Plotting

**Steps**:
1. Read plot_backtest.py or app/charts.py to understand structure
2. Add new subplot or modify existing plots
3. Ensure both `--output` and interactive modes work (for plot_backtest.py)
4. Test with real CSV data
5. Maintain style consistency

**Example** (adding new subplot):
```python
# In plot_backtest.py
def plot_new_metric(df, ax):
    """Plot new metric over time"""
    ax.plot(df.index, df['new_metric'], color='#2c3e50', linewidth=2)
    ax.set_title('New Metric Over Time')
    ax.set_ylabel('New Metric Value')
    ax.grid(True, alpha=0.3)

# Add to main plotting function
fig, axes = plt.subplots(3, 2, figsize=(16, 12))  # Changed from 2x2 to 3x2
plot_new_metric(df, axes[2, 0])
```

### Scenario 4: Fix Data Alignment Bug

**Steps**:
1. Understand the date alignment logic in `compute_metrics()`
2. Review alignment code (find common start date)
3. Test with tickers that have different start dates
4. Ensure forward-fill logic is correct
5. Validate no data leakage (look-ahead bias)
6. Add regression test

**Test Example**:
```python
def test_different_start_dates_alignment(self):
    """Test alignment when tickers have different start dates"""
    # AAPL starts Jan 1, MSFT starts Feb 1
    dates_aapl = pd.date_range("2020-01-01", periods=100, freq="D")
    dates_msft = pd.date_range("2020-02-01", periods=70, freq="D")

    # ... setup data with different start dates ...

    result = compute_metrics(portfolio, weights, benchmark, 100000)

    # Should align to Feb 1 (latest start)
    assert result.index[0] >= pd.Timestamp("2020-02-01")
```

### Scenario 5: Improve Error Messages

**Steps**:
1. Identify where errors occur (ValueError, SystemExit)
2. Make messages more actionable
3. Include relevant context (ticker, date range, etc.)
4. Test error paths explicitly
5. Ensure errors don't expose sensitive info (not applicable here)

**Before**:
```python
raise ValueError("Invalid ticker")
```

**After**:
```python
raise ValueError(
    f"Invalid ticker format: '{ticker}'\n"
    f"Supported formats:\n"
    f"  - Standard: AAPL, MSFT\n"
    f"  - UK exchange: VWRA.L, VDCP.L\n"
    f"  - Indices: ^GSPC, ^DJI\n"
    f"  - Currency pairs: EURUSD=X"
)
```

---

## Quick Reference

### Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run web UI (interactive dashboard)
streamlit run app.py

# Run backtest (CLI)
python backtest.py --tickers AAPL MSFT --weights 0.6 0.4 --benchmark SPY

# Plot results (CLI)
python plot_backtest.py --csv results/backtest.csv --output charts/test

# Run ALL tests (208 tests)
pytest -v

# Run tests by module
pytest tests/test_backtest.py -v     # 92 tests
pytest tests/test_app.py -v          # 63 tests
pytest tests/test_ticker_data.py -v  # 32 tests
pytest tests/test_integration.py -v  # 21 tests

# Check test coverage
pytest --cov=backtest --cov=app --cov-report=term-missing

# Clear cache
rm -rf .cache/

# Git workflow
git checkout -b claude/<description>-<session-id>
git add .
git commit -m "feat: description"
git push -u origin claude/<description>-<session-id>
```

### Default Values

- **Tickers**: VDCP.L, VHYD.L
- **Weights**: 0.5, 0.5 (auto-normalized)
- **Benchmark**: VWRA.L
- **Start Date**: 2018-01-01
- **End Date**: Today
- **Capital**: 100,000
- **Cache**: Enabled (use `--no-cache` to disable)
- **Cache TTL**: 24 hours (use `--cache-ttl` to change)
- **Plot Style**: seaborn-v0_8 (matplotlib)

### Performance Metrics Available

- Ending Value
- Total Return
- CAGR (Compound Annual Growth Rate)
- Volatility (annualized standard deviation)
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk-adjusted return)
- Maximum Drawdown (largest peak-to-trough decline)

### Key File Locations

See [FILE_REFERENCE.md](FILE_REFERENCE.md) for comprehensive file documentation.

**Quick Links**:
- Core logic: backtest.py:199-307 (compute_metrics, summarize)
- Data download: backtest.py:364-538 (download_prices, validation)
- CLI interface: backtest.py:206-260 (parse_args)
- Caching: backtest.py:263-338 (cache functions)
- Web UI: app/main.py:75-459 (main orchestration)
- Charts: app/charts.py (Plotly visualizations)
- Configuration: app/config.py (32 constants)

---

**Last Updated**: 2025-11-15
**For**: Portfolio Backtester v2.1.0
**See Also**: [FILE_REFERENCE.md](FILE_REFERENCE.md), [TESTING_GUIDE.md](TESTING_GUIDE.md), [CLAUDE.md](../CLAUDE.md)
