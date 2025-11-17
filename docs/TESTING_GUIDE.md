# Testing Guide - Test-Driven Development Rules

**Purpose**: Comprehensive testing guidelines and best practices for the portfolio-backtester project.

For file-specific documentation, see [FILE_REFERENCE.md](FILE_REFERENCE.md).
For AI development guidance, see [CLAUDE.md](../CLAUDE.md).

---

## Core Principles

**CRITICAL**: Always write tests for new functionality. Testing is not optional.

### Current Test Coverage Status

**Canonical Source (last validated via `pytest -v`):**
- **Total Tests**: 256 (93 backtest + 71 UI + 39 state manager + 32 ticker data + 21 integration) ✅
- **Pass Rate**: 100% (256/256 passing) ✅
- **Test-to-Code Ratio**: ~0.90:1 ✅
- **Overall Coverage**: ~88% ✅

**Breakdown by Module**:
- Backtest engine (`tests/test_backtest.py`): 93 tests covering metrics, caching, validation, DCA/IRR logic
- Streamlit UI (`tests/test_app.py`): 71 tests spanning presets, validation, downloads, rolling metrics
- State manager (`tests/test_state_manager.py`): 39 tests covering centralized session logic
- Ticker utilities (`tests/test_ticker_data.py`): 32 tests covering curated lists/search helpers (`tests/test_ticker_names.py` currently holds scaffolding for future scenarios)
- Integration (`tests/test_integration.py`): 21 end-to-end scenarios

**Target**: Maintain 85%+ coverage for all new code ✅ **ACHIEVED**

---

## When to Write Tests

### ALWAYS Write Tests For:

1. **New functions**: Any new function in backtest.py or app/ modules
2. **New features**: UI components, workflows, calculations
3. **Bug fixes**: Add regression test before fixing the bug
4. **Refactoring**: Ensure tests pass before and after changes
5. **Edge cases**: Error handling, empty data, boundary conditions

### Test Writing Workflow

#### For New Features (TDD Approach):
```
1. Write failing test first (red)
2. Implement minimal code to pass test (green)
3. Refactor while keeping tests green
4. Add additional edge case tests
5. Verify coverage meets minimum requirements
```

#### For Bug Fixes:
```
1. Write test that reproduces the bug (should fail)
2. Verify test fails as expected
3. Fix the bug
4. Verify test now passes
5. Add related edge case tests
6. Run full test suite to prevent regressions
```

---

## What to Test

### Backtest Engine (test_backtest.py)

**Test Classes** (11 total):
- `TestParseArgs`: CLI argument parsing (7 tests)
- `TestCacheFunctions`: Cache with TTL and migration (6 tests)
- `TestSummarize`: Statistics calculation (4 tests)
- `TestComputeMetrics`: Backtest computation (4 tests)
- `TestRetryLogic`: Exponential backoff decorator (4 tests)
- `TestTickerValidation`: Ticker format validation (11 tests)
- `TestDateValidation`: Date parsing and ranges (7 tests)
- `TestDownloadPrices`: Price fetching with batch caching (9 tests)
- `TestDataValidation`: Data quality checks (10 tests)
- `TestMain`: Integration tests (6 tests)

**Coverage Areas**:
- ✅ Function inputs/outputs (all pure functions)
- ✅ Edge cases (empty data, NaN values, date misalignments)
- ✅ Error conditions (invalid tickers, network failures)
- ✅ Calculations (metrics, returns, statistics)
- ✅ CLI argument parsing and validation
- ✅ Caching behavior (TTL, expiration, migration)
- ✅ Batch download optimization
- ✅ Data quality validation
- ✅ Integration (main() workflow)

### Web UI (test_app.py)

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

**Coverage Areas**:
- ✅ Integration with backtest module
- ✅ Metric formatting (currency, percentage, ratio)
- ✅ Error handling (invalid inputs, failed backtests)
- ✅ Data transformations (drawdown, active return)
- ✅ Export functionality (CSV, charts)
- ✅ Portfolio preset behavior
- ✅ Date preset calculations
- ✅ Multiple benchmark logic
- ✅ Delta indicator calculations
- ✅ Rolling returns calculations

### Integration Tests (test_integration.py)

**Test Classes** (6 total):
- `TestEndToEndWorkflow`: CLI to CSV workflow, multi-ticker, caching (3 tests)
- `TestEdgeCases`: Leap years, extreme drawdowns, zero volatility, etc. (8 tests)
- `TestDataQuality`: NaN detection, negative prices, extreme changes (5 tests)
- `TestValidation`: Ticker/date format validation, future dates (5 tests)
- `TestStatisticalEdgeCases`: Sharpe/Sortino edge cases, CAGR precision (4 tests)
- `TestMultiTickerEdgeCases`: Different start dates alignment (1 test)

**Coverage Areas**:
- ✅ End-to-end workflows (CLI → download → compute → CSV)
- ✅ Edge cases and boundary conditions
- ✅ Data quality validation
- ✅ Input validation
- ✅ Statistical calculation edge cases
- ✅ Multi-ticker scenarios

---

## Test Structure Patterns

### AAA Pattern (Arrange-Act-Assert)

**Good Test Structure**:
```python
def test_feature_name(self):
    """Clear description of what is being tested"""
    # ARRANGE: Set up test data
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    prices = pd.DataFrame({"AAPL": [100, 110, 105]}, index=dates[:3])
    weights = np.array([1.0])
    benchmark = pd.Series([400, 410, 405], index=dates[:3])
    capital = 100000

    # ACT: Execute the function being tested
    result = backtest.compute_metrics(prices, weights, benchmark, capital)

    # ASSERT: Verify expected behavior
    assert result is not None
    assert 'portfolio_value' in result.columns
    assert len(result) == 3
    assert result['portfolio_value'].iloc[0] == capital
```

### Test Class Organization

**Recommended Structure**:
```python
class TestFeatureName:
    """Test suite for specific feature or function"""

    def test_basic_case(self):
        """Test normal/happy path"""
        # Test the most common use case
        pass

    def test_edge_case_empty_data(self):
        """Test with empty input"""
        # Verify graceful handling of empty data
        pass

    def test_edge_case_single_value(self):
        """Test with minimal input"""
        # Test boundary conditions
        pass

    def test_error_handling(self):
        """Test expected errors are raised"""
        with pytest.raises(ValueError, match="expected message"):
            # Code that should raise ValueError
            pass

    def test_edge_case_extreme_values(self):
        """Test with extreme/unusual values"""
        # Test with very large, very small, or unusual values
        pass
```

### Parametrized Tests

**For Multiple Similar Test Cases**:
```python
import pytest

@pytest.mark.parametrize("ticker,expected_valid", [
    ("AAPL", True),
    ("VWRA.L", True),
    ("^GSPC", True),
    ("EURUSD=X", True),
    ("BRK-B", True),
    ("", False),
    ("TOOLONGTICKER", False),
    ("123", False),
])
def test_ticker_validation(ticker, expected_valid):
    """Test ticker validation with multiple inputs"""
    is_valid, msg = validate_ticker(ticker)
    assert is_valid == expected_valid
```

---

## Mocking Patterns

### Mock External Dependencies

**yfinance Downloads**:
```python
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

@patch('backtest.yf.download')
def test_download_prices(mock_download):
    """Test price downloading with mocked yfinance"""
    # ARRANGE: Mock yfinance response
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    mock_download.return_value = pd.DataFrame({
        "AAPL": np.linspace(150, 180, len(dates)),
        "SPY": np.linspace(400, 450, len(dates))
    }, index=dates)

    # ACT: Call download function
    result = backtest.download_prices(
        ["AAPL", "SPY"],
        "2023-01-01",
        "2023-12-31",
        use_cache=False
    )

    # ASSERT: Verify results
    assert not result.empty
    assert "AAPL" in result.columns
    assert "SPY" in result.columns
    assert len(result) == 100
```

**Streamlit Components**:
```python
import sys
from unittest.mock import MagicMock

# Mock streamlit before importing app modules
sys.modules['streamlit'] = MagicMock()

import backtest
# Now import modules that use backtest
```

**File I/O Operations**:
```python
from pathlib import Path
import tempfile

def test_cache_save_and_load(tmp_path):
    """Test cache operations with temporary directory"""
    # tmp_path is a pytest fixture for temporary directories
    cache_file = tmp_path / "test_cache.pkl"

    # Test save
    data = pd.DataFrame({"AAPL": [100, 110, 120]})
    save_cached_prices(cache_file, data)

    # Test load
    loaded = load_cached_prices(cache_file)
    pd.testing.assert_frame_equal(data, loaded)
```

---

## Coverage Expectations

### Minimum Coverage Requirements

- **New functions**: 90%+ coverage required
- **New features**: 80%+ coverage required
- **Bug fixes**: Must include regression test
- **Overall codebase**: Maintain 85%+ coverage

### How to Check Coverage

```bash
# Run all tests with coverage
pytest --cov=backtest --cov=app --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=backtest --cov=app --cov-report=html
open htmlcov/index.html

# Check coverage for specific module
pytest tests/test_backtest.py --cov=backtest --cov-report=term-missing

# Check coverage with missing line numbers
pytest --cov=backtest --cov-report=term-missing:skip-covered
```

### Coverage Report Interpretation

```
Name               Stmts   Miss  Cover   Missing
------------------------------------------------
backtest.py          391     20    95%   45-47, 231-235
app/main.py          459     46    90%   123-125, 340-342
app/charts.py        306     15    95%   78-82
------------------------------------------------
TOTAL              1174    117    88%
```

**Coverage Grades**:
- **95%+**: ✅ Excellent
- **85-94%**: ✅ Good (target range)
- **70-84%**: ⚠️ Acceptable but needs improvement
- **< 70%**: ❌ Insufficient, add tests

---

## Common Testing Mistakes

### DON'T ❌

- ❌ Skip writing tests for "simple" code (all code needs tests)
- ❌ Test implementation details instead of behavior
- ❌ Write tests that depend on external services (mock them)
- ❌ Write tests that depend on current date/time (use fixed dates)
- ❌ Commit code without running tests first
- ❌ Ignore failing tests ("I'll fix it later")
- ❌ Write tests that test the same thing multiple times
- ❌ Use real file I/O without temp files or mocks
- ❌ Write tests with shared state between test functions
- ❌ Use sleep() for timing (makes tests slow and flaky)

### DO ✅

- ✅ Write descriptive test names: `test_portfolio_value_increases_with_positive_returns`
- ✅ Use pytest fixtures for repeated setup code
- ✅ Mock external dependencies (yfinance, file I/O, network calls)
- ✅ Test edge cases (empty data, NaN, None, negative values)
- ✅ Use `pytest.raises()` for expected errors with match pattern
- ✅ Keep tests fast (entire suite < 10 seconds)
- ✅ Make tests independent (no shared state)
- ✅ Use `@pytest.mark.parametrize` for testing multiple inputs
- ✅ Add docstrings to test classes and complex tests
- ✅ Use clear assertion messages for debugging

---

## Example: Adding Tests for New Feature

### Scenario: Portfolio Presets Feature

**Feature**: Added 6 pre-configured portfolio presets to UI

**Required Tests**:
```python
class TestPortfolioPresets:
    """Test portfolio preset functionality"""

    def test_preset_portfolios_exist(self):
        """Verify all preset portfolios are defined"""
        presets = get_portfolio_presets()
        expected_presets = [
            "Custom (Manual Entry)",
            "Default UK ETFs",
            "60/40 US Stocks/Bonds",
            "Tech Giants",
            "Dividend Aristocrats",
            "Global Diversified"
        ]
        assert len(presets) == len(expected_presets)
        for expected in expected_presets:
            assert expected in presets

    def test_tech_giants_preset_values(self):
        """Verify Tech Giants preset has correct values"""
        presets = get_portfolio_presets()
        tech_giants = presets["Tech Giants"]

        # Expected: 4 tickers, equal weights, QQQ benchmark
        assert tech_giants["tickers"] == ["AAPL", "MSFT", "GOOGL", "AMZN"]
        assert tech_giants["weights"] == [0.25, 0.25, 0.25, 0.25]
        assert tech_giants["benchmark"] == "QQQ"

    def test_preset_selection_populates_inputs(self):
        """Verify selecting preset updates session state"""
        # Mock session state
        mock_state = {}

        # Select "Tech Giants"
        apply_preset(mock_state, "Tech Giants")

        # Assert tickers/weights/benchmark are populated
        assert mock_state["tickers"] == ["AAPL", "MSFT", "GOOGL", "AMZN"]
        assert mock_state["weights"] == [0.25, 0.25, 0.25, 0.25]
        assert mock_state["benchmark"] == "QQQ"

    def test_custom_preset_allows_manual_entry(self):
        """Verify Custom preset doesn't override inputs"""
        mock_state = {"tickers": ["TEST"], "weights": [1.0]}

        # Select "Custom"
        apply_preset(mock_state, "Custom (Manual Entry)")

        # Assert inputs remain unchanged
        assert mock_state["tickers"] == ["TEST"]
        assert mock_state["weights"] == [1.0]

    def test_all_presets_have_valid_structure(self):
        """Verify all presets have required fields"""
        presets = get_portfolio_presets()

        for name, preset in presets.items():
            if name == "Custom (Manual Entry)":
                continue

            assert "tickers" in preset
            assert "weights" in preset
            assert "benchmark" in preset
            assert len(preset["tickers"]) == len(preset["weights"])
            assert abs(sum(preset["weights"]) - 1.0) < 0.01  # Weights sum to 1
```

---

## Test Metrics and Goals

### Achievement Summary

**Phase 1: Reliability & Validation** (+28 tests):
- Cache Expiration System: 6 tests ✅
- Retry Logic with Exponential Backoff: 4 tests ✅
- Ticker Validation (multiple formats): 11 tests ✅
- Date Validation & Normalization: 7 tests ✅

**Phase 2: UI Enhancements** (+39 tests):
- Portfolio Presets: 8 tests ✅
- Date Range Presets: 7 tests ✅
- Multiple Benchmarks: 9 tests ✅
- Delta Indicators: 7 tests ✅
- Rolling Returns: 8 tests ✅

**Phase 3: Performance & Data Quality** (+40 tests):
- Batch Download Optimization: 5 tests ✅
- Data Quality Validation: 10 tests ✅
- Integration Tests: 25 tests ✅

**Goals vs. Actual**:

| Metric | Goal | Actual | Status |
|--------|------|--------|--------|
| Total tests | 200+ | 256 | ✅ Exceeded |
| Pass rate | 100% | 100% | ✅ Perfect |
| Test-to-code ratio | >0.80:1 | ~0.90:1 | ✅ Exceeded |
| Coverage | 85%+ | ~88% | ✅ Achieved |

---

## Running Tests

### Development Workflow

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests (recommended before commits)
pytest -v

# Run all tests with coverage
pytest --cov=backtest --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_backtest.py -v
pytest tests/test_app.py -v
pytest tests/test_integration.py -v

# Run specific test class
pytest tests/test_backtest.py::TestTickerValidation -v

# Run specific test function
pytest tests/test_app.py::TestPortfolioPresets::test_tech_giants_preset_values -v

# Run with verbose output and show all print statements
pytest -vv -s

# Run tests matching a keyword
pytest -k "validation" -v

# Run tests and stop at first failure
pytest -x

# Watch mode (re-run on file changes) - requires pytest-watch
pytest-watch

# Ensure all tests pass before pushing
pytest -v && git push
```

### Continuous Integration

```bash
# Pre-commit check
pytest -v --cov=backtest --cov=app --cov-report=term-missing

# Coverage must be ≥85%
pytest --cov=backtest --cov=app --cov-report=term --cov-fail-under=85
```

### Test Output Examples

**Successful Test Run**:
```
============================= test session starts ==============================
collected 256 items

test_backtest.py::TestParseArgs::test_default_values PASSED           [  0%]
test_backtest.py::TestCacheFunctions::test_cache_key_generation PASSED [  0%]
...
test_integration.py::TestEndToEndWorkflow::test_cli_to_csv_workflow PASSED [100%]

============================= 256 passed in 2.02s ===============================
```

**Failed Test Example**:
```
FAILED test_backtest.py::TestComputeMetrics::test_minimum_data_validation
E   AssertionError: ValueError not raised
E   Expected: ValueError with message matching "Insufficient.*data"
E   Actual: Function completed without error
```

---

## Test Documentation Requirements

### Every Test Class Should Have:

1. **Docstring**: Explaining what feature/function is being tested
2. **Individual test docstrings**: For complex or non-obvious tests
3. **Clear assertion messages**: For debugging when tests fail
4. **Organized structure**: Group related tests together

### Example:
```python
class TestDataValidation:
    """Test suite for price data quality validation.

    Covers:
    - All-NaN detection
    - Excessive missing data (>50%)
    - Zero/negative price detection
    - Extreme price change detection (>90%/day)
    """

    def test_all_nan_data_rejected(self):
        """Test that tickers with all-NaN data are rejected."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        prices = pd.DataFrame({"BAD": [np.nan] * 100}, index=dates)

        with pytest.raises(ValueError, match="all values are NaN"):
            validate_price_data(prices, ["BAD"])
```

---

## Quick Reference

### Test Commands Cheat Sheet

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=backtest --cov=app --cov-report=term-missing

# Run specific file
pytest tests/test_backtest.py -v

# Run specific test
pytest tests/test_backtest.py::TestTickerValidation::test_valid_formats -v

# Run and stop at first failure
pytest -x

# HTML coverage report
pytest --cov=backtest --cov=app --cov-report=html
```

### Current Test Statistics

- **Total Tests**: 256 (93 backtest + 71 UI + 39 state manager + 32 ticker data + 21 integration)
- **Pass Rate**: 100% (last run: `pytest -v`)
- **Coverage**: ~88% overall (see `pytest --cov` reports)
- **Test Files**: 6 primary suites (`test_backtest.py`, `test_app.py`, `test_state_manager.py`, `test_ticker_data.py`, `test_ticker_names.py`, `test_integration.py`)
- **Execution Time**: ~2-3 seconds on a modern laptop

---

**Last Updated**: 2025-11-17
**For**: Portfolio Backtester v2.1.0
**See Also**: [FILE_REFERENCE.md](FILE_REFERENCE.md), [CLAUDE.md](../CLAUDE.md), [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
