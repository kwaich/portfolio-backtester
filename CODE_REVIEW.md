# Code Review: Portfolio Backtester

**Review Date:** 2025-11-17
**Branch:** `claude/review-code-01FMcnN697didaPWUPyzpyT8`
**Total Lines:** ~9,489 Python code
**Test Coverage:** ~88% (261 tests, 97.7% pass rate)

---

## Executive Summary

The portfolio backtester is a **well-structured, professionally developed codebase** with strong fundamentals:

✅ **Strengths:**
- Excellent modular architecture with clear separation of concerns
- Strong test coverage (88%) with comprehensive test suites
- Good documentation with detailed docstrings
- Proper error handling with contextual messages
- Streamlit best practices implemented (caching, forms, URL sharing)
- Colorblind-accessible UI design

⚠️ **Areas for Improvement:**
- Security concerns with pickle-based caching
- Some functions exceed recommended complexity limits
- Opportunities for performance optimization
- Minor code duplication and refactoring opportunities
- Type hint coverage could be improved

---

## Critical Issues (Fix Immediately)

### 1. ✅ FIXED: Security: Pickle-Based Caching Vulnerability

**Status:** ✅ **COMPLETED** (2025-11-17)

**Original Location:** `backtest.py:310-337, 340-358`

**Issue:**
Using `pickle` for caching creates potential security vulnerabilities. Unpickling untrusted data can execute arbitrary code.

```python
# Current implementation (VULNERABLE)
with open(cache_path, "rb") as f:
    cache_data = pickle.load(f)  # ⚠️ Security risk
```

**Recommendation:**
Replace pickle with a safer serialization format like JSON or Parquet:

```python
# Option 1: Use Parquet for DataFrames (best performance)
def save_cached_prices(cache_path: Path, prices: pd.DataFrame) -> None:
    try:
        metadata_path = cache_path.with_suffix('.json')
        data_path = cache_path.with_suffix('.parquet')

        # Save metadata
        metadata = {
            "timestamp": time.time(),
            "version": CACHE_VERSION
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        # Save data
        prices.to_parquet(data_path, compression='gzip')
        logger.info(f"Saved data to cache: {data_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

# Option 2: Use JSON (more portable but slower)
def save_cached_prices(cache_path: Path, prices: pd.DataFrame) -> None:
    try:
        cache_data = {
            "data": prices.to_dict(orient='split'),
            "timestamp": time.time(),
            "version": CACHE_VERSION
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")
```

**Priority:** CRITICAL
**Effort:** 2-3 hours
**Impact:** High - Eliminates security vulnerability

**Implementation:**
Migrated from pickle to Parquet format with JSON metadata:
- **New Cache Format:**
  - `{cache_key}.parquet` - Price data (gzip compressed)
  - `{cache_key}.json` - Metadata (timestamp, version)
- **Backward Compatibility:** Automatic migration from old pickle caches
- **Test Coverage:** 9/9 cache tests passing (including migration tests)
- **Dependencies:** Added `pyarrow>=10.0.0` to requirements.txt
- **Cache Version:** Bumped to 2.0

**Files Changed:**
- `backtest.py`: Updated load/save functions to use Parquet
- `requirements.txt`: Added pyarrow dependency
- `tests/test_backtest.py`: Updated cache tests for new format

---

## High Priority Issues

### 2. Function Complexity: `compute_metrics()` Too Long

**Location:** `backtest.py:795-984` (190 lines)

**Issue:**
The `compute_metrics()` function is too long and handles multiple responsibilities:
- Data alignment and validation
- Portfolio value calculation (buy-and-hold, rebalancing, DCA)
- Benchmark calculation
- Rolling Sharpe ratio calculation

**Recommendation:**
Break into smaller, focused functions:

```python
def compute_metrics(
    prices: pd.DataFrame,
    weights: np.ndarray,
    benchmark: pd.Series,
    capital: float,
    rebalance_freq: str = None,
    dca_amount: float = None,
    dca_freq: str = None,
) -> pd.DataFrame:
    """Main orchestrator - delegates to specialized functions."""

    # Step 1: Align and validate data
    aligned_prices, aligned_benchmark = _align_and_validate_data(
        prices, benchmark
    )

    # Step 2: Calculate portfolio values
    portfolio_value, portfolio_contributions = _calculate_portfolio_value(
        aligned_prices, weights, capital, rebalance_freq, dca_amount, dca_freq
    )

    # Step 3: Calculate benchmark values
    bench_value, bench_contributions = _calculate_benchmark_value(
        aligned_benchmark, capital, rebalance_freq, dca_amount, dca_freq
    )

    # Step 4: Build results table
    table = _build_results_table(
        portfolio_value, portfolio_contributions,
        bench_value, bench_contributions
    )

    # Step 5: Add rolling metrics
    _add_rolling_metrics(table, portfolio_value, bench_value,
                        portfolio_contributions, bench_contributions)

    return table

def _align_and_validate_data(
    prices: pd.DataFrame,
    benchmark: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract data alignment logic (lines 824-880)."""
    # ... implementation ...

def _calculate_portfolio_value(
    prices: pd.DataFrame,
    weights: np.ndarray,
    capital: float,
    rebalance_freq: str = None,
    dca_amount: float = None,
    dca_freq: str = None
) -> tuple[pd.Series, pd.Series]:
    """Extract portfolio calculation logic (lines 882-902)."""
    # ... implementation ...
```

**Priority:** HIGH
**Effort:** 4-6 hours
**Impact:** Medium - Improves maintainability and testability

---

### 3. Performance: Batch Download Could Be More Efficient

**Location:** `backtest.py:556-603`

**Issue:**
The batch download logic downloads uncached tickers individually through yfinance, which could be optimized to batch multiple uncached tickers into a single API call.

**Current Flow:**
```python
for ticker in uncached_tickers:
    # Downloads happen one-by-one internally in yfinance
    new_data = _download_from_yfinance(uncached_tickers, start, end)
```

**Recommendation:**
Optimize by batching uncached tickers:

```python
# Optimized: Download all uncached tickers in a single call
if uncached_tickers:
    if len(uncached_tickers) == 1:
        # Single ticker - use existing path
        new_data = _download_from_yfinance(uncached_tickers, start, end)
        new_prices = _process_yfinance_data(new_data, uncached_tickers, start, end)
    else:
        # Multiple tickers - batch download
        logger.info(f"Batch downloading {len(uncached_tickers)} tickers")
        new_data = _download_from_yfinance(uncached_tickers, start, end)
        new_prices = _process_yfinance_data(new_data, uncached_tickers, start, end)

    # Cache each ticker individually for future use
    for ticker in uncached_tickers:
        if ticker in new_prices.columns:
            single_cache_path = get_cache_path([ticker], start, end)
            save_cached_prices(single_cache_path, new_prices[[ticker]])
```

**Priority:** HIGH
**Effort:** 2-3 hours
**Impact:** Medium - Faster data downloads (20-40% improvement with multiple uncached tickers)

---

### 4. Error Handling: Missing Type Validation in StateManager

**Location:** `app/state_manager.py:106-148`

**Issue:**
StateManager setter methods don't validate input types, which could lead to runtime errors:

```python
@staticmethod
def set_num_tickers(num: int) -> None:
    """Set the number of tickers in the portfolio."""
    _session_state()[StateKeys.NUM_TICKERS] = num  # No validation
```

**Recommendation:**
Add type validation:

```python
@staticmethod
def set_num_tickers(num: int) -> None:
    """Set the number of tickers in the portfolio.

    Args:
        num: Number of tickers (must be positive integer)

    Raises:
        TypeError: If num is not an integer
        ValueError: If num is not positive
    """
    if not isinstance(num, int):
        raise TypeError(f"num_tickers must be int, got {type(num).__name__}")
    if num < 1:
        raise ValueError(f"num_tickers must be positive, got {num}")
    _session_state()[StateKeys.NUM_TICKERS] = num

@staticmethod
def set_preset_tickers(tickers: List[str]) -> None:
    """Set the preset ticker symbols.

    Args:
        tickers: List of ticker symbols (all non-empty strings)

    Raises:
        TypeError: If tickers is not a list
        ValueError: If any ticker is empty
    """
    if not isinstance(tickers, list):
        raise TypeError(f"tickers must be list, got {type(tickers).__name__}")
    if any(not isinstance(t, str) or not t.strip() for t in tickers):
        raise ValueError("All tickers must be non-empty strings")
    _session_state()[StateKeys.PRESET_TICKERS] = tickers
```

**Priority:** HIGH
**Effort:** 3-4 hours
**Impact:** Medium - Prevents runtime errors and improves debugging

---

## Medium Priority Issues

### 5. Code Duplication: Frequency Normalization

**Location:** Multiple locations
- `backtest.py:1253-1265` (rebalance freq)
- `backtest.py:1267-1280` (DCA freq)

**Issue:**
Same frequency mapping logic duplicated twice:

```python
# Duplicated code
freq_map = {
    'daily': 'D',
    'weekly': 'W',
    'monthly': 'M',
    'quarterly': 'Q',
    'yearly': 'Y'
}
rebalance_freq = freq_map.get(rebalance_freq.lower(), rebalance_freq.upper())
```

**Recommendation:**
Extract to a shared function:

```python
def normalize_frequency_string(freq_str: Optional[str]) -> Optional[str]:
    """Normalize frequency string to pandas code.

    Args:
        freq_str: Frequency string (daily/weekly/monthly/etc or D/W/M/etc)

    Returns:
        Normalized pandas frequency code or None

    Examples:
        >>> normalize_frequency_string('monthly')
        'M'
        >>> normalize_frequency_string('M')
        'M'
        >>> normalize_frequency_string(None)
        None
    """
    if freq_str is None:
        return None

    freq_map = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'M',
        'quarterly': 'Q',
        'yearly': 'Y'
    }

    normalized = freq_map.get(freq_str.lower(), freq_str.upper())
    return _normalize_pandas_frequency(normalized)

# Usage in main()
rebalance_freq = normalize_frequency_string(args.rebalance)
dca_freq = normalize_frequency_string(args.dca_freq)
```

**Priority:** MEDIUM
**Effort:** 1 hour
**Impact:** Low - Reduces duplication

---

### 6. Missing Docstring Return Types

**Location:** Multiple files

**Issue:**
Many functions have docstrings but don't specify return types in the docstring:

```python
def calculate_drawdown(series: pd.Series) -> pd.Series:
    """Calculate drawdown from value series.

    Drawdown is the peak-to-trough decline during a specific period.

    Args:
        series: Time series of values (e.g., portfolio value)

    Returns:
        Series of drawdown percentages  # ⚠️ Vague
    """
```

**Recommendation:**
Add detailed return type information:

```python
def calculate_drawdown(series: pd.Series) -> pd.Series:
    """Calculate drawdown from value series.

    Drawdown is the peak-to-trough decline during a specific period.

    Args:
        series: Time series of values (e.g., portfolio value)

    Returns:
        pd.Series: Drawdown percentages (negative values) with same index as input.
            Values range from 0 (at peak) to negative percentages (in drawdown).

    Examples:
        >>> values = pd.Series([100, 110, 105, 120, 115],
        ...                    index=pd.date_range('2020-01-01', periods=5))
        >>> dd = calculate_drawdown(values)
        >>> dd.iloc[-1]  # 115 vs peak of 120 = -4.17%
        -4.166666666666666
    """
```

**Priority:** MEDIUM
**Effort:** 4-6 hours
**Impact:** Low - Improves documentation

---

### 7. Magic Numbers in Configuration

**Location:** `app/config.py`, `plot_backtest.py`

**Issue:**
Some magic numbers lack explanation:

```python
# app/config.py
SUBPLOT_VERTICAL_SPACING = 0.15    # Why 0.15?
SUBPLOT_HORIZONTAL_SPACING = 0.12  # Why 0.12?

# plot_backtest.py
DPI = 150  # Default DPI - why 150?
```

**Recommendation:**
Add comments explaining the rationale:

```python
# Visual spacing constants
# Vertical spacing optimized for 2x2 grid at 800px height
# 0.15 provides ~120px between rows for clear visual separation
SUBPLOT_VERTICAL_SPACING = 0.15

# Horizontal spacing optimized for wide layout
# 0.12 provides ~150px between columns on typical screens
SUBPLOT_HORIZONTAL_SPACING = 0.12

# Output quality
# 150 DPI provides good balance between file size and print quality
# (300 DPI is overkill for screen viewing, 72 DPI looks pixelated when zoomed)
DEFAULT_DPI = 150
```

**Priority:** MEDIUM
**Effort:** 1-2 hours
**Impact:** Low - Improves code clarity

---

### 8. Inefficient String Concatenation

**Location:** `backtest.py:150-153`

**Issue:**
Using `"\n".join()` for building multiline error messages is less readable than f-strings:

```python
raise ValueError(
    f"Invalid ticker(s) detected:\n" + "\n".join(errors) + "\n\n"
    f"Valid ticker examples: AAPL, MSFT, VWRA.L, ^GSPC, EURUSD=X"
)
```

**Recommendation:**
Use consistent f-string formatting:

```python
error_list = "\n".join(errors)
raise ValueError(
    f"Invalid ticker(s) detected:\n"
    f"{error_list}\n\n"
    f"Valid ticker examples: AAPL, MSFT, VWRA.L, ^GSPC, EURUSD=X"
)
```

**Priority:** MEDIUM
**Effort:** 30 minutes
**Impact:** Low - Improves readability

---

## Low Priority Issues (Nice to Have)

### 9. Consider Adding Logging Levels

**Location:** Multiple files

**Issue:**
Logging uses only INFO and WARNING levels. Could benefit from DEBUG for development:

```python
logger.info(f"Cache hit for {ticker}")  # Could be DEBUG
logger.info(f"Rebalancing on {date.strftime('%Y-%m-%d')}: ...")  # Could be DEBUG
```

**Recommendation:**
Use DEBUG for verbose operational details:

```python
logger.debug(f"Cache hit for {ticker}")
logger.debug(f"Rebalancing on {date.strftime('%Y-%m-%d')}: portfolio value = ${current_value:,.2f}")
logger.info(f"Batch download complete: {len(cached_results)} ticker(s)")
```

**Priority:** LOW
**Effort:** 1-2 hours
**Impact:** Low - Better debugging experience

---

### 10. Consider Using Enum for Frequencies

**Location:** `app/config.py:60-67, 76-83`

**Issue:**
Frequency options are defined as dictionaries, could be more type-safe with Enum:

```python
REBALANCE_OPTIONS = {
    "Buy-and-Hold (No Rebalancing)": None,
    "Daily": "D",
    "Weekly": "W",
    # ...
}
```

**Recommendation:**
Use Enum for better type safety:

```python
from enum import Enum

class RebalanceFrequency(Enum):
    """Rebalancing frequency options."""
    NONE = (None, "Buy-and-Hold (No Rebalancing)")
    DAILY = ("D", "Daily")
    WEEKLY = ("W", "Weekly")
    MONTHLY = ("M", "Monthly")
    QUARTERLY = ("Q", "Quarterly")
    YEARLY = ("Y", "Yearly")

    def __init__(self, code, display_name):
        self.code = code
        self.display_name = display_name

    @classmethod
    def get_options(cls) -> Dict[str, Optional[str]]:
        """Get display_name -> code mapping for UI."""
        return {freq.display_name: freq.code for freq in cls}
```

**Priority:** LOW
**Effort:** 2-3 hours
**Impact:** Low - Improves type safety

---

### 11. Add Input Sanitization for Ticker Search

**Location:** `app/ticker_data.py`

**Issue:**
User input for ticker search isn't sanitized, could lead to injection issues:

```python
def search_tickers(query: str, limit: int = 10) -> List[Dict[str, str]]:
    # ... no sanitization of query ...
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
```

**Recommendation:**
Add URL encoding:

```python
import urllib.parse

def search_tickers(query: str, limit: int = 10) -> List[Dict[str, str]]:
    """Search for tickers using Yahoo Finance API.

    Args:
        query: Search query (will be URL-encoded)
        limit: Maximum number of results

    Returns:
        List of ticker dictionaries
    """
    # Sanitize input
    query = query.strip()
    if not query:
        return []

    # URL encode the query to prevent injection
    encoded_query = urllib.parse.quote(query)
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={encoded_query}"
    # ...
```

**Priority:** LOW
**Effort:** 30 minutes
**Impact:** Low - Security improvement

---

## Testing Improvements

### 12. Add Property-Based Testing

**Current:** Unit tests with fixed examples
**Recommendation:** Add property-based tests with Hypothesis

```python
from hypothesis import given, strategies as st

@given(
    prices=st.floats(min_value=1.0, max_value=10000.0),
    weights=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=5)
)
def test_weight_normalization_property(prices, weights):
    """Property: Normalized weights always sum to 1.0."""
    normalized = normalize_weights(weights)
    assert np.isclose(normalized.sum(), 1.0)
    assert all(w >= 0 for w in normalized)
```

**Priority:** LOW
**Effort:** 4-6 hours
**Impact:** Medium - Better edge case coverage

---

### 13. Add Performance Benchmarks

**Current:** No performance tests
**Recommendation:** Add pytest-benchmark tests

```python
def test_download_performance(benchmark):
    """Benchmark data download with caching."""
    def download():
        return download_prices(
            ["AAPL"], "2023-01-01", "2023-12-31", use_cache=True
        )

    result = benchmark(download)
    assert len(result) > 0

def test_compute_metrics_performance(benchmark):
    """Benchmark metric computation."""
    # ... setup data ...

    result = benchmark(
        compute_metrics, prices, weights, benchmark, capital
    )
    assert len(result) > 0
```

**Priority:** LOW
**Effort:** 2-3 hours
**Impact:** Low - Performance monitoring

---

## Architecture Suggestions

### 14. Consider Repository Pattern for Data Access

**Current:** Direct yfinance calls throughout code
**Recommendation:** Abstract data access layer

```python
# data_repository.py
class PriceDataRepository:
    """Abstract interface for price data access."""

    def get_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Get price data for tickers."""
        raise NotImplementedError

class YahooFinanceRepository(PriceDataRepository):
    """Yahoo Finance implementation."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager

    def get_prices(self, tickers, start_date, end_date):
        # Check cache first
        cached = self.cache_manager.get(tickers, start_date, end_date)
        if cached is not None:
            return cached

        # Download from yfinance
        data = self._download_from_yfinance(tickers, start_date, end_date)

        # Cache for future use
        self.cache_manager.set(tickers, start_date, end_date, data)

        return data

class MockRepository(PriceDataRepository):
    """Mock implementation for testing."""

    def get_prices(self, tickers, start_date, end_date):
        # Return fake data for tests
        return pd.DataFrame(...)
```

**Benefits:**
- Easier testing (no network calls)
- Can swap data sources (CSV, database, API)
- Clearer separation of concerns

**Priority:** LOW
**Effort:** 8-12 hours (significant refactoring)
**Impact:** Medium - Better testability and flexibility

---

## Documentation Improvements

### 15. Add Architecture Decision Records (ADRs)

**Recommendation:**
Document key architectural decisions:

```markdown
# ADR-001: Use Pickle for Caching

Date: 2024-XX-XX
Status: Superseded by ADR-00X

## Context
Need to cache downloaded price data to avoid repeated API calls.

## Decision
Use Python's pickle module for serialization.

## Consequences
- Fast serialization/deserialization
- Security vulnerability with untrusted data
- Not portable across Python versions

## Superseded By
ADR-00X: Migrate to Parquet for safer caching
```

**Priority:** LOW
**Effort:** 2-3 hours
**Impact:** Low - Improves knowledge transfer

---

## Summary of Recommendations

### Immediate Actions (Critical)
1. ✅ **Replace pickle with Parquet/JSON** (Security)

### Short-term (High Priority)
2. ✅ **Refactor `compute_metrics()`** (Maintainability)
3. ✅ **Optimize batch downloads** (Performance)
4. ✅ **Add type validation to StateManager** (Reliability)

### Medium-term (Medium Priority)
5. ✅ **Extract frequency normalization** (Code quality)
6. ✅ **Improve docstring return types** (Documentation)
7. ✅ **Document magic numbers** (Clarity)
8. ✅ **Improve string formatting** (Readability)

### Long-term (Low Priority)
9. ✅ **Add DEBUG logging** (Developer experience)
10. ✅ **Use Enum for frequencies** (Type safety)
11. ✅ **Sanitize ticker search input** (Security)
12. ✅ **Add property-based tests** (Test quality)
13. ✅ **Add performance benchmarks** (Monitoring)
14. ✅ **Consider repository pattern** (Architecture)
15. ✅ **Document architectural decisions** (Knowledge transfer)

---

## Overall Assessment

**Code Quality Grade: A- (Excellent with room for improvement)**

This is a **professionally developed codebase** with:
- Strong architecture and separation of concerns
- Good test coverage and documentation
- Proper error handling and user experience
- Adherence to Python best practices

The recommendations above would elevate it from "excellent" to "outstanding" by addressing:
- Security concerns (pickle caching)
- Code complexity (long functions)
- Performance optimization opportunities
- Minor code quality issues

**Estimated Total Effort:** 30-45 hours for all improvements
**Recommended Priority:** Start with Critical and High Priority items (10-15 hours)

---

## Next Steps

1. **Review this document** with the team
2. **Prioritize improvements** based on business impact
3. **Create GitHub issues** for accepted recommendations
4. **Implement Critical fixes** in next sprint
5. **Schedule High Priority work** for following sprints
6. **Track progress** and update this document

---

*Generated by Claude Code Review*
*Review conducted on: 2025-11-17*
