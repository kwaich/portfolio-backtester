# Design: Repository Pattern for Data Access (Issue 14)

**Date:** 2026-05-01
**Status:** Approved
**Scope:** B (price data + ticker search + ticker name lookup)
**Integration:** A (module-level default with override hook)

---

## 1. Goals

- Abstract all Yahoo Finance data access behind a clean interface.
- Enable easy swapping of data sources (CSV, mock, database) without touching business logic.
- Improve testability: tests can inject a `MockRepository` instead of patching scattered `yf.download` and `requests.get` calls.
- Maintain 100% backward compatibility: `download_prices`, `search_tickers_with_yahoo`, `get_ticker_name` remain as public APIs.

## 2. Repository Interface

A single `DataRepository` abstract class in `app/data_repository.py`:

```python
class DataRepository(ABC):
    @abstractmethod
    def get_prices(self, tickers: List[str], start: str, end: str) -> pd.DataFrame: ...
    @abstractmethod
    def search_tickers(self, query: str, limit: int = 10) -> List[Tuple[str, str]]: ...
    @abstractmethod
    def get_ticker_name(self, ticker: str) -> str: ...
```

## 3. Concrete Implementations

- **`YahooFinanceRepository`** — wraps `yf.download`, ticker search, and name lookup. Includes per-ticker Parquet caching logic migrated from `backtest.py`.
- **`MockRepository`** — returns fake data for tests.

## 4. Integration Strategy

Module-level singleton in `app/data_repository.py`:

```python
_default_repo: DataRepository | None = None

def get_repository() -> DataRepository:
    global _default_repo
    if _default_repo is None:
        _default_repo = YahooFinanceRepository()
    return _default_repo

def set_repository(repo: DataRepository) -> None:
    global _default_repo
    _default_repo = repo
```

`download_prices` in `backtest.py` becomes a thin wrapper delegating to `get_repository().get_prices(...)`. Similarly for ticker search and name lookup in `app/ticker_data.py`.

## 5. File Changes

- **New:** `app/data_repository.py`
- **Modified:** `backtest.py` — `download_prices` delegates; cache helpers move into repository
- **Modified:** `app/ticker_data.py` — delegates search and name lookup
- **New tests:** `tests/test_data_repository.py`

## 6. Testing Strategy

- Existing `yf.download` patches continue working (repository still calls yfinance internally).
- New pattern: `set_repository(MockRepository(fake_data))` for clean injection.
- Cache tests migrate to `test_data_repository.py` and test `YahooFinanceRepository` directly.

## 7. Migration Order

1. Create `app/data_repository.py` with ABC + `YahooFinanceRepository`.
2. Update `backtest.py` `download_prices` to delegate.
3. Update `app/ticker_data.py` to delegate.
4. Add `MockRepository`.
5. Migrate/add tests.
6. Run full test suite.
