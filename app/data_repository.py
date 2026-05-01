"""Repository pattern for portfolio data access.

Abstracts all external data sources (Yahoo Finance, CSV, mocks) behind a
single interface. Use get_repository() / set_repository() for the module-level
singleton pattern.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_VERSION = "2.0"
DEFAULT_CACHE_TTL_HOURS = 24


class DataRepository(ABC):
    """Abstract interface for portfolio data access."""

    @abstractmethod
    def get_prices(
        self,
        tickers: List[str],
        start: str,
        end: str,
        use_cache: bool = True,
        cache_ttl_hours: int = DEFAULT_CACHE_TTL_HOURS,
    ) -> pd.DataFrame:
        """Fetch adjusted close prices for tickers."""
        ...

    @abstractmethod
    def search_tickers(self, query: str, limit: int = 10) -> List[Tuple[str, str]]:
        """Search for tickers matching a query."""
        ...

    @abstractmethod
    def get_ticker_name(self, ticker: str) -> str:
        """Get the full name for a ticker symbol."""
        ...


class YahooFinanceRepository(DataRepository):
    """Yahoo Finance implementation with per-ticker Parquet caching."""

    # ------------------------------------------------------------------
    # Cache helpers (migrated from backtest.py)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_cache_key(tickers: List[str], start: str, end: str) -> str:
        key_str = f"{'_'.join(sorted(tickers))}_{start}_{end}"
        return hashlib.md5(key_str.encode()).hexdigest()

    @staticmethod
    def _get_cache_path(tickers: List[str], start: str, end: str) -> Path:
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        cache_key = YahooFinanceRepository._get_cache_key(tickers, start, end)
        return cache_dir / cache_key

    def load_cached_prices(
        self, cache_path: Path, max_age_hours: int = DEFAULT_CACHE_TTL_HOURS
    ) -> pd.DataFrame | None:
        parquet_path = cache_path.with_suffix(".parquet")
        metadata_path = cache_path.with_suffix(".json")
        old_pickle_path = cache_path.with_suffix(".pkl")

        # Migrate old pickle caches
        if old_pickle_path.exists() and not parquet_path.exists():
            logger.warning(f"Old pickle cache detected at {old_pickle_path}, migrating to Parquet")
            try:
                with open(old_pickle_path, "rb") as f:
                    import pickle

                    cache_data = pickle.load(f)

                if isinstance(cache_data, dict) and "data" in cache_data:
                    df = cache_data["data"]
                    timestamp = cache_data.get("timestamp", time.time())
                elif isinstance(cache_data, pd.DataFrame):
                    df = cache_data
                    timestamp = time.time()
                else:
                    raise ValueError("Unknown pickle cache format")

                df.to_parquet(parquet_path, compression="gzip", index=True)
                metadata = {"timestamp": timestamp, "version": CACHE_VERSION}
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)

                old_pickle_path.unlink()
                logger.info(f"Migrated pickle cache to Parquet: {parquet_path}")
            except Exception as e:
                logger.warning(f"Failed to migrate old cache: {e}")
                try:
                    old_pickle_path.unlink()
                except Exception:
                    pass
                return None

        if not parquet_path.exists() or not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            cache_age_hours = (time.time() - metadata["timestamp"]) / 3600
            if cache_age_hours > max_age_hours:
                logger.info(f"Cache expired (age: {cache_age_hours:.1f}h, max: {max_age_hours}h)")
                parquet_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                return None

            df = pd.read_parquet(parquet_path)
            logger.debug(f"Loaded cached data (age: {cache_age_hours:.1f}h from {parquet_path})")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            parquet_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            return None

    def save_cached_prices(self, cache_path: Path, prices: pd.DataFrame) -> None:
        parquet_path = cache_path.with_suffix(".parquet")
        metadata_path = cache_path.with_suffix(".json")

        try:
            prices.to_parquet(parquet_path, compression="gzip", index=True)
            metadata = {"timestamp": time.time(), "version": CACHE_VERSION}
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
            logger.debug(f"Saved data to cache: {parquet_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
            parquet_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # yfinance download helpers (migrated from backtest.py)
    # ------------------------------------------------------------------
    def _download_from_yfinance(self, tickers: List[str], start: str, end: str) -> Any:
        logger.info(f"Downloading data for {len(tickers)} ticker(s) from {start} to {end}")
        return yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

    def _process_yfinance_data(
        self, data: Any, tickers: List[str], start: str, end: str
    ) -> pd.DataFrame:
        if data.empty:
            raise ValueError(
                f"No price data returned for period {start} to {end}.\n"
                f"Tickers requested: {', '.join(tickers)}\n"
                f"Please verify:\n"
                f"  1. Tickers are valid symbols\n"
                f"  2. Tickers were trading during this period\n"
                f"  3. Date range is valid (not in the future)"
            )

        if isinstance(data, pd.Series):
            prices = data.to_frame(name=tickers[0])
        elif isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0)
            price_field = None
            for candidate in ("Adj Close", "Close"):
                if candidate in level0:
                    price_field = candidate
                    break
            if price_field is None:
                price_field = level0[0]
            prices = data.xs(price_field, axis=1, level=0)
        else:
            preferred = next(
                (c for c in ("Adj Close", "Close") if c in data.columns), None
            )
            column = preferred or data.columns[0]
            prices = data[[column]].rename(columns={column: tickers[0]})

        prices = prices.dropna(how="all")
        if prices.empty:
            raise ValueError(
                f"No price data returned for period {start} to {end}.\n"
                f"Tickers requested: {', '.join(tickers)}\n"
                f"Please verify:\n"
                f"  1. Tickers are valid symbols\n"
                f"  2. Tickers were trading during this period\n"
                f"  3. Date range is valid (not in the future)"
            )

        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        missing = [ticker for ticker in tickers if ticker not in prices.columns]
        if missing:
            available = [t for t in tickers if t not in missing]
            raise ValueError(
                f"Missing data for ticker(s): {', '.join(missing)}\n"
                f"Date range: {start} to {end}\n"
                f"Available ticker(s): {', '.join(available) if available else 'None'}\n"
                f"Verify the missing tickers were trading during this period."
            )

        prices = prices.loc[:, tickers]
        return prices

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def get_prices(
        self,
        tickers: List[str],
        start: str,
        end: str,
        use_cache: bool = True,
        cache_ttl_hours: int = DEFAULT_CACHE_TTL_HOURS,
    ) -> pd.DataFrame:
        """Fetch adjusted closes with per-ticker caching."""

        # Optimized batch caching: check each ticker individually
        if use_cache and len(tickers) > 1:
            cached_results: dict[str, pd.Series] = {}
            uncached_tickers: List[str] = []

            for ticker in tickers:
                single_cache_path = self._get_cache_path([ticker], start, end)
                cached_data = self.load_cached_prices(
                    single_cache_path, max_age_hours=cache_ttl_hours
                )
                if cached_data is not None:
                    cached_results[ticker] = cached_data[ticker]
                    logger.debug(f"Cache hit for {ticker}")
                else:
                    uncached_tickers.append(ticker)

            if uncached_tickers:
                logger.info(
                    f"Downloading {len(uncached_tickers)} uncached ticker(s): "
                    f"{', '.join(uncached_tickers)}"
                )
                new_data = self._download_from_yfinance(uncached_tickers, start, end)
                new_prices = self._process_yfinance_data(
                    new_data, uncached_tickers, start, end
                )

                for ticker in uncached_tickers:
                    if ticker in new_prices.columns:
                        single_cache_path = self._get_cache_path([ticker], start, end)
                        self.save_cached_prices(single_cache_path, new_prices[[ticker]])
                        cached_results[ticker] = new_prices[ticker]

            if cached_results:
                combined_prices = pd.DataFrame(
                    {ticker: cached_results[ticker] for ticker in tickers if ticker in cached_results}
                )
                missing = [ticker for ticker in tickers if ticker not in combined_prices.columns]
                if missing:
                    raise ValueError(
                        f"Missing data for ticker(s): {', '.join(missing)}\n"
                        f"Date range: {start} to {end}\n"
                        f"Verify the missing tickers were trading during this period."
                    )
                logger.info(
                    f"Batch download complete: {len(cached_results)} ticker(s) "
                    f"({len(cached_results) - len(uncached_tickers)} cached, "
                    f"{len(uncached_tickers)} downloaded)"
                )
                return combined_prices
            else:
                pass  # fallback to standard path

        # Standard path: single ticker or cache disabled
        if use_cache:
            cache_path = self._get_cache_path(tickers, start, end)
            cached_data = self.load_cached_prices(cache_path, max_age_hours=cache_ttl_hours)
            if cached_data is not None:
                return cached_data

        data = self._download_from_yfinance(tickers, start, end)
        prices = self._process_yfinance_data(data, tickers, start, end)

        if use_cache:
            cache_path = self._get_cache_path(tickers, start, end)
            self.save_cached_prices(cache_path, prices)

        return prices

    def search_tickers(self, query: str, limit: int = 10) -> List[Tuple[str, str]]:
        """Search Yahoo Finance autocomplete API."""
        if not query or len(query) < 1:
            return []

        try:
            url = "https://query2.finance.yahoo.com/v1/finance/search"
            params = {
                "q": query,
                "quotesCount": limit,
                "newsCount": 0,
                "enableFuzzyQuery": False,
                "quotesQueryId": "tss_match_phrase_query",
            }
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://finance.yahoo.com",
                "Origin": "https://finance.yahoo.com",
            }

            response = requests.get(url, params=params, headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            quotes = data.get("quotes", [])

            results = []
            for quote in quotes[:limit]:
                symbol = quote.get("symbol", "")
                name = quote.get("longname") or quote.get("shortname", "")
                if symbol and name:
                    results.append((symbol, name))

            logger.info(f"Yahoo Finance search for '{query}': found {len(results)} results")
            return results

        except Exception as e:
            logger.warning(f"Yahoo Finance search failed for '{query}': {e}")
            return []

    def get_ticker_name(self, ticker: str) -> str:
        """Fetch full name from yfinance Ticker.info."""
        if not ticker or not ticker.strip():
            return ""

        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            if not info:
                logger.warning(f"No info available for ticker: {ticker}")
                return ""

            name = info.get("longName") or info.get("shortName") or ""
            if name:
                logger.info(f"Fetched ticker name for {ticker}: {name}")
            else:
                logger.warning(f"No name fields available for ticker: {ticker}")
            return name

        except Exception as e:
            logger.warning(f"Error fetching ticker name for {ticker}: {e}")
            return ""


class MockRepository(DataRepository):
    """Mock implementation for testing.

    Args:
        prices: DataFrame to return from get_prices
        search_results: List of (ticker, name) tuples for search_tickers
        ticker_names: Dict mapping ticker -> name for get_ticker_name
    """

    def __init__(
        self,
        prices: pd.DataFrame | None = None,
        search_results: List[Tuple[str, str]] | None = None,
        ticker_names: dict[str, str] | None = None,
    ):
        self.prices = prices
        self.search_results = search_results or []
        self.ticker_names = ticker_names or {}
        self.get_prices_calls: List[tuple] = []
        self.search_tickers_calls: List[tuple] = []
        self.get_ticker_name_calls: List[str] = []

    def get_prices(
        self,
        tickers: List[str],
        start: str,
        end: str,
        use_cache: bool = True,
        cache_ttl_hours: int = DEFAULT_CACHE_TTL_HOURS,
    ) -> pd.DataFrame:
        self.get_prices_calls.append((tickers, start, end, use_cache, cache_ttl_hours))
        if self.prices is None:
            dates = pd.date_range(start, end, freq="B")
            data = {t: np.linspace(100, 110, len(dates)) for t in tickers}
            return pd.DataFrame(data, index=dates)
        return self.prices

    def search_tickers(self, query: str, limit: int = 10) -> List[Tuple[str, str]]:
        self.search_tickers_calls.append((query, limit))
        return self.search_results[:limit]

    def get_ticker_name(self, ticker: str) -> str:
        self.get_ticker_name_calls.append(ticker)
        return self.ticker_names.get(ticker, "")


# Module-level singleton
_default_repo: DataRepository | None = None


def get_repository() -> DataRepository:
    """Get the default data repository instance.

    Lazily initializes a YahooFinanceRepository on first call.
    """
    global _default_repo
    if _default_repo is None:
        _default_repo = YahooFinanceRepository()
    return _default_repo


def set_repository(repo: DataRepository) -> None:
    """Replace the default data repository instance.

    Use this in tests to inject MockRepository.
    """
    global _default_repo
    _default_repo = repo
