"""Simple ETF backtest for VDCP.L/VHYD.L versus VWRA.L.

Typical workflow:

source .venv/bin/activate
python backtest.py --start 2018-01-01 --end 2024-12-31 \
    --capital 100000 --weights 0.5 0.5 --benchmark VWRA.L \
    --output results/backtest_series.csv

The script downloads daily adjusted closes via yfinance, models a
buy-and-hold portfolio with static weights, and prints summary stats plus
an optional CSV of the full time series if --output is provided.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
import re
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency yfinance. Install it via 'pip install yfinance'."
    ) from exc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
TRADING_DAYS_PER_YEAR = 252
DEFAULT_CACHE_TTL_HOURS = 24
CACHE_VERSION = "1.0"


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Decorator to retry function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 2.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exceptions: Tuple of exception types to catch (default: all Exception)

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        # Last attempt failed, raise the exception
                        raise

                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            return None  # Should never reach here
        return wrapper
    return decorator


def validate_ticker(ticker: str) -> tuple[bool, str]:
    """Validate ticker symbol format.

    Args:
        ticker: Ticker symbol to validate

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty string.

    Examples:
        >>> validate_ticker("AAPL")
        (True, "")
        >>> validate_ticker("")
        (False, "Ticker cannot be empty")
        >>> validate_ticker("123")
        (False, "Ticker cannot be all numbers: 123")
    """
    if not ticker:
        return False, "Ticker cannot be empty"

    if len(ticker) > 10:
        return False, f"Ticker too long: {ticker} (max 10 characters)"

    # Allow: letters, numbers, dots (for UK tickers like VWRA.L),
    # hyphens, carets (for indices like ^GSPC), equals (for currencies)
    if not re.match(r'^[A-Z0-9\.\-\^=]+$', ticker.upper()):
        return False, f"Invalid ticker format: {ticker} (use only letters, numbers, ., -, ^, =)"

    # Check for common invalid patterns
    if ticker.isdigit():
        return False, f"Ticker cannot be all numbers: {ticker}"

    return True, ""


def validate_tickers(tickers: List[str]) -> None:
    """Validate list of ticker symbols, raise ValueError if any invalid.

    Args:
        tickers: List of ticker symbols to validate

    Raises:
        ValueError: If any ticker is invalid, with detailed error messages

    Examples:
        >>> validate_tickers(["AAPL", "MSFT"])
        >>> validate_tickers(["AAPL", ""])
        ValueError: Invalid ticker(s) detected...
    """
    if not tickers:
        raise ValueError("No tickers provided")

    errors = []
    for ticker in tickers:
        is_valid, error_msg = validate_ticker(ticker)
        if not is_valid:
            errors.append(f"  • {error_msg}")

    if errors:
        raise ValueError(
            f"Invalid ticker(s) detected:\n" + "\n".join(errors) + "\n\n"
            f"Valid ticker examples: AAPL, MSFT, VWRA.L, ^GSPC, EURUSD=X"
        )


def validate_date_string(date_str: str) -> str:
    """Validate and normalize date string to YYYY-MM-DD format.

    Args:
        date_str: Date string in various formats (YYYY-MM-DD, YYYY/MM/DD, etc.)

    Returns:
        Normalized date string in YYYY-MM-DD format

    Raises:
        argparse.ArgumentTypeError: If date format is invalid or date is invalid

    Examples:
        >>> validate_date_string("2020-01-01")
        '2020-01-01'
        >>> validate_date_string("2020/01/01")
        '2020-01-01'
        >>> validate_date_string("invalid")
        ArgumentTypeError: Invalid date format...
    """
    try:
        # Try parsing with pandas (accepts many formats)
        dt = pd.Timestamp(date_str)

        # Check if date is not too far in the past
        if dt.year < 1970:
            raise argparse.ArgumentTypeError(
                f"Date too far in the past: {date_str} (minimum: 1970-01-01)"
            )

        # Check if date is not in the future
        if dt > pd.Timestamp.today():
            raise argparse.ArgumentTypeError(
                f"Date is in the future: {date_str}"
            )

        # Return normalized format
        return dt.strftime("%Y-%m-%d")

    except argparse.ArgumentTypeError:
        # Re-raise our custom errors
        raise
    except (ValueError, TypeError) as e:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: '{date_str}'\n"
            f"Expected format: YYYY-MM-DD (e.g., 2020-01-01)\n"
            f"Error: {str(e)}"
        )


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETF backtest helper")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["VDCP.L", "VHYD.L"],
        help="Portfolio tickers in the order matching --weights",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.5, 0.5],
        help="Portfolio weights; will be normalized if they do not sum to 1",
    )
    parser.add_argument(
        "--benchmark",
        default="VWRA.L",
        help="Benchmark ticker for comparison",
    )
    parser.add_argument(
        "--start",
        type=validate_date_string,
        default="2018-01-01",
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=validate_date_string,
        default=pd.Timestamp.today().strftime("%Y-%m-%d"),
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Initial capital applied to the portfolio and benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV destination for the full time-series table",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable price data caching (always download fresh data)",
    )
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=DEFAULT_CACHE_TTL_HOURS,
        help=f"Cache time-to-live in hours (default: {DEFAULT_CACHE_TTL_HOURS})",
    )
    parser.add_argument(
        "--rebalance",
        type=str,
        default=None,
        choices=['D', 'W', 'M', 'Q', 'Y', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
        help="Rebalancing frequency: D/daily, W/weekly, M/monthly, Q/quarterly, Y/yearly (default: None = buy-and-hold)",
    )
    parser.add_argument(
        "--dca-amount",
        type=float,
        default=None,
        help="Dollar-cost averaging: amount to contribute at each interval (default: None = lump sum)",
    )
    parser.add_argument(
        "--dca-freq",
        type=str,
        default=None,
        choices=['D', 'W', 'M', 'Q', 'Y', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
        help="DCA frequency: D/daily, W/weekly, M/monthly, Q/quarterly, Y/yearly (requires --dca-amount)",
    )
    return parser.parse_args(argv)


def get_cache_key(tickers: List[str], start: str, end: str) -> str:
    """Generate a unique cache key for the given parameters."""
    key_str = f"{'_'.join(sorted(tickers))}_{start}_{end}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cache_path(tickers: List[str], start: str, end: str) -> Path:
    """Get the cache file path for the given parameters."""
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    cache_key = get_cache_key(tickers, start, end)
    return cache_dir / f"{cache_key}.pkl"


def load_cached_prices(cache_path: Path, max_age_hours: int = DEFAULT_CACHE_TTL_HOURS) -> pd.DataFrame | None:
    """Load cached price data if it exists and is not stale.

    Args:
        cache_path: Path to cache file
        max_age_hours: Maximum age of cache in hours (default: 24)

    Returns:
        DataFrame if cache is valid and fresh, None otherwise
    """
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)

        # Handle old cache format (migration from plain DataFrame)
        if isinstance(cache_data, pd.DataFrame):
            logger.warning(f"Old cache format detected at {cache_path}, will re-download")
            cache_path.unlink()  # Delete old format cache
            return None

        # Check cache age
        cache_age_hours = (time.time() - cache_data["timestamp"]) / 3600
        if cache_age_hours > max_age_hours:
            logger.info(f"Cache expired (age: {cache_age_hours:.1f}h, max: {max_age_hours}h)")
            cache_path.unlink()  # Delete stale cache
            return None

        logger.info(f"Loaded cached data (age: {cache_age_hours:.1f}h from {cache_path})")
        return cache_data["data"]

    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        # Try to clean up corrupted cache file
        try:
            cache_path.unlink()
        except:
            pass
        return None


def save_cached_prices(cache_path: Path, prices: pd.DataFrame) -> None:
    """Save price data to cache with metadata.

    Args:
        cache_path: Path where cache file will be saved
        prices: DataFrame containing price data to cache
    """
    try:
        cache_data = {
            "data": prices,
            "timestamp": time.time(),
            "version": CACHE_VERSION
        }
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved data to cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


@retry_with_backoff(max_retries=3, base_delay=2.0)
def _download_from_yfinance(tickers: List[str], start: str, end: str) -> Any:
    """Internal function to download from yfinance with retry logic.

    Args:
        tickers: List of ticker symbols
        start: Start date
        end: End date

    Returns:
        Raw data from yfinance (format varies based on number of tickers)
    """
    logger.info(f"Downloading data for {len(tickers)} ticker(s) from {start} to {end}")

    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )


def validate_price_data(df: pd.DataFrame, tickers: List[str]) -> None:
    """Validate price data quality.

    Args:
        df: DataFrame containing price data
        tickers: List of ticker symbols

    Raises:
        ValueError: If data quality issues are detected
    """
    issues = []

    for ticker in tickers:
        if ticker not in df.columns:
            continue  # Will be caught by other validation

        series = df[ticker]

        # Check for all NaN
        if series.isna().all():
            issues.append(f"{ticker}: all values are NaN (no price data available)")
            continue

        # Check for excessive NaN (>50%)
        nan_pct = series.isna().sum() / len(series)
        if nan_pct > 0.5:
            issues.append(
                f"{ticker}: {nan_pct:.1%} missing values (>50% threshold)"
            )

        # Check for zero/negative prices (after dropping NaN)
        valid_prices = series.dropna()
        if len(valid_prices) > 0:
            if (valid_prices <= 0).any():
                zero_count = (valid_prices == 0).sum()
                neg_count = (valid_prices < 0).sum()
                issue_parts = []
                if zero_count > 0:
                    issue_parts.append(f"{zero_count} zero price(s)")
                if neg_count > 0:
                    issue_parts.append(f"{neg_count} negative price(s)")
                issues.append(f"{ticker}: contains {', '.join(issue_parts)}")

            # Check for extreme price changes (>90% in single day - likely data error)
            price_changes = valid_prices.pct_change().dropna()
            if len(price_changes) > 0:
                extreme_changes = price_changes[price_changes.abs() > 0.9]
                if len(extreme_changes) > 0:
                    max_change = extreme_changes.abs().max()
                    issues.append(
                        f"{ticker}: contains extreme price change ({max_change:.1%}/day - possible data error)"
                    )

    if issues:
        raise ValueError(
            "Price data quality issues detected:\n" +
            "\n".join(f"  • {issue}" for issue in issues)
        )


def _process_yfinance_data(data: Any, tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Process raw yfinance data into standardized DataFrame format.

    Args:
        data: Raw data from yfinance
        tickers: List of ticker symbols
        start: Start date for error messages
        end: End date for error messages

    Returns:
        DataFrame with adjusted close prices for all tickers

    Raises:
        ValueError: If data is empty or missing required tickers
    """
    # Check if data is empty
    if data.empty:
        raise ValueError(
            f"No price data returned for period {start} to {end}.\n"
            f"Tickers requested: {', '.join(tickers)}\n"
            f"Please verify:\n"
            f"  1. Tickers are valid symbols\n"
            f"  2. Tickers were trading during this period\n"
            f"  3. Date range is valid (not in the future)"
        )

    # Handle different yfinance return formats
    if isinstance(data, pd.Series):  # single ticker result
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
        preferred = next((c for c in ("Adj Close", "Close") if c in data.columns), None)
        column = preferred or data.columns[0]
        prices = data[[column]].rename(columns={column: tickers[0]})

    # Drop rows with all NaN values
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

    # Ensure DataFrame format
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    # Check for missing tickers
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        available = [t for t in tickers if t not in missing]
        raise ValueError(
            f"Missing data for ticker(s): {', '.join(missing)}\n"
            f"Date range: {start} to {end}\n"
            f"Available ticker(s): {', '.join(available) if available else 'None'}\n"
            f"Verify the missing tickers were trading during this period."
        )

    # Return tickers in requested order
    prices = prices.loc[:, tickers]

    # Validate data quality
    validate_price_data(prices, tickers)

    return prices


def download_prices(
    tickers: List[str],
    start: str,
    end: str,
    use_cache: bool = True,
    cache_ttl_hours: int = DEFAULT_CACHE_TTL_HOURS
) -> pd.DataFrame:
    """Fetch adjusted closes for the requested tickers.

    Implements optimized batch downloading with per-ticker caching:
    - Checks cache individually for each ticker
    - Downloads only uncached tickers
    - Combines cached and fresh data efficiently

    Args:
        tickers: List of ticker symbols to download
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        use_cache: Whether to use cached data (default: True)
        cache_ttl_hours: Cache time-to-live in hours (default: 24)

    Returns:
        DataFrame with adjusted close prices for all tickers

    Raises:
        ValueError: If any ticker is invalid
    """

    # Validate tickers before attempting download
    validate_tickers(tickers)

    # Optimized batch caching: check each ticker individually
    if use_cache and len(tickers) > 1:
        cached_results = {}
        uncached_tickers = []

        for ticker in tickers:
            # Check individual ticker cache
            single_cache_path = get_cache_path([ticker], start, end)
            cached_data = load_cached_prices(single_cache_path, max_age_hours=cache_ttl_hours)

            if cached_data is not None:
                cached_results[ticker] = cached_data[ticker]
                logger.info(f"Cache hit for {ticker}")
            else:
                uncached_tickers.append(ticker)

        # Download only uncached tickers
        if uncached_tickers:
            logger.info(f"Downloading {len(uncached_tickers)} uncached ticker(s): {', '.join(uncached_tickers)}")
            new_data = _download_from_yfinance(uncached_tickers, start, end)

            # Process downloaded data
            new_prices = _process_yfinance_data(new_data, uncached_tickers, start, end)

            # Cache each newly downloaded ticker individually
            for ticker in uncached_tickers:
                if ticker in new_prices.columns:
                    single_cache_path = get_cache_path([ticker], start, end)
                    save_cached_prices(single_cache_path, new_prices[[ticker]])
                    cached_results[ticker] = new_prices[ticker]

        # Combine all results in original order
        if cached_results:
            combined_prices = pd.DataFrame({ticker: cached_results[ticker] for ticker in tickers if ticker in cached_results})

            # Ensure all requested tickers are present
            missing = [ticker for ticker in tickers if ticker not in combined_prices.columns]
            if missing:
                raise ValueError(
                    f"Missing data for ticker(s): {', '.join(missing)}\n"
                    f"Date range: {start} to {end}\n"
                    f"Verify the missing tickers were trading during this period."
                )

            logger.info(f"Batch download complete: {len(cached_results)} ticker(s) ({len(cached_results) - len(uncached_tickers)} cached, {len(uncached_tickers)} downloaded)")
            return combined_prices
        else:
            # Fallback to normal download if no cache hits
            pass

    # Standard path: single ticker or cache disabled
    if use_cache:
        cache_path = get_cache_path(tickers, start, end)
        cached_data = load_cached_prices(cache_path, max_age_hours=cache_ttl_hours)
        if cached_data is not None:
            return cached_data

    # Download with retry logic
    data = _download_from_yfinance(tickers, start, end)

    # Process the downloaded data
    prices = _process_yfinance_data(data, tickers, start, end)

    # Save to cache
    if use_cache:
        save_cached_prices(cache_path, prices)

    return prices


def _calculate_rebalanced_portfolio(
    prices: pd.DataFrame,
    weights: np.ndarray,
    capital: float,
    rebalance_freq: str
) -> pd.Series:
    """Calculate portfolio value with periodic rebalancing.

    Args:
        prices: DataFrame with aligned prices for all portfolio tickers
        weights: Array of target portfolio weights
        capital: Initial capital amount
        rebalance_freq: Pandas frequency string ('D', 'W', 'M', 'Q', 'Y')

    Returns:
        Series with portfolio value for each date
    """
    # Generate rebalancing dates
    rebalance_dates = pd.date_range(
        start=prices.index[0],
        end=prices.index[-1],
        freq=rebalance_freq
    )

    # Ensure first date is included
    if prices.index[0] not in rebalance_dates:
        rebalance_dates = rebalance_dates.insert(0, prices.index[0])

    # Filter to only dates that exist in our price data
    rebalance_dates = rebalance_dates.intersection(prices.index)

    if len(rebalance_dates) == 0:
        logger.warning(f"No rebalancing dates found for frequency '{rebalance_freq}'. Using buy-and-hold.")
        first_prices = prices.iloc[0]
        units = (capital * weights) / first_prices
        return (prices * units).sum(axis=1)

    # Initialize portfolio value series
    portfolio_values = pd.Series(index=prices.index, dtype=float)

    # Track current units and portfolio value
    current_value = capital
    current_units = None

    # Process each date
    for date in prices.index:
        # Check if we need to rebalance
        if date in rebalance_dates or current_units is None:
            # Rebalance: allocate current_value according to target weights
            current_prices = prices.loc[date]
            current_units = (current_value * weights) / current_prices
            logger.debug(f"Rebalancing on {date.strftime('%Y-%m-%d')}: portfolio value = ${current_value:,.2f}")

        # Calculate portfolio value for this date
        current_value = (prices.loc[date] * current_units).sum()
        portfolio_values[date] = current_value

    logger.info(
        f"Rebalancing strategy '{rebalance_freq}': "
        f"{len(rebalance_dates)} rebalances from {rebalance_dates[0].strftime('%Y-%m-%d')} "
        f"to {rebalance_dates[-1].strftime('%Y-%m-%d')}"
    )

    return portfolio_values


def _calculate_dca_portfolio(
    prices: pd.DataFrame,
    weights: np.ndarray,
    capital: float,
    dca_amount: float,
    dca_freq: str
) -> tuple[pd.Series, pd.Series]:
    """Calculate portfolio value with Dollar-Cost Averaging (regular contributions).

    Args:
        prices: DataFrame with aligned prices for all portfolio tickers
        weights: Array of target portfolio weights
        capital: Initial capital amount
        dca_amount: Amount to contribute at each DCA interval
        dca_freq: Pandas frequency string for contributions ('D', 'W', 'M', 'Q', 'Y')

    Returns:
        Tuple of (portfolio_values, cumulative_contributions):
            - portfolio_values: Series with portfolio market value for each date
            - cumulative_contributions: Series with total amount invested up to each date
    """
    # Generate DCA contribution dates
    dca_dates = pd.date_range(
        start=prices.index[0],
        end=prices.index[-1],
        freq=dca_freq
    )

    # Ensure first date is included for initial investment
    if prices.index[0] not in dca_dates:
        dca_dates = dca_dates.insert(0, prices.index[0])

    # Map DCA dates to actual trading days (handle weekends/holidays)
    # If a DCA date falls on a weekend/holiday, use the next available trading day
    actual_dca_dates = []
    for dca_date in dca_dates:
        # Find the next available trading day on or after the DCA date
        available_dates = prices.index[prices.index >= dca_date]
        if len(available_dates) > 0:
            actual_dca_dates.append(available_dates[0])

    dca_dates = pd.DatetimeIndex(actual_dca_dates).unique()  # Remove duplicates

    if len(dca_dates) == 0:
        logger.warning(f"No DCA dates found for frequency '{dca_freq}'. Using lump sum.")
        first_prices = prices.iloc[0]
        units = (capital * weights) / first_prices
        portfolio_values = (prices * units).sum(axis=1)
        cumulative_contributions = pd.Series(capital, index=prices.index, dtype=float)
        return portfolio_values, cumulative_contributions

    # Initialize series
    portfolio_values = pd.Series(index=prices.index, dtype=float)
    cumulative_contributions = pd.Series(index=prices.index, dtype=float)

    # Track current units held for each ticker and total invested
    current_units = np.zeros(len(weights))
    total_invested = 0.0

    # Process each date
    for date in prices.index:
        # Check if we have a contribution on this date
        if date in dca_dates:
            # Determine contribution amount (initial capital on first date, then DCA amount)
            contribution = capital if date == prices.index[0] else dca_amount
            total_invested += contribution

            # Buy shares according to target weights at current prices
            current_prices = prices.loc[date]
            new_units = (contribution * weights) / current_prices
            current_units += new_units

            logger.debug(
                f"DCA contribution on {date.strftime('%Y-%m-%d')}: "
                f"${contribution:,.2f} invested (total: ${total_invested:,.2f})"
            )

        # Calculate portfolio value and cumulative contributions for this date
        current_value = (prices.loc[date] * current_units).sum()
        portfolio_values[date] = current_value
        cumulative_contributions[date] = total_invested

    total_contributions = capital + (dca_amount * (len(dca_dates) - 1))
    logger.info(
        f"DCA strategy '{dca_freq}': "
        f"{len(dca_dates)} contributions (initial ${capital:,.2f} + "
        f"{len(dca_dates) - 1} × ${dca_amount:,.2f} = ${total_contributions:,.2f} total)"
    )

    return portfolio_values, cumulative_contributions


def compute_metrics(
    prices: pd.DataFrame,
    weights: np.ndarray,
    benchmark: pd.Series,
    capital: float,
    rebalance_freq: str = None,
    dca_amount: float = None,
    dca_freq: str = None,
) -> pd.DataFrame:
    """Builds the backtest table and summary columns.

    Args:
        prices: DataFrame with adjusted close prices for portfolio tickers
        weights: Array of portfolio weights (must sum to 1.0)
        benchmark: Series with benchmark prices
        capital: Initial capital amount
        rebalance_freq: Rebalancing frequency - None (buy-and-hold), 'M' (monthly),
                       'Q' (quarterly), 'Y' (yearly), 'W' (weekly), 'D' (daily)
        dca_amount: Dollar-cost averaging contribution amount (None for lump sum)
        dca_freq: DCA frequency - None (lump sum), 'M' (monthly), 'Q' (quarterly),
                 'Y' (yearly), 'W' (weekly), 'D' (daily)

    Returns:
        DataFrame with portfolio metrics including value, returns, and active return

    Note:
        DCA and rebalancing are mutually exclusive. If both are specified, DCA takes precedence.
    """

    prices = prices.sort_index()
    first_valid_points = [series.first_valid_index() for _, series in prices.items()]
    if any(idx is None for idx in first_valid_points):
        tickers_status = [
            f"{ticker}: {'OK' if idx is not None else 'NO DATA'}"
            for ticker, idx in zip(prices.columns, first_valid_points)
        ]
        raise ValueError(
            f"One or more tickers have no valid prices in this window:\n"
            + "\n".join(f"  - {s}" for s in tickers_status)
        )
    start_date = max(first_valid_points)
    aligned = prices.loc[start_date:].ffill().dropna()
    if aligned.empty:
        earliest_dates = {
            ticker: idx.strftime("%Y-%m-%d") if idx else "N/A"
            for ticker, idx in zip(prices.columns, first_valid_points)
        }
        raise ValueError(
            f"Not enough overlapping history for portfolio tickers.\n"
            f"Ticker start dates: {earliest_dates}\n"
            f"Common start would be: {start_date.strftime('%Y-%m-%d')}\n"
            f"Try using a later start date or different tickers."
        )

    # Align benchmark to the same index, forward-filling gaps once it starts trading
    benchmark = benchmark.sort_index()
    bench_start = benchmark.first_valid_index()
    if bench_start is None:
        raise ValueError(
            "Benchmark has no data in the requested window.\n"
            "Verify the benchmark ticker is valid and traded during this period."
        )
    combined_start = max(aligned.index[0], bench_start)
    aligned = aligned.loc[combined_start:]
    benchmark = benchmark.loc[combined_start:].reindex(aligned.index).ffill().dropna()
    if benchmark.empty:
        raise ValueError(
            f"Benchmark has no overlapping data in this window.\n"
            f"Portfolio starts: {aligned.index[0].strftime('%Y-%m-%d')}\n"
            f"Benchmark starts: {bench_start.strftime('%Y-%m-%d')}\n"
            f"Try adjusting the date range or choose a different benchmark."
        )

    # Validate sufficient data after alignment
    if len(aligned) < 2:
        raise ValueError(
            f"Insufficient overlapping data: only {len(aligned)} trading day(s).\n"
            f"Need at least 2 days for meaningful backtest.\n"
            f"Try using a longer date range or different tickers."
        )

    if len(aligned) < 30:
        logger.warning(
            f"Limited data: only {len(aligned)} trading days. "
            f"Statistics may be unreliable for periods < 30 days."
        )

    # Calculate portfolio value based on strategy
    # Priority: DCA > Rebalancing > Buy-and-hold
    portfolio_contributions = None  # Track for DCA
    if dca_freq is not None and dca_amount is not None and dca_amount > 0:
        # DCA strategy (dollar-cost averaging with regular contributions)
        if rebalance_freq is not None:
            logger.warning(
                "Both DCA and rebalancing specified. Using DCA strategy only. "
                "DCA and rebalancing are mutually exclusive."
            )
        portfolio_value, portfolio_contributions = _calculate_dca_portfolio(aligned, weights, capital, dca_amount, dca_freq)
    elif rebalance_freq is None:
        # Buy-and-hold strategy (lump sum, no rebalancing)
        first_prices = aligned.iloc[0]
        units = (capital * weights) / first_prices
        portfolio_value = (aligned * units).sum(axis=1)
        portfolio_contributions = pd.Series(capital, index=aligned.index, dtype=float)
    else:
        # Rebalancing strategy (lump sum with periodic rebalancing)
        portfolio_value = _calculate_rebalanced_portfolio(aligned, weights, capital, rebalance_freq)
        portfolio_contributions = pd.Series(capital, index=aligned.index, dtype=float)

    # Calculate returns based on cumulative contributions (accurate for DCA)
    portfolio_return = (portfolio_value - portfolio_contributions) / portfolio_contributions

    # Calculate benchmark value with same strategy as portfolio for fair comparison
    bench_contributions = None  # Track for DCA
    if dca_freq is not None and dca_amount is not None and dca_amount > 0:
        # Apply DCA to benchmark for fair comparison
        benchmark_df = benchmark.to_frame(name='benchmark')
        bench_weights = np.array([1.0])  # 100% in benchmark
        bench_value, bench_contributions = _calculate_dca_portfolio(benchmark_df, bench_weights, capital, dca_amount, dca_freq)
    elif rebalance_freq is not None:
        # Apply rebalancing to benchmark (though it's a single asset, so effectively same as buy-and-hold)
        benchmark_df = benchmark.to_frame(name='benchmark')
        bench_weights = np.array([1.0])
        bench_value = _calculate_rebalanced_portfolio(benchmark_df, bench_weights, capital, rebalance_freq)
        bench_contributions = pd.Series(capital, index=benchmark_df.index, dtype=float)
    else:
        # Buy-and-hold for benchmark
        bench_units = capital / benchmark.iloc[0]
        bench_value = benchmark * bench_units
        bench_contributions = pd.Series(capital, index=benchmark.index, dtype=float)

    # Calculate returns based on cumulative contributions (accurate for DCA)
    bench_return = (bench_value - bench_contributions) / bench_contributions

    table = pd.DataFrame(
        {
            "portfolio_value": portfolio_value,
            "portfolio_return": portfolio_return,
            "portfolio_contributions": portfolio_contributions,
            "benchmark_value": bench_value,
            "benchmark_return": bench_return,
            "benchmark_contributions": bench_contributions,
            "active_return": portfolio_return - bench_return,
        }
    )

    # Calculate rolling 12-month Sharpe ratio (252 trading days)
    # This shows how risk-adjusted performance evolves over time
    window = 252  # Approximately 12 months of trading days

    def calculate_rolling_sharpe(values: pd.Series, window_size: int) -> pd.Series:
        """Calculate rolling Sharpe ratio over a specified window.

        Args:
            values: Series of portfolio/benchmark values
            window_size: Rolling window size in trading days

        Returns:
            Series of rolling Sharpe ratios (NaN for insufficient data)
        """
        # Calculate daily returns
        daily_returns = values.pct_change()

        # Calculate rolling mean and std of returns
        rolling_mean = daily_returns.rolling(window=window_size).mean()
        rolling_std = daily_returns.rolling(window=window_size).std()

        # Annualize: mean * 252, std * sqrt(252)
        annualized_return = rolling_mean * TRADING_DAYS_PER_YEAR
        annualized_volatility = rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Sharpe ratio = annualized_return / annualized_volatility
        # Division will produce NaN for insufficient data (from rolling window)
        # and inf for zero volatility
        sharpe = annualized_return / annualized_volatility

        # Replace inf/-inf (from zero volatility) with 0, but preserve NaN
        sharpe = sharpe.replace([np.inf, -np.inf], 0.0)

        return sharpe

    table["portfolio_rolling_sharpe_12m"] = calculate_rolling_sharpe(portfolio_value, window)
    table["benchmark_rolling_sharpe_12m"] = calculate_rolling_sharpe(bench_value, window)

    logger.info(
        f"Calculated rolling 12-month Sharpe ratio "
        f"({len(table[table['portfolio_rolling_sharpe_12m'].notna()])} valid values)"
    )

    return table


def _calculate_xirr(cashflows: np.ndarray, dates_in_days: np.ndarray, guess: float = 0.1, max_iterations: int = 100, tolerance: float = 1e-6) -> float:
    """Calculate XIRR (time-weighted Internal Rate of Return) using Newton-Raphson method.

    Args:
        cashflows: Array of cashflows (negative for outflows, positive for inflows)
        dates_in_days: Array of days since first cashflow
        guess: Initial guess for annual rate
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance

    Returns:
        Annualized IRR as a decimal (e.g., 0.10 for 10%)
        Returns None if calculation fails or doesn't converge
    """
    rate = guess

    for iteration in range(max_iterations):
        # Calculate NPV and its derivative using actual time periods
        npv = 0.0
        npv_derivative = 0.0

        for cf, days in zip(cashflows, dates_in_days):
            years = days / 365.0
            discount_factor = (1 + rate) ** years

            npv += cf / discount_factor

            if years > 0:
                npv_derivative -= years * cf / (discount_factor * (1 + rate))

        # Check for convergence
        if abs(npv) < tolerance:
            return rate

        # Avoid division by zero
        if abs(npv_derivative) < 1e-10:
            return None

        # Newton-Raphson update
        new_rate = rate - npv / npv_derivative

        # Avoid extreme values
        if new_rate < -0.99 or new_rate > 10.0:
            return None

        rate = new_rate

    # Didn't converge
    return None


def summarize(
    series: pd.Series,
    capital: float,
    total_contributions: float = None,
    contributions_series: pd.Series = None
) -> dict[str, float]:
    """Calculate summary statistics for a portfolio or benchmark series.

    Args:
        series: Time series of portfolio/benchmark values
        capital: Initial capital amount
        total_contributions: Total amount contributed (for DCA). If None, uses capital.
        contributions_series: Series of cumulative contributions over time (for IRR calculation)

    Returns:
        Dictionary of performance metrics

    Note:
        For DCA strategies, total_contributions should include all contributions.
        Returns and CAGR are calculated based on total invested amount.
        IRR is calculated when contributions_series is provided.
    """
    if series.empty:
        raise ValueError("Cannot summarize an empty series")

    # Use total contributions if provided (for DCA), otherwise use initial capital
    if total_contributions is None:
        total_contributions = capital

    ending_value = float(series.iloc[-1])
    total_return = (ending_value - total_contributions) / total_contributions
    days = max(1, (series.index[-1] - series.index[0]).days)

    # CAGR calculation for DCA is approximate (true metric would be IRR)
    # This gives a reasonable approximation for comparison purposes
    cagr = (ending_value / total_contributions) ** (365 / days) - 1

    # Calculate IRR for DCA strategies (more accurate than CAGR)
    irr = None
    if contributions_series is not None:
        # Build cashflow array: contributions as outflows (negative), final value as inflow (positive)
        contrib_changes = contributions_series.diff().fillna(contributions_series.iloc[0])

        # Find dates where contributions occurred (non-zero changes)
        contrib_dates = contrib_changes[contrib_changes > 0].index

        if len(contrib_dates) > 1:  # Only calculate IRR if multiple contributions
            # Build cashflows: each contribution as negative, final value as positive
            cashflows = []
            cashflow_days = []

            for date in contrib_dates:
                cashflows.append(-contrib_changes[date])  # Contribution as outflow (negative)
                cashflow_days.append((date - series.index[0]).days)

            # Add final value as positive inflow
            cashflows.append(ending_value)
            cashflow_days.append((series.index[-1] - series.index[0]).days)

            # Calculate XIRR using time-weighted cashflows
            cashflows_array = np.array(cashflows)
            days_array = np.array(cashflow_days)

            irr = _calculate_xirr(cashflows_array, days_array)

            # Validate IRR result
            if irr is not None:
                # Sanity check: IRR should be reasonable (-99% to 1000%)
                if irr < -0.99 or irr > 10.0:
                    logger.warning(
                        f"IRR calculation produced unrealistic value ({irr:.2%}). "
                        f"Using CAGR instead."
                    )
                    irr = None

    # Calculate daily returns for additional metrics
    # For DCA strategies, we need to exclude the impact of contributions
    if contributions_series is not None:
        # Calculate contribution-adjusted returns
        # Daily change in value minus any new contributions
        value_changes = series.diff()
        contribution_changes = contributions_series.diff().fillna(0)
        market_value_changes = value_changes - contribution_changes

        # Calculate returns based on previous day's value
        # This gives the true market return, excluding contribution impact
        daily_returns = market_value_changes / series.shift(1)
        daily_returns = daily_returns.dropna()
    else:
        # For lump sum investments, use simple percentage change
        daily_returns = series.pct_change().dropna()

    # Volatility (annualized standard deviation)
    volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    # For DCA strategies, we could use IRR instead of CAGR for better accuracy
    annualized_return = irr if (irr is not None) else cagr
    sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0.0

    # Maximum drawdown
    if contributions_series is not None:
        # For DCA, calculate drawdown on return percentage (value - contributions) / contributions
        # This shows the maximum decline from a peak return percentage
        return_pct = (series - contributions_series) / contributions_series

        # Track the running maximum return percentage
        running_max = return_pct.expanding().max()

        # Drawdown at each point = current return % - peak return %
        drawdown_series = return_pct - running_max

        # Max drawdown is the worst (most negative) drawdown
        drawdown = drawdown_series.min()
    else:
        # For lump sum, use traditional drawdown calculation on absolute value
        cumulative = series / series.expanding().max()
        drawdown = (cumulative - 1).min()

    # Sortino ratio (downside deviation only)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside_returns) > 0 else 0.0
    sortino_ratio = (annualized_return / downside_std) if downside_std > 0 else 0.0

    metrics = {
        "ending_value": ending_value,
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": drawdown,
        "total_contributions": total_contributions,  # Include for reference
    }

    # Add IRR if calculated (for DCA strategies)
    if irr is not None:
        metrics["irr"] = irr

    return metrics


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    tickers = args.tickers
    weights = np.array(args.weights, dtype=float)

    # Validate tickers early
    try:
        validate_tickers(tickers)
        validate_tickers([args.benchmark])
    except ValueError as e:
        raise SystemExit(f"Ticker validation failed:\n{e}")

    # Validate date range
    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)

    if start_dt >= end_dt:
        raise SystemExit(
            f"Invalid date range: start ({args.start}) must be before end ({args.end})"
        )

    days_in_range = (end_dt - start_dt).days
    if days_in_range < 30:
        logger.warning(
            f"Short backtest period: {days_in_range} days. "
            f"Results may be unreliable for periods < 30 days."
        )

    if len(tickers) != len(weights):
        raise SystemExit("Number of weights must match number of tickers")
    if not np.isclose(weights.sum(), 1.0):
        logger.info(f"Normalizing weights from {weights} to sum to 1.0")
        weights = weights / weights.sum()

    universe = list(dict.fromkeys(tickers + [args.benchmark]))
    use_cache = not args.no_cache
    prices = download_prices(
        universe,
        args.start,
        args.end,
        use_cache=use_cache,
        cache_ttl_hours=args.cache_ttl
    )

    portfolio_prices = prices[tickers]
    benchmark_prices = prices[args.benchmark]

    # Normalize rebalancing frequency
    rebalance_freq = args.rebalance
    if rebalance_freq:
        # Convert long form to pandas frequency codes
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M',
            'quarterly': 'Q',
            'yearly': 'Y'
        }
        rebalance_freq = freq_map.get(rebalance_freq.lower(), rebalance_freq.upper())

    # Normalize DCA frequency
    dca_freq = args.dca_freq
    dca_amount = args.dca_amount
    if dca_freq:
        # Convert long form to pandas frequency codes
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M',
            'quarterly': 'Q',
            'yearly': 'Y'
        }
        dca_freq = freq_map.get(dca_freq.lower(), dca_freq.upper())

    # Validate DCA parameters
    if (dca_freq is not None and dca_amount is None) or (dca_amount is not None and dca_freq is None):
        raise SystemExit("DCA requires both --dca-amount and --dca-freq to be specified")

    if dca_amount is not None and dca_amount <= 0:
        raise SystemExit("DCA amount must be positive")

    logger.info("Computing backtest metrics...")
    table = compute_metrics(
        portfolio_prices,
        weights,
        benchmark_prices,
        args.capital,
        rebalance_freq=rebalance_freq,
        dca_amount=dca_amount,
        dca_freq=dca_freq
    )

    # Get total contributions for accurate return calculations
    portfolio_total_contrib = table["portfolio_contributions"].iloc[-1]
    benchmark_total_contrib = table["benchmark_contributions"].iloc[-1]

    # Pass contributions series for IRR calculation (only for DCA strategies)
    portfolio_stats = summarize(
        table["portfolio_value"],
        args.capital,
        portfolio_total_contrib,
        contributions_series=table["portfolio_contributions"] if (dca_freq and dca_amount) else None
    )
    benchmark_stats = summarize(
        table["benchmark_value"],
        args.capital,
        benchmark_total_contrib,
        contributions_series=table["benchmark_contributions"] if (dca_freq and dca_amount) else None
    )

    # Print results
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print(f"Initial Capital: ${args.capital:,.2f}")
    if dca_freq and dca_amount:
        print(f"Total Contributions: ${portfolio_total_contrib:,.2f} (Portfolio & Benchmark)")
    print(f"Time Span: {table.index[0].date()} → {table.index[-1].date()}")
    print(f"Portfolio: {', '.join(f'{t} ({w:.1%})' for t, w in zip(tickers, weights))}")
    print(f"Benchmark: {args.benchmark}")

    # Show strategy
    if dca_freq and dca_amount:
        freq_names = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly', 'Y': 'Yearly'}
        strategy_name = freq_names.get(dca_freq, dca_freq)
        print(f"Strategy: Dollar-Cost Averaging ({strategy_name}, ${dca_amount:,.2f}/contribution)")
    elif rebalance_freq:
        freq_names = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly', 'Y': 'Yearly'}
        strategy_name = freq_names.get(rebalance_freq, rebalance_freq)
        print(f"Strategy: {strategy_name} Rebalancing")
    else:
        print(f"Strategy: Buy-and-Hold")

    print("-"*70)

    print("\nPORTFOLIO PERFORMANCE:")
    print(f"  Ending Value:    ${portfolio_stats['ending_value']:>15,.2f}")
    print(f"  Total Return:    {portfolio_stats['total_return']:>15.2%}")
    print(f"  CAGR:            {portfolio_stats['cagr']:>15.2%}")
    if 'irr' in portfolio_stats:
        print(f"  IRR:             {portfolio_stats['irr']:>15.2%}")
    print(f"  Volatility:      {portfolio_stats['volatility']:>15.2%}")
    print(f"  Sharpe Ratio:    {portfolio_stats['sharpe_ratio']:>15.2f}")
    print(f"  Sortino Ratio:   {portfolio_stats['sortino_ratio']:>15.2f}")
    print(f"  Max Drawdown:    {portfolio_stats['max_drawdown']:>15.2%}")

    print("\nBENCHMARK PERFORMANCE:")
    print(f"  Ending Value:    ${benchmark_stats['ending_value']:>15,.2f}")
    print(f"  Total Return:    {benchmark_stats['total_return']:>15.2%}")
    print(f"  CAGR:            {benchmark_stats['cagr']:>15.2%}")
    if 'irr' in benchmark_stats:
        print(f"  IRR:             {benchmark_stats['irr']:>15.2%}")
    print(f"  Volatility:      {benchmark_stats['volatility']:>15.2%}")
    print(f"  Sharpe Ratio:    {benchmark_stats['sharpe_ratio']:>15.2f}")
    print(f"  Sortino Ratio:   {benchmark_stats['sortino_ratio']:>15.2f}")
    print(f"  Max Drawdown:    {benchmark_stats['max_drawdown']:>15.2%}")

    print("\nRELATIVE PERFORMANCE:")
    active_return = portfolio_stats["total_return"] - benchmark_stats["total_return"]
    active_cagr = portfolio_stats["cagr"] - benchmark_stats["cagr"]
    print(f"  Active Return:   {active_return:>15.2%}")
    print(f"  Active CAGR:     {active_cagr:>15.2%}")
    print("="*70 + "\n")

    if args.output:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index_label="date")
        logger.info(f"Saved detailed series to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
