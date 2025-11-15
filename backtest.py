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
        default="2018-01-01",
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
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


def download_prices(
    tickers: List[str],
    start: str,
    end: str,
    use_cache: bool = True,
    cache_ttl_hours: int = DEFAULT_CACHE_TTL_HOURS
) -> pd.DataFrame:
    """Fetch adjusted closes for the requested tickers.

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

    # Try to load from cache first
    if use_cache:
        cache_path = get_cache_path(tickers, start, end)
        cached_data = load_cached_prices(cache_path, max_age_hours=cache_ttl_hours)
        if cached_data is not None:
            return cached_data

    # Download with retry logic
    data = _download_from_yfinance(tickers, start, end)

    # Check if data is empty before processing
    if data.empty:
        raise ValueError(
            f"No price data returned for period {start} to {end}.\n"
            f"Tickers requested: {', '.join(tickers)}\n"
            f"Please verify:\n"
            f"  1. Tickers are valid symbols\n"
            f"  2. Tickers were trading during this period\n"
            f"  3. Date range is valid (not in the future)"
        )

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

    # Save to cache
    if use_cache:
        save_cached_prices(cache_path, prices)

    return prices


def compute_metrics(
    prices: pd.DataFrame,
    weights: np.ndarray,
    benchmark: pd.Series,
    capital: float,
) -> pd.DataFrame:
    """Builds the backtest table and summary columns."""

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

    # Calculate portfolio value with the aligned data
    first_prices = aligned.iloc[0]
    units = (capital * weights) / first_prices
    portfolio_value = (aligned * units).sum(axis=1)
    portfolio_return = portfolio_value / capital - 1

    # Calculate benchmark value
    bench_units = capital / benchmark.iloc[0]
    bench_value = benchmark * bench_units
    bench_return = bench_value / capital - 1

    table = pd.DataFrame(
        {
            "portfolio_value": portfolio_value,
            "portfolio_return": portfolio_return,
            "benchmark_value": bench_value,
            "benchmark_return": bench_return,
            "active_return": portfolio_return - bench_return,
        }
    )

    return table


def summarize(series: pd.Series, capital: float) -> dict[str, float]:
    if series.empty:
        raise ValueError("Cannot summarize an empty series")

    ending_value = float(series.iloc[-1])
    total_return = ending_value / capital - 1
    days = max(1, (series.index[-1] - series.index[0]).days)
    cagr = (ending_value / capital) ** (365 / days) - 1

    # Calculate daily returns for additional metrics
    daily_returns = series.pct_change().dropna()

    # Volatility (annualized standard deviation)
    volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    sharpe_ratio = (cagr / volatility) if volatility > 0 else 0.0

    # Maximum drawdown
    cumulative = series / series.expanding().max()
    drawdown = (cumulative - 1).min()

    # Sortino ratio (downside deviation only)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside_returns) > 0 else 0.0
    sortino_ratio = (cagr / downside_std) if downside_std > 0 else 0.0

    return {
        "ending_value": ending_value,
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": drawdown,
    }


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

    logger.info("Computing backtest metrics...")
    table = compute_metrics(portfolio_prices, weights, benchmark_prices, args.capital)

    portfolio_stats = summarize(table["portfolio_value"], args.capital)
    benchmark_stats = summarize(table["benchmark_value"], args.capital)

    # Print results
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print(f"Capital: ${args.capital:,.2f}")
    print(f"Time Span: {table.index[0].date()} → {table.index[-1].date()}")
    print(f"Portfolio: {', '.join(f'{t} ({w:.1%})' for t, w in zip(tickers, weights))}")
    print(f"Benchmark: {args.benchmark}")
    print("-"*70)

    print("\nPORTFOLIO PERFORMANCE:")
    print(f"  Ending Value:    ${portfolio_stats['ending_value']:>15,.2f}")
    print(f"  Total Return:    {portfolio_stats['total_return']:>15.2%}")
    print(f"  CAGR:            {portfolio_stats['cagr']:>15.2%}")
    print(f"  Volatility:      {portfolio_stats['volatility']:>15.2%}")
    print(f"  Sharpe Ratio:    {portfolio_stats['sharpe_ratio']:>15.2f}")
    print(f"  Sortino Ratio:   {portfolio_stats['sortino_ratio']:>15.2f}")
    print(f"  Max Drawdown:    {portfolio_stats['max_drawdown']:>15.2%}")

    print("\nBENCHMARK PERFORMANCE:")
    print(f"  Ending Value:    ${benchmark_stats['ending_value']:>15,.2f}")
    print(f"  Total Return:    {benchmark_stats['total_return']:>15.2%}")
    print(f"  CAGR:            {benchmark_stats['cagr']:>15.2%}")
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
