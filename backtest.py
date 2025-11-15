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
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency yfinance. Install it via 'pip install yfinance'."
    ) from exc


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
    return parser.parse_args(argv)


def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted closes for the requested tickers."""

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
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
        raise ValueError("No price rows returned; check tickers or date range")
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise ValueError(f"Missing data for tickers: {', '.join(missing)}")
    prices = prices.loc[:, tickers]
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
        raise ValueError("One or more tickers have no valid prices in this window")
    start_date = max(first_valid_points)
    aligned = prices.loc[start_date:].ffill().dropna()
    if aligned.empty:
        raise ValueError(
            "Not enough overlapping history for portfolio tickers; try a later start"
        )

    first_prices = aligned.iloc[0]
    units = (capital * weights) / first_prices
    portfolio_value = (aligned * units).sum(axis=1)
    portfolio_return = portfolio_value / capital - 1

    # Align benchmark to the same index, forward-filling gaps once it starts trading
    benchmark = benchmark.sort_index()
    bench_start = benchmark.first_valid_index()
    if bench_start is None:
        raise ValueError("Benchmark has no data in this window")
    combined_start = max(aligned.index[0], bench_start)
    aligned = aligned.loc[combined_start:]
    benchmark = benchmark.loc[combined_start:].reindex(aligned.index).ffill().dropna()
    if benchmark.empty:
        raise ValueError("Benchmark has no overlapping data in this window")
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
    return {
        "ending_value": ending_value,
        "total_return": total_return,
        "cagr": cagr,
    }


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    tickers = args.tickers
    weights = np.array(args.weights, dtype=float)
    if len(tickers) != len(weights):
        raise SystemExit("Number of weights must match number of tickers")
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()

    universe = list(dict.fromkeys(tickers + [args.benchmark]))
    prices = download_prices(universe, args.start, args.end)

    portfolio_prices = prices[tickers]
    benchmark_prices = prices[args.benchmark]

    table = compute_metrics(portfolio_prices, weights, benchmark_prices, args.capital)

    portfolio_stats = summarize(table["portfolio_value"], args.capital)
    benchmark_stats = summarize(table["benchmark_value"], args.capital)

    print("Portfolio vs Benchmark (capital: {:.2f})".format(args.capital))
    print("Time span: {} → {}".format(table.index[0].date(), table.index[-1].date()))
    print(
        "Portfolio Ending={ending_value:,.2f} Total={total_return:.2%} CAGR={cagr:.2%}".format(
            **portfolio_stats
        )
    )
    print(
        "Benchmark Ending={ending_value:,.2f} Total={total_return:.2%} CAGR={cagr:.2%}".format(
            **benchmark_stats
        )
    )
    active = portfolio_stats["total_return"] - benchmark_stats["total_return"]
    print("Active Return vs Benchmark: {:+.2%}".format(active))

    if args.output:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index_label="date")
        print(f"Saved detailed series to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
