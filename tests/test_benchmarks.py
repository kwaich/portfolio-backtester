"""Performance benchmarks for the Portfolio Backtester.

Uses pytest-benchmark to establish baselines for key computational paths.
Run with: pytest tests/test_benchmarks.py --benchmark-only -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backtest
from app.charts import calculate_drawdown
from app.validation import normalize_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_benchmark_dataset(years: int = 5, n_tickers: int = 5, seed: int = 42) -> tuple:
    """Generate deterministic synthetic data for benchmarking.

    Returns:
        Tuple of (prices DataFrame, weights array, benchmark Series)
    """
    rng = np.random.RandomState(seed)
    n_days = years * 252
    idx = pd.date_range("2020-01-01", periods=n_days, freq="D")

    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (n_days, n_tickers)), axis=0),
        index=idx,
        columns=[f"T{i}" for i in range(n_tickers)],
    )
    weights = np.ones(n_tickers) / n_tickers
    benchmark = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n_days)),
        index=idx,
    )
    return prices, weights, benchmark


def _make_flat_prices(years: int = 5, n_tickers: int = 5) -> tuple:
    """Generate flat prices (no growth) for deterministic benchmarking."""
    n_days = years * 252
    idx = pd.date_range("2020-01-01", periods=n_days, freq="D")
    prices = pd.DataFrame(100.0, index=idx, columns=[f"T{i}" for i in range(n_tickers)])
    weights = np.ones(n_tickers) / n_tickers
    benchmark = pd.Series(100.0, index=idx)
    return prices, weights, benchmark


# ---------------------------------------------------------------------------
# Issue 13 – Performance Benchmarks
# ---------------------------------------------------------------------------


class TestBenchmarkNormalizeWeights:
    """Benchmarks for normalize_weights."""

    def test_benchmark_normalize_weights(self, benchmark):
        """Benchmark weight normalization with 10 weights."""
        weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = benchmark(normalize_weights, weights)
        assert np.isclose(result.sum(), 1.0)


class TestBenchmarkComputeMetrics:
    """Benchmarks for compute_metrics under different strategies."""

    def test_benchmark_compute_metrics_lump_sum(self, benchmark):
        """Benchmark compute_metrics for lump-sum buy-and-hold."""
        prices, weights, benchmark_series = _make_benchmark_dataset(years=5, n_tickers=5)
        result = benchmark(backtest.compute_metrics, prices, weights, benchmark_series, 100_000)
        assert len(result) > 0
        assert "portfolio_value" in result.columns

    def test_benchmark_compute_metrics_rebalance(self, benchmark):
        """Benchmark compute_metrics with monthly rebalancing."""
        prices, weights, benchmark_series = _make_benchmark_dataset(years=5, n_tickers=5)
        result = benchmark(
            backtest.compute_metrics,
            prices,
            weights,
            benchmark_series,
            100_000,
            rebalance_freq="ME",
        )
        assert len(result) > 0

    def test_benchmark_compute_metrics_dca(self, benchmark):
        """Benchmark compute_metrics with monthly DCA."""
        prices, weights, benchmark_series = _make_benchmark_dataset(years=5, n_tickers=5)
        result = benchmark(
            backtest.compute_metrics,
            prices,
            weights,
            benchmark_series,
            100_000,
            dca_amount=1_000,
            dca_freq="ME",
        )
        assert len(result) > 0


class TestBenchmarkRebalancedPortfolio:
    """Benchmarks for _calculate_rebalanced_portfolio."""

    def test_benchmark_rebalanced_portfolio(self, benchmark):
        """Benchmark monthly rebalancing with 10 tickers over 5 years."""
        prices, weights, _ = _make_benchmark_dataset(years=5, n_tickers=10)
        result = benchmark(
            backtest._calculate_rebalanced_portfolio,
            prices,
            weights,
            100_000,
            "ME",
        )
        assert len(result) > 0


class TestBenchmarkDCAPortfolio:
    """Benchmarks for _calculate_dca_portfolio."""

    def test_benchmark_dca_portfolio(self, benchmark):
        """Benchmark monthly DCA with 5 tickers over 5 years."""
        prices, weights, _ = _make_benchmark_dataset(years=5, n_tickers=5)
        result = benchmark(
            backtest._calculate_dca_portfolio,
            prices,
            weights,
            100_000,
            1_000,
            "ME",
        )
        assert len(result) == 2  # (portfolio_values, cumulative_contributions)


class TestBenchmarkXIRR:
    """Benchmarks for _calculate_xirr."""

    def test_benchmark_xirr_small(self, benchmark):
        """Benchmark XIRR with 10 cashflows."""
        cashflows = np.array([-1000.0] * 5 + [6000.0])
        days = np.array([0, 30, 60, 90, 120, 365])
        result = benchmark(backtest._calculate_xirr, cashflows, days)
        assert result is None or isinstance(result, (float, np.floating))

    def test_benchmark_xirr_large(self, benchmark):
        """Benchmark XIRR with 500 cashflows."""
        n = 500
        cashflows = np.concatenate([np.full(n - 1, -100.0), [n * 100 * 1.1]])
        days = np.linspace(0, 365 * 10, n)
        result = benchmark(backtest._calculate_xirr, cashflows, days)
        assert result is None or isinstance(result, (float, np.floating))


class TestBenchmarkSummarize:
    """Benchmarks for summarize."""

    def test_benchmark_summarize(self, benchmark):
        """Benchmark summarize with 10-year daily series."""
        n_days = 10 * 252
        idx = pd.date_range("2010-01-01", periods=n_days, freq="D")
        rng = np.random.RandomState(42)
        returns = rng.normal(0.0003, 0.01, n_days)
        values = pd.Series(10_000 * np.cumprod(1 + returns), index=idx)
        result = benchmark(backtest.summarize, values, 10_000)
        assert isinstance(result, dict)
        assert "cagr" in result


class TestBenchmarkCalculateDrawdown:
    """Benchmarks for calculate_drawdown."""

    def test_benchmark_calculate_drawdown(self, benchmark):
        """Benchmark drawdown calculation with 10-year daily series."""
        n_days = 10 * 252
        idx = pd.date_range("2010-01-01", periods=n_days, freq="D")
        rng = np.random.RandomState(42)
        returns = rng.normal(0.0003, 0.01, n_days)
        values = pd.Series(10_000 * np.cumprod(1 + returns), index=idx)
        result = benchmark(calculate_drawdown, values)
        assert len(result) == n_days
        assert (result <= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
