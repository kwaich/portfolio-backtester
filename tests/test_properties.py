"""Property-based tests for the Portfolio Backtester.

Uses Hypothesis to verify mathematical invariants and behavioral properties
across a wide range of automatically generated inputs.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

import backtest
from app.charts import calculate_drawdown
from app.validation import normalize_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FREQ_OPTIONS = [
    "D", "W", "M", "Q", "Y",
    "ME", "QE", "YE",
    "daily", "weekly", "monthly", "quarterly", "yearly",
]


def _make_price_df(
    prices_list: List[List[float]] | List[float],
    n_tickers: Optional[int] = None,
) -> pd.DataFrame:
    """Build a valid price DataFrame from a list of float lists."""
    if n_tickers is None:
        n_tickers = len(prices_list)
    n_days = len(prices_list[0]) if prices_list else 2
    idx = pd.date_range("2020-01-01", periods=n_days, freq="D")
    data = {}
    for i in range(n_tickers):
        col = [row[i] for row in prices_list] if isinstance(prices_list[0], list) else prices_list
        data[f"T{i}"] = col
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# Issue 12 – Property-Based Testing
# ---------------------------------------------------------------------------


class TestNormalizeWeightsProperties:
    """Properties for normalize_weights()."""

    @given(weights=st.lists(st.floats(min_value=0.0, max_value=1000.0), min_size=1, max_size=20))
    @settings(max_examples=200)
    def test_sum_always_one(self, weights):
        """Property: Normalized weights always sum to 1.0."""
        assume(sum(weights) > 0)
        result = normalize_weights(weights)
        assert np.isclose(result.sum(), 1.0)

    @given(weights=st.lists(st.floats(min_value=0.0, max_value=1000.0), min_size=1, max_size=20))
    @settings(max_examples=200)
    def test_all_weights_non_negative(self, weights):
        """Property: All normalized weights are >= 0."""
        assume(sum(weights) > 0)
        result = normalize_weights(weights)
        assert all(w >= 0 for w in result)

    @given(weights=st.lists(st.floats(min_value=0.0, max_value=1000.0), min_size=1, max_size=20))
    @settings(max_examples=200)
    def test_idempotent(self, weights):
        """Property: Applying normalization twice yields the same result."""
        assume(sum(weights) > 0)
        first = normalize_weights(weights)
        second = normalize_weights(first.tolist())
        np.testing.assert_allclose(first, second)


class TestNormalizeFrequencyProperties:
    """Properties for normalize_frequency()."""

    @given(
        freq=st.one_of(
            st.sampled_from(FREQ_OPTIONS),
            st.none(),
        )
    )
    @settings(max_examples=150)
    def test_idempotent(self, freq):
        """Property: normalize_frequency is idempotent."""
        first = backtest.normalize_frequency(freq)
        second = backtest.normalize_frequency(first)
        assert first == second

    def test_none_returns_none(self):
        """Property: None input returns None."""
        assert backtest.normalize_frequency(None) is None

    @given(freq=st.sampled_from(FREQ_OPTIONS))
    @settings(max_examples=100)
    def test_case_insensitive_names(self, freq):
        """Property: Frequency names are case-insensitive."""
        lower_result = backtest.normalize_frequency(freq.lower())
        upper_result = backtest.normalize_frequency(freq.upper())
        assert lower_result == upper_result


class TestValidateTickerProperties:
    """Properties for validate_ticker()."""

    @given(ticker=st.text(min_size=1, max_size=10))
    @settings(max_examples=300)
    def test_empty_ticker_is_invalid(self, ticker):
        """Property: Empty string is always invalid."""
        is_valid, _ = backtest.validate_ticker("")
        assert not is_valid

    @given(ticker=st.from_regex(r"^[A-Z]{1,5}$", fullmatch=True))
    @settings(max_examples=100)
    def test_valid_tickers_pass(self, ticker):
        """Property: Standard uppercase tickers are valid."""
        is_valid, _ = backtest.validate_ticker(ticker)
        assert is_valid

    @given(ticker=st.from_regex(r"^\d+$", fullmatch=True))
    @settings(max_examples=50)
    def test_all_digits_invalid(self, ticker):
        """Property: All-numeric tickers are invalid."""
        is_valid, _ = backtest.validate_ticker(ticker)
        assert not is_valid


class TestCalculateDrawdownProperties:
    """Properties for calculate_drawdown()."""

    @given(
        values=st.lists(
            st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=500,
        )
    )
    @settings(max_examples=200)
    def test_all_values_less_than_or_equal_to_zero(self, values):
        """Property: Drawdown is never positive."""
        series = pd.Series(values)
        dd = calculate_drawdown(series)
        assert (dd <= 0).all()

    @given(
        values=st.lists(
            st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=500,
        )
    )
    @settings(max_examples=200)
    def test_first_value_is_zero(self, values):
        """Property: First drawdown value is always 0."""
        series = pd.Series(values)
        dd = calculate_drawdown(series)
        assert dd.iloc[0] == pytest.approx(0.0)

    @given(
        values=st.lists(
            st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=500,
        )
    )
    @settings(max_examples=200)
    def test_zero_at_new_peaks(self, values):
        """Property: Drawdown is 0 whenever the series reaches a new high."""
        series = pd.Series(values)
        dd = calculate_drawdown(series)
        running_max = series.expanding().max()
        at_peak = series == running_max
        peak_drawdowns = dd[at_peak].dropna()
        np.testing.assert_allclose(peak_drawdowns.values, 0.0, atol=1e-10)


class TestRollingSharpeProperties:
    """Properties for _calculate_rolling_sharpe()."""

    @given(
        n_days=st.integers(min_value=10, max_value=1000),
        window=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=100)
    def test_first_window_minus_one_are_nan(self, n_days, window):
        """Property: First (window-1) values are NaN."""
        idx = pd.date_range("2020-01-01", periods=n_days, freq="D")
        rng = np.random.RandomState(0)
        values = pd.Series(100 + rng.randn(n_days).cumsum(), index=idx)
        result = backtest._calculate_rolling_sharpe(values, window)
        assert result.iloc[: window - 1].isna().all()

    @given(
        n_days=st.integers(min_value=100, max_value=500),
        window=st.integers(min_value=10, max_value=50),
    )
    @settings(max_examples=100)
    def test_finite_output_for_valid_input(self, n_days, window):
        """Property: No infinite values in output for normal price series."""
        idx = pd.date_range("2020-01-01", periods=n_days, freq="D")
        rng = np.random.RandomState(0)
        values = pd.Series(100 + rng.randn(n_days).cumsum(), index=idx)
        result = backtest._calculate_rolling_sharpe(values, window)
        valid = result.dropna()
        assert np.isfinite(valid).all()


class TestXIRRProperties:
    """Properties for _calculate_xirr()."""

    @given(
        principal=st.floats(min_value=1.0, max_value=1e6, allow_nan=False),
        return_factor=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
    )
    @settings(max_examples=150)
    def test_break_even_returns_zero(self, principal, return_factor):
        """Property: Invest X, get back X → IRR ≈ 0."""
        cashflows = np.array([-principal, principal])
        days = np.array([0.0, 365.0])
        irr = backtest._calculate_xirr(cashflows, days)
        if irr is not None:
            assert abs(irr) < 0.01

    @given(
        principal=st.floats(min_value=1.0, max_value=1e6, allow_nan=False),
        gain=st.floats(min_value=0.01, max_value=1e6, allow_nan=False),
    )
    @settings(max_examples=150)
    def test_profit_returns_positive(self, principal, gain):
        """Property: Final value > principal → IRR > 0."""
        cashflows = np.array([-principal, principal + gain])
        days = np.array([0.0, 365.0])
        irr = backtest._calculate_xirr(cashflows, days)
        if irr is not None:
            assert irr > 0

    @given(
        principal=st.floats(min_value=1.0, max_value=1e6, allow_nan=False),
        final=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
    )
    @settings(max_examples=150)
    def test_loss_returns_negative(self, principal, final):
        """Property: Final value < principal → IRR < 0."""
        cashflows = np.array([-principal, principal * final])
        days = np.array([0.0, 365.0])
        irr = backtest._calculate_xirr(cashflows, days)
        if irr is not None:
            assert irr < 0


class TestComputeMetricsProperties:
    """Properties for compute_metrics()."""

    @given(
        n_days=st.integers(min_value=30, max_value=500),
        n_tickers=st.integers(min_value=1, max_value=5),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_portfolio_and_benchmark_values_non_negative(self, n_days, n_tickers):
        """Property: Portfolio and benchmark values are always >= 0."""
        idx = pd.date_range("2020-01-01", periods=n_days, freq="D")
        rng = np.random.RandomState(42)
        prices = pd.DataFrame(
            100 * np.cumprod(
                1 + rng.normal(0.0003, 0.01, (n_days, n_tickers)), axis=0
            ),
            index=idx,
            columns=[f"T{i}" for i in range(n_tickers)],
        )
        weights = np.ones(n_tickers) / n_tickers
        benchmark = pd.Series(
            100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n_days)),
            index=idx,
        )

        table = backtest.compute_metrics(prices, weights, benchmark, 10_000)
        assert (table["portfolio_value"] >= 0).all()
        assert (table["benchmark_value"] >= 0).all()

    @given(
        n_days=st.integers(min_value=30, max_value=500),
        n_tickers=st.integers(min_value=1, max_value=5),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_active_return_equals_difference(self, n_days, n_tickers):
        """Property: active_return = portfolio_return - benchmark_return."""
        idx = pd.date_range("2020-01-01", periods=n_days, freq="D")
        rng = np.random.RandomState(42)
        prices = pd.DataFrame(
            100 * np.cumprod(
                1 + rng.normal(0.0003, 0.01, (n_days, n_tickers)), axis=0
            ),
            index=idx,
            columns=[f"T{i}" for i in range(n_tickers)],
        )
        weights = np.ones(n_tickers) / n_tickers
        benchmark = pd.Series(
            100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n_days)),
            index=idx,
        )

        table = backtest.compute_metrics(prices, weights, benchmark, 10_000)
        expected = table["portfolio_return"] - table["benchmark_return"]
        pd.testing.assert_series_equal(
            table["active_return"], expected, check_names=False
        )


class TestSummarizeProperties:
    """Properties for summarize()."""

    @given(
        n_days=st.integers(min_value=10, max_value=500),
        growth_factor=st.floats(min_value=0.5, max_value=3.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_returns_dict_with_expected_keys(self, n_days, growth_factor):
        """Property: summarize always returns a dict with core keys."""
        idx = pd.date_range("2020-01-01", periods=n_days, freq="D")
        values = pd.Series(np.linspace(10_000, 10_000 * growth_factor, n_days), index=idx)
        result = backtest.summarize(values, 10_000)
        expected_keys = {
            "ending_value", "total_return", "cagr", "volatility",
            "sharpe_ratio", "sortino_ratio", "max_drawdown",
        }
        assert expected_keys.issubset(result.keys())

    @given(
        n_days=st.integers(min_value=250, max_value=260),
    )
    @settings(max_examples=50)
    def test_cagr_approximates_total_return_for_one_year(self, n_days):
        """Property: For ~1 trading year, CAGR ≈ total_return."""
        idx = pd.date_range("2020-01-01", periods=n_days, freq="D")
        values = pd.Series(np.linspace(10_000, 11_000, n_days), index=idx)
        result = backtest.summarize(values, 10_000)
        # For a 10% return over ~1 year, CAGR should be close to total_return
        assert abs(result["cagr"] - result["total_return"]) < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
