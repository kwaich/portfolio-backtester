"""Integration tests for the backtester system.

These tests verify end-to-end workflows, edge cases, and system integration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import backtest
from backtest import (
    download_prices,
    compute_metrics,
    summarize,
    validate_ticker,
    validate_tickers,
    validate_date_string,
    validate_price_data,
)


class TestEndToEndWorkflow:
    """Test complete workflows from start to finish."""

    # Removed test_cli_to_csv_workflow - requires network access to Yahoo Finance
    # Removed test_multi_ticker_portfolio_workflow - requires network access to Yahoo Finance

    @patch("backtest.yf.download")
    def test_cache_workflow(self, mock_download, tmp_path):
        """Test caching workflow: download → cache → reload."""
        dates = pd.date_range("2023-01-01", periods=50, freq="B")
        data = pd.DataFrame({"AAPL": np.linspace(150, 180, len(dates))}, index=dates)
        mock_download.return_value = data

        # First call: should download and cache
        with patch("backtest.Path.mkdir"):  # Mock cache dir creation
            with patch("backtest.save_cached_prices") as mock_save:
                with patch("backtest.load_cached_prices", return_value=None):
                    prices1 = download_prices(["AAPL"], "2023-01-01", "2023-12-31", use_cache=True)
                    mock_save.assert_called_once()

        # Second call: should use cache
        with patch("backtest.load_cached_prices", return_value=data):
            prices2 = download_prices(["AAPL"], "2023-01-01", "2023-12-31", use_cache=True)
            # yfinance should not be called again
            assert mock_download.call_count == 1  # Only from first call

        pd.testing.assert_frame_equal(prices1, prices2)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_day_backtest(self):
        """Test backtest with only 1 trading day (should fail)."""
        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        prices = pd.DataFrame({"AAPL": [150]}, index=dates)
        benchmark = pd.Series([400], index=dates)
        weights = np.array([1.0])

        # Should raise error for insufficient data
        with pytest.raises(ValueError, match="Insufficient.*data"):
            compute_metrics(prices, weights, benchmark, 100000)

    def test_leap_year_dates(self):
        """Test date calculations across leap years."""
        # 2020 was a leap year (Feb 29, 2020 exists)
        dates = pd.date_range("2019-12-01", "2020-03-01", freq="B")
        values = pd.Series(np.linspace(100000, 110000, len(dates)), index=dates)

        stats = summarize(values, 100000)

        # Should handle Feb 29, 2020 correctly
        assert stats['cagr'] > 0
        assert not np.isnan(stats['cagr'])
        assert not np.isinf(stats['cagr'])

    def test_extreme_drawdown(self):
        """Test with >90% drawdown."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        # Drop from 100 to 5 (95% drawdown)
        values = pd.Series(
            np.concatenate([
                np.linspace(100000, 5000, 50),
                np.linspace(5000, 10000, 50)
            ]),
            index=dates
        )

        stats = summarize(values, 100000)

        assert stats['max_drawdown'] < -0.9
        assert stats['max_drawdown'] > -1.0
        assert stats['total_return'] < 0  # Lost money overall

    def test_zero_volatility_period(self):
        """Test period with no price movement."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        values = pd.Series([100000] * 100, index=dates)

        stats = summarize(values, 100000)

        assert stats['volatility'] == 0.0
        assert stats['sharpe_ratio'] == 0.0
        assert stats['sortino_ratio'] == 0.0
        assert stats['max_drawdown'] == 0.0
        assert stats['cagr'] == 0.0

    def test_very_short_date_range(self):
        """Test with minimal date range (2 days - minimum)."""
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        prices = pd.DataFrame({"AAPL": [150, 151]}, index=dates)
        benchmark = pd.Series([400, 401], index=dates)
        weights = np.array([1.0])

        # Should succeed with warning
        result = compute_metrics(prices, weights, benchmark, 100000)
        assert len(result) == 2

    @patch("backtest.yf.download")
    def test_missing_ticker_data(self, mock_download):
        """Test handling of tickers with missing data."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        # Only return data for one ticker, not both
        mock_download.return_value = pd.DataFrame({
            "AAPL": np.linspace(150, 180, len(dates))
            # MISSING: "INVALID" ticker
        }, index=dates)

        with pytest.raises(ValueError, match="Missing data for ticker"):
            download_prices(["AAPL", "INVALID"], "2020-01-01", "2020-12-31", use_cache=False)

    # Removed test_negative_returns - requires network access to Yahoo Finance


class TestDataQuality:
    """Test data validation and quality checks."""

    def test_all_nan_data(self):
        """Test rejection of all-NaN data."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        prices = pd.DataFrame({"BAD": [np.nan] * 100}, index=dates)

        with pytest.raises(ValueError, match="all values are NaN"):
            validate_price_data(prices, ["BAD"])

    def test_excessive_missing_data(self):
        """Test handling of tickers with >50% missing data."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        # 60% NaN
        data = [100 if i < 40 else np.nan for i in range(100)]
        prices = pd.DataFrame({"SPARSE": data}, index=dates)

        with pytest.raises(ValueError, match="60.0% missing values"):
            validate_price_data(prices, ["SPARSE"])

    def test_negative_prices_detection(self):
        """Test detection of negative prices."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        prices = pd.DataFrame({
            "BAD": [100, 101, 102, -50, 104, 105, 106, 107, 108, 109]
        }, index=dates)

        with pytest.raises(ValueError, match="negative price"):
            validate_price_data(prices, ["BAD"])

    def test_zero_prices_detection(self):
        """Test detection of zero prices."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        prices = pd.DataFrame({
            "BAD": [100, 101, 0, 103, 104, 105, 106, 107, 108, 109]
        }, index=dates)

        with pytest.raises(ValueError, match="zero price"):
            validate_price_data(prices, ["BAD"])

    def test_extreme_price_change_detection(self):
        """Test detection of extreme single-day price changes."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")

        # 100 -> 1 is a -99% change (extreme)
        prices = pd.DataFrame({
            "VOLATILE": [100, 101, 102, 1, 2, 3, 4, 5, 6, 7]
        }, index=dates)

        with pytest.raises(ValueError, match="extreme price change"):
            validate_price_data(prices, ["VOLATILE"])


class TestValidation:
    """Test input validation functions."""

    def test_ticker_validation_valid(self):
        """Test valid ticker formats."""
        valid_tickers = ["AAPL", "MSFT", "VWRA.L", "^GSPC", "EURUSD=X", "BRK-B"]

        for ticker in valid_tickers:
            is_valid, msg = validate_ticker(ticker)
            assert is_valid, f"{ticker} should be valid"

    def test_ticker_validation_invalid(self):
        """Test invalid ticker formats."""
        invalid = [
            ("", "empty"),
            ("TOOLONGTICKER", "too long"),
            ("123", "all numbers"),
            ("A@B", "special char"),
        ]

        for ticker, reason in invalid:
            is_valid, msg = validate_ticker(ticker)
            assert not is_valid, f"{ticker} should be invalid ({reason})"

    def test_date_validation_valid(self):
        """Test valid date formats."""
        valid_dates = ["2020-01-01", "2023-12-31", "2022-06-15"]

        for date_str in valid_dates:
            try:
                result = validate_date_string(date_str)
                assert result == date_str  # Should return normalized format
            except Exception as e:
                pytest.fail(f"{date_str} should be valid: {e}")

    def test_date_validation_future(self):
        """Test rejection of future dates."""
        future = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")

        with pytest.raises(Exception, match="future"):
            validate_date_string(future)

    def test_date_validation_too_old(self):
        """Test rejection of dates before 1970."""
        with pytest.raises(Exception, match="too far in the past"):
            validate_date_string("1969-12-31")


class TestStatisticalEdgeCases:
    """Test statistical calculation edge cases."""

    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio calculation with zero volatility."""
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        # Constant value = zero volatility
        values = pd.Series([100000] * 252, index=dates)

        stats = summarize(values, 100000)

        assert stats['sharpe_ratio'] == 0.0
        assert not np.isnan(stats['sharpe_ratio'])

    def test_sortino_ratio_no_downside(self):
        """Test Sortino ratio when there's no downside volatility."""
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        # Only positive returns
        values = pd.Series(np.linspace(100000, 120000, 252), index=dates)

        stats = summarize(values, 100000)

        # Sortino should be 0 when no downside volatility
        assert stats['sortino_ratio'] == 0.0

    def test_cagr_calculation_precision(self):
        """Test CAGR calculation precision."""
        # Exactly 1 year
        dates = pd.date_range("2020-01-01", "2021-01-01", freq="D")
        # 20% gain
        values = pd.Series(np.linspace(100000, 120000, len(dates)), index=dates)

        stats = summarize(values, 100000)

        # Should be approximately 20% CAGR
        assert abs(stats['cagr'] - 0.20) < 0.01  # Within 1%

    def test_max_drawdown_recovery(self):
        """Test max drawdown when portfolio recovers."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        # Drop 50%, then recover to original level
        values = pd.Series(
            np.concatenate([
                np.linspace(100000, 50000, 50),  # -50%
                np.linspace(50000, 100000, 50)   # +100% (back to even)
            ]),
            index=dates
        )

        stats = summarize(values, 100000)

        # Max drawdown should still be -50% even after recovery
        assert abs(stats['max_drawdown'] - (-0.5)) < 0.01


class TestMultiTickerEdgeCases:
    """Test edge cases with multiple tickers."""

    # Removed test_different_start_dates_alignment - requires network access to Yahoo Finance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
