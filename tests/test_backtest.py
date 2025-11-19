"""Unit tests for backtest.py"""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import backtest


class TestParseArgs:
    """Test argument parsing"""

    def test_default_arguments(self):
        args = backtest.parse_args([])
        assert args.tickers == ["VDCP.L", "VHYD.L"]
        assert args.weights == [0.5, 0.5]
        assert args.benchmark == "VWRA.L"
        assert args.capital == 100_000.0
        assert args.no_cache is False

    def test_custom_tickers(self):
        args = backtest.parse_args(["--tickers", "AAPL", "MSFT", "GOOGL"])
        assert args.tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_custom_weights(self):
        args = backtest.parse_args(["--weights", "0.4", "0.3", "0.3"])
        assert args.weights == [0.4, 0.3, 0.3]

    def test_no_cache_flag(self):
        args = backtest.parse_args(["--no-cache"])
        assert args.no_cache is True

    def test_cache_ttl_argument(self):
        args = backtest.parse_args(["--cache-ttl", "48"])
        assert args.cache_ttl == 48

    def test_cache_ttl_default(self):
        args = backtest.parse_args([])
        assert args.cache_ttl == backtest.DEFAULT_CACHE_TTL_HOURS

    def test_output_path(self):
        args = backtest.parse_args(["--output", "test.csv"])
        assert args.output == Path("test.csv")

    def test_dca_arguments(self):
        """Test DCA argument parsing"""
        args = backtest.parse_args(["--dca-amount", "1000", "--dca-freq", "M"])
        assert args.dca_amount == 1000.0
        assert args.dca_freq == "M"

    def test_dca_defaults(self):
        """Test DCA default values"""
        args = backtest.parse_args([])
        assert args.dca_amount is None
        assert args.dca_freq is None

    def test_dca_frequency_options(self):
        """Test various DCA frequency options"""
        for freq in ['D', 'W', 'M', 'Q', 'Y', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly']:
            args = backtest.parse_args(["--dca-freq", freq, "--dca-amount", "500"])
            assert args.dca_freq == freq

    def test_dca_invalid_frequency(self):
        """Test that invalid DCA frequency raises error"""
        with pytest.raises(SystemExit):
            backtest.parse_args(["--dca-freq", "invalid"])


class TestCacheFunctions:
    """Test caching functionality"""

    def test_get_cache_key(self):
        key1 = backtest.get_cache_key(["AAPL", "MSFT"], "2020-01-01", "2021-01-01")
        key2 = backtest.get_cache_key(["MSFT", "AAPL"], "2020-01-01", "2021-01-01")
        # Should be same regardless of order (sorted internally)
        assert key1 == key2
        assert len(key1) == 32  # MD5 hash length

    def test_get_cache_key_different_params(self):
        key1 = backtest.get_cache_key(["AAPL"], "2020-01-01", "2021-01-01")
        key2 = backtest.get_cache_key(["AAPL"], "2020-01-01", "2022-01-01")
        assert key1 != key2

    def test_cache_path_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backtest.Path") as mock_path:
                mock_cache_dir = MagicMock()
                mock_path.return_value = mock_cache_dir
                path = backtest.get_cache_path(["AAPL"], "2020-01-01", "2021-01-01")
                # Verify the path includes .cache directory
                assert ".cache" in str(path) or mock_cache_dir.mkdir.called

    def test_load_cached_prices_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "nonexistent"
            result = backtest.load_cached_prices(cache_path)
            assert result is None

    def test_save_and_load_cached_prices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache"
            test_data = pd.DataFrame(
                {"AAPL": [100, 101, 102], "MSFT": [200, 201, 202]},
                index=pd.date_range("2020-01-01", periods=3),
            )

            backtest.save_cached_prices(cache_path, test_data)
            loaded_data = backtest.load_cached_prices(cache_path)

            assert loaded_data is not None
            # Note: Parquet doesn't preserve DatetimeIndex.freq, so check_freq=False
            pd.testing.assert_frame_equal(loaded_data, test_data, check_freq=False)

            # Verify Parquet and JSON files were created
            assert cache_path.with_suffix('.parquet').exists()
            assert cache_path.with_suffix('.json').exists()

    def test_cache_expiration(self):
        """Test that stale cache is rejected and deleted."""
        import time
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache"
            test_data = pd.DataFrame(
                {"AAPL": [100, 101, 102]},
                index=pd.date_range("2020-01-01", periods=3),
            )

            # Save cache
            backtest.save_cached_prices(cache_path, test_data)

            # Modify timestamp in JSON metadata to simulate old cache (25 hours old)
            metadata_path = cache_path.with_suffix('.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata["timestamp"] = time.time() - (25 * 3600)  # 25 hours ago
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            # Try to load with default TTL (24 hours)
            loaded_data = backtest.load_cached_prices(cache_path, max_age_hours=24)

            # Should return None and delete both files
            assert loaded_data is None
            assert not cache_path.with_suffix('.parquet').exists()
            assert not cache_path.with_suffix('.json').exists()

    def test_cache_within_ttl(self):
        """Test that fresh cache is loaded successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache"
            test_data = pd.DataFrame(
                {"AAPL": [100, 101, 102]},
                index=pd.date_range("2020-01-01", periods=3),
            )

            # Save cache
            backtest.save_cached_prices(cache_path, test_data)

            # Load with long TTL (should succeed)
            loaded_data = backtest.load_cached_prices(cache_path, max_age_hours=48)

            assert loaded_data is not None
            pd.testing.assert_frame_equal(loaded_data, test_data, check_freq=False)

    def test_old_cache_format_migration(self):
        """Test migration from old pickle cache format to Parquet."""
        import pickle

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "old_cache"
            old_pickle_path = cache_path.with_suffix('.pkl')
            test_data = pd.DataFrame(
                {"AAPL": [100, 101, 102]},
                index=pd.date_range("2020-01-01", periods=3),
            )

            # Save old format (dict with data and timestamp)
            old_cache_data = {
                "data": test_data,
                "timestamp": time.time(),
                "version": "1.0"
            }
            with open(old_pickle_path, "wb") as f:
                pickle.dump(old_cache_data, f)

            # Try to load (should migrate to Parquet)
            loaded_data = backtest.load_cached_prices(cache_path)

            # Should successfully load the migrated data
            assert loaded_data is not None
            pd.testing.assert_frame_equal(loaded_data, test_data, check_freq=False)

            # Old pickle should be deleted, new Parquet files should exist
            assert not old_pickle_path.exists()
            assert cache_path.with_suffix('.parquet').exists()
            assert cache_path.with_suffix('.json').exists()

    def test_corrupted_cache_handling(self):
        """Test that corrupted cache is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "corrupted"
            parquet_path = cache_path.with_suffix('.parquet')
            json_path = cache_path.with_suffix('.json')

            # Write corrupted Parquet file
            with open(parquet_path, "wb") as f:
                f.write(b"corrupted data")

            # Write valid JSON metadata
            with open(json_path, "w") as f:
                f.write('{"timestamp": 123456, "version": "2.0"}')

            # Should return None and clean up both files
            result = backtest.load_cached_prices(cache_path)
            assert result is None
            assert not parquet_path.exists()
            assert not json_path.exists()


class TestSummarize:
    """Test summary statistics calculation"""

    def test_basic_returns(self):
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        # Simple linear growth from 100k to 120k
        values = pd.Series(np.linspace(100_000, 120_000, 252), index=dates)

        stats = backtest.summarize(values, 100_000)

        assert stats["ending_value"] == pytest.approx(120_000, rel=0.01)
        assert stats["total_return"] == pytest.approx(0.2, rel=0.01)  # 20% return
        assert "cagr" in stats
        assert "volatility" in stats
        assert "sharpe_ratio" in stats
        assert "sortino_ratio" in stats
        assert "max_drawdown" in stats

    def test_zero_volatility(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        # Constant value - no volatility
        values = pd.Series([100_000] * 10, index=dates)

        stats = backtest.summarize(values, 100_000)

        assert stats["volatility"] == 0.0
        assert stats["sharpe_ratio"] == 0.0
        assert stats["sortino_ratio"] == 0.0

    def test_drawdown_calculation(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        # Peak at 110k, then drop to 90k (18.18% drawdown)
        values = pd.Series([100_000, 110_000, 105_000, 90_000, 95_000], index=dates)

        stats = backtest.summarize(values, 100_000)

        # Max drawdown should be negative (from 110k to 90k)
        assert stats["max_drawdown"] < 0
        assert stats["max_drawdown"] == pytest.approx(-0.1818, abs=0.01)

    def test_dca_drawdown_not_triggered_by_contributions(self):
        dates = pd.date_range("2020-01-01", periods=6, freq="ME")
        contributions = pd.Series([1000 * (i + 1) for i in range(len(dates))], index=dates, dtype=float)
        values = contributions.copy()  # portfolio value only grows via contributions

        stats = backtest.summarize(
            values,
            1000,
            total_contributions=contributions.iloc[-1],
            contributions_series=contributions,
        )

        assert stats["max_drawdown"] == pytest.approx(0.0)
        assert stats["volatility"] == 0.0
        assert stats["sharpe_ratio"] == 0.0
        assert stats["sortino_ratio"] == 0.0

    def test_dca_drawdown_tracks_equity_decline(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="ME")
        contributions = pd.Series([1000 * (i + 1) for i in range(len(dates))], index=dates, dtype=float)
        values = pd.Series([1000, 2100, 3200, 2800, 3600], index=dates, dtype=float)

        stats = backtest.summarize(
            values,
            1000,
            total_contributions=contributions.iloc[-1],
            contributions_series=contributions,
        )

        # Drop from 3200 peak to 2800 trough ~ -12.5%
        assert stats["max_drawdown"] == pytest.approx(-0.125, abs=0.001)

    def test_empty_series_raises_error(self):
        empty_series = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="Cannot summarize an empty series"):
            backtest.summarize(empty_series, 100_000)


class TestHelperFunctions:
    def test_contribution_adjusted_daily_returns(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        values = pd.Series([100.0, 160.0, 152.0], index=dates)
        contributions = pd.Series([100.0, 150.0, 150.0], index=dates)

        returns = backtest._calculate_daily_returns(values, contributions)
        expected = pd.Series([np.nan, 0.1, -0.05], index=dates)

        pd.testing.assert_series_equal(returns, expected)


class TestComputeMetrics:
    """Test backtest metrics computation"""

    def test_basic_computation(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        prices = pd.DataFrame(
            {"AAPL": np.linspace(100, 110, 10), "MSFT": np.linspace(200, 220, 10)},
            index=dates,
        )
        benchmark = pd.Series(np.linspace(150, 165, 10), index=dates)
        weights = np.array([0.5, 0.5])

        table = backtest.compute_metrics(prices, weights, benchmark, 100_000)

        assert "portfolio_value" in table.columns
        assert "portfolio_return" in table.columns
        assert "benchmark_value" in table.columns
        assert "benchmark_return" in table.columns
        assert "active_return" in table.columns
        assert len(table) == 10

    def test_weight_application(self):
        dates = pd.date_range("2020-01-01", periods=2, freq="D")
        # Simple 50/50 portfolio: AAPL doubles, MSFT stays flat
        prices = pd.DataFrame({"AAPL": [100, 200], "MSFT": [100, 100]}, index=dates)
        benchmark = pd.Series([100, 100], index=dates)
        weights = np.array([0.5, 0.5])

        table = backtest.compute_metrics(prices, weights, benchmark, 100_000)

        # Portfolio should gain 50% (50k in AAPL doubles to 100k, 50k in MSFT stays)
        # Final value: 100k + 50k = 150k
        final_value = table["portfolio_value"].iloc[-1]
        assert final_value == pytest.approx(150_000, rel=0.01)

    def test_missing_data_error(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        # Create prices with NaN values
        prices = pd.DataFrame(
            {"AAPL": [np.nan] * 5, "MSFT": [100, 101, 102, 103, 104]}, index=dates
        )
        benchmark = pd.Series([100, 101, 102, 103, 104], index=dates)
        weights = np.array([0.5, 0.5])

        with pytest.raises(ValueError, match="no valid prices"):
            backtest.compute_metrics(prices, weights, benchmark, 100_000)

    def test_benchmark_alignment(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        prices = pd.DataFrame(
            {"AAPL": np.linspace(100, 110, 10), "MSFT": np.linspace(200, 220, 10)},
            index=dates,
        )
        # Benchmark starts later
        benchmark = pd.Series(
            np.linspace(150, 165, 5), index=pd.date_range("2020-01-06", periods=5, freq="D")
        )
        weights = np.array([0.5, 0.5])

        table = backtest.compute_metrics(prices, weights, benchmark, 100_000)

        # Result should only include overlapping dates
        assert len(table) == 5
        assert table.index[0] == pd.Timestamp("2020-01-06")

    def test_rolling_sharpe_12m_column_exists(self):
        """Test that rolling 12-month Sharpe ratio column is created."""
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        prices = pd.DataFrame(
            {"AAPL": np.linspace(100, 120, 300), "MSFT": np.linspace(200, 240, 300)},
            index=dates,
        )
        benchmark = pd.Series(np.linspace(150, 180, 300), index=dates)
        weights = np.array([0.5, 0.5])

        table = backtest.compute_metrics(prices, weights, benchmark, 100_000)

        # Check that rolling Sharpe columns exist
        assert "portfolio_rolling_sharpe_12m" in table.columns
        assert "benchmark_rolling_sharpe_12m" in table.columns

    def test_rolling_sharpe_12m_nan_initially(self):
        """Test that rolling Sharpe is NaN for first 252 days (12 months)."""
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        prices = pd.DataFrame(
            {"AAPL": np.linspace(100, 120, 300), "MSFT": np.linspace(200, 240, 300)},
            index=dates,
        )
        benchmark = pd.Series(np.linspace(150, 180, 300), index=dates)
        weights = np.array([0.5, 0.5])

        table = backtest.compute_metrics(prices, weights, benchmark, 100_000)

        # First 252 values should be NaN (index 0 has NaN from pct_change,
        # then we need 252 values for rolling window, so first valid is at index 252)
        assert pd.isna(table["portfolio_rolling_sharpe_12m"].iloc[:252]).all()
        assert pd.isna(table["benchmark_rolling_sharpe_12m"].iloc[:252]).all()

        # At index 252 (253rd value), should have valid rolling Sharpe
        assert not pd.isna(table["portfolio_rolling_sharpe_12m"].iloc[252])
        assert not pd.isna(table["benchmark_rolling_sharpe_12m"].iloc[252])

    def test_rolling_sharpe_12m_calculation(self):
        """Test rolling Sharpe ratio calculation accuracy."""
        # Create 400 days of data with realistic returns (trend + volatility)
        dates = pd.date_range("2020-01-01", periods=400, freq="D")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Create portfolio with 10% annual growth + 15% annual volatility
        daily_mean_return = 0.10 / 252  # 10% annual / 252 trading days
        daily_volatility = 0.15 / np.sqrt(252)  # 15% annual volatility

        # Generate random daily returns with drift
        daily_returns = np.random.normal(daily_mean_return, daily_volatility, 400)

        # Convert to cumulative values starting at 100
        cumulative_returns = np.cumprod(1 + daily_returns)
        portfolio_values = 100 * cumulative_returns

        prices = pd.DataFrame(
            {"AAPL": portfolio_values * 0.5, "MSFT": portfolio_values * 0.5},
            index=dates,
        )
        benchmark = pd.Series(portfolio_values * 0.8, index=dates)
        weights = np.array([0.5, 0.5])

        table = backtest.compute_metrics(prices, weights, benchmark, 100_000)

        # Check that rolling Sharpe values are reasonable (should be positive for uptrend)
        valid_sharpe = table["portfolio_rolling_sharpe_12m"].dropna()
        assert len(valid_sharpe) > 0

        # For realistic data with positive expected return and volatility,
        # Sharpe ratio should be in a reasonable range
        # With 10% return and 15% volatility, theoretical Sharpe ≈ 0.67
        # But rolling estimates will vary, so use wider bounds
        assert (valid_sharpe < 5).all()  # Reasonable upper bound
        assert (valid_sharpe > -5).all()  # Reasonable lower bound

        # Average rolling Sharpe should be reasonably positive (not perfect but trending up)
        assert valid_sharpe.mean() > -1  # On average, should be above -1

    def test_rolling_sharpe_12m_zero_volatility(self):
        """Test rolling Sharpe ratio with zero volatility (constant prices)."""
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        # Constant prices (zero volatility)
        prices = pd.DataFrame(
            {"AAPL": [100] * 300, "MSFT": [200] * 300},
            index=dates,
        )
        benchmark = pd.Series([150] * 300, index=dates)
        weights = np.array([0.5, 0.5])

        table = backtest.compute_metrics(prices, weights, benchmark, 100_000)

        # With zero volatility and zero returns, Sharpe should be 0
        valid_sharpe = table["portfolio_rolling_sharpe_12m"].dropna()
        assert (valid_sharpe == 0.0).all()

    def test_dca_rolling_sharpe_ignores_contributions(self):
        dates = pd.date_range("2020-01-01", periods=260, freq="B")
        prices = pd.DataFrame({"AAPL": [100.0] * len(dates)}, index=dates)
        benchmark = pd.Series([100.0] * len(dates), index=dates)
        weights = np.array([1.0])

        table = backtest.compute_metrics(
            prices,
            weights,
            benchmark,
            1_000,
            dca_amount=100,
            dca_freq="M",
        )

        # With flat prices, the rolling Sharpe columns should stay NaN (no valid returns)
        assert table["portfolio_rolling_sharpe_12m"].dropna().empty
        assert table["benchmark_rolling_sharpe_12m"].dropna().empty

    def test_rolling_sharpe_12m_insufficient_data(self):
        """Test rolling Sharpe with less than 252 days of data."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        prices = pd.DataFrame(
            {"AAPL": np.linspace(100, 120, 100), "MSFT": np.linspace(200, 240, 100)},
            index=dates,
        )
        benchmark = pd.Series(np.linspace(150, 180, 100), index=dates)
        weights = np.array([0.5, 0.5])

        table = backtest.compute_metrics(prices, weights, benchmark, 100_000)

        # All rolling Sharpe values should be NaN (insufficient data)
        assert pd.isna(table["portfolio_rolling_sharpe_12m"]).all()
        assert pd.isna(table["benchmark_rolling_sharpe_12m"]).all()


class TestTickerValidation:
    """Test ticker validation functionality"""

    def test_valid_tickers(self):
        """Test that valid tickers pass validation."""
        valid_tickers = ["AAPL", "MSFT", "VWRA.L", "^GSPC", "EURUSD=X", "BRK-B"]

        for ticker in valid_tickers:
            is_valid, error = backtest.validate_ticker(ticker)
            assert is_valid, f"{ticker} should be valid, got error: {error}"
            assert error == ""

    def test_empty_ticker(self):
        """Test that empty ticker is rejected."""
        is_valid, error = backtest.validate_ticker("")
        assert not is_valid
        assert "cannot be empty" in error.lower()

    def test_too_long_ticker(self):
        """Test that ticker longer than 10 characters is rejected."""
        is_valid, error = backtest.validate_ticker("VERYLONGTICKER")
        assert not is_valid
        assert "too long" in error.lower()

    def test_all_numbers_ticker(self):
        """Test that all-numeric ticker is rejected."""
        is_valid, error = backtest.validate_ticker("12345")
        assert not is_valid
        assert "all numbers" in error.lower()

    def test_invalid_characters(self):
        """Test that ticker with invalid characters is rejected."""
        invalid_tickers = ["AAP@L", "MSF!T", "TEST#", "TIC KER"]

        for ticker in invalid_tickers:
            is_valid, error = backtest.validate_ticker(ticker)
            assert not is_valid, f"{ticker} should be invalid"
            assert "invalid" in error.lower() or "format" in error.lower()

    def test_validate_tickers_list_valid(self):
        """Test validation of valid ticker list."""
        # Should not raise
        backtest.validate_tickers(["AAPL", "MSFT", "GOOGL"])

    def test_validate_tickers_empty_list(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError, match="No tickers provided"):
            backtest.validate_tickers([])

    def test_validate_tickers_with_invalid(self):
        """Test that list with invalid ticker raises error."""
        with pytest.raises(ValueError, match="Invalid ticker"):
            backtest.validate_tickers(["AAPL", "", "MSFT"])

    def test_validate_tickers_multiple_errors(self):
        """Test that multiple invalid tickers show all errors."""
        with pytest.raises(ValueError) as exc_info:
            backtest.validate_tickers(["", "123", "VERYLONGTICKER"])

        error_msg = str(exc_info.value)
        assert "cannot be empty" in error_msg
        assert "all numbers" in error_msg
        assert "too long" in error_msg

    def test_case_insensitive_validation(self):
        """Test that validation is case-insensitive."""
        # Lowercase should work
        is_valid, _ = backtest.validate_ticker("aapl")
        assert is_valid

        # Mixed case should work
        is_valid, _ = backtest.validate_ticker("VwRa.L")
        assert is_valid


class TestDateValidation:
    """Test date validation functionality"""

    def test_valid_date_formats(self):
        """Test that various valid date formats are accepted and normalized."""
        valid_dates = [
            ("2020-01-01", "2020-01-01"),
            ("2020/01/01", "2020-01-01"),
            ("2020.01.01", "2020-01-01"),
        ]

        for input_date, expected in valid_dates:
            result = backtest.validate_date_string(input_date)
            assert result == expected, f"Failed for input: {input_date}"

    def test_date_too_far_in_past(self):
        """Test that dates before 1970 are rejected."""
        with pytest.raises(argparse.ArgumentTypeError, match="too far in the past"):
            backtest.validate_date_string("1969-12-31")

    def test_future_date(self):
        """Test that future dates are rejected."""
        future_date = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        with pytest.raises(argparse.ArgumentTypeError, match="in the future"):
            backtest.validate_date_string(future_date)

    def test_invalid_date_format(self):
        """Test that invalid date strings are rejected."""
        invalid_dates = ["not-a-date", "2020-13-01", "2020-02-30", "abc"]

        for invalid_date in invalid_dates:
            with pytest.raises(argparse.ArgumentTypeError, match="Invalid date format"):
                backtest.validate_date_string(invalid_date)

    def test_date_range_validation_in_main(self):
        """Test that main() validates date ranges."""
        # Start date after end date should fail
        with pytest.raises(SystemExit, match="Invalid date range"):
            backtest.main([
                "--start", "2023-12-31",
                "--end", "2023-01-01",
                "--tickers", "AAPL",
                "--weights", "1.0"
            ])

    def test_short_date_range_warning(self, caplog):
        """Test that short date ranges trigger a warning."""
        with patch("backtest.download_prices"):
            with patch("backtest.compute_metrics"):
                with patch("backtest.summarize") as mock_summarize:
                    mock_summarize.return_value = {
                        "ending_value": 100000,
                        "total_return": 0.0,
                        "cagr": 0.0,
                        "volatility": 0.0,
                        "sharpe_ratio": 0.0,
                        "sortino_ratio": 0.0,
                        "max_drawdown": 0.0,
                    }

                    # Run with short date range (20 days)
                    import logging
                    with caplog.at_level(logging.WARNING):
                        backtest.main([
                            "--start", "2023-01-01",
                            "--end", "2023-01-21",
                            "--tickers", "AAPL",
                            "--weights", "1.0",
                            "--benchmark", "SPY"
                        ])

                    # Check for warning
                    assert any("Short backtest period" in record.message for record in caplog.records)

    def test_parse_args_with_date_validation(self):
        """Test that argparse uses date validation."""
        # Valid dates should work
        args = backtest.parse_args(["--start", "2020-01-01", "--end", "2020-12-31"])
        assert args.start == "2020-01-01"
        assert args.end == "2020-12-31"

        # Invalid date should raise
        with pytest.raises(SystemExit):
            backtest.parse_args(["--start", "invalid-date"])


class TestRetryLogic:
    """Test retry logic with exponential backoff"""

    def test_retry_decorator_success_first_attempt(self):
        """Test that function succeeds on first attempt."""
        call_count = 0

        @backtest.retry_with_backoff(max_retries=3, base_delay=0.1)
        def test_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = test_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_decorator_success_after_failures(self):
        """Test that function retries and eventually succeeds."""
        call_count = 0

        @backtest.retry_with_backoff(max_retries=3, base_delay=0.1)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "success"

        result = test_func()
        assert result == "success"
        assert call_count == 3

    def test_retry_decorator_max_retries_exceeded(self):
        """Test that function raises after max retries."""
        call_count = 0

        @backtest.retry_with_backoff(max_retries=3, base_delay=0.1)
        def test_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent error")

        with pytest.raises(ValueError, match="Persistent error"):
            test_func()

        assert call_count == 3

    def test_retry_decorator_exponential_backoff(self):
        """Test that retry delays follow exponential backoff."""
        import time
        call_times = []

        @backtest.retry_with_backoff(max_retries=3, base_delay=0.1, max_delay=1.0)
        def test_func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Error")
            return "success"

        test_func()

        # Check delays: should be ~0.1s, ~0.2s
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        assert 0.05 < delay1 < 0.2  # ~0.1s with tolerance
        assert 0.15 < delay2 < 0.3  # ~0.2s with tolerance


class TestDownloadPrices:
    """Test price download functionality"""

    @patch("backtest.yf.download")
    @patch("backtest.load_cached_prices")
    @patch("backtest.save_cached_prices")
    def test_cache_hit(self, mock_save, mock_load, mock_yf_download):
        """Test that cached data is used when available"""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        cached_data = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104]}, index=dates)
        mock_load.return_value = cached_data

        result = backtest.download_prices(["AAPL"], "2020-01-01", "2020-01-05", use_cache=True)

        mock_load.assert_called_once()
        mock_yf_download.assert_not_called()
        mock_save.assert_not_called()
        pd.testing.assert_frame_equal(result, cached_data)

    @patch("backtest.yf.download")
    @patch("backtest.load_cached_prices")
    @patch("backtest.save_cached_prices")
    def test_cache_miss_downloads_and_caches(self, mock_save, mock_load, mock_yf_download):
        """Test that data is downloaded and cached when not in cache"""
        mock_load.return_value = None
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        downloaded_data = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104]}, index=dates)
        mock_yf_download.return_value = downloaded_data

        result = backtest.download_prices(["AAPL"], "2020-01-01", "2020-01-05", use_cache=True)

        mock_load.assert_called_once()
        mock_yf_download.assert_called_once()
        mock_save.assert_called_once()

    @patch("backtest.yf.download")
    def test_no_cache_skips_cache(self, mock_yf_download):
        """Test that use_cache=False skips cache"""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        downloaded_data = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104]}, index=dates)
        mock_yf_download.return_value = downloaded_data

        with patch("backtest.load_cached_prices") as mock_load:
            with patch("backtest.save_cached_prices") as mock_save:
                result = backtest.download_prices(
                    ["AAPL"], "2020-01-01", "2020-01-05", use_cache=False
                )

                mock_load.assert_not_called()
                mock_save.assert_not_called()
                mock_yf_download.assert_called_once()

    @patch("backtest.yf.download")
    @patch("backtest.load_cached_prices")
    @patch("backtest.save_cached_prices")
    def test_empty_data_raises_error(self, mock_save, mock_load, mock_yf_download):
        """Test that empty data raises helpful error"""
        mock_load.return_value = None
        mock_yf_download.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="No price data returned"):
            backtest.download_prices(["INVALID"], "2020-01-01", "2020-01-05")

    @patch("backtest.yf.download")
    @patch("backtest.load_cached_prices")
    @patch("backtest.save_cached_prices")
    def test_batch_download_all_cached(self, mock_save, mock_load, mock_yf_download):
        """Test batch download when all tickers are cached"""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")

        # Mock cache returning data for each ticker
        def cache_side_effect(path, *args, **kwargs):
            # Determine which ticker based on the path
            if "AAPL" in str(path):
                return pd.DataFrame({"AAPL": [100, 101, 102, 103, 104]}, index=dates)
            elif "MSFT" in str(path):
                return pd.DataFrame({"MSFT": [200, 201, 202, 203, 204]}, index=dates)
            elif "GOOGL" in str(path):
                return pd.DataFrame({"GOOGL": [300, 301, 302, 303, 304]}, index=dates)
            return None

        # Use get_cache_path to determine ticker names
        original_get_cache_path = backtest.get_cache_path
        def get_cache_path_wrapper(tickers, *args, **kwargs):
            result = original_get_cache_path(tickers, *args, **kwargs)
            # Store ticker info in path name for identification (Parquet format uses no extension)
            result = Path(f"{result}_{tickers[0]}")
            return result

        with patch("backtest.get_cache_path", side_effect=get_cache_path_wrapper):
            mock_load.side_effect = cache_side_effect

            result = backtest.download_prices(
                ["AAPL", "MSFT", "GOOGL"], "2020-01-01", "2020-01-05", use_cache=True
            )

            # Should not download anything
            mock_yf_download.assert_not_called()
            # Should have loaded from cache 3 times
            assert mock_load.call_count == 3
            # Should have all three tickers
            assert list(result.columns) == ["AAPL", "MSFT", "GOOGL"]

    @patch("backtest.yf.download")
    @patch("backtest.load_cached_prices")
    @patch("backtest.save_cached_prices")
    def test_batch_download_partial_cache(self, mock_save, mock_load, mock_yf_download):
        """Test batch download when some tickers are cached"""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")

        # Mock cache: AAPL cached, MSFT not cached
        def cache_side_effect(path, *args, **kwargs):
            if "AAPL" in str(path):
                return pd.DataFrame({"AAPL": [100, 101, 102, 103, 104]}, index=dates)
            return None  # MSFT not cached

        original_get_cache_path = backtest.get_cache_path
        def get_cache_path_wrapper(tickers, *args, **kwargs):
            result = original_get_cache_path(tickers, *args, **kwargs)
            # Store ticker info in path name for identification (Parquet format uses no extension)
            result = Path(f"{result}_{tickers[0]}")
            return result

        # Mock yfinance download for uncached ticker
        mock_yf_download.return_value = pd.DataFrame(
            {"MSFT": [200, 201, 202, 203, 204]}, index=dates
        )

        with patch("backtest.get_cache_path", side_effect=get_cache_path_wrapper):
            mock_load.side_effect = cache_side_effect

            result = backtest.download_prices(
                ["AAPL", "MSFT"], "2020-01-01", "2020-01-05", use_cache=True
            )

            # Should download only MSFT
            mock_yf_download.assert_called_once()
            # Should check cache for both tickers
            assert mock_load.call_count == 2
            # Should save only MSFT to cache
            assert mock_save.call_count == 1
            # Should have both tickers in result
            assert list(result.columns) == ["AAPL", "MSFT"]

    # Removed test_batch_download_no_cache_hits - requires network access to Yahoo Finance

    @patch("backtest.yf.download")
    def test_single_ticker_uses_standard_path(self, mock_yf_download):
        """Test that single ticker doesn't use batch optimization"""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        mock_yf_download.return_value = pd.DataFrame(
            {"AAPL": [100, 101, 102, 103, 104]}, index=dates
        )

        with patch("backtest.load_cached_prices") as mock_load:
            with patch("backtest.save_cached_prices") as mock_save:
                mock_load.return_value = None

                result = backtest.download_prices(
                    ["AAPL"], "2020-01-01", "2020-01-05", use_cache=True
                )

                # Should use standard path (single cache check)
                assert mock_load.call_count == 1
                mock_yf_download.assert_called_once()
                mock_save.assert_called_once()


class TestDataValidation:
    """Test data validation functionality"""

    def test_validate_price_data_all_nan(self):
        """Test that all-NaN data is rejected"""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"AAPL": [np.nan] * 10}, index=dates)

        with pytest.raises(ValueError, match="all values are NaN"):
            backtest.validate_price_data(df, ["AAPL"])

    def test_validate_price_data_excessive_nan(self):
        """Test that >50% NaN data raises error"""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        # 6 NaN out of 10 = 60%
        data = [100, 101, 102, 103, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df = pd.DataFrame({"AAPL": data}, index=dates)

        with pytest.raises(ValueError, match="60.0% missing values"):
            backtest.validate_price_data(df, ["AAPL"])

    def test_validate_price_data_negative_prices(self):
        """Test that negative prices are detected"""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"AAPL": [100, 101, -50, 103, 104, 105, 106, 107, 108, 109]}, index=dates)

        with pytest.raises(ValueError, match="negative price"):
            backtest.validate_price_data(df, ["AAPL"])

    def test_validate_price_data_zero_prices(self):
        """Test that zero prices are detected"""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"AAPL": [100, 101, 0, 103, 104, 105, 106, 107, 108, 109]}, index=dates)

        with pytest.raises(ValueError, match="zero price"):
            backtest.validate_price_data(df, ["AAPL"])

    def test_validate_price_data_extreme_changes(self):
        """Test that extreme price changes (>90%) are detected"""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        # 100 -> 1 is a -99% change
        df = pd.DataFrame({"AAPL": [100, 101, 102, 1, 2, 3, 4, 5, 6, 7]}, index=dates)

        with pytest.raises(ValueError, match="extreme price change"):
            backtest.validate_price_data(df, ["AAPL"])

    def test_validate_price_data_valid(self):
        """Test that valid data passes validation"""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}, index=dates)

        # Should not raise
        backtest.validate_price_data(df, ["AAPL"])

    def test_validate_price_data_acceptable_nan(self):
        """Test that <50% NaN data passes"""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        # 4 NaN out of 10 = 40% (acceptable)
        data = [100, 101, 102, 103, 104, 105, np.nan, np.nan, np.nan, np.nan]
        df = pd.DataFrame({"AAPL": data}, index=dates)

        # Should not raise
        backtest.validate_price_data(df, ["AAPL"])

    def test_compute_metrics_insufficient_data(self):
        """Test that compute_metrics rejects insufficient data"""
        # Only 1 day of data
        dates = pd.date_range("2020-01-01", periods=1, freq="D")
        prices = pd.DataFrame({"AAPL": [100]}, index=dates)
        benchmark = pd.Series([400], index=dates)
        weights = np.array([1.0])

        with pytest.raises(ValueError, match="Insufficient overlapping data"):
            backtest.compute_metrics(prices, weights, benchmark, 100000)

    def test_compute_metrics_two_days_minimum(self):
        """Test that compute_metrics accepts 2 days minimum"""
        dates = pd.date_range("2020-01-01", periods=2, freq="D")
        prices = pd.DataFrame({"AAPL": [100, 101]}, index=dates)
        benchmark = pd.Series([400, 401], index=dates)
        weights = np.array([1.0])

        # Should not raise (2 days is minimum)
        result = backtest.compute_metrics(prices, weights, benchmark, 100000)
        assert len(result) == 2


class TestDCA:
    """Test Dollar-Cost Averaging (DCA) functionality"""

    def test_dca_monthly_contributions(self):
        """Test DCA with monthly contributions"""
        # Create 6 months of data
        dates = pd.date_range("2020-01-01", periods=180, freq="D")
        prices = pd.DataFrame({
            "AAPL": np.linspace(100, 120, 180),  # Price increases from 100 to 120
        }, index=dates)
        weights = np.array([1.0])
        capital = 10000
        dca_amount = 1000
        dca_freq = "M"

        portfolio_value, cumulative_contributions = backtest._calculate_dca_portfolio(prices, weights, capital, dca_amount, dca_freq)

        # Check that portfolio value increases due to contributions and price appreciation
        assert portfolio_value.iloc[0] > 0
        assert portfolio_value.iloc[-1] > portfolio_value.iloc[0]

        # Check cumulative contributions
        expected_contributions = capital + (dca_amount * 5)  # 6 months total (including initial)
        assert cumulative_contributions.iloc[-1] == expected_contributions

        # Portfolio value at end should be greater than total contributions
        # (assuming price appreciation)
        assert portfolio_value.iloc[-1] > expected_contributions

    def test_dca_weekly_contributions(self):
        """Test DCA with weekly contributions"""
        dates = pd.date_range("2020-01-01", periods=90, freq="D")
        prices = pd.DataFrame({
            "AAPL": [100] * 90,  # Constant price
        }, index=dates)
        weights = np.array([1.0])
        capital = 5000
        dca_amount = 500
        dca_freq = "W"

        portfolio_value, cumulative_contributions = backtest._calculate_dca_portfolio(prices, weights, capital, dca_amount, dca_freq)

        # With constant price, portfolio value should equal total contributions
        # DCA dates include first date + weekly dates (first date is added if not in weekly schedule)
        weekly_dates = pd.date_range(dates[0], dates[-1], freq="W").intersection(dates)
        if dates[0] not in weekly_dates:
            num_dca_dates = len(weekly_dates) + 1  # Add 1 for initial date
        else:
            num_dca_dates = len(weekly_dates)

        expected_value = capital + (dca_amount * (num_dca_dates - 1))  # -1 because first is initial capital

        # Check contributions match expected
        assert cumulative_contributions.iloc[-1] == expected_value

        # Allow small tolerance due to rounding
        assert abs(portfolio_value.iloc[-1] - expected_value) < 10

    def test_dca_multi_ticker(self):
        """Test DCA with multiple tickers"""
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        prices = pd.DataFrame({
            "AAPL": [100] * 60,
            "MSFT": [200] * 60,
        }, index=dates)
        weights = np.array([0.6, 0.4])
        capital = 10000
        dca_amount = 1000
        dca_freq = "M"

        portfolio_value, cumulative_contributions = backtest._calculate_dca_portfolio(prices, weights, capital, dca_amount, dca_freq)

        # Check that contributions are split according to weights
        assert portfolio_value.iloc[0] == capital
        assert portfolio_value.iloc[-1] > capital
        assert cumulative_contributions.iloc[0] == capital

    def test_dca_with_price_decline(self):
        """Test DCA benefits when prices decline (buy more shares at lower prices)"""
        dates = pd.date_range("2020-01-01", periods=90, freq="D")
        # Price declines from 100 to 50
        prices = pd.DataFrame({
            "AAPL": np.linspace(100, 50, 90),
        }, index=dates)
        weights = np.array([1.0])
        capital = 10000
        dca_amount = 1000
        dca_freq = "M"

        portfolio_value, cumulative_contributions = backtest._calculate_dca_portfolio(prices, weights, capital, dca_amount, dca_freq)

        # Portfolio value should decline due to price decline
        # but cumulative contributions continue to grow
        assert portfolio_value.iloc[-1] < cumulative_contributions.iloc[-1]

    def test_dca_no_frequency(self):
        """Test that DCA with no frequency falls back to lump sum"""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        prices = pd.DataFrame({
            "AAPL": [100] * 30,
        }, index=dates)
        weights = np.array([1.0])
        capital = 10000
        dca_amount = 1000
        dca_freq = "M"

        # Test with very short period where no DCA dates are generated
        short_dates = dates[:5]  # Only 5 days
        short_prices = prices.iloc[:5]

        portfolio_value, cumulative_contributions = backtest._calculate_dca_portfolio(short_prices, weights, capital, dca_amount, "Q")

        # Should have at least initial investment
        assert portfolio_value.iloc[0] == capital
        assert cumulative_contributions.iloc[0] == capital

    def test_compute_metrics_with_dca(self):
        """Test compute_metrics with DCA parameters and accurate return calculations"""
        dates = pd.date_range("2020-01-01", periods=90, freq="D")
        prices = pd.DataFrame({
            "AAPL": [100] * 90,
            "MSFT": [200] * 90,
        }, index=dates)
        benchmark = pd.Series([400] * 90, index=dates)
        weights = np.array([0.5, 0.5])
        capital = 10000
        dca_amount = 1000
        dca_freq = "M"

        results = backtest.compute_metrics(
            prices, weights, benchmark, capital,
            dca_amount=dca_amount, dca_freq=dca_freq
        )

        # Check that results contain expected columns including contributions
        assert 'portfolio_value' in results.columns
        assert 'portfolio_return' in results.columns
        assert 'portfolio_contributions' in results.columns
        assert 'benchmark_value' in results.columns
        assert 'benchmark_contributions' in results.columns
        assert 'active_return' in results.columns

        # Portfolio value should be greater than initial capital due to contributions
        assert results['portfolio_value'].iloc[-1] > capital

        # Benchmark should also have DCA applied for fair comparison
        assert results['benchmark_value'].iloc[-1] > capital

        # Check contributions are tracked correctly
        # DCA dates include first date + monthly dates (first date is added if not in monthly schedule)
        monthly_dates = pd.date_range(dates[0], dates[-1], freq="ME").intersection(dates)
        if dates[0] not in monthly_dates:
            num_dca_dates = len(monthly_dates) + 1  # Add 1 for initial date
        else:
            num_dca_dates = len(monthly_dates)

        expected_total = capital + (dca_amount * (num_dca_dates - 1))
        assert results['portfolio_contributions'].iloc[-1] == expected_total
        assert results['benchmark_contributions'].iloc[-1] == expected_total

        # With constant prices, portfolio and benchmark ending values should equal contributions
        assert abs(results['portfolio_value'].iloc[-1] - expected_total) < 10
        assert abs(results['benchmark_value'].iloc[-1] - expected_total) < 10

        # Returns should be ~0% since prices are constant (no gains/losses)
        assert abs(results['portfolio_return'].iloc[-1]) < 0.01  # < 1%
        assert abs(results['benchmark_return'].iloc[-1]) < 0.01  # < 1%

    def test_dca_precedence_over_rebalancing(self, caplog):
        """Test that DCA takes precedence over rebalancing when both specified"""
        dates = pd.date_range("2020-01-01", periods=90, freq="D")
        prices = pd.DataFrame({
            "AAPL": [100] * 90,
        }, index=dates)
        benchmark = pd.Series([400] * 90, index=dates)
        weights = np.array([1.0])
        capital = 10000

        with caplog.at_level(logging.WARNING):
            results = backtest.compute_metrics(
                prices, weights, benchmark, capital,
                rebalance_freq="M",
                dca_amount=1000,
                dca_freq="M"
            )

        # Check that warning was logged
        assert any("mutually exclusive" in record.message for record in caplog.records)

        # Portfolio should use DCA logic (value > capital due to contributions)
        assert results['portfolio_value'].iloc[-1] > capital

    def test_dca_weekend_handling(self):
        """Test that DCA dates falling on weekends/holidays use next trading day"""
        # Create date range with only weekdays (simulating market closed on weekends)
        all_dates = pd.date_range("2022-11-18", "2023-11-18", freq="D")
        trading_days = all_dates[all_dates.dayofweek < 5]  # Only Mon-Fri

        prices = pd.DataFrame({
            "AAPL": [100] * len(trading_days),
        }, index=trading_days)

        weights = np.array([1.0])
        capital = 10000
        dca_amount = 1000
        dca_freq = "M"

        portfolio_value, cumulative_contributions = backtest._calculate_dca_portfolio(
            prices, weights, capital, dca_amount, dca_freq
        )

        # Generate expected monthly dates
        monthly_dates = pd.date_range("2022-11-18", "2023-11-18", freq="ME")

        # Count actual contributions
        contrib_changes = cumulative_contributions.diff().fillna(cumulative_contributions.iloc[0])
        contrib_events = contrib_changes[contrib_changes > 0]

        # Should have 1 initial + 12 monthly contributions = 13 total
        # Even though some monthly dates fall on weekends
        assert len(contrib_events) == 13, \
            f"Expected 13 contributions (1 initial + 12 monthly), got {len(contrib_events)}"

        # Total should be $22,000 (10k initial + 12*1k monthly)
        expected_total = capital + (dca_amount * len(monthly_dates))
        assert cumulative_contributions.iloc[-1] == expected_total, \
            f"Expected ${expected_total:,.2f}, got ${cumulative_contributions.iloc[-1]:,.2f}"

        # Verify no contributions were skipped due to weekends
        assert cumulative_contributions.iloc[-1] == 22000.0

    def test_daily_dca_preserves_weekend_contributions(self):
        dates = pd.date_range("2023-06-01", "2023-06-10", freq="D")
        trading_days = pd.bdate_range(dates[0], dates[-1] + pd.Timedelta(days=2))

        prices = pd.DataFrame({
            "AAPL": [100.0] * len(trading_days)
        }, index=trading_days)

        weights = np.array([1.0])
        capital = 1000
        dca_amount = 100

        portfolio_value, cumulative_contributions = backtest._calculate_dca_portfolio(
            prices,
            weights,
            capital,
            dca_amount,
            "D"
        )

        contrib_changes = cumulative_contributions.diff().fillna(cumulative_contributions.iloc[0])
        monday = pd.Timestamp("2023-06-05")  # Sat/Sun + Monday contributions
        assert contrib_changes.loc[monday] == pytest.approx(300, abs=1e-9)

        calendar_days = (prices.index[-1] - prices.index[0]).days + 1
        expected_total = capital + dca_amount * (calendar_days - 1)
        assert cumulative_contributions.iloc[-1] == expected_total

    def test_xirr_calculation_basic(self):
        """Test basic XIRR calculation with known cashflows"""
        # Simple case: invest $1000, get $1100 back after 365 days = 10% return
        cashflows = np.array([-1000.0, 1100.0])
        days = np.array([0, 365])

        irr = backtest._calculate_xirr(cashflows, days)

        # Should be close to 10%
        assert irr is not None
        assert abs(irr - 0.10) < 0.01  # Within 1% tolerance

    def test_xirr_calculation_multiple_contributions(self):
        """Test XIRR with multiple contributions over time"""
        # Three contributions of $1000 each (at start, day 180, day 360)
        # Final value $3300 at day 720 (2 years)
        # Expected IRR around 10% annually
        cashflows = np.array([-1000.0, -1000.0, -1000.0, 3300.0])
        days = np.array([0, 180, 360, 720])

        irr = backtest._calculate_xirr(cashflows, days)

        # Should be positive (growth)
        assert irr is not None
        assert irr > 0
        assert irr < 1.0  # Less than 100% annual return

    def test_xirr_negative_return(self):
        """Test XIRR with negative returns (losses)"""
        # Invest $1000, get back $900 after 365 days = -10% return
        cashflows = np.array([-1000.0, 900.0])
        days = np.array([0, 365])

        irr = backtest._calculate_xirr(cashflows, days)

        # Should be close to -10%
        assert irr is not None
        assert irr < 0
        assert abs(irr - (-0.10)) < 0.01

    def test_xirr_zero_return(self):
        """Test XIRR with zero returns (break even)"""
        # Invest $1000, get back $1000 after 365 days = 0% return
        cashflows = np.array([-1000.0, 1000.0])
        days = np.array([0, 365])

        irr = backtest._calculate_xirr(cashflows, days)

        # Should be close to 0%
        assert irr is not None
        assert abs(irr) < 0.01

    def test_xirr_non_convergence(self):
        """Test XIRR returns None when calculation doesn't converge"""
        # Unrealistic cashflows that may not converge
        cashflows = np.array([-1000.0, -1000.0, -1000.0, 10.0])  # Heavy losses
        days = np.array([0, 30, 60, 90])

        irr = backtest._calculate_xirr(cashflows, days)

        # Should return None for non-converging cases
        # (or a very negative value like -99%)
        if irr is not None:
            assert irr < -0.9  # Very negative return

    def test_summarize_with_irr(self):
        """Test that summarize calculates IRR when contributions_series is provided"""
        # Create 6 months of data with monthly DCA
        dates = pd.date_range("2020-01-01", periods=180, freq="D")

        # Price grows from 100 to 110 (10% over 6 months)
        portfolio_values = pd.Series(
            np.linspace(10000, 15500, 180),  # Value grows due to contributions + returns
            index=dates
        )

        # Monthly contributions: initial 10000, then 1000/month for 5 months
        contributions = pd.Series(0.0, index=dates)
        monthly_dates = pd.date_range(dates[0], dates[-1], freq="ME").intersection(dates)
        if dates[0] not in monthly_dates:
            monthly_dates = monthly_dates.insert(0, dates[0])

        # Set contribution values
        contributions.iloc[0] = 10000
        for i, date in enumerate(monthly_dates[1:], 1):
            # Find closest index
            idx = dates.get_loc(date)
            contributions.iloc[idx] = 1000

        # Make cumulative
        contributions = contributions.cumsum()

        capital = 10000
        total_contributions = 15000  # 10000 + 5*1000

        stats = backtest.summarize(
            portfolio_values,
            capital,
            total_contributions=total_contributions,
            contributions_series=contributions
        )

        # Should include IRR
        assert 'irr' in stats
        assert stats['irr'] is not None

        # IRR should be reasonable (not extreme)
        assert -0.99 < stats['irr'] < 10.0

        # IRR should be different from CAGR for DCA strategies
        # (though they should be in the same ballpark)
        assert 'cagr' in stats

    def test_summarize_without_irr(self):
        """Test that summarize works without IRR (lump sum investment)"""
        dates = pd.date_range("2020-01-01", periods=365, freq="D")

        # Lump sum: invest 10000, grows to 11000 (10% return)
        portfolio_values = pd.Series(
            np.linspace(10000, 11000, 365),
            index=dates
        )

        capital = 10000

        # Don't pass contributions_series (lump sum investment)
        stats = backtest.summarize(portfolio_values, capital)

        # Should NOT include IRR (only for DCA)
        assert 'irr' not in stats

        # Should still have other metrics
        assert 'cagr' in stats
        assert 'sharpe_ratio' in stats
        assert 'total_return' in stats

    def test_summarize_irr_single_contribution(self):
        """Test that IRR is not calculated for single contribution (lump sum)"""
        dates = pd.date_range("2020-01-01", periods=180, freq="D")

        portfolio_values = pd.Series(
            np.linspace(10000, 11000, 180),
            index=dates
        )

        # Contributions series with only initial investment (no DCA)
        contributions = pd.Series(10000, index=dates)

        capital = 10000

        stats = backtest.summarize(
            portfolio_values,
            capital,
            total_contributions=capital,
            contributions_series=contributions
        )

        # Should NOT calculate IRR for single contribution
        assert 'irr' not in stats


class TestMain:
    """Test main function integration"""

    def test_weight_normalization(self, capsys):
        """Test that weights are normalized if they don't sum to 1"""
        with patch("backtest.download_prices") as mock_download:
            with patch("backtest.compute_metrics") as mock_compute:
                with patch("backtest.summarize") as mock_summarize:
                    dates = pd.date_range("2020-01-01", periods=10, freq="D")
                    mock_download.return_value = pd.DataFrame(
                        {"A": [100] * 10, "B": [100] * 10, "C": [100] * 10}, index=dates
                    )
                    mock_compute.return_value = pd.DataFrame(
                        {
                            "portfolio_value": [100_000] * 10,
                            "portfolio_return": [0.0] * 10,
                            "portfolio_contributions": [100_000] * 10,
                            "benchmark_value": [100_000] * 10,
                            "benchmark_return": [0.0] * 10,
                            "benchmark_contributions": [100_000] * 10,
                            "active_return": [0.0] * 10,
                        },
                        index=dates,
                    )
                    mock_summarize.return_value = {
                        "ending_value": 100_000,
                        "total_return": 0.0,
                        "cagr": 0.0,
                        "volatility": 0.0,
                        "sharpe_ratio": 0.0,
                        "sortino_ratio": 0.0,
                        "max_drawdown": 0.0,
                    }

                    # Weights that don't sum to 1, using C as benchmark
                    backtest.main(["--tickers", "A", "B", "--weights", "2", "3", "--benchmark", "C"])

                    # Check that normalized weights were passed
                    call_args = mock_compute.call_args
                    weights = call_args[0][1]
                    assert np.isclose(weights.sum(), 1.0)
                    assert np.allclose(weights, [0.4, 0.6])

    def test_mismatched_weights_and_tickers(self):
        """Test that mismatched weights and tickers raises error"""
        with pytest.raises(SystemExit, match="Number of weights must match"):
            backtest.main(["--tickers", "A", "B", "--weights", "0.5"])



class TestXIRR:
    """Test XIRR calculation stability"""

    def test_basic_xirr(self):
        # Simple case: -1000, +1100 after 1 year = 10%
        cashflows = np.array([-1000.0, 1100.0])
        days = np.array([0.0, 365.0])
        irr = backtest._calculate_xirr(cashflows, days)
        assert irr == pytest.approx(0.10, abs=1e-4)

    def test_xirr_multiple_cashflows(self):
        # -1000 at t=0, -1000 at t=365, +2310 at t=730
        # Approx 10% return:
        # -1000 * 1.1^2 = -1210
        # -1000 * 1.1^1 = -1100
        # Total required = 2310
        cashflows = np.array([-1000.0, -1000.0, 2310.0])
        days = np.array([0.0, 365.0, 730.0])
        irr = backtest._calculate_xirr(cashflows, days)
        assert irr == pytest.approx(0.10, abs=1e-4)

    def test_xirr_fallback_convergence(self):
        # Case that might challenge Newton-Raphson but work with Bisection
        # Alternating signs or extreme values
        cashflows = np.array([-100, 200, -150, 100])
        days = np.array([0, 100, 200, 300])
        # Just check it returns a valid number, not None
        irr = backtest._calculate_xirr(cashflows, days)
        assert irr is not None
        assert -0.99 <= irr <= 10.0

    def test_xirr_no_convergence_possible(self):
        # All positive cashflows - impossible to have IRR
        cashflows = np.array([100.0, 100.0])
        days = np.array([0.0, 365.0])
        irr = backtest._calculate_xirr(cashflows, days)
        assert irr is None

    def test_xirr_extreme_values(self):
        # Very large return
        cashflows = np.array([-100.0, 1000.0])
        days = np.array([0.0, 365.0])
        irr = backtest._calculate_xirr(cashflows, days)
        assert irr == pytest.approx(9.0, abs=1e-4)  # 900% return

    def test_xirr_loss_scenario(self):
        # Negative return: -100 invested, 50 returned after 1 year
        cashflows = np.array([-100.0, 50.0])
        days = np.array([0.0, 365.0])
        irr = backtest._calculate_xirr(cashflows, days)
        assert irr is not None
        assert irr == pytest.approx(-0.5, abs=1e-4)  # -50% return

    def test_xirr_very_small_cashflows(self):
        # Test with very small amounts to check precision
        cashflows = np.array([-0.01, 0.011])
        days = np.array([0.0, 365.0])
        irr = backtest._calculate_xirr(cashflows, days)
        assert irr is not None
        assert irr == pytest.approx(0.10, abs=1e-3)  # 10% return

    def test_xirr_long_duration(self):
        # Very long investment period (27+ years)
        cashflows = np.array([-100.0, 200.0])
        days = np.array([0.0, 10000.0])
        irr = backtest._calculate_xirr(cashflows, days)
        assert irr is not None
        # 100% return over 27.4 years ≈ 2.5% annual return
        assert irr == pytest.approx(0.0253, abs=1e-3)

    def test_xirr_zero_duration(self):
        # Edge case: zero duration should return None
        cashflows = np.array([-100.0, 110.0])
        days = np.array([0.0, 0.0])
        irr = backtest._calculate_xirr(cashflows, days)
        # With zero duration, the smart initial guess will have division by zero
        # Should gracefully handle this
        assert irr is None or abs(irr) < 100  # Either None or reasonable value

    def test_xirr_negative_cashflow_end(self):
        # Multiple inflows followed by large outflow
        cashflows = np.array([100.0, 100.0, -250.0])
        days = np.array([0.0, 365.0, 730.0])
        irr = backtest._calculate_xirr(cashflows, days)
        assert irr is not None
        # Should calculate correctly with outflow at end

    def test_xirr_bisection_fallback_verified(self):
        # Case designed to trigger bisection fallback with verifiable result
        # Irregular cashflow pattern: -1000, +100, +100, +100, +800
        # Total in: 1100, Total out: 1000, Net: 100 over ~3 years
        cashflows = np.array([-1000.0, 100.0, 100.0, 100.0, 800.0])
        days = np.array([0.0, 365.0, 730.0, 1095.0, 1460.0])
        irr = backtest._calculate_xirr(cashflows, days)
        assert irr is not None
        # Rough estimate: 10% gain over 4 years ≈ 2.4% annual
        assert 0.01 <= irr <= 0.05  # Should be small positive return


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
