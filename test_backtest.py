"""Unit tests for backtest.py"""

from __future__ import annotations

import tempfile
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

    def test_output_path(self):
        args = backtest.parse_args(["--output", "test.csv"])
        assert args.output == Path("test.csv")


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
            cache_path = Path(tmpdir) / "nonexistent.pkl"
            result = backtest.load_cached_prices(cache_path)
            assert result is None

    def test_save_and_load_cached_prices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.pkl"
            test_data = pd.DataFrame(
                {"AAPL": [100, 101, 102], "MSFT": [200, 201, 202]},
                index=pd.date_range("2020-01-01", periods=3),
            )

            backtest.save_cached_prices(cache_path, test_data)
            loaded_data = backtest.load_cached_prices(cache_path)

            assert loaded_data is not None
            pd.testing.assert_frame_equal(loaded_data, test_data)


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

    def test_empty_series_raises_error(self):
        empty_series = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="Cannot summarize an empty series"):
            backtest.summarize(empty_series, 100_000)


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
                            "benchmark_value": [100_000] * 10,
                            "benchmark_return": [0.0] * 10,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
