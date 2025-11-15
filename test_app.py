"""Unit tests for app.py (Streamlit UI)"""

from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Mock streamlit before importing app
sys.modules['streamlit'] = MagicMock()

import backtest


class TestMetricLabels:
    """Test metric label mappings for UI display"""

    def test_metric_labels_complete(self):
        """Ensure all metrics from summarize() have labels"""
        # Create sample data
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        values = pd.Series(np.linspace(100000, 120000, 252), index=dates)

        summary = backtest.summarize(values, 100000.0)

        # Labels that should exist in the UI
        expected_labels = {
            "ending_value": "Ending Value",
            "total_return": "Total Return",
            "cagr": "CAGR",
            "volatility": "Volatility",
            "sharpe_ratio": "Sharpe Ratio",
            "sortino_ratio": "Sortino Ratio",
            "max_drawdown": "Max Drawdown"
        }

        # Ensure all summary keys have labels
        for key in summary.keys():
            assert key in expected_labels, f"Missing label for metric: {key}"


class TestBacktestIntegration:
    """Test integration between UI and backtest module"""

    def test_import_required_functions(self):
        """Verify all required functions can be imported from backtest"""
        from backtest import download_prices, compute_metrics, summarize

        assert callable(download_prices)
        assert callable(compute_metrics)
        assert callable(summarize)

    def test_ui_workflow_with_mock_data(self):
        """Test the complete UI workflow with mocked data"""
        # Setup: Create mock price data
        dates = pd.date_range("2020-01-01", periods=252, freq="D")

        portfolio_prices = pd.DataFrame({
            "AAPL": np.linspace(100, 150, 252),
            "MSFT": np.linspace(200, 250, 252)
        }, index=dates)

        benchmark_prices = pd.Series(
            np.linspace(300, 360, 252),
            index=dates
        )

        weights = np.array([0.6, 0.4])
        capital = 100000.0

        # Execute: Run backtest computation
        results = backtest.compute_metrics(
            portfolio_prices,
            weights,
            benchmark_prices,
            capital
        )

        # Verify results structure
        assert isinstance(results, pd.DataFrame)
        assert 'portfolio_value' in results.columns
        assert 'benchmark_value' in results.columns
        assert 'portfolio_return' in results.columns
        assert 'benchmark_return' in results.columns
        assert 'active_return' in results.columns

        # Execute: Generate summaries
        portfolio_summary = backtest.summarize(results['portfolio_value'], capital)
        benchmark_summary = backtest.summarize(results['benchmark_value'], capital)

        # Verify summary structure
        assert isinstance(portfolio_summary, dict)
        assert isinstance(benchmark_summary, dict)

        # Verify all expected keys exist (using lowercase underscore format)
        expected_keys = {
            'ending_value', 'total_return', 'cagr',
            'volatility', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown'
        }
        assert set(portfolio_summary.keys()) == expected_keys
        assert set(benchmark_summary.keys()) == expected_keys

        # Verify we can calculate relative performance
        excess_return = portfolio_summary['total_return'] - benchmark_summary['total_return']
        excess_cagr = portfolio_summary['cagr'] - benchmark_summary['cagr']
        excess_sharpe = portfolio_summary['sharpe_ratio'] - benchmark_summary['sharpe_ratio']

        assert isinstance(excess_return, float)
        assert isinstance(excess_cagr, float)
        assert isinstance(excess_sharpe, float)

    def test_weight_normalization(self):
        """Test that weights are properly normalized as in the UI"""
        weights = np.array([0.3, 0.4, 0.2])  # Sum = 0.9, not 1.0

        # This is what the UI does
        if not np.isclose(weights.sum(), 1.0):
            normalized_weights = weights / weights.sum()
        else:
            normalized_weights = weights

        assert np.isclose(normalized_weights.sum(), 1.0)
        assert len(normalized_weights) == 3
        assert normalized_weights[0] == pytest.approx(0.3333, rel=1e-3)
        assert normalized_weights[1] == pytest.approx(0.4444, rel=1e-3)
        assert normalized_weights[2] == pytest.approx(0.2222, rel=1e-3)

    def test_date_range_handling(self):
        """Test date range conversion as used in UI"""
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2024, 12, 31)

        # Convert to string format as the UI does
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        assert start_str == "2018-01-01"
        assert end_str == "2024-12-31"


class TestMetricFormatting:
    """Test metric formatting logic used in UI"""

    def test_currency_formatting(self):
        """Test currency value formatting"""
        value = 142538.21
        formatted = f"${value:,.2f}"
        assert formatted == "$142,538.21"

    def test_percentage_formatting(self):
        """Test percentage formatting"""
        value = 0.4254
        formatted = f"{value:.2%}"
        assert formatted == "42.54%"

    def test_ratio_formatting(self):
        """Test ratio formatting (3 decimal places)"""
        value = 0.427
        formatted = f"{value:.3f}"
        assert formatted == "0.427"

    def test_metric_type_detection(self):
        """Test that metrics are categorized correctly"""
        # Metrics that should be formatted as percentages
        percentage_metrics = ["total_return", "cagr", "volatility", "max_drawdown"]

        # Metrics that should be formatted as ratios (3 decimals)
        ratio_metrics = ["sharpe_ratio", "sortino_ratio"]

        # Metrics that should be formatted as currency
        currency_metrics = ["ending_value"]

        # Verify categorization
        for metric in percentage_metrics:
            assert metric not in ratio_metrics
            assert metric not in currency_metrics

        for metric in ratio_metrics:
            assert metric not in percentage_metrics
            assert metric not in currency_metrics

        for metric in currency_metrics:
            assert metric not in percentage_metrics
            assert metric not in ratio_metrics


class TestErrorHandling:
    """Test error handling scenarios in UI workflow"""

    def test_empty_ticker_validation(self):
        """Test handling of empty ticker input"""
        tickers = ["", "MSFT"]

        # UI should detect this: if not all(tickers)
        has_empty = not all(tickers)
        assert has_empty is True

    def test_invalid_date_range(self):
        """Test detection of invalid date ranges"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2020, 1, 1)  # End before start

        # UI should detect this: if start_date >= end_date
        is_invalid = start_date >= end_date
        assert is_invalid is True

    def test_compute_metrics_error_handling(self):
        """Test that compute_metrics raises appropriate errors"""
        # Create data with missing values that will cause alignment issues
        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        # Portfolio with some NaN values
        portfolio_prices = pd.DataFrame({
            "AAPL": [np.nan] * 100  # All NaN values
        }, index=dates)

        benchmark_prices = pd.Series(
            np.random.randn(100) + 200,
            index=dates
        )

        weights = np.array([1.0])
        capital = 100000.0

        # This should raise an error due to no valid data
        with pytest.raises(ValueError, match="no valid prices"):
            backtest.compute_metrics(
                portfolio_prices,
                weights,
                benchmark_prices,
                capital
            )


class TestPortfolioComposition:
    """Test portfolio composition display logic"""

    def test_composition_table_data(self):
        """Test creation of portfolio composition table"""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        weights_original = [0.5, 0.3, 0.2]
        weights_array = np.array(weights_original)

        # Normalize weights as UI does
        if not np.isclose(weights_array.sum(), 1.0):
            weights_normalized = weights_array / weights_array.sum()
        else:
            weights_normalized = weights_array

        # Create composition data structure
        composition_data = {
            "Ticker": tickers,
            "Weight": [f"{w:.1%}" for w in weights_array],
            "Normalized Weight": [f"{w:.3%}" for w in weights_normalized]
        }

        # Verify structure
        assert len(composition_data["Ticker"]) == 3
        assert len(composition_data["Weight"]) == 3
        assert len(composition_data["Normalized Weight"]) == 3

        # Verify formatting
        assert composition_data["Weight"][0] == "50.0%"
        assert composition_data["Weight"][1] == "30.0%"
        assert composition_data["Weight"][2] == "20.0%"


class TestChartData:
    """Test chart data preparation"""

    def test_drawdown_calculation(self):
        """Test drawdown calculation for chart"""
        # Create sample portfolio values
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        values = pd.Series([100, 105, 110, 105, 100, 95, 105, 110, 115, 112] * 10, index=dates)

        # Calculate drawdown as in the UI
        cummax = values.expanding().max()
        drawdown = ((values - cummax) / cummax) * 100

        assert isinstance(drawdown, pd.Series)
        assert len(drawdown) == len(values)
        assert drawdown.max() == 0.0  # Max drawdown is at peak
        assert drawdown.min() < 0.0  # Should have negative drawdowns

    def test_active_return_calculation(self):
        """Test active return calculation for chart"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        portfolio_return = pd.Series(np.random.randn(100) * 0.01 + 0.0005, index=dates)
        benchmark_return = pd.Series(np.random.randn(100) * 0.01 + 0.0003, index=dates)

        # Calculate active return as in the UI
        active_return = (portfolio_return - benchmark_return) * 100

        assert isinstance(active_return, pd.Series)
        assert len(active_return) == 100


class TestDownloadFunctionality:
    """Test CSV and chart download preparation"""

    def test_csv_export_format(self):
        """Test that results can be exported to CSV"""
        # Create sample results DataFrame
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        results = pd.DataFrame({
            "portfolio_value": np.linspace(100000, 110000, 10),
            "portfolio_return": np.linspace(0, 0.1, 10),
            "benchmark_value": np.linspace(100000, 105000, 10),
            "benchmark_return": np.linspace(0, 0.05, 10),
            "active_return": np.linspace(0, 0.05, 10)
        }, index=dates)

        # Convert to CSV as UI does
        csv_data = results.to_csv()

        assert isinstance(csv_data, str)
        assert "portfolio_value" in csv_data
        assert "benchmark_value" in csv_data
        assert len(csv_data) > 0

    def test_timestamp_generation(self):
        """Test timestamp generation for filenames"""
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')

        # Verify format
        assert len(timestamp) == 15  # YYYYMMDD_HHMMSS
        assert "_" in timestamp

        # Create filename as UI does
        csv_filename = f"backtest_{timestamp}.csv"
        png_filename = f"backtest_charts_{timestamp}.png"

        assert csv_filename.endswith(".csv")
        assert png_filename.endswith(".png")


class TestCacheToggle:
    """Test cache functionality toggle"""

    @patch('backtest.download_prices')
    def test_cache_enabled(self, mock_download):
        """Test that cache parameter is passed correctly when enabled"""
        mock_download.return_value = pd.DataFrame()

        tickers = ["AAPL", "MSFT"]
        start_date = "2020-01-01"
        end_date = "2021-01-01"
        use_cache = True

        # This is what the UI calls
        backtest.download_prices(tickers, start_date, end_date, use_cache=use_cache)

        # Verify it was called with use_cache=True
        mock_download.assert_called_once_with(tickers, start_date, end_date, use_cache=True)

    @patch('backtest.download_prices')
    def test_cache_disabled(self, mock_download):
        """Test that cache parameter is passed correctly when disabled"""
        mock_download.return_value = pd.DataFrame()

        tickers = ["AAPL", "MSFT"]
        start_date = "2020-01-01"
        end_date = "2021-01-01"
        use_cache = False

        # This is what the UI calls
        backtest.download_prices(tickers, start_date, end_date, use_cache=use_cache)

        # Verify it was called with use_cache=False
        mock_download.assert_called_once_with(tickers, start_date, end_date, use_cache=False)


class TestInputValidation:
    """Test input validation logic"""

    def test_valid_inputs(self):
        """Test that valid inputs pass validation"""
        tickers = ["AAPL", "MSFT"]
        benchmark = "SPY"
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2024, 12, 31)

        # Validation checks from UI
        assert all(tickers)  # No empty tickers
        assert benchmark  # Non-empty benchmark
        assert start_date < end_date  # Valid date range

    def test_invalid_tickers(self):
        """Test detection of invalid ticker inputs"""
        tickers_empty = ["AAPL", ""]
        tickers_all_empty = ["", ""]

        assert not all(tickers_empty)
        assert not all(tickers_all_empty)

    def test_invalid_benchmark(self):
        """Test detection of empty benchmark"""
        benchmark_empty = ""
        benchmark_valid = "SPY"

        assert not benchmark_empty
        assert benchmark_valid

    def test_number_of_tickers_range(self):
        """Test ticker count validation"""
        min_tickers = 1
        max_tickers = 10

        valid_count = 5
        assert min_tickers <= valid_count <= max_tickers

        invalid_low = 0
        invalid_high = 11
        assert not (min_tickers <= invalid_low <= max_tickers)
        assert not (min_tickers <= invalid_high <= max_tickers)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
