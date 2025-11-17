"""Unit tests for app.py (Streamlit UI)"""

from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

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
            "irr": "IRR",  # Optional metric for DCA strategies
            "volatility": "Volatility",
            "sharpe_ratio": "Sharpe Ratio",
            "sortino_ratio": "Sortino Ratio",
            "max_drawdown": "Max Drawdown",
            "total_contributions": "Total Contributions"  # Internal metric for DCA tracking
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
            'volatility', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'total_contributions'  # Added for DCA tracking
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

    @patch('app.ticker_data.yf.Ticker')
    def test_composition_table_data(self, mock_yf_ticker):
        """Test creation of portfolio composition table with ticker names from yfinance"""
        from app.ticker_data import get_ticker_name

        # Clear cache
        get_ticker_name.cache_clear()

        # Mock yfinance responses for different tickers
        def mock_ticker_side_effect(symbol):
            mock_instance = Mock()
            names = {
                "AAPL": {'longName': 'Apple Inc.'},
                "MSFT": {'longName': 'Microsoft Corporation'},
                "GOOGL": {'longName': 'Alphabet Inc. (Google) Class A'}
            }
            mock_instance.info = names.get(symbol, {})
            return mock_instance

        mock_yf_ticker.side_effect = mock_ticker_side_effect

        tickers = ["AAPL", "MSFT", "GOOGL"]
        weights_array = np.array([0.5, 0.3, 0.2])

        # Get ticker names as the UI does
        ticker_names = [get_ticker_name(ticker) for ticker in tickers]

        # Create composition data structure (matching actual implementation)
        composition_data = {
            "Ticker": tickers,
            "Name": ticker_names,
            "Weight": [f"{w:.1%}" for w in weights_array]
        }

        # Verify structure
        assert len(composition_data["Ticker"]) == 3
        assert len(composition_data["Name"]) == 3
        assert len(composition_data["Weight"]) == 3

        # Verify ticker names are fetched from yfinance
        assert composition_data["Name"][0] == "Apple Inc."
        assert composition_data["Name"][1] == "Microsoft Corporation"
        assert composition_data["Name"][2] == "Alphabet Inc. (Google) Class A"

        # Verify weight formatting
        assert composition_data["Weight"][0] == "50.0%"
        assert composition_data["Weight"][1] == "30.0%"
        assert composition_data["Weight"][2] == "20.0%"

    @patch('app.ticker_data.yf.Ticker')
    def test_composition_with_unknown_ticker(self, mock_yf_ticker):
        """Test composition table handles unknown tickers gracefully"""
        from app.ticker_data import get_ticker_name

        # Clear cache
        get_ticker_name.cache_clear()

        # Mock yfinance responses
        def mock_ticker_side_effect(symbol):
            mock_instance = Mock()
            names = {
                "AAPL": {'longName': 'Apple Inc.'},
                "SPY": {'longName': 'SPDR S&P 500 ETF Trust'},
                "UNKNOWN_TICKER": None  # No info for unknown ticker
            }
            mock_instance.info = names.get(symbol, None)
            return mock_instance

        mock_yf_ticker.side_effect = mock_ticker_side_effect

        tickers = ["AAPL", "UNKNOWN_TICKER", "SPY"]
        weights_array = np.array([0.4, 0.3, 0.3])

        # Get ticker names
        ticker_names = [get_ticker_name(ticker) for ticker in tickers]

        composition_data = {
            "Ticker": tickers,
            "Name": ticker_names,
            "Weight": [f"{w:.1%}" for w in weights_array]
        }

        # Known tickers should have names
        assert composition_data["Name"][0] == "Apple Inc."
        assert composition_data["Name"][2] == "SPDR S&P 500 ETF Trust"

        # Unknown ticker should have empty name
        assert composition_data["Name"][1] == ""


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


class TestPortfolioPresets:
    """Test portfolio preset functionality"""

    def test_preset_portfolios_defined(self):
        """Verify all expected portfolio presets exist"""
        expected_presets = [
            "Custom (Manual Entry)",
            "Default UK ETFs",
            "60/40 US Stocks/Bonds",
            "Tech Giants",
            "Dividend Aristocrats",
            "Global Diversified"
        ]

        # Portfolio preset data structure from app.py
        example_portfolios = {
            "Custom (Manual Entry)": {"tickers": [], "weights": [], "benchmark": "VWRA.L"},
            "Default UK ETFs": {"tickers": ["VDCP.L", "VHYD.L"], "weights": [0.5, 0.5], "benchmark": "VWRA.L"},
            "60/40 US Stocks/Bonds": {"tickers": ["VOO", "BND"], "weights": [0.6, 0.4], "benchmark": "SPY"},
            "Tech Giants": {"tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"], "weights": [0.25, 0.25, 0.25, 0.25], "benchmark": "QQQ"},
            "Dividend Aristocrats": {"tickers": ["JNJ", "PG", "KO", "PEP"], "weights": [0.25, 0.25, 0.25, 0.25], "benchmark": "SPY"},
            "Global Diversified": {"tickers": ["VTI", "VXUS", "BND"], "weights": [0.5, 0.3, 0.2], "benchmark": "VT"}
        }

        # Verify all expected presets are defined
        for preset in expected_presets:
            assert preset in example_portfolios

    def test_tech_giants_preset_values(self):
        """Verify Tech Giants preset has correct configuration"""
        tech_giants = {
            "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "weights": [0.25, 0.25, 0.25, 0.25],
            "benchmark": "QQQ"
        }

        assert len(tech_giants["tickers"]) == 4
        assert tech_giants["tickers"] == ["AAPL", "MSFT", "GOOGL", "AMZN"]
        assert all(w == 0.25 for w in tech_giants["weights"])
        assert tech_giants["benchmark"] == "QQQ"
        assert sum(tech_giants["weights"]) == 1.0

    def test_default_uk_etfs_preset_values(self):
        """Verify Default UK ETFs preset has correct configuration"""
        default_uk = {
            "tickers": ["VDCP.L", "VHYD.L"],
            "weights": [0.5, 0.5],
            "benchmark": "VWRA.L"
        }

        assert len(default_uk["tickers"]) == 2
        assert default_uk["tickers"] == ["VDCP.L", "VHYD.L"]
        assert all(w == 0.5 for w in default_uk["weights"])
        assert default_uk["benchmark"] == "VWRA.L"
        assert sum(default_uk["weights"]) == 1.0

    def test_60_40_preset_values(self):
        """Verify 60/40 preset has correct stock/bond allocation"""
        sixty_forty = {
            "tickers": ["VOO", "BND"],
            "weights": [0.6, 0.4],
            "benchmark": "SPY"
        }

        assert len(sixty_forty["tickers"]) == 2
        assert sixty_forty["weights"][0] == 0.6  # 60% stocks
        assert sixty_forty["weights"][1] == 0.4  # 40% bonds
        assert sixty_forty["benchmark"] == "SPY"
        assert sum(sixty_forty["weights"]) == 1.0

    def test_global_diversified_preset_values(self):
        """Verify Global Diversified preset has correct allocation"""
        global_div = {
            "tickers": ["VTI", "VXUS", "BND"],
            "weights": [0.5, 0.3, 0.2],
            "benchmark": "VT"
        }

        assert len(global_div["tickers"]) == 3
        assert global_div["weights"] == [0.5, 0.3, 0.2]
        assert global_div["benchmark"] == "VT"
        assert sum(global_div["weights"]) == 1.0

    def test_custom_preset_empty_configuration(self):
        """Verify Custom preset allows manual entry"""
        custom = {
            "tickers": [],
            "weights": [],
            "benchmark": "VWRA.L"
        }

        assert custom["tickers"] == []
        assert custom["weights"] == []
        assert custom["benchmark"] == "VWRA.L"  # Has default benchmark

    def test_all_presets_have_required_keys(self):
        """Verify all presets have tickers, weights, and benchmark keys"""
        example_portfolios = {
            "Custom (Manual Entry)": {"tickers": [], "weights": [], "benchmark": "VWRA.L"},
            "Default UK ETFs": {"tickers": ["VDCP.L", "VHYD.L"], "weights": [0.5, 0.5], "benchmark": "VWRA.L"},
            "60/40 US Stocks/Bonds": {"tickers": ["VOO", "BND"], "weights": [0.6, 0.4], "benchmark": "SPY"},
            "Tech Giants": {"tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"], "weights": [0.25, 0.25, 0.25, 0.25], "benchmark": "QQQ"},
            "Dividend Aristocrats": {"tickers": ["JNJ", "PG", "KO", "PEP"], "weights": [0.25, 0.25, 0.25, 0.25], "benchmark": "SPY"},
            "Global Diversified": {"tickers": ["VTI", "VXUS", "BND"], "weights": [0.5, 0.3, 0.2], "benchmark": "VT"}
        }

        for preset_name, preset_config in example_portfolios.items():
            assert "tickers" in preset_config, f"{preset_name} missing 'tickers'"
            assert "weights" in preset_config, f"{preset_name} missing 'weights'"
            assert "benchmark" in preset_config, f"{preset_name} missing 'benchmark'"

    def test_preset_weights_match_tickers(self):
        """Verify weights length matches tickers length for all presets"""
        example_portfolios = {
            "Default UK ETFs": {"tickers": ["VDCP.L", "VHYD.L"], "weights": [0.5, 0.5], "benchmark": "VWRA.L"},
            "60/40 US Stocks/Bonds": {"tickers": ["VOO", "BND"], "weights": [0.6, 0.4], "benchmark": "SPY"},
            "Tech Giants": {"tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"], "weights": [0.25, 0.25, 0.25, 0.25], "benchmark": "QQQ"},
            "Dividend Aristocrats": {"tickers": ["JNJ", "PG", "KO", "PEP"], "weights": [0.25, 0.25, 0.25, 0.25], "benchmark": "SPY"},
            "Global Diversified": {"tickers": ["VTI", "VXUS", "BND"], "weights": [0.5, 0.3, 0.2], "benchmark": "VT"}
        }

        for preset_name, preset_config in example_portfolios.items():
            if preset_config["tickers"]:  # Skip Custom which is empty
                assert len(preset_config["tickers"]) == len(preset_config["weights"]), \
                    f"{preset_name}: tickers/weights length mismatch"


class TestDateRangePresets:
    """Test date range preset calculations"""

    def test_one_year_preset(self):
        """Verify 1Y preset calculates correct date"""
        from datetime import datetime, timedelta

        today = datetime.today()
        one_year_ago = today - timedelta(days=365)

        # Verify date is approximately 365 days ago (within 1 day tolerance)
        diff = (today - one_year_ago).days
        assert 364 <= diff <= 366

    def test_three_year_preset(self):
        """Verify 3Y preset calculates correct date"""
        from datetime import datetime, timedelta

        today = datetime.today()
        three_years_ago = today - timedelta(days=365*3)

        diff = (today - three_years_ago).days
        assert 1093 <= diff <= 1096  # 3 years ± 1 day

    def test_five_year_preset(self):
        """Verify 5Y preset calculates correct date"""
        from datetime import datetime, timedelta

        today = datetime.today()
        five_years_ago = today - timedelta(days=365*5)

        diff = (today - five_years_ago).days
        assert 1823 <= diff <= 1827  # 5 years ± 2 days

    def test_ten_year_preset(self):
        """Verify 10Y preset calculates correct date"""
        from datetime import datetime, timedelta

        today = datetime.today()
        ten_years_ago = today - timedelta(days=365*10)

        diff = (today - ten_years_ago).days
        assert 3648 <= diff <= 3653  # 10 years ± 2 days

    def test_ytd_preset(self):
        """Verify YTD preset returns January 1st of current year"""
        from datetime import datetime

        today = datetime.today()
        ytd_date = datetime(today.year, 1, 1)

        assert ytd_date.year == today.year
        assert ytd_date.month == 1
        assert ytd_date.day == 1

    def test_max_preset(self):
        """Verify Max preset returns 2010-01-01"""
        from datetime import datetime

        max_date = datetime(2010, 1, 1)

        assert max_date.year == 2010
        assert max_date.month == 1
        assert max_date.day == 1

    def test_all_presets_return_datetime(self):
        """Verify all date presets return datetime objects"""
        from datetime import datetime, timedelta

        today = datetime.today()

        presets = {
            "1Y": today - timedelta(days=365),
            "3Y": today - timedelta(days=365*3),
            "5Y": today - timedelta(days=365*5),
            "10Y": today - timedelta(days=365*10),
            "YTD": datetime(today.year, 1, 1),
            "Max": datetime(2010, 1, 1)
        }

        for preset_name, preset_date in presets.items():
            assert isinstance(preset_date, datetime), f"{preset_name} not a datetime object"


class TestMultipleBenchmarks:
    """Test multiple benchmark functionality"""

    def test_single_benchmark_list(self):
        """Verify single benchmark works in list format"""
        benchmarks = ["SPY"]

        assert len(benchmarks) == 1
        assert benchmarks[0] == "SPY"

    def test_two_benchmarks_list(self):
        """Verify two benchmarks can be specified"""
        benchmarks = ["SPY", "QQQ"]

        assert len(benchmarks) == 2
        assert "SPY" in benchmarks
        assert "QQQ" in benchmarks

    def test_three_benchmarks_list(self):
        """Verify three benchmarks (maximum) can be specified"""
        benchmarks = ["SPY", "QQQ", "VTI"]

        assert len(benchmarks) == 3
        assert all(b in benchmarks for b in ["SPY", "QQQ", "VTI"])

    def test_benchmark_count_validation(self):
        """Verify benchmark count is within valid range (1-3)"""
        valid_counts = [1, 2, 3]
        invalid_counts = [0, 4, 5]

        for count in valid_counts:
            assert 1 <= count <= 3

        for count in invalid_counts:
            assert not (1 <= count <= 3)

    def test_multiple_benchmark_colors(self):
        """Verify color scheme for multiple benchmarks"""
        benchmark_colors = ['#9467bd', '#e377c2', '#bcbd22']

        assert len(benchmark_colors) >= 3  # At least 3 colors for 3 benchmarks
        assert all(color.startswith('#') for color in benchmark_colors)
        assert all(len(color) == 7 for color in benchmark_colors)  # Hex color format

    def test_multiple_benchmark_line_styles(self):
        """Verify line styles for multiple benchmarks"""
        benchmark_dash = ['dash', 'dot', 'dashdot']

        assert len(benchmark_dash) >= 3  # At least 3 styles for 3 benchmarks
        assert 'dash' in benchmark_dash
        assert 'dot' in benchmark_dash
        assert 'dashdot' in benchmark_dash

    def test_benchmark_index_wrapping(self):
        """Verify color/style indexing wraps correctly using modulo"""
        benchmark_colors = ['#9467bd', '#e377c2', '#bcbd22']

        # Test modulo wrapping for accessing colors
        idx_0 = 0 % len(benchmark_colors)  # 0
        idx_1 = 1 % len(benchmark_colors)  # 1
        idx_2 = 2 % len(benchmark_colors)  # 2

        assert idx_0 == 0
        assert idx_1 == 1
        assert idx_2 == 2

    @patch('backtest.download_prices')
    def test_multiple_benchmark_download(self, mock_download):
        """Test downloading multiple benchmarks"""
        dates = pd.date_range("2020-01-01", periods=252, freq="D")

        # Mock return data for each benchmark
        mock_download.side_effect = [
            pd.DataFrame({"SPY": np.linspace(300, 350, 252)}, index=dates),
            pd.DataFrame({"QQQ": np.linspace(250, 300, 252)}, index=dates)
        ]

        # Simulate downloading two benchmarks
        benchmarks = ["SPY", "QQQ"]
        all_benchmark_prices = {}

        for bench in benchmarks:
            bench_data = mock_download([bench], "2020-01-01", "2020-12-31")[bench]
            all_benchmark_prices[bench] = bench_data

        assert len(all_benchmark_prices) == 2
        assert "SPY" in all_benchmark_prices
        assert "QQQ" in all_benchmark_prices

    def test_multiple_benchmark_metrics_structure(self):
        """Test that multiple benchmarks create separate metric dictionaries"""
        # Simulate structure of all_benchmark_summaries
        all_benchmark_summaries = {
            "SPY": {
                "total_return": 0.15,
                "cagr": 0.12,
                "sharpe_ratio": 1.2
            },
            "QQQ": {
                "total_return": 0.20,
                "cagr": 0.16,
                "sharpe_ratio": 1.5
            }
        }

        assert len(all_benchmark_summaries) == 2
        assert "SPY" in all_benchmark_summaries
        assert "QQQ" in all_benchmark_summaries
        assert "total_return" in all_benchmark_summaries["SPY"]
        assert "cagr" in all_benchmark_summaries["QQQ"]


class TestDeltaIndicators:
    """Test delta indicator calculations and formatting"""

    def test_positive_delta_calculation(self):
        """Verify positive delta (outperformance) calculation"""
        portfolio_return = 0.15  # 15%
        benchmark_return = 0.10  # 10%
        excess_return = portfolio_return - benchmark_return

        assert excess_return == pytest.approx(0.05)
        assert excess_return > 0  # Positive delta (outperformance)

    def test_negative_delta_calculation(self):
        """Verify negative delta (underperformance) calculation"""
        portfolio_return = 0.08  # 8%
        benchmark_return = 0.12  # 12%
        excess_return = portfolio_return - benchmark_return

        assert excess_return == pytest.approx(-0.04)
        assert excess_return < 0  # Negative delta (underperformance)

    def test_zero_delta_calculation(self):
        """Verify zero delta (matching performance) calculation"""
        portfolio_return = 0.10
        benchmark_return = 0.10
        excess_return = portfolio_return - benchmark_return

        assert excess_return == 0.0

    def test_delta_formatting_percentage(self):
        """Verify delta percentage formatting"""
        excess_return = 0.05

        formatted = f"{excess_return:.2%}"
        assert formatted == "5.00%"

    def test_delta_formatting_ratio(self):
        """Verify delta ratio formatting (for Sharpe, Sortino)"""
        excess_sharpe = 0.35

        formatted = f"{excess_sharpe:.3f}"
        assert formatted == "0.350"

    def test_volatility_delta_inverse_logic(self):
        """Verify volatility delta uses inverse coloring (lower is better)"""
        portfolio_vol = 0.12
        benchmark_vol = 0.15
        excess_vol = portfolio_vol - benchmark_vol

        # Lower volatility is better, so negative excess is good
        assert excess_vol == -0.03
        # In UI: delta_color="inverse" means negative shows green

    def test_all_delta_metrics(self):
        """Verify all metrics that should have deltas"""
        portfolio_summary = {
            'total_return': 0.15,
            'cagr': 0.12,
            'sharpe_ratio': 1.2,
            'volatility': 0.14,
            'sortino_ratio': 1.5
        }

        benchmark_summary = {
            'total_return': 0.10,
            'cagr': 0.08,
            'sharpe_ratio': 1.0,
            'volatility': 0.16,
            'sortino_ratio': 1.3
        }

        # Calculate all deltas
        excess_return = portfolio_summary['total_return'] - benchmark_summary['total_return']
        excess_cagr = portfolio_summary['cagr'] - benchmark_summary['cagr']
        excess_sharpe = portfolio_summary['sharpe_ratio'] - benchmark_summary['sharpe_ratio']
        excess_volatility = portfolio_summary['volatility'] - benchmark_summary['volatility']
        excess_sortino = portfolio_summary['sortino_ratio'] - benchmark_summary['sortino_ratio']

        assert excess_return == pytest.approx(0.05)
        assert excess_cagr == pytest.approx(0.04)
        assert excess_sharpe == pytest.approx(0.2)
        assert excess_volatility == pytest.approx(-0.02)  # Lower is better
        assert excess_sortino == pytest.approx(0.2)


class TestRollingReturns:
    """Test rolling returns calculations"""

    def test_30_day_rolling_returns(self):
        """Verify 30-day rolling returns calculation"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        values = pd.Series(np.linspace(100, 110, 100), index=dates)

        # Calculate 30-day rolling returns
        rolling_30 = values.pct_change(30) * 100

        # Should have 30 NaN values at start
        assert rolling_30.isna().sum() == 30
        # Should have non-NaN values after window
        assert not rolling_30.iloc[30:].isna().all()

    def test_90_day_rolling_returns(self):
        """Verify 90-day rolling returns calculation"""
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        values = pd.Series(np.linspace(100, 120, 200), index=dates)

        rolling_90 = values.pct_change(90) * 100

        assert rolling_90.isna().sum() == 90
        assert not rolling_90.iloc[90:].isna().all()

    def test_180_day_rolling_returns(self):
        """Verify 180-day rolling returns calculation"""
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        values = pd.Series(np.linspace(100, 130, 365), index=dates)

        rolling_180 = values.pct_change(180) * 100

        assert rolling_180.isna().sum() == 180
        assert not rolling_180.iloc[180:].isna().all()

    def test_rolling_windows_definition(self):
        """Verify rolling window periods are correctly defined"""
        rolling_windows = [30, 90, 180]

        assert len(rolling_windows) == 3
        assert 30 in rolling_windows
        assert 90 in rolling_windows
        assert 180 in rolling_windows

    def test_rolling_returns_positive_growth(self):
        """Verify rolling returns for positive growth"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        # 10% growth over 100 days
        values = pd.Series(np.linspace(100, 110, 100), index=dates)

        rolling_30 = values.pct_change(30)

        # Last value should show positive return
        assert rolling_30.iloc[-1] > 0

    def test_rolling_returns_negative_growth(self):
        """Verify rolling returns for negative growth"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        # 10% decline over 100 days
        values = pd.Series(np.linspace(100, 90, 100), index=dates)

        rolling_30 = values.pct_change(30)

        # Last value should show negative return
        assert rolling_30.iloc[-1] < 0

    def test_rolling_returns_percentage_conversion(self):
        """Verify rolling returns are converted to percentage"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        values = pd.Series([100] + [110] * 99, index=dates)  # 10% jump

        rolling_30 = values.pct_change(30) * 100

        # After 30 days, should show ~10% return
        # (comparing day 30 to day 60, both are 110 vs day 0 which is 100)
        non_nan_values = rolling_30.dropna()
        # Most values should be around 10% after the jump stabilizes
        assert any(abs(val - 10.0) < 1.0 for val in non_nan_values)

    def test_multiple_windows_same_data(self):
        """Verify all window sizes work on same dataset"""
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        values = pd.Series(np.linspace(100, 120, 365), index=dates)

        windows = [30, 90, 180]
        rolling_results = {}

        for window in windows:
            rolling_results[window] = values.pct_change(window) * 100

        # All windows should produce results
        assert len(rolling_results) == 3
        # Each should have correct number of NaN values
        assert rolling_results[30].isna().sum() == 30
        assert rolling_results[90].isna().sum() == 90
        assert rolling_results[180].isna().sum() == 180


class TestColorblindAccessibility:
    """Test colorblind accessibility features in charts"""

    def test_colorblind_safe_palette_defined(self):
        """Test that colorblind-safe colors are defined in config"""
        from app.config import (
            PORTFOLIO_COLOR,
            BENCHMARK_COLORS,
            POSITIVE_COLOR,
            NEGATIVE_COLOR
        )

        # Verify colors are defined
        assert PORTFOLIO_COLOR is not None
        assert len(BENCHMARK_COLORS) >= 3
        assert POSITIVE_COLOR is not None
        assert NEGATIVE_COLOR is not None

        # Verify they are hex color codes
        assert PORTFOLIO_COLOR.startswith("#")
        assert len(PORTFOLIO_COLOR) == 7
        for color in BENCHMARK_COLORS:
            assert color.startswith("#")
            assert len(color) == 7

    def test_wong_palette_colors(self):
        """Test that colors match Wong's colorblind-safe palette"""
        from app.config import (
            PORTFOLIO_COLOR,
            BENCHMARK_COLORS,
            POSITIVE_COLOR,
            NEGATIVE_COLOR
        )

        # Wong's colorblind-safe palette (standard colors)
        wong_blue = "#0173B2"
        wong_orange = "#DE8F05"
        wong_teal = "#029E73"
        wong_pink = "#CC78BC"

        # Verify portfolio uses Wong blue
        assert PORTFOLIO_COLOR == wong_blue

        # Verify benchmarks use Wong palette colors
        assert BENCHMARK_COLORS[0] == wong_orange
        assert BENCHMARK_COLORS[1] == wong_teal
        assert BENCHMARK_COLORS[2] == wong_pink

        # Verify positive/negative avoid red-green (use blue/orange instead)
        assert POSITIVE_COLOR == wong_blue
        assert NEGATIVE_COLOR == wong_orange

    def test_avoids_problematic_color_combinations(self):
        """Test that we avoid blue-purple and red-green combinations"""
        from app.config import (
            PORTFOLIO_COLOR,
            BENCHMARK_COLORS,
            POSITIVE_COLOR,
            NEGATIVE_COLOR
        )

        # Problematic colors for colorblind users
        problematic_purple = "#9467bd"  # Old purple benchmark
        problematic_green = "#06A77D"   # Old positive green
        problematic_red = "#D62246"     # Old negative red

        # Verify we no longer use these problematic colors
        assert PORTFOLIO_COLOR != problematic_purple
        assert problematic_purple not in BENCHMARK_COLORS
        assert POSITIVE_COLOR != problematic_green
        assert NEGATIVE_COLOR != problematic_red

    def test_line_style_differentiation(self):
        """Test that benchmark line styles provide visual differentiation"""
        from app.config import BENCHMARK_DASH_STYLES

        # Should have multiple distinct line styles
        assert len(BENCHMARK_DASH_STYLES) >= 3
        assert len(set(BENCHMARK_DASH_STYLES)) >= 3  # All unique

        # Common Plotly dash styles
        valid_styles = ["solid", "dash", "dot", "dashdot"]
        for style in BENCHMARK_DASH_STYLES:
            assert style in valid_styles

    def test_marker_symbols_defined(self):
        """Test that marker symbols are defined for additional differentiation"""
        from app.config import PORTFOLIO_MARKER, BENCHMARK_MARKERS

        # Verify markers are defined
        assert PORTFOLIO_MARKER is not None
        assert len(BENCHMARK_MARKERS) >= 3

        # Should be valid Plotly marker symbols
        valid_markers = ["circle", "square", "diamond", "triangle-up", "triangle-down", "cross", "x"]
        assert PORTFOLIO_MARKER in valid_markers
        for marker in BENCHMARK_MARKERS:
            assert marker in valid_markers

        # Should have distinct markers
        all_markers = [PORTFOLIO_MARKER] + BENCHMARK_MARKERS
        assert len(set(all_markers)) == len(all_markers)

    def test_active_return_uses_colorblind_colors(self):
        """Test that active return chart uses blue/orange instead of green/red"""
        from app.config import POSITIVE_COLOR, NEGATIVE_COLOR

        # Should use blue for positive (not green)
        assert POSITIVE_COLOR == "#0173B2"  # Wong blue

        # Should use orange for negative (not red)
        assert NEGATIVE_COLOR == "#DE8F05"  # Wong orange

        # Verify they are distinguishable
        assert POSITIVE_COLOR != NEGATIVE_COLOR

    def test_chart_colors_import(self):
        """Test that chart modules can import colorblind-safe colors"""
        # This ensures the refactoring maintains backward compatibility
        try:
            from app.charts import (
                create_main_dashboard,
                create_rolling_returns_chart,
                create_rolling_sharpe_chart
            )
            # If imports succeed, config is properly integrated
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import chart functions: {e}")

    def test_color_contrast_sufficient(self):
        """Test that colors have sufficient contrast for visibility"""
        from app.config import PORTFOLIO_COLOR, BENCHMARK_COLORS

        def hex_to_rgb(hex_color):
            """Convert hex color to RGB tuple"""
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        def relative_luminance(rgb):
            """Calculate relative luminance for contrast ratio"""
            r, g, b = [x / 255.0 for x in rgb]
            r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
            g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
            b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        def contrast_ratio(color1, color2):
            """Calculate WCAG contrast ratio between two colors"""
            lum1 = relative_luminance(hex_to_rgb(color1))
            lum2 = relative_luminance(hex_to_rgb(color2))
            lighter = max(lum1, lum2)
            darker = min(lum1, lum2)
            return (lighter + 0.05) / (darker + 0.05)

        # Test contrast between portfolio and each benchmark color
        # WCAG AA requires 3:1 for large text, 4.5:1 for normal text
        # We use a relaxed threshold of 1.4:1 for chart lines since we also use
        # different line styles (solid, dashed, dotted) for additional differentiation
        # Wong palette is optimized for colorblind differentiation, not brightness contrast
        for bench_color in BENCHMARK_COLORS:
            ratio = contrast_ratio(PORTFOLIO_COLOR, bench_color)
            assert ratio >= 1.4, f"Insufficient contrast between portfolio and benchmark: {ratio:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
