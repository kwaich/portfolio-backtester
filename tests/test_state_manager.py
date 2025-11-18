"""Tests for app/state_manager.py - Centralized session state management.

This test suite validates:
- State initialization
- Portfolio configuration management
- Date range management
- Backtest results storage and retrieval
- Widget state management
- Utility methods (reset, validate, etc.)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
import pandas as pd

# Mock streamlit before importing app modules
import sys

from functools import lru_cache

streamlit_mock = MagicMock()

def mock_cache_data(*args, **kwargs):
    """Mock cache_data that actually caches using lru_cache."""
    def decorator(func):
        # Use lru_cache to actually cache
        cached_func = lru_cache(maxsize=128)(func)
        # Add Streamlit-compatible clear method
        cached_func.clear = cached_func.cache_clear
        return cached_func
    return decorator

streamlit_mock.cache_data = mock_cache_data
streamlit_mock.query_params = {}
streamlit_mock.session_state = {}
sys.modules['streamlit'] = streamlit_mock

from app.state_manager import StateManager, StateKeys, ValidationError


class TestStateKeys:
    """Test that state key constants are defined correctly."""

    def test_all_keys_are_strings(self):
        """All state key constants should be strings."""
        assert isinstance(StateKeys.SELECTED_PORTFOLIO, str)
        assert isinstance(StateKeys.NUM_TICKERS, str)
        assert isinstance(StateKeys.PRESET_TICKERS, str)
        assert isinstance(StateKeys.PRESET_WEIGHTS, str)
        assert isinstance(StateKeys.PRESET_BENCHMARK, str)
        assert isinstance(StateKeys.START_DATE, str)
        assert isinstance(StateKeys.END_DATE, str)
        assert isinstance(StateKeys.BACKTEST_COMPLETED, str)
        assert isinstance(StateKeys.BACKTEST_RESULTS, str)
        assert isinstance(StateKeys.WIDGET_PREFIX, str)

    def test_keys_are_unique(self):
        """All state keys should be unique."""
        keys = [
            StateKeys.SELECTED_PORTFOLIO,
            StateKeys.NUM_TICKERS,
            StateKeys.PRESET_TICKERS,
            StateKeys.PRESET_WEIGHTS,
            StateKeys.PRESET_BENCHMARK,
            StateKeys.START_DATE,
            StateKeys.END_DATE,
            StateKeys.BACKTEST_COMPLETED,
            StateKeys.BACKTEST_RESULTS,
        ]
        assert len(keys) == len(set(keys)), "State keys must be unique"


class TestStateManagerInitialization:
    """Test state initialization functionality."""

    def setup_method(self):
        """Set up mock session state before each test."""
        import streamlit as st
        st.session_state = {}

    def test_initialize_creates_all_required_keys(self):
        """Initialization should create all required state keys."""
        import streamlit as st

        StateManager.initialize()

        assert StateKeys.SELECTED_PORTFOLIO in st.session_state
        assert StateKeys.NUM_TICKERS in st.session_state
        assert StateKeys.START_DATE in st.session_state
        assert StateKeys.END_DATE in st.session_state
        assert StateKeys.PRESET_TICKERS in st.session_state
        assert StateKeys.PRESET_WEIGHTS in st.session_state
        assert StateKeys.PRESET_BENCHMARK in st.session_state
        assert StateKeys.BACKTEST_COMPLETED in st.session_state

    def test_initialize_sets_correct_defaults(self):
        """Initialization should set correct default values."""
        import streamlit as st

        StateManager.initialize()

        assert st.session_state[StateKeys.SELECTED_PORTFOLIO] == "Custom (Manual Entry)"
        assert st.session_state[StateKeys.NUM_TICKERS] == 2
        assert isinstance(st.session_state[StateKeys.START_DATE], datetime)
        assert isinstance(st.session_state[StateKeys.END_DATE], datetime)
        assert st.session_state[StateKeys.PRESET_TICKERS] == []
        assert st.session_state[StateKeys.PRESET_WEIGHTS] == []
        assert st.session_state[StateKeys.PRESET_BENCHMARK] == "VWRA.L"
        assert st.session_state[StateKeys.BACKTEST_COMPLETED] is False

    def test_initialize_is_idempotent(self):
        """Multiple calls to initialize should not override existing values."""
        import streamlit as st

        StateManager.initialize()
        original_portfolio = st.session_state[StateKeys.SELECTED_PORTFOLIO]

        # Modify state
        st.session_state[StateKeys.SELECTED_PORTFOLIO] = "Tech Portfolio"

        # Re-initialize
        StateManager.initialize()

        # Value should not be overridden
        assert st.session_state[StateKeys.SELECTED_PORTFOLIO] == "Tech Portfolio"


class TestPortfolioConfiguration:
    """Test portfolio configuration management."""

    def setup_method(self):
        """Set up mock session state before each test."""
        import streamlit as st
        st.session_state = {}
        StateManager.initialize()

    def test_get_selected_portfolio_default(self):
        """Should return default portfolio when not set."""
        portfolio = StateManager.get_selected_portfolio()
        assert portfolio == "Custom (Manual Entry)"

    def test_set_and_get_selected_portfolio(self):
        """Should set and retrieve selected portfolio."""
        StateManager.set_selected_portfolio("Tech Portfolio")
        assert StateManager.get_selected_portfolio() == "Tech Portfolio"

    def test_get_num_tickers_default(self):
        """Should return default number of tickers when not set."""
        num = StateManager.get_num_tickers()
        assert num == 2

    def test_set_and_get_num_tickers(self):
        """Should set and retrieve number of tickers."""
        StateManager.set_num_tickers(5)
        assert StateManager.get_num_tickers() == 5

    def test_get_preset_tickers_default(self):
        """Should return empty list when not set."""
        tickers = StateManager.get_preset_tickers()
        assert tickers == []

    def test_set_and_get_preset_tickers(self):
        """Should set and retrieve preset tickers."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        StateManager.set_preset_tickers(tickers)
        assert StateManager.get_preset_tickers() == tickers

    def test_get_preset_weights_default(self):
        """Should return empty list when not set."""
        weights = StateManager.get_preset_weights()
        assert weights == []

    def test_set_and_get_preset_weights(self):
        """Should set and retrieve preset weights."""
        weights = [0.4, 0.3, 0.3]
        StateManager.set_preset_weights(weights)
        assert StateManager.get_preset_weights() == weights

    def test_get_preset_benchmark_default(self):
        """Should return default benchmark when not set."""
        benchmark = StateManager.get_preset_benchmark()
        assert benchmark == "VWRA.L"

    def test_set_and_get_preset_benchmark(self):
        """Should set and retrieve preset benchmark."""
        StateManager.set_preset_benchmark("SPY")
        assert StateManager.get_preset_benchmark() == "SPY"

    def test_update_portfolio_preset_atomic(self):
        """Should update all preset values atomically."""
        preset_config = {
            "tickers": ["AAPL", "MSFT"],
            "weights": [0.6, 0.4],
            "benchmark": "SPY"
        }

        StateManager.update_portfolio_preset("Tech Portfolio", preset_config)

        assert StateManager.get_num_tickers() == 2
        assert StateManager.get_preset_tickers() == ["AAPL", "MSFT"]
        assert StateManager.get_preset_weights() == [0.6, 0.4]
        assert StateManager.get_preset_benchmark() == "SPY"

    def test_update_portfolio_preset_ignores_custom(self):
        """Should not update state when preset is Custom."""
        import streamlit as st

        # Set initial values
        StateManager.set_num_tickers(3)
        StateManager.set_preset_tickers(["A", "B", "C"])

        # Try to update with Custom preset
        preset_config = {
            "tickers": ["X", "Y"],
            "weights": [0.5, 0.5],
            "benchmark": "Z"
        }
        StateManager.update_portfolio_preset("Custom (Manual Entry)", preset_config)

        # Values should not change
        assert StateManager.get_num_tickers() == 3
        assert StateManager.get_preset_tickers() == ["A", "B", "C"]


class TestDateRange:
    """Test date range management."""

    def setup_method(self):
        """Set up mock session state before each test."""
        import streamlit as st
        st.session_state = {}
        StateManager.initialize()

    def test_get_start_date_default(self):
        """Should return default start date."""
        start_date = StateManager.get_start_date()
        assert isinstance(start_date, datetime)
        assert start_date.year == 2018

    def test_get_end_date_default(self):
        """Should return today as default end date."""
        end_date = StateManager.get_end_date()
        assert isinstance(end_date, datetime)
        # Should be close to today
        assert (datetime.today() - end_date).days < 2

    def test_set_date_range(self):
        """Should set both start and end dates atomically."""
        start = datetime(2020, 1, 1)
        end = datetime(2023, 12, 31)

        StateManager.set_date_range(start, end)

        assert StateManager.get_start_date() == start
        assert StateManager.get_end_date() == end

    def test_set_date_preset(self):
        """Should set start date to preset and end date to today."""
        preset_date = datetime(2019, 6, 15)

        StateManager.set_date_preset(preset_date)

        assert StateManager.get_start_date() == preset_date
        # End date should be today
        end_date = StateManager.get_end_date()
        assert (datetime.today() - end_date).days < 2


class TestBacktestResults:
    """Test backtest results storage and retrieval."""

    def setup_method(self):
        """Set up mock session state before each test."""
        import streamlit as st
        st.session_state = {}
        StateManager.initialize()

    def test_is_backtest_completed_default(self):
        """Should return False when no backtest completed."""
        assert StateManager.is_backtest_completed() is False

    def test_get_backtest_results_none_when_not_completed(self):
        """Should return None when backtest not completed."""
        assert StateManager.get_backtest_results() is None

    def test_store_backtest_results(self):
        """Should store backtest results and set completion flag."""
        # Create mock data
        results = pd.DataFrame({'value': [100, 110, 120]})
        all_benchmark_results = {
            'SPY': pd.DataFrame({'value': [100, 105, 115]})
        }
        tickers = ['AAPL', 'MSFT']
        benchmarks = ['SPY']
        weights_array = np.array([0.6, 0.4])
        capital = 100000.0
        rebalance_strategy = "Buy-and-Hold"
        rebalance_freq = None

        StateManager.store_backtest_results(
            results=results,
            all_benchmark_results=all_benchmark_results,
            tickers=tickers,
            benchmarks=benchmarks,
            weights_array=weights_array,
            capital=capital,
            rebalance_strategy=rebalance_strategy,
            rebalance_freq=rebalance_freq
        )

        assert StateManager.is_backtest_completed() is True
        stored = StateManager.get_backtest_results()
        assert stored is not None
        assert 'results' in stored
        assert 'all_benchmark_results' in stored
        assert stored['tickers'] == tickers
        assert stored['benchmarks'] == benchmarks
        assert np.array_equal(stored['weights_array'], weights_array)
        assert stored['capital'] == capital

    def test_store_backtest_results_with_dca(self):
        """Should store backtest results with DCA parameters."""
        results = pd.DataFrame({'value': [100, 110, 120]})
        all_benchmark_results = {'SPY': pd.DataFrame({'value': [100, 105, 115]})}

        StateManager.store_backtest_results(
            results=results,
            all_benchmark_results=all_benchmark_results,
            tickers=['AAPL'],
            benchmarks=['SPY'],
            weights_array=np.array([1.0]),
            capital=100000.0,
            rebalance_strategy="DCA",
            rebalance_freq=None,
            dca_frequency="Monthly",
            dca_freq="M",
            dca_amount=1000.0
        )

        stored = StateManager.get_backtest_results()
        assert stored['dca_frequency'] == "Monthly"
        assert stored['dca_freq'] == "M"
        assert stored['dca_amount'] == 1000.0

    def test_clear_backtest_results(self):
        """Should clear results and reset completion flag."""
        # Store some results first
        results = pd.DataFrame({'value': [100, 110, 120]})
        StateManager.store_backtest_results(
            results=results,
            all_benchmark_results={},
            tickers=['AAPL'],
            benchmarks=['SPY'],
            weights_array=np.array([1.0]),
            capital=100000.0,
            rebalance_strategy="Buy-and-Hold",
            rebalance_freq=None
        )

        assert StateManager.is_backtest_completed() is True

        # Clear results
        StateManager.clear_backtest_results()

        assert StateManager.is_backtest_completed() is False
        assert StateManager.get_backtest_results() is None


class TestWidgetState:
    """Test widget state management."""

    def setup_method(self):
        """Set up mock session state before each test."""
        import streamlit as st
        st.session_state = {}
        StateManager.initialize()

    def test_get_widget_state_returns_default(self):
        """Should return default when widget state not set."""
        state = StateManager.get_widget_state("ticker_0", {"default": True})
        assert state == {"default": True}

    def test_set_and_get_widget_state(self):
        """Should set and retrieve widget state."""
        widget_state = {"value": "AAPL", "show_search": False}
        StateManager.set_widget_state("ticker_0", widget_state)

        retrieved = StateManager.get_widget_state("ticker_0")
        assert retrieved == widget_state

    def test_widget_state_uses_prefix(self):
        """Widget state should use WIDGET_PREFIX internally."""
        import streamlit as st

        StateManager.set_widget_state("ticker_0", {"test": True})

        # Check that the actual key has the prefix
        expected_key = f"{StateKeys.WIDGET_PREFIX}ticker_0"
        assert expected_key in st.session_state

    def test_clear_widget_state(self):
        """Should clear widget state."""
        import streamlit as st

        StateManager.set_widget_state("ticker_0", {"value": "AAPL"})
        assert StateManager.get_widget_state("ticker_0") is not None

        StateManager.clear_widget_state("ticker_0")

        # Should return default after clearing
        assert StateManager.get_widget_state("ticker_0", None) is None

    def test_clear_widget_state_handles_missing_key(self):
        """Should not error when clearing non-existent widget state."""
        # Should not raise exception
        StateManager.clear_widget_state("nonexistent_widget")


class TestUtilityMethods:
    """Test utility methods (reset, validate, get_all_state)."""

    def setup_method(self):
        """Set up mock session state before each test."""
        import streamlit as st
        st.session_state = {}
        StateManager.initialize()

    def test_validate_state_returns_true_when_valid(self):
        """Should return True when all required keys exist."""
        assert StateManager.validate_state() is True

    def test_validate_state_returns_false_when_missing_keys(self):
        """Should return False when required keys are missing."""
        import streamlit as st

        # Remove a required key
        del st.session_state[StateKeys.NUM_TICKERS]

        assert StateManager.validate_state() is False

    def test_get_all_state(self):
        """Should return dictionary of all managed state."""
        StateManager.set_selected_portfolio("Tech Portfolio")
        StateManager.set_num_tickers(3)

        all_state = StateManager.get_all_state()

        assert isinstance(all_state, dict)
        assert all_state['selected_portfolio'] == "Tech Portfolio"
        assert all_state['num_tickers'] == 3
        assert 'start_date' in all_state
        assert 'end_date' in all_state
        assert 'backtest_completed' in all_state

    def test_reset_all_clears_state(self):
        """Should clear all state and reinitialize with defaults."""
        import streamlit as st

        # Set some custom values
        StateManager.set_selected_portfolio("Custom Portfolio")
        StateManager.set_num_tickers(5)
        StateManager.set_widget_state("widget_1", {"test": True})

        # Reset all
        StateManager.reset_all()

        # Should be back to defaults
        assert StateManager.get_selected_portfolio() == "Custom (Manual Entry)"
        assert StateManager.get_num_tickers() == 2
        assert StateManager.get_widget_state("widget_1", None) is None

    def test_reset_all_clears_widget_state(self):
        """Should clear all widget state when resetting."""
        import streamlit as st

        # Create multiple widget states
        StateManager.set_widget_state("widget_1", {"a": 1})
        StateManager.set_widget_state("widget_2", {"b": 2})
        StateManager.set_widget_state("widget_3", {"c": 3})

        # Reset all
        StateManager.reset_all()

        # All widget states should be cleared
        assert StateManager.get_widget_state("widget_1", None) is None
        assert StateManager.get_widget_state("widget_2", None) is None
        assert StateManager.get_widget_state("widget_3", None) is None

        # Non-widget keys should still exist (reinitialized)
        assert StateKeys.NUM_TICKERS in st.session_state


class TestStateManagerEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up mock session state before each test."""
        import streamlit as st
        st.session_state = {}
        StateManager.initialize()

    def test_get_methods_work_without_initialization(self):
        """Get methods should return defaults even without initialization."""
        import streamlit as st
        st.session_state = {}  # Clear state, don't initialize

        # Should still return defaults
        assert StateManager.get_selected_portfolio() == "Custom (Manual Entry)"
        assert StateManager.get_num_tickers() == 2
        assert StateManager.get_preset_tickers() == []

    def test_multiple_widget_states_independent(self):
        """Multiple widget states should not interfere with each other."""
        StateManager.set_widget_state("widget_1", {"value": "A"})
        StateManager.set_widget_state("widget_2", {"value": "B"})
        StateManager.set_widget_state("widget_3", {"value": "C"})

        assert StateManager.get_widget_state("widget_1")["value"] == "A"
        assert StateManager.get_widget_state("widget_2")["value"] == "B"
        assert StateManager.get_widget_state("widget_3")["value"] == "C"

    def test_backtest_results_persists_complex_data(self):
        """Backtest results should handle complex nested data structures."""
        # Create complex data with multiple benchmarks
        results = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=3),
            'value': [100, 110, 120]
        })
        all_benchmark_results = {
            'SPY': pd.DataFrame({'value': [100, 105, 110]}),
            'QQQ': pd.DataFrame({'value': [100, 108, 112]}),
            'IWM': pd.DataFrame({'value': [100, 103, 107]})
        }

        StateManager.store_backtest_results(
            results=results,
            all_benchmark_results=all_benchmark_results,
            tickers=['AAPL', 'MSFT', 'GOOGL'],
            benchmarks=['SPY', 'QQQ', 'IWM'],
            weights_array=np.array([0.4, 0.3, 0.3]),
            capital=250000.0,
            rebalance_strategy="Monthly",
            rebalance_freq="M"
        )

        stored = StateManager.get_backtest_results()
        assert len(stored['all_benchmark_results']) == 3
        assert 'SPY' in stored['all_benchmark_results']
        assert 'QQQ' in stored['all_benchmark_results']
        assert len(stored['tickers']) == 3
        assert len(stored['benchmarks']) == 3


class TestStateManagerValidation:
    """Test input validation for StateManager setters."""

    def setup_method(self):
        """Set up mock session state before each test."""
        import streamlit as st
        st.session_state = {}
        StateManager.initialize()

    def test_set_num_tickers_valid(self):
        """set_num_tickers should accept valid positive integers."""
        StateManager.set_num_tickers(5)
        assert StateManager.get_num_tickers() == 5

    def test_set_num_tickers_rejects_non_integer(self):
        """set_num_tickers should reject non-integer values."""
        with pytest.raises(ValidationError, match="must be an integer"):
            StateManager.set_num_tickers("5")

        with pytest.raises(ValidationError, match="must be an integer"):
            StateManager.set_num_tickers(5.5)

    def test_set_num_tickers_rejects_zero_or_negative(self):
        """set_num_tickers should reject zero or negative values."""
        with pytest.raises(ValidationError, match="must be at least 1"):
            StateManager.set_num_tickers(0)

        with pytest.raises(ValidationError, match="must be at least 1"):
            StateManager.set_num_tickers(-1)

    def test_set_num_tickers_rejects_too_large(self):
        """set_num_tickers should reject values > 10."""
        with pytest.raises(ValidationError, match="must be at most 10"):
            StateManager.set_num_tickers(11)

    def test_set_preset_tickers_valid(self):
        """set_preset_tickers should accept valid list of strings."""
        StateManager.set_preset_tickers(["AAPL", "MSFT", "GOOGL"])
        assert StateManager.get_preset_tickers() == ["AAPL", "MSFT", "GOOGL"]

    def test_set_preset_tickers_accepts_empty_list(self):
        """set_preset_tickers should accept empty list."""
        StateManager.set_preset_tickers([])
        assert StateManager.get_preset_tickers() == []

    def test_set_preset_tickers_rejects_non_list(self):
        """set_preset_tickers should reject non-list values."""
        with pytest.raises(ValidationError, match="must be a list"):
            StateManager.set_preset_tickers("AAPL")

        with pytest.raises(ValidationError, match="must be a list"):
            StateManager.set_preset_tickers(("AAPL", "MSFT"))

    def test_set_preset_tickers_rejects_non_string_elements(self):
        """set_preset_tickers should reject lists with non-string elements."""
        with pytest.raises(ValidationError, match="must be a string"):
            StateManager.set_preset_tickers(["AAPL", 123, "GOOGL"])

    def test_set_preset_tickers_rejects_empty_strings(self):
        """set_preset_tickers should reject empty or whitespace-only strings."""
        with pytest.raises(ValidationError, match="cannot be an empty string"):
            StateManager.set_preset_tickers(["AAPL", "", "GOOGL"])

        with pytest.raises(ValidationError, match="cannot be an empty string"):
            StateManager.set_preset_tickers(["AAPL", "   ", "GOOGL"])

    def test_set_preset_weights_valid(self):
        """set_preset_weights should accept valid list of floats."""
        StateManager.set_preset_weights([0.5, 0.3, 0.2])
        assert StateManager.get_preset_weights() == [0.5, 0.3, 0.2]

    def test_set_preset_weights_accepts_ints(self):
        """set_preset_weights should accept integers."""
        StateManager.set_preset_weights([1, 2, 3])
        assert StateManager.get_preset_weights() == [1, 2, 3]

    def test_set_preset_weights_rejects_non_list(self):
        """set_preset_weights should reject non-list values."""
        with pytest.raises(ValidationError, match="must be a list"):
            StateManager.set_preset_weights(0.5)

    def test_set_preset_weights_rejects_non_numeric(self):
        """set_preset_weights should reject non-numeric elements."""
        with pytest.raises(ValidationError, match="must be a number"):
            StateManager.set_preset_weights([0.5, "0.3", 0.2])

    def test_set_preset_weights_rejects_negative(self):
        """set_preset_weights should reject negative weights."""
        with pytest.raises(ValidationError, match="must be non-negative"):
            StateManager.set_preset_weights([0.5, -0.3, 0.2])

    def test_set_preset_benchmark_valid(self):
        """set_preset_benchmark should accept valid ticker strings."""
        StateManager.set_preset_benchmark("SPY")
        assert StateManager.get_preset_benchmark() == "SPY"

    def test_set_preset_benchmark_rejects_non_string(self):
        """set_preset_benchmark should reject non-string values."""
        with pytest.raises(ValidationError, match="must be a string"):
            StateManager.set_preset_benchmark(123)

    def test_set_preset_benchmark_rejects_empty(self):
        """set_preset_benchmark should reject empty strings."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            StateManager.set_preset_benchmark("")

        with pytest.raises(ValidationError, match="cannot be empty"):
            StateManager.set_preset_benchmark("   ")

    def test_set_date_range_valid(self):
        """set_date_range should accept valid date pairs."""
        start = datetime(2020, 1, 1)
        end = datetime(2021, 1, 1)
        StateManager.set_date_range(start, end)
        assert StateManager.get_start_date() == start
        assert StateManager.get_end_date() == end

    def test_set_date_range_rejects_non_datetime(self):
        """set_date_range should reject non-datetime values."""
        with pytest.raises(ValidationError, match="must be a datetime object"):
            StateManager.set_date_range("2020-01-01", datetime(2021, 1, 1))

        with pytest.raises(ValidationError, match="must be a datetime object"):
            StateManager.set_date_range(datetime(2020, 1, 1), "2021-01-01")

    def test_set_date_range_rejects_start_after_end(self):
        """set_date_range should reject start_date >= end_date."""
        with pytest.raises(ValidationError, match="must be before"):
            StateManager.set_date_range(datetime(2021, 1, 1), datetime(2020, 1, 1))

        with pytest.raises(ValidationError, match="must be before"):
            StateManager.set_date_range(datetime(2020, 1, 1), datetime(2020, 1, 1))

    def test_set_selected_portfolio_valid(self):
        """set_selected_portfolio should accept valid portfolio names."""
        StateManager.set_selected_portfolio("Tech Giants")
        assert StateManager.get_selected_portfolio() == "Tech Giants"

    def test_set_selected_portfolio_rejects_non_string(self):
        """set_selected_portfolio should reject non-string values."""
        with pytest.raises(ValidationError, match="must be a string"):
            StateManager.set_selected_portfolio(123)

    def test_set_selected_portfolio_rejects_empty(self):
        """set_selected_portfolio should reject empty strings."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            StateManager.set_selected_portfolio("")

    def test_store_backtest_results_valid(self):
        """store_backtest_results should accept valid parameters."""
        results = pd.DataFrame({'value': [100, 110, 120]})
        all_benchmark_results = {'SPY': pd.DataFrame({'value': [100, 105, 110]})}

        StateManager.store_backtest_results(
            results=results,
            all_benchmark_results=all_benchmark_results,
            tickers=['AAPL', 'MSFT'],
            benchmarks=['SPY'],
            weights_array=np.array([0.6, 0.4]),
            capital=100000.0,
            rebalance_strategy="Buy-and-Hold",
            rebalance_freq=None
        )

        assert StateManager.is_backtest_completed()

    def test_store_backtest_results_rejects_non_dataframe(self):
        """store_backtest_results should reject non-DataFrame results."""
        with pytest.raises(ValidationError, match="must be a DataFrame"):
            StateManager.store_backtest_results(
                results=[1, 2, 3],
                all_benchmark_results={},
                tickers=['AAPL'],
                benchmarks=['SPY'],
                weights_array=np.array([1.0]),
                capital=100000.0,
                rebalance_strategy="Buy-and-Hold",
                rebalance_freq=None
            )

    def test_store_backtest_results_rejects_invalid_tickers(self):
        """store_backtest_results should reject invalid ticker lists."""
        results = pd.DataFrame({'value': [100, 110, 120]})

        with pytest.raises(ValidationError, match="must be a list"):
            StateManager.store_backtest_results(
                results=results,
                all_benchmark_results={},
                tickers="AAPL",
                benchmarks=['SPY'],
                weights_array=np.array([1.0]),
                capital=100000.0,
                rebalance_strategy="Buy-and-Hold",
                rebalance_freq=None
            )

    def test_store_backtest_results_rejects_negative_capital(self):
        """store_backtest_results should reject negative or zero capital."""
        results = pd.DataFrame({'value': [100, 110, 120]})

        with pytest.raises(ValidationError, match="must be a positive number"):
            StateManager.store_backtest_results(
                results=results,
                all_benchmark_results={},
                tickers=['AAPL'],
                benchmarks=['SPY'],
                weights_array=np.array([1.0]),
                capital=-100000.0,
                rebalance_strategy="Buy-and-Hold",
                rebalance_freq=None
            )

        with pytest.raises(ValidationError, match="must be a positive number"):
            StateManager.store_backtest_results(
                results=results,
                all_benchmark_results={},
                tickers=['AAPL'],
                benchmarks=['SPY'],
                weights_array=np.array([1.0]),
                capital=0,
                rebalance_strategy="Buy-and-Hold",
                rebalance_freq=None
            )

    def test_store_backtest_results_rejects_non_numpy_weights(self):
        """store_backtest_results should reject non-numpy array weights."""
        results = pd.DataFrame({'value': [100, 110, 120]})

        with pytest.raises(ValidationError, match="must be a numpy array"):
            StateManager.store_backtest_results(
                results=results,
                all_benchmark_results={},
                tickers=['AAPL'],
                benchmarks=['SPY'],
                weights_array=[0.6, 0.4],
                capital=100000.0,
                rebalance_strategy="Buy-and-Hold",
                rebalance_freq=None
            )
