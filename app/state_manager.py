"""Centralized session state management for the ETF Backtester.

This module provides a single source of truth for all Streamlit session state
operations, eliminating scattered state access across components.

Key Benefits:
- Type-safe state keys (constants instead of magic strings)
- Single source of truth for state operations
- Easier testing and debugging
- Clear API for state access and modification
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, Optional, List, MutableMapping

import importlib
import numpy as np
import pandas as pd

from .config import DEFAULT_NUM_TICKERS, DEFAULT_START_DATE, DEFAULT_BENCHMARK


def _session_state() -> MutableMapping[str, Any]:
    """Return the active Streamlit session_state mapping.

    Streamlit may be swapped with a mock after this module is imported
    (e.g., in tests). Import it dynamically so StateManager always uses
    the latest module instance.
    """
    return importlib.import_module("streamlit").session_state


class StateKeys:
    """Constants for all session state keys.

    Eliminates magic strings and provides autocomplete support.
    """
    # Portfolio configuration
    SELECTED_PORTFOLIO = "selected_portfolio"
    NUM_TICKERS = "num_tickers"
    PRESET_TICKERS = "preset_tickers"
    PRESET_WEIGHTS = "preset_weights"
    PRESET_BENCHMARK = "preset_benchmark"

    # Date range
    START_DATE = "start_date"
    END_DATE = "end_date"

    # Backtest results
    BACKTEST_COMPLETED = "backtest_completed"
    BACKTEST_RESULTS = "backtest_results"

    # Widget state prefix
    WIDGET_PREFIX = "_widget_"


class StateManager:
    """Centralized session state manager.

    Provides type-safe access to all session state with consistent
    patterns and validation.

    Examples:
        >>> StateManager.initialize()
        >>> portfolio = StateManager.get_selected_portfolio()
        >>> StateManager.set_date_range(start, end)
        >>> StateManager.store_backtest_results({...})
    """

    @staticmethod
    def initialize() -> None:
        """Initialize all session state variables with defaults.

        This should be called once at app startup to ensure all
        session state variables exist.

        Safe to call multiple times - only initializes missing keys.
        """
        defaults = {
            StateKeys.SELECTED_PORTFOLIO: "Custom (Manual Entry)",
            StateKeys.NUM_TICKERS: DEFAULT_NUM_TICKERS,
            StateKeys.START_DATE: DEFAULT_START_DATE,
            StateKeys.END_DATE: datetime.today(),
            StateKeys.PRESET_TICKERS: [],
            StateKeys.PRESET_WEIGHTS: [],
            StateKeys.PRESET_BENCHMARK: DEFAULT_BENCHMARK,
            StateKeys.BACKTEST_COMPLETED: False,
        }

        for key, value in defaults.items():
            if key not in _session_state():
                _session_state()[key] = value

    # =========================================================================
    # Portfolio Configuration
    # =========================================================================

    @staticmethod
    def get_selected_portfolio() -> str:
        """Get the currently selected portfolio preset name."""
        return _session_state().get(StateKeys.SELECTED_PORTFOLIO, "Custom (Manual Entry)")

    @staticmethod
    def set_selected_portfolio(name: str) -> None:
        """Set the selected portfolio preset name."""
        _session_state()[StateKeys.SELECTED_PORTFOLIO] = name

    @staticmethod
    def get_num_tickers() -> int:
        """Get the number of tickers in the portfolio."""
        return _session_state().get(StateKeys.NUM_TICKERS, DEFAULT_NUM_TICKERS)

    @staticmethod
    def set_num_tickers(num: int) -> None:
        """Set the number of tickers in the portfolio."""
        _session_state()[StateKeys.NUM_TICKERS] = num

    @staticmethod
    def get_preset_tickers() -> List[str]:
        """Get the preset ticker symbols."""
        return _session_state().get(StateKeys.PRESET_TICKERS, [])

    @staticmethod
    def set_preset_tickers(tickers: List[str]) -> None:
        """Set the preset ticker symbols."""
        _session_state()[StateKeys.PRESET_TICKERS] = tickers

    @staticmethod
    def get_preset_weights() -> List[float]:
        """Get the preset portfolio weights."""
        return _session_state().get(StateKeys.PRESET_WEIGHTS, [])

    @staticmethod
    def set_preset_weights(weights: List[float]) -> None:
        """Set the preset portfolio weights."""
        _session_state()[StateKeys.PRESET_WEIGHTS] = weights

    @staticmethod
    def get_preset_benchmark() -> str:
        """Get the preset benchmark ticker."""
        return _session_state().get(StateKeys.PRESET_BENCHMARK, DEFAULT_BENCHMARK)

    @staticmethod
    def set_preset_benchmark(benchmark: str) -> None:
        """Set the preset benchmark ticker."""
        _session_state()[StateKeys.PRESET_BENCHMARK] = benchmark

    @staticmethod
    def update_portfolio_preset(preset_name: str, preset_config: Dict[str, Any]) -> None:
        """Update all portfolio preset values atomically.

        Args:
            preset_name: Name of the selected preset
            preset_config: Configuration dictionary with keys:
                - tickers: List[str]
                - weights: List[float]
                - benchmark: str

        Examples:
            >>> config = {
            ...     "tickers": ["AAPL", "MSFT"],
            ...     "weights": [0.6, 0.4],
            ...     "benchmark": "SPY"
            ... }
            >>> StateManager.update_portfolio_preset("Tech Portfolio", config)
        """
        if preset_name == "Custom (Manual Entry)":
            # Don't override for custom entry
            return

        # Atomic update of all preset values
        StateManager.set_num_tickers(len(preset_config["tickers"]))
        StateManager.set_preset_tickers(preset_config["tickers"])
        StateManager.set_preset_weights(preset_config["weights"])
        StateManager.set_preset_benchmark(preset_config["benchmark"])

    # =========================================================================
    # Date Range
    # =========================================================================

    @staticmethod
    def get_start_date() -> datetime:
        """Get the backtest start date."""
        return _session_state().get(StateKeys.START_DATE, DEFAULT_START_DATE)

    @staticmethod
    def get_end_date() -> datetime:
        """Get the backtest end date."""
        return _session_state().get(StateKeys.END_DATE, datetime.today())

    @staticmethod
    def set_date_range(start_date: datetime, end_date: datetime) -> None:
        """Set the backtest date range atomically.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date

        Examples:
            >>> StateManager.set_date_range(
            ...     datetime(2020, 1, 1),
            ...     datetime.today()
            ... )
        """
        _session_state()[StateKeys.START_DATE] = start_date
        _session_state()[StateKeys.END_DATE] = end_date

    @staticmethod
    def set_date_preset(preset_date: datetime) -> None:
        """Set date range from a preset (start_date = preset, end_date = today).

        Args:
            preset_date: The preset start date

        Examples:
            >>> StateManager.set_date_preset(datetime(2020, 1, 1))
            # Sets start to 2020-01-01, end to today
        """
        StateManager.set_date_range(preset_date, datetime.today())

    # =========================================================================
    # Backtest Results
    # =========================================================================

    @staticmethod
    def is_backtest_completed() -> bool:
        """Check if a backtest has been completed."""
        return _session_state().get(StateKeys.BACKTEST_COMPLETED, False)

    @staticmethod
    def get_backtest_results() -> Optional[Dict[str, Any]]:
        """Get the stored backtest results.

        Returns:
            Dictionary containing backtest results, or None if not completed.

            Expected keys in result dictionary:
            - results: pd.DataFrame
            - all_benchmark_results: Dict[str, pd.DataFrame]
            - tickers: List[str]
            - benchmarks: List[str]
            - weights_array: np.ndarray
            - capital: float
            - rebalance_strategy: str
            - rebalance_freq: Optional[str]
            - dca_frequency: Optional[str]
            - dca_freq: Optional[str]
            - dca_amount: Optional[float]
        """
        if not StateManager.is_backtest_completed():
            return None
        return _session_state().get(StateKeys.BACKTEST_RESULTS)

    @staticmethod
    def store_backtest_results(
        results: pd.DataFrame,
        all_benchmark_results: Dict[str, pd.DataFrame],
        tickers: List[str],
        benchmarks: List[str],
        weights_array: np.ndarray,
        capital: float,
        rebalance_strategy: str,
        rebalance_freq: Optional[str],
        dca_frequency: Optional[str] = None,
        dca_freq: Optional[str] = None,
        dca_amount: Optional[float] = None
    ) -> None:
        """Store backtest results atomically.

        Args:
            results: Primary backtest results DataFrame
            all_benchmark_results: Dictionary of benchmark results
            tickers: List of portfolio ticker symbols
            benchmarks: List of benchmark ticker symbols
            weights_array: Normalized portfolio weights
            capital: Initial capital amount
            rebalance_strategy: Rebalancing strategy name
            rebalance_freq: Rebalancing frequency code
            dca_frequency: DCA frequency name (optional)
            dca_freq: DCA frequency code (optional)
            dca_amount: DCA contribution amount (optional)

        Examples:
            >>> StateManager.store_backtest_results(
            ...     results=df,
            ...     all_benchmark_results={'SPY': spy_df},
            ...     tickers=['AAPL', 'MSFT'],
            ...     benchmarks=['SPY'],
            ...     weights_array=np.array([0.6, 0.4]),
            ...     capital=100000.0,
            ...     rebalance_strategy="Buy-and-Hold",
            ...     rebalance_freq=None
            ... )
        """
        _session_state()[StateKeys.BACKTEST_RESULTS] = {
            'results': results,
            'all_benchmark_results': all_benchmark_results,
            'tickers': tickers,
            'benchmarks': benchmarks,
            'weights_array': weights_array,
            'capital': capital,
            'rebalance_strategy': rebalance_strategy,
            'rebalance_freq': rebalance_freq,
            'dca_frequency': dca_frequency,
            'dca_freq': dca_freq,
            'dca_amount': dca_amount
        }
        _session_state()[StateKeys.BACKTEST_COMPLETED] = True

    @staticmethod
    def clear_backtest_results() -> None:
        """Clear backtest results and reset completion flag.

        Useful for clearing old results before running a new backtest.
        """
        _session_state()[StateKeys.BACKTEST_COMPLETED] = False
        if StateKeys.BACKTEST_RESULTS in _session_state():
            del _session_state()[StateKeys.BACKTEST_RESULTS]

    # =========================================================================
    # Widget State Management
    # =========================================================================

    @staticmethod
    def get_widget_state(widget_key: str, default: Any = None) -> Any:
        """Get widget-specific state.

        Args:
            widget_key: The widget's unique key
            default: Default value if key doesn't exist

        Returns:
            The widget's state value, or default if not found

        Examples:
            >>> state = StateManager.get_widget_state("ticker_0", {})
        """
        full_key = f"{StateKeys.WIDGET_PREFIX}{widget_key}"
        return _session_state().get(full_key, default)

    @staticmethod
    def set_widget_state(widget_key: str, value: Any) -> None:
        """Set widget-specific state.

        Args:
            widget_key: The widget's unique key
            value: Value to store

        Examples:
            >>> StateManager.set_widget_state("ticker_0", {"value": "AAPL"})
        """
        full_key = f"{StateKeys.WIDGET_PREFIX}{widget_key}"
        _session_state()[full_key] = value

    @staticmethod
    def clear_widget_state(widget_key: str) -> None:
        """Clear widget-specific state.

        Args:
            widget_key: The widget's unique key

        Examples:
            >>> StateManager.clear_widget_state("ticker_0")
        """
        full_key = f"{StateKeys.WIDGET_PREFIX}{widget_key}"
        if full_key in _session_state():
            del _session_state()[full_key]

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def reset_all() -> None:
        """Reset all managed state to defaults.

        WARNING: This clears all session state. Use with caution.

        Useful for testing or implementing a "reset to defaults" feature.
        """
        # Clear all managed keys
        keys_to_clear = [
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

        for key in keys_to_clear:
            if key in _session_state():
                del _session_state()[key]

        # Clear all widget state
        widget_keys = [k for k in _session_state().keys()
                      if k.startswith(StateKeys.WIDGET_PREFIX)]
        for key in widget_keys:
            del _session_state()[key]

        # Re-initialize with defaults
        StateManager.initialize()

    @staticmethod
    def get_all_state() -> Dict[str, Any]:
        """Get all managed state as a dictionary.

        Useful for debugging or logging current state.

        Returns:
            Dictionary of all managed state keys and values

        Examples:
            >>> state = StateManager.get_all_state()
            >>> print(state['selected_portfolio'])
        """
        return {
            'selected_portfolio': StateManager.get_selected_portfolio(),
            'num_tickers': StateManager.get_num_tickers(),
            'preset_tickers': StateManager.get_preset_tickers(),
            'preset_weights': StateManager.get_preset_weights(),
            'preset_benchmark': StateManager.get_preset_benchmark(),
            'start_date': StateManager.get_start_date(),
            'end_date': StateManager.get_end_date(),
            'backtest_completed': StateManager.is_backtest_completed(),
            'has_results': StateManager.get_backtest_results() is not None,
        }

    @staticmethod
    def validate_state() -> bool:
        """Validate that all required state keys exist.

        Returns:
            True if all required keys exist, False otherwise

        Raises:
            ValueError: If validation fails and detailed error needed

        Examples:
            >>> if not StateManager.validate_state():
            ...     StateManager.initialize()
        """
        required_keys = [
            StateKeys.SELECTED_PORTFOLIO,
            StateKeys.NUM_TICKERS,
            StateKeys.START_DATE,
            StateKeys.END_DATE,
            StateKeys.PRESET_TICKERS,
            StateKeys.PRESET_WEIGHTS,
            StateKeys.PRESET_BENCHMARK,
        ]

        missing_keys = [key for key in required_keys if key not in _session_state()]

        if missing_keys:
            return False

        return True
