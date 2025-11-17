"""Input validation and session state management for the Portfolio Backtester.

This module handles:
- Session state initialization (delegated to StateManager)
- Input validation for tickers, dates, and other parameters
- Weight normalization
- Portfolio preset updates (delegated to StateManager)

NOTE: Session state management has been centralized in state_manager.py.
This module now primarily handles validation logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np

from .state_manager import StateManager


def initialize_session_state() -> None:
    """Initialize all session state variables with defaults.

    This should be called once at app startup to ensure all
    session state variables exist.

    Delegates to StateManager for centralized state management.
    """
    StateManager.initialize()


def update_portfolio_preset(preset_name: str, preset_config: Dict[str, Any]) -> None:
    """Update session state when portfolio preset changes.

    Args:
        preset_name: Name of the selected preset
        preset_config: Configuration dictionary for the preset with keys:
            - tickers: List[str]
            - weights: List[float]
            - benchmark: str

    Delegates to StateManager for centralized state management.

    Examples:
        >>> config = {
        ...     "tickers": ["AAPL", "MSFT"],
        ...     "weights": [0.6, 0.4],
        ...     "benchmark": "SPY"
        ... }
        >>> update_portfolio_preset("Tech Portfolio", config)
    """
    StateManager.update_portfolio_preset(preset_name, preset_config)


def validate_backtest_inputs(
    tickers: List[str],
    benchmarks: List[str],
    start_date: datetime,
    end_date: datetime
) -> Tuple[bool, str]:
    """Validate all backtest inputs.
    
    Args:
        tickers: List of portfolio ticker symbols
        benchmarks: List of benchmark ticker symbols
        start_date: Backtest start date
        end_date: Backtest end date
    
    Returns:
        Tuple of (is_valid, error_message).
        If valid, error_message is empty string.
    
    Examples:
        >>> is_valid, error = validate_backtest_inputs(
        ...     ["AAPL", "MSFT"],
        ...     ["SPY"],
        ...     datetime(2020, 1, 1),
        ...     datetime(2023, 1, 1)
        ... )
        >>> is_valid
        True
    """
    # Check tickers are not empty
    if not all(tickers):
        return False, "Please enter all ticker symbols"
    
    # Check at least one benchmark
    if not benchmarks or not all(benchmarks):
        return False, "Please enter at least one valid benchmark ticker"
    
    # Check date range
    if start_date >= end_date:
        return False, "Start date must be before end date"
    
    # Check minimum date range (at least 7 days)
    if (end_date - start_date).days < 7:
        return False, "Date range must be at least 7 days"
    
    return True, ""


def normalize_weights(weights: List[float]) -> np.ndarray:
    """Normalize weights to sum to 1.0.
    
    Args:
        weights: List of portfolio weights
    
    Returns:
        Numpy array of normalized weights
    
    Examples:
        >>> normalize_weights([0.5, 0.5])
        array([0.5, 0.5])
        >>> normalize_weights([1, 1, 1])
        array([0.333..., 0.333..., 0.333...])
    """
    weights_array = np.array(weights)
    
    if not np.isclose(weights_array.sum(), 1.0):
        weights_array = weights_array / weights_array.sum()
    
    return weights_array


def check_and_normalize_weights(weights: List[float]) -> Tuple[np.ndarray, bool]:
    """Check if weights need normalization and normalize if needed.
    
    Args:
        weights: List of portfolio weights
    
    Returns:
        Tuple of (normalized_weights, was_normalized)
    
    Examples:
        >>> weights, normalized = check_and_normalize_weights([0.5, 0.5])
        >>> normalized
        False
        >>> weights, normalized = check_and_normalize_weights([1, 1, 1])
        >>> normalized
        True
    """
    weights_array = np.array(weights)
    was_normalized = not np.isclose(weights_array.sum(), 1.0)
    
    if was_normalized:
        weights_array = weights_array / weights_array.sum()
    
    return weights_array, was_normalized
