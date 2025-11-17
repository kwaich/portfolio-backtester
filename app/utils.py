"""Utility functions for the Portfolio Backtester Streamlit app.

This module provides helper functions for:
- URL parameter parsing and state synchronization
- Error handling and user feedback
- Progress tracking
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
from datetime import datetime
import streamlit as st


def get_query_params() -> Dict[str, Any]:
    """Get and parse URL query parameters.

    Returns:
        Dictionary of parsed query parameters

    Examples:
        >>> # URL: ?tickers=AAPL,MSFT&weights=0.6,0.4&benchmark=SPY
        >>> params = get_query_params()
        >>> params['tickers']
        ['AAPL', 'MSFT']
    """
    try:
        query_params = st.query_params

        parsed = {}

        # Parse tickers (comma-separated)
        if 'tickers' in query_params:
            tickers_str = query_params['tickers']
            parsed['tickers'] = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]

        # Parse weights (comma-separated floats)
        if 'weights' in query_params:
            weights_str = query_params['weights']
            try:
                parsed['weights'] = [float(w.strip()) for w in weights_str.split(',') if w.strip()]
            except ValueError:
                pass  # Invalid weights, ignore

        # Parse benchmark
        if 'benchmark' in query_params:
            parsed['benchmark'] = query_params['benchmark'].strip().upper()

        # Parse benchmarks (multiple)
        if 'benchmarks' in query_params:
            benchmarks_str = query_params['benchmarks']
            parsed['benchmarks'] = [b.strip().upper() for b in benchmarks_str.split(',') if b.strip()]

        # Parse dates
        if 'start_date' in query_params:
            try:
                parsed['start_date'] = datetime.strptime(query_params['start_date'], '%Y-%m-%d')
            except ValueError:
                pass  # Invalid date, ignore

        if 'end_date' in query_params:
            try:
                parsed['end_date'] = datetime.strptime(query_params['end_date'], '%Y-%m-%d')
            except ValueError:
                pass  # Invalid date, ignore

        # Parse capital
        if 'capital' in query_params:
            try:
                parsed['capital'] = float(query_params['capital'])
            except ValueError:
                pass  # Invalid capital, ignore

        # Parse preset
        if 'preset' in query_params:
            parsed['preset'] = query_params['preset']

        return parsed

    except Exception as e:
        # If query params fail, return empty dict
        return {}


def set_query_params(
    tickers: List[str] = None,
    weights: List[float] = None,
    benchmarks: List[str] = None,
    start_date: datetime = None,
    end_date: datetime = None,
    capital: float = None,
    preset: str = None
) -> None:
    """Set URL query parameters for sharing configurations.

    Args:
        tickers: List of portfolio ticker symbols
        weights: List of portfolio weights
        benchmarks: List of benchmark ticker symbols
        start_date: Backtest start date
        end_date: Backtest end date
        capital: Initial capital
        preset: Portfolio preset name

    Examples:
        >>> set_query_params(
        ...     tickers=['AAPL', 'MSFT'],
        ...     weights=[0.6, 0.4],
        ...     benchmarks=['SPY']
        ... )
        # Updates URL with query parameters
    """
    try:
        params = {}

        if tickers:
            params['tickers'] = ','.join(tickers)

        if weights:
            params['weights'] = ','.join(str(w) for w in weights)

        if benchmarks:
            params['benchmarks'] = ','.join(benchmarks)

        if start_date:
            params['start_date'] = start_date.strftime('%Y-%m-%d')

        if end_date:
            params['end_date'] = end_date.strftime('%Y-%m-%d')

        if capital is not None:
            params['capital'] = str(capital)

        if preset:
            params['preset'] = preset

        # Update query params
        st.query_params.update(params)

    except Exception:
        # Silently fail if query params can't be set
        pass


def show_error(message: str, error: Optional[Exception] = None, help_text: str = None) -> None:
    """Display a user-friendly error message with optional help text.

    Args:
        message: Main error message
        error: Optional exception object
        help_text: Optional help text with suggestions

    Examples:
        >>> show_error(
        ...     "Failed to download data",
        ...     error=e,
        ...     help_text="Check your internet connection and try again"
        ... )
    """
    error_msg = f"❌ **Error:** {message}"

    if error:
        error_msg += f"\n\n**Details:** {str(error)}"

    if help_text:
        error_msg += f"\n\n💡 **Suggestion:** {help_text}"

    st.error(error_msg)


def show_warning(message: str, help_text: str = None) -> None:
    """Display a user-friendly warning message with optional help text.

    Args:
        message: Main warning message
        help_text: Optional help text with suggestions

    Examples:
        >>> show_warning(
        ...     "Weights don't sum to 1.0",
        ...     help_text="Weights will be automatically normalized"
        ... )
    """
    warning_msg = f"⚠️ {message}"

    if help_text:
        warning_msg += f"\n\n💡 {help_text}"

    st.warning(warning_msg)


def show_success(message: str) -> None:
    """Display a success message.

    Args:
        message: Success message

    Examples:
        >>> show_success("Backtest completed successfully!")
    """
    st.success(f"✅ {message}")


def show_info(message: str) -> None:
    """Display an info message.

    Args:
        message: Info message

    Examples:
        >>> show_info("Click 'Run Backtest' to begin")
    """
    st.info(f"ℹ️ {message}")


class ProgressTracker:
    """Context manager for tracking multi-step progress.

    Examples:
        >>> with ProgressTracker(["Download data", "Compute metrics", "Generate charts"]) as tracker:
        ...     tracker.step("Download data")
        ...     # ... do work ...
        ...     tracker.step("Compute metrics")
        ...     # ... do work ...
        ...     tracker.step("Generate charts")
    """

    def __init__(self, steps: List[str]):
        """Initialize progress tracker.

        Args:
            steps: List of step names
        """
        self.steps = steps
        self.current_step = 0
        self.progress_bar = None
        self.status_text = None

    def __enter__(self):
        """Enter context and create progress widgets."""
        self.status_text = st.empty()
        self.progress_bar = st.progress(0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and clear progress widgets."""
        if exc_type is None:
            # Success - show 100% completion
            self.progress_bar.progress(1.0)
            self.status_text.text("✅ Complete!")
        else:
            # Error - clear widgets
            self.progress_bar.empty()
            self.status_text.empty()

    def step(self, step_name: str = None):
        """Move to next step.

        Args:
            step_name: Optional step name (uses next in list if not provided)
        """
        if step_name is None and self.current_step < len(self.steps):
            step_name = self.steps[self.current_step]

        if step_name:
            self.status_text.text(f"🔄 {step_name}...")

        self.current_step += 1
        progress = min(self.current_step / len(self.steps), 1.0)
        self.progress_bar.progress(progress)
