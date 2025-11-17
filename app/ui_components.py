"""Reusable UI components for the ETF Backtester.

This module provides functions for rendering metrics, tables, and other
UI elements consistently throughout the application.

Widget state management has been simplified to use StateManager's
centralized widget state API, reducing complexity.
"""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st
import pandas as pd

from .config import METRIC_LABELS
from .ticker_data import get_all_tickers, format_ticker_option, search_tickers_with_yahoo, get_ticker_name
from .state_manager import StateManager


def format_metric_value(key: str, value: float) -> str:
    """Format metric value based on metric type.
    
    Args:
        key: Metric key (e.g., 'cagr', 'sharpe_ratio')
        value: Metric value to format
    
    Returns:
        Formatted string representation of the value
    
    Examples:
        >>> format_metric_value("ending_value", 150000.5)
        '$150,000.50'
        >>> format_metric_value("cagr", 0.0856)
        '8.56%'
        >>> format_metric_value("sharpe_ratio", 1.234)
        '1.234'
    """
    if key == "ending_value" or key == "total_contributions":
        return f"${value:,.2f}"
    elif key in ["total_return", "cagr", "irr", "volatility", "max_drawdown"]:
        return f"{value:.2%}"
    elif key in ["sharpe_ratio", "sortino_ratio"]:
        return f"{value:.3f}"
    else:
        return f"{value:.2f}"


def render_metric(key: str, value: float, label: str = None, delta: str = None, delta_color: str = "normal") -> None:
    """Render a single metric with appropriate formatting.
    
    Args:
        key: Metric key for determining format
        label: Optional custom label (uses METRIC_LABELS if not provided)
        value: Metric value
        delta: Optional delta value to display
        delta_color: Color for delta ("normal", "inverse", "off")
    
    Examples:
        >>> render_metric("cagr", 0.0856)
        # Displays "CAGR: 8.56%"
        
        >>> render_metric("sharpe_ratio", 1.234, delta="0.5", delta_color="normal")
        # Displays "Sharpe Ratio: 1.234" with green +0.5 delta
    """
    if label is None:
        label = METRIC_LABELS.get(key, key)
    
    formatted_value = format_metric_value(key, value)
    
    if delta is not None:
        st.metric(label, formatted_value, delta=delta, delta_color=delta_color)
    else:
        st.metric(label, formatted_value)


def render_metrics_column(summary: Dict[str, float], title: str) -> None:
    """Render a column of metrics.
    
    Args:
        summary: Dictionary of metric keys and values
        title: Column title (e.g., "Portfolio", "Benchmark")
    
    Examples:
        >>> summary = {
        ...     "ending_value": 150000,
        ...     "total_return": 0.50,
        ...     "cagr": 0.0856
        ... }
        >>> render_metrics_column(summary, "Portfolio")
        # Renders markdown header and all metrics
    """
    st.markdown(f"### {title}")
    
    for key, value in summary.items():
        label = METRIC_LABELS.get(key, key)
        formatted_value = format_metric_value(key, value)
        st.metric(label, formatted_value)


def render_relative_metrics(
    portfolio_summary: Dict[str, float],
    benchmark_summary: Dict[str, float]
) -> None:
    """Render relative performance metrics with delta indicators.
    
    Args:
        portfolio_summary: Portfolio metrics dictionary
        benchmark_summary: Benchmark metrics dictionary
    
    The function calculates and displays:
    - Excess Return (portfolio - benchmark)
    - Excess CAGR
    - Volatility Difference (inverse coloring - lower is better)
    - Sharpe Difference
    - Sortino Difference
    """
    st.markdown("### Relative Performance")
    
    # Excess Return
    excess_return = portfolio_summary["total_return"] - benchmark_summary["total_return"]
    st.metric(
        "Excess Return",
        f"{excess_return:.2%}",
        delta=f"{excess_return:.2%}",
        delta_color="normal"
    )
    
    # Excess CAGR
    excess_cagr = portfolio_summary["cagr"] - benchmark_summary["cagr"]
    st.metric(
        "Excess CAGR",
        f"{excess_cagr:.2%}",
        delta=f"{excess_cagr:.2%}",
        delta_color="normal"
    )

    # Excess IRR (only if both have IRR - for DCA strategies)
    if "irr" in portfolio_summary and "irr" in benchmark_summary:
        excess_irr = portfolio_summary["irr"] - benchmark_summary["irr"]
        st.metric(
            "Excess IRR",
            f"{excess_irr:.2%}",
            delta=f"{excess_irr:.2%}",
            delta_color="normal"
        )

    # Volatility Difference (inverse - lower is better)
    vol_diff = portfolio_summary["volatility"] - benchmark_summary["volatility"]
    st.metric(
        "Volatility Diff",
        f"{vol_diff:.2%}",
        delta=f"{vol_diff:.2%}",
        delta_color="inverse"  # Lower is better for volatility
    )
    
    # Sharpe Difference
    sharpe_diff = portfolio_summary["sharpe_ratio"] - benchmark_summary["sharpe_ratio"]
    st.metric(
        "Sharpe Difference",
        f"{sharpe_diff:.3f}",
        delta=f"{sharpe_diff:.3f}",
        delta_color="normal"
    )
    
    # Sortino Difference
    sortino_diff = portfolio_summary["sortino_ratio"] - benchmark_summary["sortino_ratio"]
    st.metric(
        "Sortino Difference",
        f"{sortino_diff:.3f}",
        delta=f"{sortino_diff:.3f}",
        delta_color="normal"
    )


def render_portfolio_composition(tickers: list, weights: list) -> None:
    """Render portfolio composition table with ticker names.

    Args:
        tickers: List of ticker symbols
        weights: List of portfolio weights (normalized)

    Examples:
        >>> render_portfolio_composition(
        ...     ["AAPL", "MSFT"],
        ...     [0.6, 0.4]
        ... )
        # Displays table with tickers, names, and weights
    """
    st.subheader("Portfolio Composition")

    # Get ticker names from ticker_data module
    ticker_names = [get_ticker_name(ticker) for ticker in tickers]

    composition_data = {
        "Ticker": tickers,
        "Name": ticker_names,
        "Weight": [f"{w:.1%}" for w in weights]
    }

    composition_df = pd.DataFrame(composition_data)
    st.table(composition_df)


def render_searchable_ticker_input(
    label: str,
    default_value: str = "",
    key: str = None,
    help_text: str = None
) -> str:
    """Render a searchable ticker input with Yahoo Finance integration.

    This component provides:
    1. Text input for direct ticker entry or search query
    2. Search button to find tickers using the input value
    3. Click-to-select from search results

    Workflow:
    - User types ticker symbol or company name (e.g., "AAPL" or "apple")
    - Click 🔍 Search button to find matching tickers
    - Click any result to populate the field

    Uses StateManager for simplified widget state (single key instead of 3).

    Args:
        label: Label for the input field
        default_value: Default ticker value
        key: Unique key for the widget
        help_text: Optional help text to display

    Returns:
        Selected or entered ticker symbol

    Examples:
        >>> ticker = render_searchable_ticker_input("Portfolio Ticker 1", "AAPL")
        # User can type "AAPL" directly or "apple" then search
    """
    if not key:
        # If no key provided, fall back to simple text input
        return st.text_input(
            label,
            value=default_value,
            help=help_text or "Enter ticker symbol",
            placeholder="e.g., AAPL"
        ).strip().upper()

    # Create unique keys for Streamlit widgets (needed before accessing widget state)
    input_key = f"{key}_input"
    search_key = f"{key}_search"

    # Get widget state from StateManager (single dict instead of 3 separate keys)
    widget_state = StateManager.get_widget_state(key, {
        'value': default_value,
        'show_search': False,
        'pending_value': None
    })

    # Handle pending ticker selection (from previous run)
    if widget_state.get('pending_value'):
        widget_state['value'] = widget_state['pending_value']
        widget_state['pending_value'] = None
        widget_state['show_search'] = False
        StateManager.set_widget_state(key, widget_state)
        # CRITICAL: Update Streamlit's widget state so text input displays the new value
        st.session_state[input_key] = widget_state['value']

    # Initialize widget state if needed (so widget has a value on first render)
    if input_key not in st.session_state:
        st.session_state[input_key] = widget_state['value']

    # Text input for ticker
    col1, col2 = st.columns([3, 1])

    with col1:
        ticker_input = st.text_input(
            label,
            key=input_key,  # Streamlit manages value via session state
            help=help_text or "Enter ticker symbol or company name, then click 🔍 to search",
            placeholder="e.g., AAPL, apple, vanguard"
        )

    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_clicked = st.button("🔍 Search", key=search_key, use_container_width=True)

    # If search button clicked, show search interface
    if search_clicked or widget_state.get('show_search', False):
        widget_state['show_search'] = True
        StateManager.set_widget_state(key, widget_state)

        # Use the ticker input value as the search query
        search_query = ticker_input.strip()

        if search_query and len(search_query) >= 2:
            with st.spinner(f"Searching for '{search_query}'..."):
                results = search_tickers_with_yahoo(search_query, limit=10)

            if results:
                st.caption(f"Found {len(results)} result(s) for '{search_query}':")

                # Display results as clickable buttons
                for idx, (ticker, name) in enumerate(results):
                    # Use index to ensure unique keys even if ticker appears multiple times
                    button_key = f"{key}_result_{idx}_{ticker}"
                    display_text = f"{ticker} - {name}"

                    if st.button(display_text, key=button_key, use_container_width=True):
                        # Store ticker in widget state for next run
                        widget_state['pending_value'] = ticker
                        widget_state['show_search'] = False
                        StateManager.set_widget_state(key, widget_state)
                        st.rerun()
            else:
                st.warning(f"No results found for '{search_query}'. Try a different search term.")
        elif not search_query:
            st.info("💡 Enter a ticker symbol or company name in the field above, then click Search")
        else:
            st.info("Enter at least 2 characters to search")

        # Close search button
        if st.button("✕ Close Search", key=f"{key}_close"):
            widget_state['show_search'] = False
            StateManager.set_widget_state(key, widget_state)
            st.rerun()

    return ticker_input.strip().upper() if ticker_input else ""
