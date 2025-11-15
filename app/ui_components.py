"""Reusable UI components for the ETF Backtester.

This module provides functions for rendering metrics, tables, and other
UI elements consistently throughout the application.
"""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st
import pandas as pd

from .config import METRIC_LABELS
from .ticker_data import get_all_tickers, format_ticker_option, search_tickers_with_yahoo


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
    if key == "ending_value":
        return f"${value:,.2f}"
    elif key in ["total_return", "cagr", "volatility", "max_drawdown"]:
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
    """Render portfolio composition table.

    Args:
        tickers: List of ticker symbols
        weights: List of portfolio weights (normalized)

    Examples:
        >>> render_portfolio_composition(
        ...     ["AAPL", "MSFT"],
        ...     [0.6, 0.4]
        ... )
        # Displays table with tickers and weights
    """
    st.subheader("Portfolio Composition")

    composition_data = {
        "Ticker": tickers,
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
    1. Text input for direct ticker entry
    2. Search button to find tickers on Yahoo Finance
    3. Click-to-select from search results

    Args:
        label: Label for the input field
        default_value: Default ticker value
        key: Unique key for the widget
        help_text: Optional help text to display

    Returns:
        Selected or entered ticker symbol

    Examples:
        >>> ticker = render_searchable_ticker_input("Portfolio Ticker 1", "AAPL")
        # User can type ticker or search Yahoo Finance
    """
    # Create unique keys for widgets
    input_key = f"{key}_input" if key else None
    search_key = f"{key}_search" if key else None
    query_key = f"{key}_query" if key else None
    pending_key = f"{key}_pending" if key else None

    # Check for pending ticker selection (from previous run)
    if key and pending_key and pending_key in st.session_state:
        pending_ticker = st.session_state[pending_key]
        del st.session_state[pending_key]
        # Set the value BEFORE creating the widget
        if input_key:
            st.session_state[input_key] = pending_ticker

    # Initialize the text input's session state if needed
    if key and input_key and input_key not in st.session_state:
        st.session_state[input_key] = default_value

    # Text input for ticker
    col1, col2 = st.columns([3, 1])

    with col1:
        ticker_input = st.text_input(
            label,
            key=input_key,
            help=help_text or "Enter ticker symbol or click Search to find tickers",
            placeholder="e.g., AAPL, MSFT, VWRA.L"
        )

    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_clicked = st.button("🔍 Search", key=search_key, use_container_width=True)

    # If search button clicked, show search interface
    if search_clicked or (key and st.session_state.get(f"{key}_show_search", False)):
        if key:
            st.session_state[f"{key}_show_search"] = True

        # Search query input
        search_query = st.text_input(
            "Search Yahoo Finance",
            key=query_key,
            placeholder="e.g., 'apple', 'vanguard s&p', 'tesla'",
            help="Search by ticker symbol or company name"
        )

        if search_query and len(search_query) >= 2:
            with st.spinner(f"Searching for '{search_query}'..."):
                results = search_tickers_with_yahoo(search_query, limit=10)

            if results:
                st.caption(f"Found {len(results)} result(s):")

                # Display results as clickable buttons
                for idx, (ticker, name) in enumerate(results):
                    # Use index to ensure unique keys even if ticker appears multiple times
                    button_key = f"{key}_result_{idx}_{ticker}" if key else None
                    display_text = f"{ticker} - {name}"

                    if st.button(display_text, key=button_key, use_container_width=True):
                        # Store ticker in pending state for next run
                        if key and pending_key:
                            st.session_state[pending_key] = ticker
                            st.session_state[f"{key}_show_search"] = False
                        st.rerun()
            else:
                st.warning(f"No results found for '{search_query}'. Try a different search term.")
        elif search_query:
            st.info("Enter at least 2 characters to search")

        # Close search button
        if st.button("✕ Close Search", key=f"{key}_close" if key else None):
            if key:
                st.session_state[f"{key}_show_search"] = False
            st.rerun()

    return ticker_input.strip().upper() if ticker_input else ""
