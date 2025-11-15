"""Reusable UI components for the ETF Backtester.

This module provides functions for rendering metrics, tables, and other
UI elements consistently throughout the application.
"""

from __future__ import annotations

from typing import Dict

import streamlit as st
import pandas as pd

from .config import METRIC_LABELS


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
