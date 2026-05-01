"""Reusable UI components for the Portfolio Backtester.

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
from app.design_system import (
    COLORS,
    TYPOGRAPHY,
    SPACING,
)


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


def display_welcome_screen() -> None:
    """Display the centered welcome hero when no backtest has been run."""
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 4rem 2rem;">
            <div style="color: {COLORS['accent']}; font-size: 48px; margin-bottom: 1rem;">📈</div>
            <div style="font-family: {TYPOGRAPHY['font_header']}; font-size: {TYPOGRAPHY['page_title_size']}; font-weight: {TYPOGRAPHY['page_title_weight']}; color: {COLORS['primary_text']}; margin-bottom: 0.5rem;">Portfolio Backtester</div>
            <div style="font-family: {TYPOGRAPHY['font_body']}; font-size: 16px; color: {COLORS['muted']};">
                Analyze historical portfolio performance<br>
                Enter tickers in the sidebar and click "Run Backtest" to get started.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_section_header(title: str) -> None:
    """Display a section header using the design system."""
    st.markdown(
        f"""
        <h3 style="
            font-family: {TYPOGRAPHY['font_header']};
            font-size: {TYPOGRAPHY['section_header_size']};
            font-weight: {TYPOGRAPHY['section_header_weight']};
            color: {COLORS['primary_text']};
            margin-top: {SPACING['section_gap']};
            margin-bottom: 1rem;
        ">{title}</h3>
        """,
        unsafe_allow_html=True,
    )


def display_info_bar(portfolio_tickers: list[str], weights: list[float], benchmarks: list[str], start_date: str, end_date: str) -> None:
    """Display a compact portfolio info bar."""
    if not portfolio_tickers or not weights:
        return

    weight_strs = [f"{t} {w:.0%}" for t, w in zip(portfolio_tickers, weights)]
    portfolio_str = " · ".join(weight_strs)
    benchmark_str = ", ".join(benchmarks) if benchmarks else "—"
    date_str = f"{start_date} – {end_date}"

    st.markdown(
        f"""
        <div style="
            background-color: {COLORS['bg_card']};
            border-radius: 8px;
            padding: 0.75rem 1rem;
            border: 1px solid {COLORS['border']};
            font-family: {TYPOGRAPHY['font_body']};
            font-size: 14px;
            color: {COLORS['muted']};
            margin-bottom: 1rem;
        ">
            <strong style="color: {COLORS['primary_text']};">{portfolio_str}</strong>
            <span style="margin: 0 0.5rem;">vs</span>
            <strong style="color: {COLORS['primary_text']};">{benchmark_str}</strong>
            <span style="margin: 0 0.5rem;">·</span>
            {date_str}
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_hero_metrics_row(metrics: dict[str, str]) -> None:
    """Display the hero metrics row as a flexbox of metric cards.

    Args:
        metrics: Dict mapping label → formatted value string.
                 Expected keys: Ending Value, Total Return, CAGR, Sharpe Ratio, Max Drawdown
    """
    cards_html = '<div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem;">'
    for label, value in metrics.items():
        value_color = COLORS["primary_text"]
        if any(k in label.lower() for k in ("return", "cagr")):
            try:
                num = float(value.replace("%", "").replace("$", "").replace(",", ""))
                value_color = COLORS["success"] if num > 0 else COLORS["danger"] if num < 0 else COLORS["primary_text"]
            except ValueError:
                pass
        elif "drawdown" in label.lower():
            value_color = COLORS["danger"]

        cards_html += f"""
        <div style="background-color: {COLORS['bg_card']}; border-radius: {SPACING['card_radius']}; padding: {SPACING['card_padding']}; box-shadow: {SPACING['card_shadow']}; border: 1px solid {COLORS['border']}; text-align: center; flex: 1; min-width: 140px;">
            <div style="font-family: {TYPOGRAPHY['font_body']}; font-size: {TYPOGRAPHY['metric_value_size']}; font-weight: {TYPOGRAPHY['metric_value_weight']}; color: {value_color}; margin-bottom: 4px;">{value}</div>
            <div style="font-family: {TYPOGRAPHY['font_body']}; font-size: {TYPOGRAPHY['metric_label_size']}; font-weight: {TYPOGRAPHY['metric_label_weight']}; color: {COLORS['muted']};">{label}</div>
        </div>
        """
    cards_html += "</div>"

    st.markdown(cards_html, unsafe_allow_html=True)


def display_metrics_tables(performance: dict[str, str], risk: dict[str, str]) -> None:
    """Display two side-by-side metrics tables.

    Args:
        performance: Dict of performance metric label → value
        risk: Dict of risk metric label → value
    """
    col1, col2 = st.columns(2)

    def _build_table(title: str, data: dict[str, str]) -> str:
        rows = ""
        for label, value in data.items():
            rows += f"""
            <tr style="border-bottom: 1px solid {COLORS['border']};">
                <td style="padding: 0.6rem 0; font-family: {TYPOGRAPHY['font_body']}; font-size: 14px; color: {COLORS['primary_text']};">{label}</td>
                <td style="padding: 0.6rem 0; font-family: {TYPOGRAPHY['font_body']}; font-size: 14px; color: {COLORS['primary_text']}; text-align: right; font-weight: 500;">{value}</td>
            </tr>
            """
        return f"""
        <div style="background-color: {COLORS['bg_card']}; border-radius: {SPACING['card_radius']}; padding: {SPACING['card_padding']}; border: 1px solid {COLORS['border']}; margin-bottom: 1rem;">
            <div style="font-family: {TYPOGRAPHY['font_header']}; font-size: 14px; font-weight: 600; color: {COLORS['muted']}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;">{title}</div>
            <table style="width: 100%; border-collapse: collapse;">{rows}</table>
        </div>
        """

    with col1:
        st.markdown(_build_table("Performance", performance), unsafe_allow_html=True)
    with col2:
        st.markdown(_build_table("Risk", risk), unsafe_allow_html=True)


def display_downloads(csv_data: bytes | None = None, chart_data: bytes | None = None) -> None:
    """Display the downloads section with styled buttons."""
    st.markdown(
        f"""
        <div style="background-color: {COLORS['bg_card']}; border-radius: {SPACING['card_radius']}; padding: {SPACING['card_padding']}; box-shadow: {SPACING['card_shadow']}; border: 1px solid {COLORS['border']}; margin-top: 1.5rem;">
            <div style="font-family: {TYPOGRAPHY['font_header']}; font-size: 16px; font-weight: 500; color: {COLORS['primary_text']}; margin-bottom: 0.75rem;">Downloads</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if csv_data:
            st.download_button(
                label="⬇ Download Results CSV",
                data=csv_data,
                file_name="backtest_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with col2:
        if chart_data:
            st.download_button(
                label="⬇ Download Chart",
                data=chart_data,
                file_name="backtest_chart.png",
                mime="image/png",
                use_container_width=True,
            )
