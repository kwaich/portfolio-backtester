"""Sidebar rendering functions for the ETF Backtester.

This module contains functions for rendering the sidebar configuration UI,
organized into logical sections for better maintainability.
"""

from __future__ import annotations

from datetime import datetime
from typing import Tuple, List, Dict, Any

import streamlit as st
import numpy as np

from .config import (
    SIDEBAR_HEADER, MAX_TICKERS, MIN_BENCHMARKS, MAX_BENCHMARKS,
    DEFAULT_CAPITAL, MIN_CAPITAL, MAX_CAPITAL,
    REBALANCE_OPTIONS, DCA_FREQUENCY_OPTIONS,
    MIN_DCA_AMOUNT, MAX_DCA_AMOUNT, DEFAULT_DCA_AMOUNT,
    DEFAULT_TICKER_1, DEFAULT_TICKER_2
)
from .presets import get_portfolio_presets, get_date_presets
from .state_manager import StateManager
from .ui_components import render_searchable_ticker_input


def render_sidebar_header() -> None:
    """Render sidebar header."""
    st.sidebar.header(SIDEBAR_HEADER)


def render_portfolio_preset_selector() -> str:
    """Render portfolio preset selector.

    Returns:
        Selected portfolio preset name
    """
    portfolio_presets = get_portfolio_presets()
    selected_portfolio = st.sidebar.selectbox(
        "Example Portfolio",
        options=list(portfolio_presets.keys()),
        index=0,
        help="Select a pre-configured portfolio or choose Custom to enter manually",
        key="portfolio_preset_selector"
    )
    return selected_portfolio


def render_portfolio_inputs(num_tickers: int) -> Tuple[List[str], List[float]]:
    """Render portfolio ticker and weight inputs.

    Args:
        num_tickers: Number of tickers to display

    Returns:
        Tuple of (tickers, weights)
    """
    st.sidebar.subheader("Portfolio Composition")

    tickers = []
    weights = []

    preset_tickers = StateManager.get_preset_tickers()
    preset_weights = StateManager.get_preset_weights()

    for i in range(num_tickers):
        # Determine default ticker
        if i < len(preset_tickers):
            default_ticker = preset_tickers[i]
        elif i == 0:
            default_ticker = DEFAULT_TICKER_1
        elif i == 1:
            default_ticker = DEFAULT_TICKER_2
        else:
            default_ticker = ""

        # Use searchable ticker input
        ticker = render_searchable_ticker_input(
            f"Ticker {i+1}",
            default_value=default_ticker,
            key=f"ticker_{i}",
            help_text="Enter ticker symbol or company name, then click 🔍 to search"
        )
        tickers.append(ticker)

        # Weight input
        if i < len(preset_weights):
            default_weight = preset_weights[i]
        else:
            default_weight = 1.0 / num_tickers

        weight = st.sidebar.number_input(
            f"Weight {i+1}",
            min_value=0.0,
            max_value=1.0,
            value=default_weight,
            step=0.05,
            key=f"weight_{i}",
            help="Portfolio weight (will be normalized)"
        )
        weights.append(weight)

    return tickers, weights


def render_benchmark_inputs(num_benchmarks: int) -> List[str]:
    """Render benchmark ticker inputs.

    Args:
        num_benchmarks: Number of benchmarks to display

    Returns:
        List of benchmark ticker symbols
    """
    st.sidebar.subheader("Benchmark")

    preset_benchmark = StateManager.get_preset_benchmark()

    # Check if benchmarks came from URL
    url_benchmarks = st.session_state.get('url_benchmarks', [])

    benchmarks = []

    for i in range(num_benchmarks):
        # Determine default benchmark for this position
        if i < len(url_benchmarks):
            # Use URL benchmark if available
            default_bench = url_benchmarks[i]
        elif i == 0:
            # First benchmark uses preset
            default_bench = preset_benchmark
        elif i == 1:
            # Second benchmark defaults to SPY
            default_bench = "SPY"
        else:
            # Additional benchmarks default to empty
            default_bench = ""

        # Use searchable ticker input for benchmarks
        bench_ticker = render_searchable_ticker_input(
            f"Benchmark {i+1}",
            default_value=default_bench,
            key=f"benchmark_{i}",
            help_text="Enter ticker symbol or company name, then click 🔍 to search"
        )
        if bench_ticker:
            benchmarks.append(bench_ticker)

    return benchmarks


def render_date_range_inputs() -> Tuple[datetime, datetime]:
    """Render date range inputs with presets.

    Returns:
        Tuple of (start_date, end_date)
    """
    st.sidebar.subheader("Date Range")
    st.sidebar.caption("Quick Presets:")

    date_presets = get_date_presets()
    preset_cols = st.sidebar.columns(6)

    # Date preset buttons
    for idx, (label, date_value) in enumerate(date_presets.items()):
        if preset_cols[idx].button(
            label,
            use_container_width=True,
            help=f"Set range to {label}",
            key=f"date_preset_{label}"
        ):
            StateManager.set_date_preset(date_value)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=StateManager.get_start_date(),
            help="Backtest start date",
            key="start_date_input"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=StateManager.get_end_date(),
            help="Backtest end date",
            key="end_date_input"
        )

    return start_date, end_date


def render_sidebar_form() -> Dict[str, Any]:
    """Render sidebar inputs as a form to reduce reruns.

    Returns:
        Dictionary containing all form inputs
    """
    render_sidebar_header()

    # Portfolio preset (outside form to enable dynamic updates)
    selected_portfolio = render_portfolio_preset_selector()

    # Handle portfolio preset selection
    if selected_portfolio != StateManager.get_selected_portfolio():
        StateManager.set_selected_portfolio(selected_portfolio)
        if selected_portfolio != "Custom (Manual Entry)":
            portfolio_presets = get_portfolio_presets()
            portfolio_config = portfolio_presets[selected_portfolio]
            StateManager.update_portfolio_preset(selected_portfolio, portfolio_config)

    # Form for main inputs (reduces reruns)
    with st.sidebar.form(key="backtest_config_form"):
        # Number of tickers
        num_tickers = st.number_input(
            "Number of Portfolio Tickers",
            min_value=1,
            max_value=MAX_TICKERS,
            value=StateManager.get_num_tickers(),
            step=1,
            help="How many different assets in your portfolio?",
            key="num_tickers_input"
        )

        # Number of benchmarks
        # Use URL benchmarks count if available, otherwise default to 1
        url_benchmarks = st.session_state.get('url_benchmarks', [])
        default_num_benchmarks = len(url_benchmarks) if url_benchmarks else 1
        num_benchmarks = st.number_input(
            "Number of Benchmarks",
            min_value=MIN_BENCHMARKS,
            max_value=MAX_BENCHMARKS,
            value=default_num_benchmarks,
            step=1,
            help="Compare against multiple benchmarks",
            key="num_benchmarks_input"
        )

        # Capital
        st.subheader("Initial Capital")
        # Use URL parameter if available, otherwise use default
        default_capital = st.session_state.get('url_capital', DEFAULT_CAPITAL)
        capital = st.number_input(
            "Capital ($)",
            min_value=float(MIN_CAPITAL),
            max_value=float(MAX_CAPITAL),
            value=float(default_capital),
            step=1000.0,
            format="%0.2f",
            help="Initial investment amount",
            key="capital_input"
        )

        # Rebalancing strategy
        st.subheader("Rebalancing Strategy")
        rebalance_strategy = st.selectbox(
            "Rebalancing Frequency",
            options=list(REBALANCE_OPTIONS.keys()),
            index=0,
            help="How often to rebalance the portfolio back to target weights",
            key="rebalance_strategy_input"
        )
        rebalance_freq = REBALANCE_OPTIONS[rebalance_strategy]

        # DCA (Dollar-Cost Averaging) strategy
        st.subheader("Dollar-Cost Averaging (DCA)")
        dca_frequency = st.selectbox(
            "DCA Contribution Frequency",
            options=list(DCA_FREQUENCY_OPTIONS.keys()),
            index=0,
            help="How often to make regular contributions (mutually exclusive with rebalancing)",
            key="dca_frequency_input"
        )
        dca_freq = DCA_FREQUENCY_OPTIONS[dca_frequency]

        # Only show DCA amount if DCA is enabled
        dca_amount = None
        if dca_freq is not None:
            dca_amount = st.number_input(
                "DCA Contribution Amount ($)",
                min_value=float(MIN_DCA_AMOUNT),
                max_value=float(MAX_DCA_AMOUNT),
                value=float(DEFAULT_DCA_AMOUNT),
                step=100.0,
                format="%0.2f",
                help="Amount to contribute at each DCA interval",
                key="dca_amount_input"
            )

            # Show warning if both DCA and rebalancing are enabled
            if rebalance_freq is not None:
                st.warning("⚠️ DCA and rebalancing are mutually exclusive. DCA will take precedence.")

        # Cache option
        st.subheader("Options")
        use_cache = st.checkbox(
            "Use cached data",
            value=True,
            help="Reuse previously downloaded data for faster results",
            key="use_cache_input"
        )

        # Submit button
        submit_button = st.form_submit_button(
            "🚀 Run Backtest",
            type="primary",
            use_container_width=True
        )

    # Render portfolio and benchmark inputs outside the form (they need interactivity)
    tickers, weights = render_portfolio_inputs(num_tickers)
    benchmarks = render_benchmark_inputs(num_benchmarks)

    # Render date range inputs outside the form (for date preset buttons)
    start_date, end_date = render_date_range_inputs()

    return {
        'submit_clicked': submit_button,
        'num_tickers': num_tickers,
        'tickers': tickers,
        'weights': weights,
        'num_benchmarks': num_benchmarks,
        'benchmarks': benchmarks,
        'start_date': start_date,
        'end_date': end_date,
        'capital': capital,
        'rebalance_strategy': rebalance_strategy,
        'rebalance_freq': rebalance_freq,
        'dca_frequency': dca_frequency,
        'dca_freq': dca_freq,
        'dca_amount': dca_amount,
        'use_cache': use_cache
    }
