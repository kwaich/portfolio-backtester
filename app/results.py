"""Results display functions for the Portfolio Backtester.

This module contains functions for rendering backtest results,
including metrics, charts, and download options.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List
import io

import streamlit as st
import pandas as pd
import numpy as np

from .ui_components import (
    render_metrics_column,
    render_relative_metrics,
    render_portfolio_composition,
    display_section_header,
    display_info_bar,
    display_hero_metrics_row,
    display_metrics_tables,
    display_downloads,
)
from .charts import create_main_dashboard, create_rolling_returns_chart, create_rolling_sharpe_chart

try:
    from backtest import summarize
except ImportError:
    # For testing
    summarize = None


def render_summary_statistics(
    results: pd.DataFrame,
    all_benchmark_results: Dict[str, pd.DataFrame],
    benchmarks: List[str],
    capital: float,
    dca_freq: str = None,
    dca_amount: float = None
) -> None:
    """Render summary statistics section.

    Args:
        results: Primary backtest results DataFrame
        all_benchmark_results: Dictionary of benchmark results
        benchmarks: List of benchmark ticker symbols
        capital: Initial capital amount
        dca_freq: DCA frequency code (optional)
        dca_amount: DCA contribution amount (optional)
    """
    st.subheader("Summary Statistics")

    # Compute summaries
    portfolio_total_contrib = results["portfolio_contributions"].iloc[-1]

    # Pass contributions series for IRR calculation (only for DCA strategies)
    portfolio_summary = summarize(
        results["portfolio_value"],
        capital,
        total_contributions=portfolio_total_contrib,
        contributions_series=results["portfolio_contributions"] if (dca_freq and dca_amount) else None
    )

    # Compute summaries for all benchmarks
    all_benchmark_summaries = {}
    for bench_name, bench_result in all_benchmark_results.items():
        benchmark_total_contrib = bench_result["benchmark_contributions"].iloc[-1]
        all_benchmark_summaries[bench_name] = summarize(
            bench_result['benchmark_value'],
            capital,
            total_contributions=benchmark_total_contrib,
            contributions_series=bench_result["benchmark_contributions"] if (dca_freq and dca_amount) else None
        )

    # Display in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        render_metrics_column(portfolio_summary, "Portfolio")

    with col2:
        render_metrics_column(all_benchmark_summaries[benchmarks[0]], f"Benchmark ({benchmarks[0]})")

    with col3:
        render_relative_metrics(portfolio_summary, all_benchmark_summaries[benchmarks[0]])

    # Additional benchmark comparisons
    if len(benchmarks) > 1:
        st.divider()
        st.subheader("Additional Benchmark Comparisons")

        for bench_name in benchmarks[1:]:
            with st.expander(f"📊 Portfolio vs {bench_name}", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    render_metrics_column(all_benchmark_summaries[bench_name], bench_name)

                with col2:
                    render_relative_metrics(portfolio_summary, all_benchmark_summaries[bench_name])


def render_portfolio_info(
    tickers: List[str],
    weights_array: np.ndarray,
    rebalance_strategy: str,
    rebalance_freq: str = None,
    dca_frequency: str = None,
    dca_freq: str = None,
    dca_amount: float = None
) -> None:
    """Render portfolio composition and strategy info.

    Args:
        tickers: List of ticker symbols
        weights_array: Normalized portfolio weights
        rebalance_strategy: Rebalancing strategy name
        rebalance_freq: Rebalancing frequency code (optional)
        dca_frequency: DCA frequency name (optional)
        dca_freq: DCA frequency code (optional)
        dca_amount: DCA contribution amount (optional)
    """
    # Portfolio composition
    render_portfolio_composition(tickers, weights_array)

    # Display investment strategy
    if dca_freq and dca_amount:
        st.info(f"📊 **Strategy**: Dollar-Cost Averaging ({dca_frequency}, ${dca_amount:,.2f}/contribution)")
    elif rebalance_freq:
        st.info(f"📊 **Strategy**: {rebalance_strategy}")
    else:
        st.info(f"📊 **Strategy**: {rebalance_strategy}")


def render_charts(
    results: pd.DataFrame,
    all_benchmark_results: Dict[str, pd.DataFrame],
    benchmarks: List[str]
) -> None:
    """Render interactive charts section.

    Args:
        results: Primary backtest results DataFrame
        all_benchmark_results: Dictionary of benchmark results
        benchmarks: List of benchmark ticker symbols
    """
    st.header("📈 Interactive Visualizations")
    st.caption("💡 Hover over the charts to see exact values")

    # Log scale toggle
    log_scale = st.checkbox(
        "Use logarithmic scale for portfolio value chart",
        value=False,
        help="Logarithmic scale is useful for viewing long-term exponential growth",
        key="log_scale_toggle"
    )

    # Create main dashboard
    fig = create_main_dashboard(results, all_benchmark_results, benchmarks, log_scale=log_scale)
    st.plotly_chart(fig, width='stretch')

    # Rolling returns analysis
    st.divider()
    st.subheader("📊 Rolling Returns Analysis")
    st.caption("💡 Rolling returns show performance consistency over different time periods")

    fig_rolling = create_rolling_returns_chart(results, all_benchmark_results, benchmarks)
    st.plotly_chart(fig_rolling, width='stretch')

    # Rolling Sharpe ratio analysis
    st.divider()
    st.subheader("📈 Rolling 12-Month Sharpe Ratio")
    st.caption("💡 Rolling Sharpe ratio shows how risk-adjusted performance evolves over time (12-month window)")

    fig_sharpe = create_rolling_sharpe_chart(results, all_benchmark_results, benchmarks)
    st.plotly_chart(fig_sharpe, width='stretch')

    return fig  # Return main figure for download


def render_download_options(results: pd.DataFrame, main_fig) -> None:
    """Render download options section.

    Args:
        results: Backtest results DataFrame
        main_fig: Main dashboard Plotly figure
    """
    st.subheader("💾 Download Results")

    col1, col2 = st.columns(2)

    with col1:
        # CSV download
        csv_buffer = io.StringIO()
        results.to_csv(csv_buffer)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width='stretch'
        )

    with col2:
        # Chart download as interactive HTML
        chart_html = main_fig.to_html(include_plotlyjs='cdn')

        st.download_button(
            label="📥 Download Interactive Charts (HTML)",
            data=chart_html,
            file_name=f"backtest_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            width='stretch'
        )


def render_raw_data(results: pd.DataFrame) -> None:
    """Render raw data section.

    Args:
        results: Backtest results DataFrame
    """
    with st.expander("📋 View Raw Data"):
        st.dataframe(results, width='stretch')


def render_results(stored_results: Dict) -> None:
    """Render all backtest results with the new fintech dashboard layout.

    Args:
        stored_results: Dictionary containing all backtest results from StateManager
    """
    # Extract stored results
    results = stored_results['results']
    all_benchmark_results = stored_results['all_benchmark_results']
    tickers = stored_results['tickers']
    benchmarks = stored_results['benchmarks']
    weights_array = stored_results['weights_array']
    capital = stored_results['capital']
    rebalance_strategy = stored_results['rebalance_strategy']
    rebalance_freq = stored_results['rebalance_freq']
    dca_frequency = stored_results.get('dca_frequency')
    dca_freq = stored_results.get('dca_freq')
    dca_amount = stored_results.get('dca_amount')

    # Compute portfolio summary for hero metrics
    portfolio_total_contrib = results["portfolio_contributions"].iloc[-1]
    if summarize is None:
        portfolio_summary = {}
    else:
        portfolio_summary = summarize(
            results["portfolio_value"],
            capital,
            total_contributions=portfolio_total_contrib,
            contributions_series=results["portfolio_contributions"] if (dca_freq and dca_amount) else None
        )

    # Get start/end dates from the results DataFrame index
    start_date = str(results.index[0].date()) if hasattr(results.index[0], 'date') else str(results.index[0])
    end_date = str(results.index[-1].date()) if hasattr(results.index[-1], 'date') else str(results.index[-1])

    # Compute primary benchmark summary for hero metrics
    benchmark_summary = {}
    primary_benchmark_name = benchmarks[0] if benchmarks else None
    if summarize is not None and primary_benchmark_name:
        bench_result = all_benchmark_results.get(primary_benchmark_name)
        if bench_result is not None:
            benchmark_total_contrib = bench_result["benchmark_contributions"].iloc[-1]
            benchmark_summary = summarize(
                bench_result["benchmark_value"],
                capital,
                total_contributions=benchmark_total_contrib,
                contributions_series=bench_result["benchmark_contributions"] if (dca_freq and dca_amount) else None
            )

    # ------------------------------------------------------------------
    # Hero Metrics Row
    # ------------------------------------------------------------------
    st.markdown("**Portfolio**")
    hero_metrics = {
        "Ending Value": f"${portfolio_summary.get('ending_value', 0):,.2f}",
        "Total Return": f"{portfolio_summary.get('total_return', 0):+.2%}",
        "CAGR": f"{portfolio_summary.get('cagr', 0):+.2%}",
        "Sharpe Ratio": f"{portfolio_summary.get('sharpe_ratio', 0):.2f}",
        "Max Drawdown": f"{portfolio_summary.get('max_drawdown', 0):.2%}",
    }
    display_hero_metrics_row(hero_metrics)

    if benchmark_summary:
        st.markdown("**Benchmark ({})**".format(primary_benchmark_name))
        benchmark_hero_metrics = {
            "Ending Value": f"${benchmark_summary.get('ending_value', 0):,.2f}",
            "Total Return": f"{benchmark_summary.get('total_return', 0):+.2%}",
            "CAGR": f"{benchmark_summary.get('cagr', 0):+.2%}",
            "Sharpe Ratio": f"{benchmark_summary.get('sharpe_ratio', 0):.2f}",
            "Max Drawdown": f"{benchmark_summary.get('max_drawdown', 0):.2%}",
        }
        display_hero_metrics_row(benchmark_hero_metrics)

    # ------------------------------------------------------------------
    # Portfolio Info Bar
    # ------------------------------------------------------------------
    display_info_bar(
        portfolio_tickers=tickers,
        weights=weights_array.tolist(),
        benchmarks=benchmarks,
        start_date=start_date,
        end_date=end_date,
    )

    # ------------------------------------------------------------------
    # Main Dashboard Chart
    # ------------------------------------------------------------------
    display_section_header("Performance Overview")
    fig = create_main_dashboard(results, all_benchmark_results, benchmarks, log_scale=False)
    st.plotly_chart(fig, width='stretch')

    # ------------------------------------------------------------------
    # Rolling Returns
    # ------------------------------------------------------------------
    display_section_header("Rolling Returns")
    fig_rolling = create_rolling_returns_chart(results, all_benchmark_results, benchmarks)
    st.plotly_chart(fig_rolling, width='stretch')

    # ------------------------------------------------------------------
    # Rolling Sharpe
    # ------------------------------------------------------------------
    display_section_header("Rolling Sharpe Ratio")
    fig_sharpe = create_rolling_sharpe_chart(results, all_benchmark_results, benchmarks)
    st.plotly_chart(fig_sharpe, width='stretch')

    # ------------------------------------------------------------------
    # Detailed Metrics Tables
    # ------------------------------------------------------------------
    display_section_header("Detailed Metrics")

    performance_metrics = {
        "Total Return": f"{portfolio_summary.get('total_return', 0):.2%}",
        "CAGR": f"{portfolio_summary.get('cagr', 0):.2%}",
        "Volatility (Annualized)": f"{portfolio_summary.get('volatility', 0):.2%}",
    }
    # Add IRR only for DCA strategies
    if 'irr' in portfolio_summary:
        performance_metrics["IRR"] = f"{portfolio_summary['irr']:.2%}"

    risk_metrics = {
        "Sharpe Ratio": f"{portfolio_summary.get('sharpe_ratio', 0):.2f}",
        "Sortino Ratio": f"{portfolio_summary.get('sortino_ratio', 0):.2f}",
        "Max Drawdown": f"{portfolio_summary.get('max_drawdown', 0):.2%}",
    }

    display_metrics_tables(performance_metrics, risk_metrics)

    # ------------------------------------------------------------------
    # Downloads
    # ------------------------------------------------------------------
    display_section_header("Downloads")

    csv_buffer = io.StringIO()
    results.to_csv(csv_buffer)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    # Generate chart HTML bytes
    chart_html = fig.to_html(include_plotlyjs='cdn')
    chart_bytes = chart_html.encode("utf-8")

    display_downloads(csv_data=csv_bytes, chart_data=chart_bytes)

    # ------------------------------------------------------------------
    # Raw Data
    # ------------------------------------------------------------------
    render_raw_data(results)


def render_welcome_screen() -> None:
    """Render welcome screen when no results are available."""
    st.info("👈 Configure your backtest in the sidebar and click '🚀 Run Backtest' to begin")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🎯 How to use:
        1. **Choose a preset** portfolio or select Custom
        2. **Enter tickers** for your portfolio (e.g., AAPL, MSFT)
        3. **Set weights** for each ticker (will auto-normalize)
        4. **Select benchmark(s)** to compare against
        5. **Select date range** for the backtest
        6. **Set initial capital** amount
        7. **Click 'Run Backtest'** to see results
        """)

    with col2:
        st.markdown("""
        #### 📊 Features:
        - **Comprehensive metrics**: CAGR, Sharpe, Sortino, Drawdown
        - **Interactive charts**: Hover to see exact values at any point
        - **Data caching**: Faster subsequent runs
        - **CSV export**: Download results for further analysis
        - **Chart export**: Save interactive visualizations as HTML
        - **Real-time data**: Fetches latest prices from Yahoo Finance
        """)
