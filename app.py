"""
Streamlit web UI for ETF backtesting.

This app provides an interactive interface for running backtests and visualizing results.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import functions from backtest module
from backtest import download_prices, compute_metrics, summarize

# Page configuration
st.set_page_config(
    page_title="ETF Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📈 ETF Backtester</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Compare portfolio performance against benchmarks</div>', unsafe_allow_html=True)

# Sidebar - Input Controls
st.sidebar.header("Backtest Configuration")

# Number of tickers
num_tickers = st.sidebar.number_input(
    "Number of Portfolio Tickers",
    min_value=1,
    max_value=10,
    value=2,
    step=1,
    help="How many different assets in your portfolio?"
)

# Dynamic ticker inputs
tickers = []
weights = []

st.sidebar.subheader("Portfolio Composition")
for i in range(num_tickers):
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        ticker = col1.text_input(
            f"Ticker {i+1}",
            value="VDCP.L" if i == 0 else "VHYD.L" if i == 1 else "",
            key=f"ticker_{i}",
            help="Enter ticker symbol (e.g., VDCP.L, AAPL)"
        )
        tickers.append(ticker)
    with col2:
        weight = col2.number_input(
            f"Weight {i+1}",
            min_value=0.0,
            max_value=1.0,
            value=1.0 / num_tickers,
            step=0.05,
            key=f"weight_{i}",
            help="Portfolio weight (will be normalized)"
        )
        weights.append(weight)

# Benchmark
st.sidebar.subheader("Benchmark")
benchmark = st.sidebar.text_input(
    "Benchmark Ticker",
    value="VWRA.L",
    help="Ticker to compare against (e.g., VWRA.L, SPY)"
)

# Date range
st.sidebar.subheader("Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime(2018, 1, 1),
        help="Beginning of backtest period"
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.today(),
        help="End of backtest period"
    )

# Capital
st.sidebar.subheader("Initial Investment")
capital = st.sidebar.number_input(
    "Initial Capital",
    min_value=1000,
    max_value=10000000,
    value=100000,
    step=10000,
    help="Starting portfolio value"
)

# Cache option
use_cache = st.sidebar.checkbox(
    "Use cached data",
    value=True,
    help="Cache price data for faster subsequent runs"
)

# Run button
run_backtest = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)

# Main content area
if run_backtest:
    # Validate inputs
    if not all(tickers):
        st.error("❌ Please enter all ticker symbols")
        st.stop()

    if not benchmark:
        st.error("❌ Please enter a benchmark ticker")
        st.stop()

    if start_date >= end_date:
        st.error("❌ Start date must be before end date")
        st.stop()

    # Normalize weights
    weights_array = np.array(weights)
    if not np.isclose(weights_array.sum(), 1.0):
        weights_array = weights_array / weights_array.sum()
        st.info(f"ℹ️ Weights normalized to sum to 1.0: {weights_array.round(3).tolist()}")

    # Progress indicator
    with st.spinner("Downloading price data..."):
        try:
            # Download prices
            portfolio_prices = download_prices(
                tickers,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                use_cache=use_cache
            )

            benchmark_prices = download_prices(
                [benchmark],
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                use_cache=use_cache
            )[benchmark]

            st.success(f"✅ Downloaded data for {len(tickers)} portfolio ticker(s) and benchmark")

        except Exception as e:
            st.error(f"❌ Error downloading data: {str(e)}")
            st.stop()

    # Compute metrics
    with st.spinner("Computing backtest metrics..."):
        try:
            results = compute_metrics(
                portfolio_prices,
                weights_array,
                benchmark_prices,
                capital
            )

            st.success(f"✅ Backtest completed: {len(results)} trading days analyzed")

        except Exception as e:
            st.error(f"❌ Error computing metrics: {str(e)}")
            st.stop()

    # Display results
    st.divider()
    st.header("📊 Results")

    # Summary statistics
    st.subheader("Summary Statistics")

    # Compute summaries
    portfolio_summary = summarize(results['portfolio_value'], capital)
    benchmark_summary = summarize(results['benchmark_value'], capital)

    # Display in columns
    col1, col2, col3 = st.columns(3)

    # Metric labels mapping
    metric_labels = {
        "ending_value": "Ending Value",
        "total_return": "Total Return",
        "cagr": "CAGR",
        "volatility": "Volatility",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "max_drawdown": "Max Drawdown"
    }

    with col1:
        st.markdown("### Portfolio")
        for key, value in portfolio_summary.items():
            label = metric_labels.get(key, key)
            if key == "ending_value":
                st.metric(label, f"${value:,.2f}")
            elif key in ["total_return", "cagr", "volatility", "max_drawdown"]:
                st.metric(label, f"{value:.2%}")
            elif key in ["sharpe_ratio", "sortino_ratio"]:
                st.metric(label, f"{value:.3f}")
            else:
                st.metric(label, f"{value:.2f}")

    with col2:
        st.markdown("### Benchmark")
        for key, value in benchmark_summary.items():
            label = metric_labels.get(key, key)
            if key == "ending_value":
                st.metric(label, f"${value:,.2f}")
            elif key in ["total_return", "cagr", "volatility", "max_drawdown"]:
                st.metric(label, f"{value:.2%}")
            elif key in ["sharpe_ratio", "sortino_ratio"]:
                st.metric(label, f"{value:.3f}")
            else:
                st.metric(label, f"{value:.2f}")

    with col3:
        st.markdown("### Relative Performance")
        excess_return = portfolio_summary['total_return'] - benchmark_summary['total_return']
        excess_cagr = portfolio_summary['cagr'] - benchmark_summary['cagr']
        excess_sharpe = portfolio_summary['sharpe_ratio'] - benchmark_summary['sharpe_ratio']

        st.metric("Excess Return", f"{excess_return:.2%}", delta=None)
        st.metric("Excess CAGR", f"{excess_cagr:.2%}", delta=None)
        st.metric("Sharpe Difference", f"{excess_sharpe:.3f}", delta=None)

    # Portfolio composition
    st.divider()
    st.subheader("Portfolio Composition")
    composition_data = {
        "Ticker": tickers,
        "Weight": [f"{w:.1%}" for w in weights_array],
        "Normalized Weight": [f"{w:.3%}" for w in weights_array]
    }
    st.table(pd.DataFrame(composition_data))

    # Charts
    st.divider()
    st.header("📈 Interactive Visualizations")
    st.caption("💡 Hover over the charts to see exact values")

    # Calculate metrics for charts
    active_return = (results['portfolio_return'] - results['benchmark_return']) * 100

    # Calculate drawdown for portfolio
    portfolio_cummax = results['portfolio_value'].expanding().max()
    portfolio_drawdown = ((results['portfolio_value'] - portfolio_cummax) / portfolio_cummax) * 100

    # Calculate drawdown for benchmark
    benchmark_cummax = results['benchmark_value'].expanding().max()
    benchmark_drawdown = ((results['benchmark_value'] - benchmark_cummax) / benchmark_cummax) * 100

    # Create 2x2 grid of charts using Plotly subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Portfolio vs Benchmark Value', 'Cumulative Returns',
                       'Active Return (Portfolio - Benchmark)', 'Drawdown Over Time'),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # Chart 1: Portfolio vs Benchmark Value (top-left)
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['portfolio_value'],
            name='Portfolio',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['benchmark_value'],
            name='Benchmark',
            line=dict(color='#9467bd', width=2, dash='dash'),
            hovertemplate='<b>Benchmark</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Chart 2: Cumulative Returns (top-right)
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['portfolio_return'] * 100,
            name='Portfolio Return',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['benchmark_return'] * 100,
            name='Benchmark Return',
            line=dict(color='#9467bd', width=2, dash='dash'),
            hovertemplate='<b>Benchmark</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )

    # Chart 3: Active Return (bottom-left)
    # Color code based on positive/negative
    colors = ['rgba(44, 160, 44, 0.3)' if x >= 0 else 'rgba(214, 39, 40, 0.3)' for x in active_return]

    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=active_return,
            name='Active Return',
            fill='tozeroy',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Active Return</b><br>Date: %{x}<br>Difference: %{y:.2f}%<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.5, row=2, col=1)

    # Chart 4: Drawdown (bottom-right)
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=portfolio_drawdown,
            name='Portfolio DD',
            fill='tozeroy',
            line=dict(color='#1f77b4', width=1.5),
            hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=benchmark_drawdown,
            name='Benchmark DD',
            fill='tozeroy',
            line=dict(color='#9467bd', width=1.5, dash='dash'),
            hovertemplate='<b>Benchmark</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.5, row=2, col=2)

    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)

    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Active Return (%)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=2)

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        template='plotly_white'
    )

    # Display interactive chart
    st.plotly_chart(fig, use_container_width=True)

    # Download options
    st.divider()
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
            use_container_width=True
        )

    with col2:
        # Chart download as interactive HTML
        chart_html = fig.to_html(include_plotlyjs='cdn')

        st.download_button(
            label="📥 Download Interactive Charts (HTML)",
            data=chart_html,
            file_name=f"backtest_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )

    # Show raw data
    with st.expander("📋 View Raw Data"):
        st.dataframe(results, use_container_width=True)

else:
    # Welcome screen
    st.info("👈 Configure your backtest parameters in the sidebar and click 'Run Backtest' to begin")

    # Quick start guide
    st.subheader("Quick Start Guide")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 📝 Steps:
        1. **Set number of tickers** in your portfolio
        2. **Enter ticker symbols** (e.g., VDCP.L, AAPL)
        3. **Assign weights** to each ticker
        4. **Choose a benchmark** (e.g., VWRA.L, SPY)
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

    st.markdown("---")

    # Example configuration
    st.subheader("Example Configuration")
    st.code("""
Tickers: VDCP.L, VHYD.L
Weights: 0.5, 0.5
Benchmark: VWRA.L
Date Range: 2018-01-01 to Today
Initial Capital: $100,000
    """, language="text")

    st.markdown("---")
    st.caption("Built with Streamlit • Data from Yahoo Finance via yfinance")
