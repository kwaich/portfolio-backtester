"""
Streamlit web UI for ETF backtesting.

This app provides an interactive interface for running backtests and visualizing results.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import io

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
                benchmark_prices,
                weights_array,
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

    with col1:
        st.markdown("### Portfolio")
        for key, value in portfolio_summary.items():
            if isinstance(value, float):
                if 'Return' in key or 'CAGR' in key or 'Drawdown' in key:
                    st.metric(key, f"{value:.2%}")
                elif 'Ratio' in key:
                    st.metric(key, f"{value:.3f}")
                elif 'Value' in key:
                    st.metric(key, f"${value:,.2f}")
                else:
                    st.metric(key, f"{value:.2%}")
            else:
                st.metric(key, value)

    with col2:
        st.markdown("### Benchmark")
        for key, value in benchmark_summary.items():
            if isinstance(value, float):
                if 'Return' in key or 'CAGR' in key or 'Drawdown' in key:
                    st.metric(key, f"{value:.2%}")
                elif 'Ratio' in key:
                    st.metric(key, f"{value:.3f}")
                elif 'Value' in key:
                    st.metric(key, f"${value:,.2f}")
                else:
                    st.metric(key, f"{value:.2%}")
            else:
                st.metric(key, value)

    with col3:
        st.markdown("### Relative Performance")
        excess_return = portfolio_summary['Total Return'] - benchmark_summary['Total Return']
        excess_cagr = portfolio_summary['CAGR'] - benchmark_summary['CAGR']
        excess_sharpe = portfolio_summary['Sharpe Ratio'] - benchmark_summary['Sharpe Ratio']

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
    st.header("📈 Visualizations")

    # Create 2x2 grid of charts
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')

    # Chart 1: Portfolio vs Benchmark Value
    ax1.plot(results.index, results['portfolio_value'], label='Portfolio', linewidth=2, color='#1f77b4')
    ax1.plot(results.index, results['benchmark_value'], label='Benchmark', linewidth=2, color='#9467bd', linestyle='--')
    ax1.set_title('Portfolio vs Benchmark Value', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Value ($)', fontsize=11)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Chart 2: Cumulative Returns
    ax2.plot(results.index, results['portfolio_return'] * 100, label='Portfolio', linewidth=2, color='#1f77b4')
    ax2.plot(results.index, results['benchmark_return'] * 100, label='Benchmark', linewidth=2, color='#9467bd', linestyle='--')
    ax2.set_title('Cumulative Returns', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Return (%)', fontsize=11)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

    # Chart 3: Active Return (Portfolio - Benchmark)
    active_return = (results['portfolio_return'] - results['benchmark_return']) * 100
    colors = ['#2ca02c' if x >= 0 else '#d62728' for x in active_return]
    ax3.fill_between(results.index, 0, active_return, alpha=0.3, color='#1f77b4')
    ax3.plot(results.index, active_return, linewidth=2, color='#1f77b4')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_title('Active Return (Portfolio - Benchmark)', fontsize=14, fontweight='bold', pad=10)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('Active Return (%)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

    # Chart 4: Drawdown
    # Calculate drawdown for portfolio
    portfolio_cummax = results['portfolio_value'].expanding().max()
    portfolio_drawdown = ((results['portfolio_value'] - portfolio_cummax) / portfolio_cummax) * 100

    # Calculate drawdown for benchmark
    benchmark_cummax = results['benchmark_value'].expanding().max()
    benchmark_drawdown = ((results['benchmark_value'] - benchmark_cummax) / benchmark_cummax) * 100

    ax4.fill_between(results.index, 0, portfolio_drawdown, alpha=0.3, color='#1f77b4', label='Portfolio')
    ax4.fill_between(results.index, 0, benchmark_drawdown, alpha=0.3, color='#9467bd', label='Benchmark')
    ax4.plot(results.index, portfolio_drawdown, linewidth=1.5, color='#1f77b4')
    ax4.plot(results.index, benchmark_drawdown, linewidth=1.5, color='#9467bd', linestyle='--')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax4.set_title('Drawdown Over Time', fontsize=14, fontweight='bold', pad=10)
    ax4.set_xlabel('Date', fontsize=11)
    ax4.set_ylabel('Drawdown (%)', fontsize=11)
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

    # Annotate max drawdowns
    max_dd_portfolio = portfolio_drawdown.min()
    max_dd_portfolio_date = portfolio_drawdown.idxmin()
    max_dd_benchmark = benchmark_drawdown.min()

    ax4.annotate(f'Max DD: {max_dd_portfolio:.2f}%',
                xy=(max_dd_portfolio_date, max_dd_portfolio),
                xytext=(10, -20), textcoords='offset points',
                fontsize=9, color='#1f77b4',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='#1f77b4'))

    plt.tight_layout()
    st.pyplot(fig)

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
        # Chart download
        chart_buffer = io.BytesIO()
        fig.savefig(chart_buffer, format='png', dpi=150, bbox_inches='tight')
        chart_data = chart_buffer.getvalue()

        st.download_button(
            label="📥 Download Charts (PNG)",
            data=chart_data,
            file_name=f"backtest_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
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
        - **Interactive charts**: Portfolio value, returns, active return
        - **Data caching**: Faster subsequent runs
        - **CSV export**: Download results for further analysis
        - **Chart export**: Save visualizations as PNG
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
