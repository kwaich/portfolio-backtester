"""
Streamlit web UI for ETF backtesting.

This app provides an interactive interface for running backtests and visualizing results.
"""

from __future__ import annotations

# Standard library imports (should always work)
from datetime import datetime, timedelta
from pathlib import Path
import io

# Check and import third-party dependencies
try:
    import streamlit as st
except ImportError as e:
    print(
        "ERROR: Streamlit is not installed.\n"
        "Please install it with: pip install streamlit>=1.28.0\n"
        f"Details: {e}"
    )
    raise SystemExit(1)

try:
    import pandas as pd
except ImportError as e:
    st.error(
        "❌ **Missing Dependency: pandas**\n\n"
        "pandas is required for data processing.\n\n"
        "**Install with:**\n"
        "```bash\n"
        "pip install pandas>=2.0.0\n"
        "```"
    )
    st.stop()

try:
    import numpy as np
except ImportError as e:
    st.error(
        "❌ **Missing Dependency: numpy**\n\n"
        "numpy is required for numerical operations.\n\n"
        "**Install with:**\n"
        "```bash\n"
        "pip install numpy>=1.24.0\n"
        "```"
    )
    st.stop()

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    st.error(
        "❌ **Missing Dependency: plotly**\n\n"
        "plotly is required for interactive charts.\n\n"
        "**Install with:**\n"
        "```bash\n"
        "pip install plotly>=5.14.0\n"
        "```"
    )
    st.stop()

# Import functions from backtest module
try:
    from backtest import download_prices, compute_metrics, summarize, validate_tickers
except ImportError as e:
    st.error(
        "❌ **Cannot Import Backtest Module**\n\n"
        f"Error: `{str(e)}`\n\n"
        "**Please ensure:**\n"
        "1. `backtest.py` is in the same directory as `app.py`\n"
        "2. All dependencies are installed:\n"
        "   ```bash\n"
        "   pip install -r requirements.txt\n"
        "   ```\n"
        "3. Python version is 3.8 or higher\n\n"
        "**Directory structure should be:**\n"
        "```\n"
        "backtester/\n"
        "├── app.py\n"
        "├── backtest.py\n"
        "└── requirements.txt\n"
        "```"
    )
    st.stop()
except Exception as e:
    st.error(
        "❌ **Error Loading Backtest Module**\n\n"
        f"Unexpected error: `{str(e)}`\n\n"
        "**Possible causes:**\n"
        "- Syntax error in `backtest.py`\n"
        "- Missing dependencies in `backtest.py`\n"
        "- Incompatible Python version\n\n"
        "**Try:**\n"
        "```bash\n"
        "python -c \"import backtest\"\n"
        "```\n"
        "to see detailed error messages."
    )
    st.stop()

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

# Example portfolio presets
example_portfolios = {
    "Custom (Manual Entry)": {"tickers": [], "weights": [], "benchmark": "VWRA.L"},
    "Default UK ETFs": {"tickers": ["VDCP.L", "VHYD.L"], "weights": [0.5, 0.5], "benchmark": "VWRA.L"},
    "60/40 US Stocks/Bonds": {"tickers": ["VOO", "BND"], "weights": [0.6, 0.4], "benchmark": "SPY"},
    "Tech Giants": {"tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"], "weights": [0.25, 0.25, 0.25, 0.25], "benchmark": "QQQ"},
    "Dividend Aristocrats": {"tickers": ["JNJ", "PG", "KO", "PEP"], "weights": [0.25, 0.25, 0.25, 0.25], "benchmark": "SPY"},
    "Global Diversified": {"tickers": ["VTI", "VXUS", "BND"], "weights": [0.5, 0.3, 0.2], "benchmark": "VT"}
}

selected_portfolio = st.sidebar.selectbox(
    "Example Portfolio",
    options=list(example_portfolios.keys()),
    index=0,
    help="Select a pre-configured portfolio or choose Custom to enter manually"
)

# Initialize session state for portfolio selection
if 'selected_portfolio' not in st.session_state:
    st.session_state.selected_portfolio = selected_portfolio

# Check if portfolio changed
if selected_portfolio != st.session_state.selected_portfolio:
    st.session_state.selected_portfolio = selected_portfolio
    if selected_portfolio != "Custom (Manual Entry)":
        portfolio_config = example_portfolios[selected_portfolio]
        st.session_state.num_tickers = len(portfolio_config["tickers"])
        st.session_state.preset_tickers = portfolio_config["tickers"]
        st.session_state.preset_weights = portfolio_config["weights"]
        st.session_state.preset_benchmark = portfolio_config["benchmark"]

# Number of tickers
num_tickers = st.sidebar.number_input(
    "Number of Portfolio Tickers",
    min_value=1,
    max_value=10,
    value=st.session_state.get('num_tickers', 2),
    step=1,
    help="How many different assets in your portfolio?"
)

# Dynamic ticker inputs
tickers = []
weights = []

st.sidebar.subheader("Portfolio Composition")

# Get preset values if available
preset_tickers = st.session_state.get('preset_tickers', [])
preset_weights = st.session_state.get('preset_weights', [])

for i in range(num_tickers):
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        # Use preset ticker if available, otherwise use defaults
        if i < len(preset_tickers):
            default_ticker = preset_tickers[i]
        elif i == 0:
            default_ticker = "VDCP.L"
        elif i == 1:
            default_ticker = "VHYD.L"
        else:
            default_ticker = ""

        ticker = col1.text_input(
            f"Ticker {i+1}",
            value=default_ticker,
            key=f"ticker_{i}",
            help="Enter ticker symbol (e.g., VDCP.L, AAPL)"
        )
        tickers.append(ticker)
    with col2:
        # Use preset weight if available, otherwise equal weight
        if i < len(preset_weights):
            default_weight = preset_weights[i]
        else:
            default_weight = 1.0 / num_tickers

        weight = col2.number_input(
            f"Weight {i+1}",
            min_value=0.0,
            max_value=1.0,
            value=default_weight,
            step=0.05,
            key=f"weight_{i}",
            help="Portfolio weight (will be normalized)"
        )
        weights.append(weight)

# Benchmark
st.sidebar.subheader("Benchmark")
preset_benchmark = st.session_state.get('preset_benchmark', "VWRA.L")

# Multiple benchmarks support
num_benchmarks = st.sidebar.number_input(
    "Number of Benchmarks",
    min_value=1,
    max_value=3,
    value=1,
    step=1,
    help="Compare against multiple benchmarks"
)

benchmarks = []
for i in range(num_benchmarks):
    if i == 0:
        default_benchmark = preset_benchmark
    elif i == 1:
        default_benchmark = "SPY"
    else:
        default_benchmark = ""

    benchmark_ticker = st.sidebar.text_input(
        f"Benchmark {i+1}",
        value=default_benchmark,
        key=f"benchmark_{i}",
        help="Ticker to compare against (e.g., VWRA.L, SPY)"
    )
    if benchmark_ticker:
        benchmarks.append(benchmark_ticker)

# Keep single benchmark variable for backward compatibility
benchmark = benchmarks[0] if benchmarks else preset_benchmark

# Date range
st.sidebar.subheader("Date Range")

# Quick preset buttons
st.sidebar.caption("Quick Presets:")
preset_cols = st.sidebar.columns(6)
today = datetime.today()

presets = {
    "1Y": today - timedelta(days=365),
    "3Y": today - timedelta(days=365*3),
    "5Y": today - timedelta(days=365*5),
    "10Y": today - timedelta(days=365*10),
    "YTD": datetime(today.year, 1, 1),
    "Max": datetime(2010, 1, 1)
}

# Initialize session state for date selection if not exists
if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime(2018, 1, 1)
if 'end_date' not in st.session_state:
    st.session_state.end_date = today

# Preset buttons
for idx, (label, date_value) in enumerate(presets.items()):
    if preset_cols[idx].button(label, use_container_width=True, help=f"Set range to {label}"):
        st.session_state.start_date = date_value
        st.session_state.end_date = today

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=st.session_state.start_date,
        help="Beginning of backtest period"
    )
    st.session_state.start_date = start_date
with col2:
    end_date = st.date_input(
        "End Date",
        value=st.session_state.end_date,
        help="End of backtest period"
    )
    st.session_state.end_date = end_date

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

    if not benchmarks:
        st.error("❌ Please enter at least one benchmark ticker")
        st.stop()

    # Validate ticker format
    try:
        validate_tickers(tickers)
        validate_tickers(benchmarks)
    except ValueError as e:
        st.error(f"❌ **Ticker Validation Failed**\n\n{str(e)}")
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

            # Download all benchmarks
            all_benchmark_prices = {}
            for bench in benchmarks:
                bench_data = download_prices(
                    [bench],
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    use_cache=use_cache
                )[bench]
                all_benchmark_prices[bench] = bench_data

            # Keep primary benchmark for backward compatibility
            benchmark_prices = all_benchmark_prices[benchmarks[0]]

            st.success(f"✅ Downloaded data for {len(tickers)} portfolio ticker(s) and {len(benchmarks)} benchmark(s)")

        except Exception as e:
            st.error(f"❌ Error downloading data: {str(e)}")
            st.stop()

    # Compute metrics
    with st.spinner("Computing backtest metrics..."):
        try:
            # Compute metrics for primary benchmark
            results = compute_metrics(
                portfolio_prices,
                weights_array,
                benchmark_prices,
                capital
            )

            # Compute metrics for additional benchmarks
            all_benchmark_results = {}
            for bench_name, bench_prices in all_benchmark_prices.items():
                bench_result = compute_metrics(
                    portfolio_prices,
                    weights_array,
                    bench_prices,
                    capital
                )
                all_benchmark_results[bench_name] = bench_result

            st.success(f"✅ Backtest completed: {len(results)} trading days analyzed across {len(benchmarks)} benchmark(s)")

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

    # Compute summaries for all benchmarks
    all_benchmark_summaries = {}
    for bench_name, bench_result in all_benchmark_results.items():
        all_benchmark_summaries[bench_name] = summarize(bench_result['benchmark_value'], capital)

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
        st.markdown(f"### Benchmark ({benchmarks[0]})")
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
        excess_volatility = portfolio_summary['volatility'] - benchmark_summary['volatility']
        excess_sortino = portfolio_summary['sortino_ratio'] - benchmark_summary['sortino_ratio']

        st.metric("Excess Return", f"{excess_return:.2%}",
                 delta=f"{excess_return:.2%}",
                 delta_color="normal")
        st.metric("Excess CAGR", f"{excess_cagr:.2%}",
                 delta=f"{excess_cagr:.2%}",
                 delta_color="normal")
        st.metric("Volatility Diff", f"{excess_volatility:.2%}",
                 delta=f"{excess_volatility:.2%}",
                 delta_color="inverse")  # Lower is better for volatility
        st.metric("Sharpe Difference", f"{excess_sharpe:.3f}",
                 delta=f"{excess_sharpe:.3f}",
                 delta_color="normal")
        st.metric("Sortino Difference", f"{excess_sortino:.3f}",
                 delta=f"{excess_sortino:.3f}",
                 delta_color="normal")

    # Additional benchmark comparisons (if multiple benchmarks)
    if len(benchmarks) > 1:
        st.divider()
        st.subheader("📊 Additional Benchmark Comparisons")

        for bench_name in benchmarks[1:]:  # Skip first benchmark (already shown above)
            with st.expander(f"Comparison vs {bench_name}", expanded=False):
                bench_summary = all_benchmark_summaries[bench_name]

                # Create 2 columns for benchmark and relative performance
                col_b1, col_b2 = st.columns(2)

                with col_b1:
                    st.markdown(f"#### {bench_name} Metrics")
                    for key, value in bench_summary.items():
                        label = metric_labels.get(key, key)
                        if key == "ending_value":
                            st.metric(label, f"${value:,.2f}")
                        elif key in ["total_return", "cagr", "volatility", "max_drawdown"]:
                            st.metric(label, f"{value:.2%}")
                        elif key in ["sharpe_ratio", "sortino_ratio"]:
                            st.metric(label, f"{value:.3f}")
                        else:
                            st.metric(label, f"{value:.2f}")

                with col_b2:
                    st.markdown(f"#### Portfolio vs {bench_name}")
                    excess_return_b = portfolio_summary['total_return'] - bench_summary['total_return']
                    excess_cagr_b = portfolio_summary['cagr'] - bench_summary['cagr']
                    excess_sharpe_b = portfolio_summary['sharpe_ratio'] - bench_summary['sharpe_ratio']
                    excess_volatility_b = portfolio_summary['volatility'] - bench_summary['volatility']
                    excess_sortino_b = portfolio_summary['sortino_ratio'] - bench_summary['sortino_ratio']

                    st.metric("Excess Return", f"{excess_return_b:.2%}",
                             delta=f"{excess_return_b:.2%}",
                             delta_color="normal")
                    st.metric("Excess CAGR", f"{excess_cagr_b:.2%}",
                             delta=f"{excess_cagr_b:.2%}",
                             delta_color="normal")
                    st.metric("Volatility Diff", f"{excess_volatility_b:.2%}",
                             delta=f"{excess_volatility_b:.2%}",
                             delta_color="inverse")
                    st.metric("Sharpe Difference", f"{excess_sharpe_b:.3f}",
                             delta=f"{excess_sharpe_b:.3f}",
                             delta_color="normal")
                    st.metric("Sortino Difference", f"{excess_sortino_b:.3f}",
                             delta=f"{excess_sortino_b:.3f}",
                             delta_color="normal")

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

    # Add all benchmarks to the chart
    benchmark_colors = ['#9467bd', '#e377c2', '#bcbd22']  # Different colors for each benchmark
    benchmark_dash = ['dash', 'dot', 'dashdot']
    for idx, bench_name in enumerate(benchmarks):
        bench_result = all_benchmark_results[bench_name]
        fig.add_trace(
            go.Scatter(
                x=bench_result.index,
                y=bench_result['benchmark_value'],
                name=bench_name,
                line=dict(color=benchmark_colors[idx % len(benchmark_colors)],
                         width=2,
                         dash=benchmark_dash[idx % len(benchmark_dash)]),
                hovertemplate=f'<b>{bench_name}</b><br>Date: %{{x}}<br>Value: $%{{y:,.2f}}<extra></extra>'
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

    # Add all benchmarks to returns chart
    for idx, bench_name in enumerate(benchmarks):
        bench_result = all_benchmark_results[bench_name]
        fig.add_trace(
            go.Scatter(
                x=bench_result.index,
                y=bench_result['benchmark_return'] * 100,
                name=f'{bench_name} Return',
                line=dict(color=benchmark_colors[idx % len(benchmark_colors)],
                         width=2,
                         dash=benchmark_dash[idx % len(benchmark_dash)]),
                hovertemplate=f'<b>{bench_name}</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>',
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

    # Add all benchmarks drawdowns
    for idx, bench_name in enumerate(benchmarks):
        bench_result = all_benchmark_results[bench_name]
        bench_cummax = bench_result['benchmark_value'].expanding().max()
        bench_dd = ((bench_result['benchmark_value'] - bench_cummax) / bench_cummax) * 100

        fig.add_trace(
            go.Scatter(
                x=bench_result.index,
                y=bench_dd,
                name=f'{bench_name} DD',
                fill='tozeroy',
                line=dict(color=benchmark_colors[idx % len(benchmark_colors)],
                         width=1.5,
                         dash=benchmark_dash[idx % len(benchmark_dash)]),
                hovertemplate=f'<b>{bench_name}</b><br>Date: %{{x}}<br>Drawdown: %{{y:.2f}}%<extra></extra>',
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

    # Rolling returns analysis
    st.divider()
    st.subheader("📊 Rolling Returns Analysis")
    st.caption("💡 Rolling returns show performance consistency over different time periods")

    # Calculate rolling returns
    rolling_windows = [30, 90, 180]

    # Create rolling returns chart
    fig_rolling = go.Figure()

    for window in rolling_windows:
        # Calculate rolling returns for portfolio
        portfolio_rolling = results['portfolio_value'].pct_change(window) * 100

        fig_rolling.add_trace(
            go.Scatter(
                x=results.index,
                y=portfolio_rolling,
                name=f'Portfolio {window}D',
                line=dict(width=2),
                hovertemplate=f'<b>Portfolio {window}D</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>'
            )
        )

    # Add all benchmarks rolling returns for comparison
    for idx, bench_name in enumerate(benchmarks):
        bench_result = all_benchmark_results[bench_name]
        for window in rolling_windows:
            bench_rolling = bench_result['benchmark_value'].pct_change(window) * 100

            fig_rolling.add_trace(
                go.Scatter(
                    x=bench_result.index,
                    y=bench_rolling,
                    name=f'{bench_name} {window}D',
                    line=dict(color=benchmark_colors[idx % len(benchmark_colors)],
                             width=1.5,
                             dash=benchmark_dash[idx % len(benchmark_dash)]),
                    hovertemplate=f'<b>{bench_name} {window}D</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>'
                )
            )

    # Add zero line
    fig_rolling.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)

    # Update layout
    fig_rolling.update_layout(
        title="Rolling Returns (30, 90, 180 Day Periods)",
        xaxis_title="Date",
        yaxis_title="Rolling Return (%)",
        height=400,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig_rolling, use_container_width=True)

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
