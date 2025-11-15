"""Main Streamlit application entry point for the ETF Backtester.

This module orchestrates the entire application workflow by integrating:
- Configuration from config.py
- Presets from presets.py
- Validation from validation.py
- UI components from ui_components.py
- Charts from charts.py
"""

from __future__ import annotations

from datetime import datetime
import io

# Third-party imports with error handling
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
    import numpy as np
except ImportError as e:
    st.error(f"❌ Missing dependency: {e}\n\nInstall with: pip install -r requirements.txt")
    st.stop()

# Import backtest functions
try:
    from backtest import download_prices, compute_metrics, summarize, validate_tickers
except ImportError as e:
    st.error(
        f"❌ **Cannot Import Backtest Module**\n\n"
        f"Error: `{str(e)}`\n\n"
        f"Ensure backtest.py is in the parent directory."
    )
    st.stop()

# Import app modules
try:
    from .config import (
        PAGE_TITLE, PAGE_ICON, PAGE_LAYOUT, SIDEBAR_STATE,
        CUSTOM_CSS, MAIN_TITLE, SUBTITLE, SIDEBAR_HEADER,
        DEFAULT_TICKER_1, DEFAULT_TICKER_2, DEFAULT_BENCHMARK,
        MAX_TICKERS, MIN_BENCHMARKS, MAX_BENCHMARKS,
        DEFAULT_CAPITAL, MIN_CAPITAL, MAX_CAPITAL
    )
    from .presets import get_portfolio_presets, get_date_presets
    from .validation import (
        initialize_session_state, update_portfolio_preset,
        validate_backtest_inputs, check_and_normalize_weights
    )
    from .ui_components import (
        render_metrics_column, render_relative_metrics,
        render_portfolio_composition
    )
    from .charts import create_main_dashboard, create_rolling_returns_chart
except ImportError as e:
    st.error(f"❌ Failed to import app modules: {e}")
    st.stop()


def main() -> None:
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=PAGE_LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE
    )
    
    # Custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown(f'<div class="main-header">{MAIN_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{SUBTITLE}</div>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # =========================================================================
    # Sidebar - Input Controls
    # =========================================================================
    
    st.sidebar.header(SIDEBAR_HEADER)
    
    # Portfolio presets
    portfolio_presets = get_portfolio_presets()
    selected_portfolio = st.sidebar.selectbox(
        "Example Portfolio",
        options=list(portfolio_presets.keys()),
        index=0,
        help="Select a pre-configured portfolio or choose Custom to enter manually"
    )
    
    # Handle portfolio preset selection
    if 'selected_portfolio' not in st.session_state:
        st.session_state.selected_portfolio = selected_portfolio
    
    if selected_portfolio != st.session_state.selected_portfolio:
        st.session_state.selected_portfolio = selected_portfolio
        if selected_portfolio != "Custom (Manual Entry)":
            portfolio_config = portfolio_presets[selected_portfolio]
            update_portfolio_preset(selected_portfolio, portfolio_config)
    
    # Number of tickers
    num_tickers = st.sidebar.number_input(
        "Number of Portfolio Tickers",
        min_value=1,
        max_value=MAX_TICKERS,
        value=st.session_state.get('num_tickers', 2),
        step=1,
        help="How many different assets in your portfolio?"
    )
    
    # Dynamic ticker inputs
    tickers = []
    weights = []
    
    st.sidebar.subheader("Portfolio Composition")
    
    preset_tickers = st.session_state.get('preset_tickers', [])
    preset_weights = st.session_state.get('preset_weights', [])
    
    for i in range(num_tickers):
        col1, col2 = st.sidebar.columns([2, 1])
        
        with col1:
            # Determine default ticker
            if i < len(preset_tickers):
                default_ticker = preset_tickers[i]
            elif i == 0:
                default_ticker = DEFAULT_TICKER_1
            elif i == 1:
                default_ticker = DEFAULT_TICKER_2
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
            # Determine default weight
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
    preset_benchmark = st.session_state.get('preset_benchmark', DEFAULT_BENCHMARK)
    
    num_benchmarks = st.sidebar.number_input(
        "Number of Benchmarks",
        min_value=MIN_BENCHMARKS,
        max_value=MAX_BENCHMARKS,
        value=1,
        step=1,
        help="Compare against multiple benchmarks"
    )
    
    benchmarks = []
    for i in range(num_benchmarks):
        if i == 0:
            default_bench = preset_benchmark
        elif i == 1:
            default_bench = "SPY"
        else:
            default_bench = ""
        
        bench_ticker = st.sidebar.text_input(
            f"Benchmark {i+1}",
            value=default_bench,
            key=f"benchmark_{i}",
            help="Ticker to compare against (e.g., VWRA.L, SPY)"
        )
        if bench_ticker:
            benchmarks.append(bench_ticker)
    
    # Keep single benchmark variable for backward compatibility
    benchmark = benchmarks[0] if benchmarks else preset_benchmark
    
    # Date range
    st.sidebar.subheader("Date Range")
    st.sidebar.caption("Quick Presets:")
    
    date_presets = get_date_presets()
    preset_cols = st.sidebar.columns(6)
    
    # Initialize date session state
    if 'start_date' not in st.session_state:
        st.session_state.start_date = datetime(2018, 1, 1)
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.today()
    
    # Preset buttons
    for idx, (label, date_value) in enumerate(date_presets.items()):
        if preset_cols[idx].button(label, use_container_width=True, help=f"Set range to {label}"):
            st.session_state.start_date = date_value
            st.session_state.end_date = datetime.today()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.start_date,
            help="Backtest start date"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=st.session_state.end_date,
            help="Backtest end date"
        )
    
    # Capital
    st.sidebar.subheader("Initial Capital")
    capital = st.sidebar.number_input(
        "Capital ($)",
        min_value=float(MIN_CAPITAL),
        max_value=float(MAX_CAPITAL),
        value=float(DEFAULT_CAPITAL),
        step=1000.0,
        format="%0.2f",
        help="Initial investment amount"
    )
    
    # Cache option
    st.sidebar.subheader("Options")
    use_cache = st.sidebar.checkbox(
        "Use cached data",
        value=True,
        help="Reuse previously downloaded data for faster results"
    )
    
    # Run button
    run_backtest = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    # =========================================================================
    # Main Content Area
    # =========================================================================
    
    if run_backtest:
        # Validate inputs
        is_valid, error_msg = validate_backtest_inputs(tickers, benchmarks, start_date, end_date)
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
            st.stop()
        
        # Validate ticker format
        try:
            validate_tickers(tickers)
            validate_tickers(benchmarks)
        except ValueError as e:
            st.error(f"❌ **Ticker Validation Failed**\n\n{str(e)}")
            st.stop()
        
        # Normalize weights
        weights_array, was_normalized = check_and_normalize_weights(weights)
        if was_normalized:
            st.info(f"ℹ️ Weights normalized to sum to 1.0: {weights_array.round(3).tolist()}")
        
        # Progress indicator
        with st.spinner("Downloading price data..."):
            try:
                # Download prices
                universe = list(dict.fromkeys(tickers + benchmarks))
                prices = download_prices(
                    universe,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    use_cache=use_cache
                )
                
                portfolio_prices = prices[tickers]
                
                # Download all benchmarks
                all_benchmark_prices = {}
                for bench in benchmarks:
                    all_benchmark_prices[bench] = prices[bench]
                
                # Keep primary benchmark for backward compatibility
                benchmark_prices = all_benchmark_prices[benchmarks[0]]
                
                st.success(
                    f"✅ Downloaded data for {len(tickers)} portfolio ticker(s) "
                    f"and {len(benchmarks)} benchmark(s)"
                )
                
            except Exception as e:
                st.error(f"❌ Error downloading data: {str(e)}")
                st.stop()
        
        # Compute metrics
        with st.spinner("Computing backtest metrics..."):
            try:
                # Compute metrics for primary benchmark
                results = compute_metrics(portfolio_prices, weights_array, benchmark_prices, capital)
                
                # Compute metrics for additional benchmarks
                all_benchmark_results = {}
                for bench_name, bench_prices in all_benchmark_prices.items():
                    bench_result = compute_metrics(portfolio_prices, weights_array, bench_prices, capital)
                    all_benchmark_results[bench_name] = bench_result
                
                st.success("✅ Backtest completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error computing metrics: {str(e)}")
                st.stop()
        
        # Display results
        st.divider()
        st.header("📊 Backtest Results")
        
        # Summary statistics
        st.subheader("Summary Statistics")
        
        # Compute summaries
        portfolio_summary = summarize(results["portfolio_value"], capital)
        
        # Compute summaries for all benchmarks
        all_benchmark_summaries = {}
        for bench_name, bench_result in all_benchmark_results.items():
            all_benchmark_summaries[bench_name] = summarize(bench_result['benchmark_value'], capital)
        
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
        
        # Portfolio composition
        st.divider()
        render_portfolio_composition(tickers, weights_array)
        
        # Charts
        st.divider()
        st.header("📈 Interactive Visualizations")
        st.caption("💡 Hover over the charts to see exact values")
        
        # Create main dashboard
        fig = create_main_dashboard(results, all_benchmark_results, benchmarks)
        st.plotly_chart(fig, use_container_width=True)
        
        # Rolling returns analysis
        st.divider()
        st.subheader("📊 Rolling Returns Analysis")
        st.caption("💡 Rolling returns show performance consistency over different time periods")
        
        fig_rolling = create_rolling_returns_chart(results, all_benchmark_results, benchmarks)
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


if __name__ == "__main__":
    main()
