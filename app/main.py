"""Main Streamlit application entry point for the Portfolio Backtester.

This module orchestrates the entire application workflow using modular components:
- Sidebar configuration with forms for better performance
- Backtest execution with progress tracking
- Results display with interactive charts
- URL parameter support for shareable configurations

IMPROVEMENTS (Streamlit Best Practices):
- ✅ Caching with @st.cache_data for expensive operations
- ✅ Forms to reduce unnecessary reruns
- ✅ Modular code organization for maintainability
- ✅ URL parameters for sharing configurations
- ✅ Better error handling with user-friendly messages
- ✅ Progress tracking for long-running operations
"""

from __future__ import annotations

from datetime import datetime

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
    from backtest import download_prices, compute_metrics, validate_tickers
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
        CUSTOM_CSS, MAIN_TITLE, SUBTITLE
    )
    from .presets import get_portfolio_presets
    from .validation import (
        initialize_session_state, update_portfolio_preset,
        validate_backtest_inputs, check_and_normalize_weights
    )
    from .state_manager import StateManager
    from .sidebar import render_sidebar_form
    from .results import render_results, render_welcome_screen
    from .utils import get_query_params, set_query_params, show_error, show_success, show_info, ProgressTracker
except ImportError as e:
    st.error(f"❌ Failed to import app modules: {e}")
    st.stop()


def _apply_url_parameters() -> None:
    """Apply URL query parameters to session state (for shareable links).

    This enables deep linking and sharing of specific backtest configurations.

    Examples:
        URL: ?tickers=AAPL,MSFT&weights=0.6,0.4&benchmarks=SPY&start_date=2020-01-01&capital=50000
    """
    # Only apply URL params once on first load
    if 'url_params_applied' not in st.session_state:
        params = get_query_params()

        if params:
            # Apply portfolio preset
            if 'preset' in params:
                portfolio_presets = get_portfolio_presets()
                if params['preset'] in portfolio_presets:
                    StateManager.set_selected_portfolio(params['preset'])
                    update_portfolio_preset(params['preset'], portfolio_presets[params['preset']])

            # Apply tickers and weights
            if 'tickers' in params:
                StateManager.set_preset_tickers(params['tickers'])
                if 'weights' in params and len(params['weights']) == len(params['tickers']):
                    StateManager.set_preset_weights(params['weights'])

            # Apply benchmarks (plural - matches what set_query_params writes)
            if 'benchmarks' in params:
                # Store first benchmark for backward compatibility
                if params['benchmarks']:
                    StateManager.set_preset_benchmark(params['benchmarks'][0])
                # Store all benchmarks for UI
                st.session_state['url_benchmarks'] = params['benchmarks']
            # Fall back to singular 'benchmark' for backward compatibility
            elif 'benchmark' in params:
                StateManager.set_preset_benchmark(params['benchmark'])
                st.session_state['url_benchmarks'] = [params['benchmark']]

            # Apply date range
            if 'start_date' in params:
                StateManager.set_date_range(
                    params['start_date'],
                    params.get('end_date', StateManager.get_end_date())
                )

            # Apply capital
            if 'capital' in params:
                st.session_state['url_capital'] = params['capital']

        st.session_state['url_params_applied'] = True


def _run_backtest(config: dict) -> None:
    """Execute backtest with the given configuration.

    Args:
        config: Dictionary containing all backtest configuration

    This function handles the entire backtest workflow:
    1. Input validation
    2. Data download with progress tracking
    3. Metric computation
    4. Result storage in session state
    5. URL parameter updates for sharing
    """
    # Extract configuration
    tickers = config['tickers']
    benchmarks = config['benchmarks']
    weights = config['weights']
    start_date = config['start_date']
    end_date = config['end_date']
    capital = config['capital']
    rebalance_freq = config['rebalance_freq']
    dca_freq = config['dca_freq']
    dca_amount = config['dca_amount']
    use_cache = config['use_cache']
    rebalance_strategy = config['rebalance_strategy']
    dca_frequency = config['dca_frequency']

    # Validate inputs
    is_valid, error_msg = validate_backtest_inputs(tickers, benchmarks, start_date, end_date)

    if not is_valid:
        show_error(error_msg, help_text="Please check your inputs and try again")
        return

    # Validate ticker format
    try:
        validate_tickers(tickers)
        validate_tickers(benchmarks)
    except ValueError as e:
        show_error("Ticker Validation Failed", error=e)
        return

    # Normalize weights
    weights_array, was_normalized = check_and_normalize_weights(weights)
    if was_normalized:
        show_info(f"Weights normalized to sum to 1.0: {weights_array.round(3).tolist()}")

    # Use progress tracker for better UX
    try:
        with ProgressTracker(["Downloading data", "Computing metrics", "Generating results"]) as tracker:
            # Step 1: Download data
            tracker.step()
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

            show_success(
                f"Downloaded data for {len(tickers)} portfolio ticker(s) "
                f"and {len(benchmarks)} benchmark(s)"
            )

            # Step 2: Compute metrics
            tracker.step()

            # Compute metrics for primary benchmark
            results = compute_metrics(
                portfolio_prices,
                weights_array,
                benchmark_prices,
                capital,
                rebalance_freq=rebalance_freq,
                dca_amount=dca_amount,
                dca_freq=dca_freq
            )

            # Compute metrics for additional benchmarks
            all_benchmark_results = {}
            for bench_name, bench_prices in all_benchmark_prices.items():
                bench_result = compute_metrics(
                    portfolio_prices,
                    weights_array,
                    bench_prices,
                    capital,
                    rebalance_freq=rebalance_freq,
                    dca_amount=dca_amount,
                    dca_freq=dca_freq
                )
                all_benchmark_results[bench_name] = bench_result

            # Step 3: Store results
            tracker.step()

            StateManager.store_backtest_results(
                results=results,
                all_benchmark_results=all_benchmark_results,
                tickers=tickers,
                benchmarks=benchmarks,
                weights_array=weights_array,
                capital=capital,
                rebalance_strategy=rebalance_strategy,
                rebalance_freq=rebalance_freq,
                dca_frequency=dca_frequency,
                dca_freq=dca_freq,
                dca_amount=dca_amount
            )

            # Update URL parameters for sharing
            set_query_params(
                tickers=tickers,
                weights=list(weights_array),
                benchmarks=benchmarks,
                start_date=start_date,
                end_date=end_date,
                capital=capital
            )

            show_success("Backtest completed successfully!")

    except Exception as e:
        show_error(
            "An error occurred during backtest execution",
            error=e,
            help_text="Please check your inputs and try again. If the problem persists, try clearing the cache."
        )


def main() -> None:
    """Main application entry point with improved organization and performance.

    This function implements Streamlit best practices:
    - Page configuration at the start
    - Session state initialization
    - URL parameter support for deep linking
    - Form-based sidebar for reduced reruns
    - Modular rendering functions
    - Progress tracking and error handling
    """

    # Page configuration (must be first Streamlit command)
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=PAGE_LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE
    )

    # Custom CSS for improved styling
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown(f'<div class="main-header">{MAIN_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{SUBTITLE}</div>', unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()

    # Check for URL parameters and apply if present (for shareable links)
    _apply_url_parameters()

    # =========================================================================
    # Sidebar - Input Controls (Using Form for Better Performance)
    # =========================================================================

    # Render sidebar and get configuration
    # Using forms prevents reruns on every input change, dramatically improving performance
    config = render_sidebar_form()

    # =========================================================================
    # Main Content Area
    # =========================================================================

    # Run backtest if form submitted
    if config['submit_clicked']:
        _run_backtest(config)

    # Display results if available
    if StateManager.is_backtest_completed():
        stored_results = StateManager.get_backtest_results()
        render_results(stored_results)
    else:
        render_welcome_screen()


if __name__ == "__main__":
    main()
