"""
ETF Backtester Streamlit Application

This package contains the refactored Streamlit application for ETF backtesting.

Modules:
- config: Configuration constants and styling
- presets: Portfolio and date range presets
- validation: Input validation and session state management
- state_manager: Centralized session state management
- ticker_data: Ticker search and Yahoo Finance integration
- ui_components: Reusable UI components for metrics and tables
- charts: Plotly chart generation functions
- sidebar: Sidebar rendering with form support
- results: Results display functions
- utils: Utility functions for URL parameters and error handling
- main: Main application entry point

Improvements (Streamlit Best Practices):
- ✅ Caching with @st.cache_data for expensive operations
- ✅ Forms to reduce unnecessary reruns
- ✅ Modular code organization for maintainability
- ✅ URL parameters for sharing configurations
- ✅ Better error handling with user-friendly messages
- ✅ Progress tracking for long-running operations
"""

__version__ = "2.0.0"  # Updated for best practices refactor
__all__ = ["main"]
