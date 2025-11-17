"""Configuration constants for the ETF Backtester UI.

This module centralizes all configuration values, including:
- Page settings
- UI limits and defaults
- Chart styling and colors
- Metric labels
- CSS styling
"""

from __future__ import annotations

from datetime import datetime

# =============================================================================
# Page Configuration
# =============================================================================

PAGE_TITLE = "ETF Backtester"
PAGE_ICON = "📈"
PAGE_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# =============================================================================
# UI Limits
# =============================================================================

# Tickers
MIN_TICKERS = 1
MAX_TICKERS = 10
DEFAULT_NUM_TICKERS = 2

# Benchmarks
MIN_BENCHMARKS = 1
MAX_BENCHMARKS = 3
DEFAULT_NUM_BENCHMARKS = 1

# Capital
DEFAULT_CAPITAL = 100_000
MIN_CAPITAL = 1_000
MAX_CAPITAL = 10_000_000

# =============================================================================
# Default Values
# =============================================================================

DEFAULT_START_DATE = datetime(2018, 1, 1)
MAX_START_DATE = datetime(2010, 1, 1)  # For "Max" preset

# Default tickers for manual entry
DEFAULT_TICKER_1 = "VDCP.L"
DEFAULT_TICKER_2 = "VHYD.L"
DEFAULT_BENCHMARK = "VWRA.L"

# =============================================================================
# Rebalancing Configuration
# =============================================================================

# Rebalancing frequency options
REBALANCE_OPTIONS = {
    "Buy-and-Hold (No Rebalancing)": None,
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "M",
    "Quarterly": "Q",
    "Yearly": "Y"
}

DEFAULT_REBALANCE_STRATEGY = "Buy-and-Hold (No Rebalancing)"

# =============================================================================
# DCA (Dollar-Cost Averaging) Configuration
# =============================================================================

# DCA frequency options (same as rebalancing)
DCA_FREQUENCY_OPTIONS = {
    "None (Lump Sum)": None,
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "M",
    "Quarterly": "Q",
    "Yearly": "Y"
}

DEFAULT_DCA_FREQUENCY = "None (Lump Sum)"
DEFAULT_DCA_AMOUNT = 1000.0
MIN_DCA_AMOUNT = 100.0
MAX_DCA_AMOUNT = 100_000.0

# =============================================================================
# Chart Configuration
# =============================================================================

# Portfolio color (blue)
PORTFOLIO_COLOR = "#1f77b4"

# Benchmark colors (purple, pink, yellow-green)
BENCHMARK_COLORS = ["#9467bd", "#e377c2", "#bcbd22"]

# Benchmark line styles
BENCHMARK_DASH_STYLES = ["dash", "dot", "dashdot"]

# Rolling windows for returns analysis (in days)
ROLLING_WINDOWS = [30, 90, 180]

# Chart dimensions
CHART_HEIGHT = 400
DASHBOARD_HEIGHT = 800

# =============================================================================
# Metric Labels
# =============================================================================

METRIC_LABELS = {
    "ending_value": "Ending Value",
    "total_return": "Total Return",
    "cagr": "CAGR",
    "irr": "IRR",
    "volatility": "Volatility",
    "sharpe_ratio": "Sharpe Ratio",
    "sortino_ratio": "Sortino Ratio",
    "max_drawdown": "Max Drawdown",
    "total_contributions": "Total Contributions"
}

# =============================================================================
# CSS Styling
# =============================================================================

CUSTOM_CSS = """
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
"""

# =============================================================================
# UI Text
# =============================================================================

MAIN_TITLE = "📈 ETF Backtester"
SUBTITLE = "Compare portfolio performance against benchmarks"
SIDEBAR_HEADER = "Backtest Configuration"
