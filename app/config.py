"""Configuration constants for the Portfolio Backtester UI.

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

PAGE_TITLE = "Portfolio Backtester"
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

# Colorblind-friendly palette (Wong palette - accessible for all types of colorblindness)
# Portfolio color (blue) - distinguishable for all colorblind types
PORTFOLIO_COLOR = "#0173B2"

# Benchmark colors (orange, teal, pink) - colorblind-safe Wong palette
# Avoids problematic blue-purple and red-green combinations
BENCHMARK_COLORS = ["#DE8F05", "#029E73", "#CC78BC"]

# Benchmark line styles (provides visual differentiation beyond color)
BENCHMARK_DASH_STYLES = ["dash", "dot", "dashdot"]

# Marker symbols for additional visual differentiation
# Used sparingly to avoid clutter (every 20th point)
PORTFOLIO_MARKER = "circle"
BENCHMARK_MARKERS = ["square", "diamond", "triangle-up"]

# Positive/negative colors (blue/orange instead of green/red for colorblind accessibility)
POSITIVE_COLOR = "#0173B2"  # Blue
NEGATIVE_COLOR = "#DE8F05"  # Orange

# Rolling windows for returns analysis (in days)
ROLLING_WINDOWS = [30, 90, 180]

# Chart dimensions
CHART_HEIGHT = 400
DASHBOARD_HEIGHT = 800

# =============================================================================
# Visual Hierarchy Configuration
# =============================================================================

# Line widths - establishes visual priority through weight
PORTFOLIO_LINE_WIDTH = 2.5  # Primary data - thickest
BENCHMARK_LINE_WIDTH = 2.0  # Secondary data - medium
REFERENCE_LINE_WIDTH = 1.0  # Reference lines (zero lines, etc.) - thin
GRID_LINE_WIDTH = 0.5       # Grid lines - thinnest

# Opacity levels - creates depth and reduces visual clutter
PORTFOLIO_OPACITY = 1.0     # Primary data - fully opaque
BENCHMARK_OPACITY = 0.85    # Secondary data - slightly transparent
FILL_OPACITY = 0.25         # Fill areas - subtle
GRID_OPACITY = 0.2          # Grid lines - very subtle
REFERENCE_LINE_OPACITY = 0.4  # Reference lines - moderate

# Font sizes - establishes typographic hierarchy (Plotly uses px)
TITLE_FONT_SIZE = 16        # Main chart titles
SUBPLOT_TITLE_FONT_SIZE = 13  # Subplot titles
AXIS_TITLE_FONT_SIZE = 12   # Axis labels
LEGEND_FONT_SIZE = 11       # Legend text
TICK_FONT_SIZE = 10         # Axis tick labels
ANNOTATION_FONT_SIZE = 10   # Annotations and notes

# Spacing and layout
SUBPLOT_VERTICAL_SPACING = 0.15    # Vertical spacing between subplots (increased from 0.12)
SUBPLOT_HORIZONTAL_SPACING = 0.12  # Horizontal spacing between subplots (increased from 0.10)

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

MAIN_TITLE = "📈 Portfolio Backtester"
SUBTITLE = "Compare portfolio performance against benchmarks"
SIDEBAR_HEADER = "Backtest Configuration"
