"""Plot helper for the portfolio backtest CSV output.

Example:

source .venv/bin/activate
python plot_backtest.py --csv results/backtest_series.csv \
    --output charts/backtest

This will save multiple PNGs (value, returns, active return, drawdown, dashboard).
If --output is omitted the plots are shown interactively using matplotlib's default backend.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Colorblind-friendly color scheme (Wong palette)
# Avoids problematic blue-purple and red-green combinations
PORTFOLIO_COLOR = "#0173B2"  # Blue
BENCHMARK_COLOR = "#DE8F05"  # Orange (instead of purple)
POSITIVE_COLOR = "#0173B2"   # Blue (instead of green)
NEGATIVE_COLOR = "#DE8F05"   # Orange (instead of red)
ACTIVE_COLOR = "#0173B2"     # Blue

# Visual hierarchy constants (matplotlib uses different units than Plotly)
# Line widths
PORTFOLIO_LINE_WIDTH = 2.5    # Primary data - thickest
BENCHMARK_LINE_WIDTH = 2.0    # Secondary data - medium
REFERENCE_LINE_WIDTH = 0.8    # Reference lines - thin
GRID_LINE_WIDTH = 0.5         # Grid lines - thinnest

# Opacity/alpha values
PORTFOLIO_OPACITY = 1.0       # Primary data - fully opaque
BENCHMARK_OPACITY = 0.8       # Secondary data - slightly transparent
FILL_OPACITY = 0.3            # Fill areas - subtle
GRID_OPACITY = 0.3            # Grid lines - subtle
REFERENCE_OPACITY = 0.5       # Reference lines - moderate

# Font sizes (matplotlib uses points)
TITLE_FONT_SIZE = 14          # Main titles
AXIS_LABEL_FONT_SIZE = 11     # Axis labels
LEGEND_FONT_SIZE = 10         # Legend text
TICK_LABEL_FONT_SIZE = 9      # Tick labels
ANNOTATION_FONT_SIZE = 9      # Annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot portfolio vs benchmark")
    parser.add_argument("--csv", required=True, help="CSV exported via backtest.py")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional prefix for PNGs; e.g. charts/run will create multiple PNG files",
    )
    parser.add_argument(
        "--style",
        default="seaborn-v0_8",
        help="Matplotlib style (default: seaborn-v0_8)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI for PNG files (default: 150)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Create a single dashboard with all plots in subplots",
    )
    return parser.parse_args()


def format_currency(x, p):
    """Format tick labels as currency."""
    return f'${x:,.0f}'


def format_percentage(x, p):
    """Format tick labels as percentage."""
    return f'{x*100:.1f}%'


def create_value_plot(df: pd.DataFrame, style: str) -> tuple:
    """Create portfolio vs benchmark value plot with visual hierarchy."""
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(12, 6))

    df["portfolio_value"].plot(
        ax=ax,
        color=PORTFOLIO_COLOR,
        linewidth=PORTFOLIO_LINE_WIDTH,
        linestyle='-',
        alpha=PORTFOLIO_OPACITY,
        label="Portfolio"
    )
    df["benchmark_value"].plot(
        ax=ax,
        color=BENCHMARK_COLOR,
        linewidth=BENCHMARK_LINE_WIDTH,
        linestyle='--',  # Dashed line for visual differentiation
        alpha=BENCHMARK_OPACITY,
        label="Benchmark"
    )

    ax.set_title("Portfolio vs Benchmark Value", fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.set_ylabel("Value ($)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlabel("Date", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    ax.legend(loc="upper left", framealpha=0.9, fontsize=LEGEND_FONT_SIZE)
    ax.grid(True, alpha=GRID_OPACITY, linestyle='--', linewidth=GRID_LINE_WIDTH)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=TICK_LABEL_FONT_SIZE)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=TICK_LABEL_FONT_SIZE)

    return fig, ax


def create_returns_plot(df: pd.DataFrame, style: str) -> tuple:
    """Create cumulative returns comparison plot with visual hierarchy."""
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(12, 6))

    (df["portfolio_return"] * 100).plot(
        ax=ax,
        color=PORTFOLIO_COLOR,
        linewidth=PORTFOLIO_LINE_WIDTH,
        linestyle='-',
        alpha=PORTFOLIO_OPACITY,
        label="Portfolio"
    )
    (df["benchmark_return"] * 100).plot(
        ax=ax,
        color=BENCHMARK_COLOR,
        linewidth=BENCHMARK_LINE_WIDTH,
        linestyle='--',  # Dashed line for visual differentiation
        alpha=BENCHMARK_OPACITY,
        label="Benchmark"
    )

    ax.axhline(0, color='black', linewidth=REFERENCE_LINE_WIDTH, linestyle='-', alpha=REFERENCE_OPACITY)
    ax.set_title("Cumulative Returns Comparison", fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.set_ylabel("Return (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlabel("Date", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=LEGEND_FONT_SIZE)
    ax.grid(True, alpha=GRID_OPACITY, linestyle='--', linewidth=GRID_LINE_WIDTH)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=TICK_LABEL_FONT_SIZE)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=TICK_LABEL_FONT_SIZE)

    return fig, ax


def create_active_return_plot(df: pd.DataFrame, style: str) -> tuple:
    """Create active return plot with colored fill and visual hierarchy."""
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(12, 5))

    active_return_pct = df["active_return"] * 100

    # Fill positive and negative areas with different colors
    ax.fill_between(
        df.index,
        active_return_pct,
        0,
        where=(active_return_pct >= 0),
        color=POSITIVE_COLOR,
        alpha=FILL_OPACITY,
        interpolate=True,
        label="Outperformance"
    )
    ax.fill_between(
        df.index,
        active_return_pct,
        0,
        where=(active_return_pct < 0),
        color=NEGATIVE_COLOR,
        alpha=FILL_OPACITY,
        interpolate=True,
        label="Underperformance"
    )

    # Plot the line
    active_return_pct.plot(ax=ax, color=ACTIVE_COLOR, linewidth=PORTFOLIO_LINE_WIDTH,
                          alpha=PORTFOLIO_OPACITY, label="Active Return")

    ax.axhline(0, color='black', linewidth=REFERENCE_LINE_WIDTH, linestyle='-', alpha=REFERENCE_OPACITY)
    ax.set_title("Active Return (Portfolio - Benchmark)", fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.set_ylabel("Active Return (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlabel("Date", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=LEGEND_FONT_SIZE)
    ax.grid(True, alpha=GRID_OPACITY, linestyle='--', linewidth=GRID_LINE_WIDTH)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=TICK_LABEL_FONT_SIZE)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=TICK_LABEL_FONT_SIZE)

    return fig, ax


def create_drawdown_plot(df: pd.DataFrame, style: str) -> tuple:
    """Create drawdown plot for portfolio and benchmark."""
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(12, 5))

    # Calculate drawdowns
    portfolio_dd = (df["portfolio_value"] / df["portfolio_value"].cummax() - 1) * 100
    benchmark_dd = (df["benchmark_value"] / df["benchmark_value"].cummax() - 1) * 100

    # Fill drawdowns
    ax.fill_between(
        df.index,
        portfolio_dd,
        0,
        alpha=0.4,
        color=PORTFOLIO_COLOR,
        label="Portfolio Drawdown"
    )
    ax.fill_between(
        df.index,
        benchmark_dd,
        0,
        alpha=0.3,
        color=BENCHMARK_COLOR,
        label="Benchmark Drawdown"
    )

    # Plot lines
    portfolio_dd.plot(ax=ax, color=PORTFOLIO_COLOR, linewidth=PORTFOLIO_LINE_WIDTH,
                     linestyle='-', alpha=PORTFOLIO_OPACITY)
    benchmark_dd.plot(ax=ax, color=BENCHMARK_COLOR, linewidth=BENCHMARK_LINE_WIDTH,
                     linestyle='--', alpha=BENCHMARK_OPACITY)

    ax.axhline(0, color='black', linewidth=REFERENCE_LINE_WIDTH, linestyle='-', alpha=REFERENCE_OPACITY)
    ax.set_title("Drawdown Over Time", fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.set_ylabel("Drawdown (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlabel("Date", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=LEGEND_FONT_SIZE)
    ax.grid(True, alpha=GRID_OPACITY, linestyle='--', linewidth=GRID_LINE_WIDTH)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=TICK_LABEL_FONT_SIZE)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=TICK_LABEL_FONT_SIZE)

    # Add annotation for max drawdowns
    port_min_dd = portfolio_dd.min()
    bench_min_dd = benchmark_dd.min()
    port_min_date = portfolio_dd.idxmin()
    bench_min_date = benchmark_dd.idxmin()

    stats_text = (
        f"Portfolio Max DD: {port_min_dd:.2f}% ({port_min_date.strftime('%Y-%m-%d')})\n"
        f"Benchmark Max DD: {bench_min_dd:.2f}% ({bench_min_date.strftime('%Y-%m-%d')})"
    )
    ax.text(
        0.02, 0.02, stats_text, transform=ax.transAxes,
        fontsize=ANNOTATION_FONT_SIZE, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    )

    return fig, ax


def create_rolling_sharpe_plot(df: pd.DataFrame, style: str) -> tuple:
    """Create rolling 12-month Sharpe ratio plot."""
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(12, 5))

    # Check if rolling Sharpe columns exist
    if 'portfolio_rolling_sharpe_12m' not in df.columns or 'benchmark_rolling_sharpe_12m' not in df.columns:
        logger.warning("Rolling Sharpe columns not found in CSV. Skipping rolling Sharpe plot.")
        plt.close(fig)
        return None, None

    # Plot rolling Sharpe ratios
    df["portfolio_rolling_sharpe_12m"].plot(
        ax=ax,
        color=PORTFOLIO_COLOR,
        linewidth=PORTFOLIO_LINE_WIDTH,
        linestyle='-',
        alpha=PORTFOLIO_OPACITY,
        label="Portfolio 12M Sharpe"
    )
    df["benchmark_rolling_sharpe_12m"].plot(
        ax=ax,
        color=BENCHMARK_COLOR,
        linewidth=BENCHMARK_LINE_WIDTH,
        linestyle='--',  # Dashed line for visual differentiation
        alpha=BENCHMARK_OPACITY,
        label="Benchmark 12M Sharpe"
    )

    # Add reference lines (using colorblind-safe teal color)
    ax.axhline(0, color='black', linewidth=REFERENCE_LINE_WIDTH, linestyle='-', alpha=REFERENCE_OPACITY)
    ax.axhline(1, color='#029E73', linewidth=REFERENCE_LINE_WIDTH, linestyle='--',
              alpha=REFERENCE_OPACITY, label="Sharpe = 1")
    ax.axhline(2, color='#029E73', linewidth=REFERENCE_LINE_WIDTH, linestyle='--',
              alpha=REFERENCE_OPACITY, label="Sharpe = 2")

    ax.set_title("Rolling 12-Month Sharpe Ratio", fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.set_ylabel("Rolling 12-Month Sharpe Ratio", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlabel("Date", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=LEGEND_FONT_SIZE)
    ax.grid(True, alpha=GRID_OPACITY, linestyle='--', linewidth=GRID_LINE_WIDTH)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=TICK_LABEL_FONT_SIZE)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=TICK_LABEL_FONT_SIZE)

    return fig, ax


def create_dashboard(df: pd.DataFrame, style: str) -> tuple:
    """Create a comprehensive 2x2 dashboard with all metrics."""
    plt.style.use(style)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Portfolio Performance Dashboard", fontsize=16, fontweight='bold', y=0.995)

    # Top left: Portfolio vs Benchmark Value
    df["portfolio_value"].plot(ax=axes[0, 0], color=PORTFOLIO_COLOR, linewidth=PORTFOLIO_LINE_WIDTH,
                               linestyle='-', alpha=PORTFOLIO_OPACITY, label="Portfolio")
    df["benchmark_value"].plot(ax=axes[0, 0], color=BENCHMARK_COLOR, linewidth=BENCHMARK_LINE_WIDTH,
                               linestyle='--', alpha=BENCHMARK_OPACITY, label="Benchmark")
    axes[0, 0].set_title("Portfolio Value", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("Value ($)", fontsize=10)
    axes[0, 0].yaxis.set_major_formatter(mticker.FuncFormatter(format_currency))
    axes[0, 0].legend(loc="upper left", fontsize=9)
    axes[0, 0].grid(True, alpha=GRID_OPACITY, linestyle='--', linewidth=GRID_LINE_WIDTH)

    # Top right: Cumulative Returns
    (df["portfolio_return"] * 100).plot(ax=axes[0, 1], color=PORTFOLIO_COLOR, linewidth=PORTFOLIO_LINE_WIDTH,
                                       linestyle='-', alpha=PORTFOLIO_OPACITY, label="Portfolio")
    (df["benchmark_return"] * 100).plot(ax=axes[0, 1], color=BENCHMARK_COLOR, linewidth=BENCHMARK_LINE_WIDTH,
                                       linestyle='--', alpha=BENCHMARK_OPACITY, label="Benchmark")
    axes[0, 1].axhline(0, color='black', linewidth=REFERENCE_LINE_WIDTH, linestyle='-', alpha=REFERENCE_OPACITY)
    axes[0, 1].set_title("Cumulative Returns", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("Return (%)", fontsize=10)
    axes[0, 1].legend(loc="upper left", fontsize=9)
    axes[0, 1].grid(True, alpha=GRID_OPACITY, linestyle='--', linewidth=GRID_LINE_WIDTH)

    # Bottom left: Active Return
    active_return_pct = df["active_return"] * 100
    axes[1, 0].fill_between(df.index, active_return_pct, 0, where=(active_return_pct >= 0),
                            color=POSITIVE_COLOR, alpha=FILL_OPACITY, interpolate=True)
    axes[1, 0].fill_between(df.index, active_return_pct, 0, where=(active_return_pct < 0),
                            color=NEGATIVE_COLOR, alpha=FILL_OPACITY, interpolate=True)
    active_return_pct.plot(ax=axes[1, 0], color=ACTIVE_COLOR, linewidth=PORTFOLIO_LINE_WIDTH, alpha=PORTFOLIO_OPACITY)
    axes[1, 0].axhline(0, color='black', linewidth=REFERENCE_LINE_WIDTH, linestyle='-', alpha=REFERENCE_OPACITY)
    axes[1, 0].set_title("Active Return", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Active Return (%)", fontsize=10)
    axes[1, 0].grid(True, alpha=GRID_OPACITY, linestyle='--', linewidth=GRID_LINE_WIDTH)

    # Bottom right: Drawdown
    portfolio_dd = (df["portfolio_value"] / df["portfolio_value"].cummax() - 1) * 100
    benchmark_dd = (df["benchmark_value"] / df["benchmark_value"].cummax() - 1) * 100
    axes[1, 1].fill_between(df.index, portfolio_dd, 0, alpha=FILL_OPACITY, color=PORTFOLIO_COLOR, label="Portfolio")
    axes[1, 1].fill_between(df.index, benchmark_dd, 0, alpha=FILL_OPACITY, color=BENCHMARK_COLOR, label="Benchmark")
    portfolio_dd.plot(ax=axes[1, 1], color=PORTFOLIO_COLOR, linewidth=PORTFOLIO_LINE_WIDTH,
                     linestyle='-', alpha=PORTFOLIO_OPACITY)
    benchmark_dd.plot(ax=axes[1, 1], color=BENCHMARK_COLOR, linewidth=BENCHMARK_LINE_WIDTH,
                     linestyle='--', alpha=BENCHMARK_OPACITY)
    axes[1, 1].axhline(0, color='black', linewidth=REFERENCE_LINE_WIDTH, linestyle='-', alpha=REFERENCE_OPACITY)
    axes[1, 1].set_title("Drawdown", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("Drawdown (%)", fontsize=10)
    axes[1, 1].legend(loc="lower left", fontsize=9)
    axes[1, 1].grid(True, alpha=GRID_OPACITY, linestyle='--', linewidth=GRID_LINE_WIDTH)

    # Rotate x-axis labels for all subplots
    for ax in axes.flat:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=TICK_LABEL_FONT_SIZE)
        plt.setp(ax.yaxis.get_majorticklabels(), fontsize=TICK_LABEL_FONT_SIZE)
        ax.set_xlabel("Date", fontsize=10)

    plt.tight_layout()

    return fig, axes


def main() -> None:
    args = parse_args()

    # Read and prepare data
    logger.info(f"Loading data from {args.csv}")
    df = pd.read_csv(args.csv)
    if "date" not in df.columns:
        raise SystemExit("CSV missing 'date' column; run backtest.py with --output")
    df = df.set_index(pd.to_datetime(df["date"]))

    # Verify required columns
    required_cols = ["portfolio_value", "benchmark_value", "portfolio_return",
                     "benchmark_return", "active_return"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise SystemExit(
            f"CSV missing required columns: {', '.join(missing_cols)}\n"
            f"Make sure the CSV was generated by backtest.py"
        )

    # Validate minimum data quantity
    if len(df) < 2:
        raise SystemExit(
            f"Insufficient data: only {len(df)} row(s) found.\n"
            f"Need at least 2 data points for plotting."
        )

    if len(df) < 30:
        logger.warning(
            f"Limited data: only {len(df)} rows. "
            f"Charts may not be meaningful with < 30 data points."
        )

    # Validate data quality (check for all-NaN columns)
    all_nan_cols = [col for col in required_cols if df[col].isna().all()]
    if all_nan_cols:
        raise SystemExit(
            f"Columns contain no valid data: {', '.join(all_nan_cols)}\n"
            f"Check the backtest configuration and ensure tickers have data for the date range."
        )

    # Check for excessive missing data
    for col in required_cols:
        nan_pct = df[col].isna().sum() / len(df)
        if nan_pct > 0.5:
            logger.warning(
                f"Column '{col}' has {nan_pct:.1%} missing values. "
                f"Charts may be incomplete."
            )

    logger.info(f"Data validated: {len(df)} rows, date range: {df.index[0].date()} to {df.index[-1].date()}")

    if args.dashboard:
        # Create single dashboard
        fig, axes = create_dashboard(df, args.style)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{args.output}_dashboard.png", dpi=args.dpi, bbox_inches="tight")
            logger.info(f"Saved dashboard to {args.output}_dashboard.png")
        else:
            plt.show()
    else:
        # Create individual plots
        figures = []

        # Value plot
        fig_value, _ = create_value_plot(df, args.style)
        figures.append(("value", fig_value))

        # Returns plot
        fig_returns, _ = create_returns_plot(df, args.style)
        figures.append(("returns", fig_returns))

        # Active return plot
        fig_active, _ = create_active_return_plot(df, args.style)
        figures.append(("active", fig_active))

        # Drawdown plot
        fig_drawdown, _ = create_drawdown_plot(df, args.style)
        figures.append(("drawdown", fig_drawdown))

        # Rolling Sharpe ratio plot (if columns exist)
        fig_sharpe, ax_sharpe = create_rolling_sharpe_plot(df, args.style)
        if fig_sharpe is not None:
            figures.append(("rolling_sharpe", fig_sharpe))

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            saved_files = []
            for name, fig in figures:
                filepath = f"{args.output}_{name}.png"
                fig.savefig(filepath, dpi=args.dpi, bbox_inches="tight")
                saved_files.append(filepath)
                plt.close(fig)

            logger.info(f"Saved {len(saved_files)} plots:")
            for filepath in saved_files:
                logger.info(f"  - {filepath}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
