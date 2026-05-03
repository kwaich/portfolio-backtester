"""Chart generation functions for the Portfolio Backtester.

This module provides Plotly chart generation for:
- 2x2 dashboard with portfolio value, returns, active return, and drawdown
- Rolling returns analysis
- Drawdown calculations
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import (
    PORTFOLIO_COLOR,
    BENCHMARK_COLORS,
    BENCHMARK_DASH_STYLES,
    POSITIVE_COLOR,
    NEGATIVE_COLOR,
    ROLLING_WINDOWS,
    DASHBOARD_HEIGHT,
    CHART_HEIGHT,
    # Visual hierarchy
    PORTFOLIO_LINE_WIDTH,
    BENCHMARK_LINE_WIDTH,
    REFERENCE_LINE_WIDTH,
    PORTFOLIO_OPACITY,
    BENCHMARK_OPACITY,
    FILL_OPACITY,
    REFERENCE_LINE_OPACITY,
    SUBPLOT_TITLE_FONT_SIZE,
    AXIS_TITLE_FONT_SIZE,
    LEGEND_FONT_SIZE,
    SUBPLOT_VERTICAL_SPACING,
    SUBPLOT_HORIZONTAL_SPACING
)

from app.design_system import COLORS


def _rgba(color_hex: str, opacity: float) -> str:
    color_hex = color_hex.strip()
    return (
        f"rgba({int(color_hex[1:3], 16)}, {int(color_hex[3:5], 16)}, "
        f"{int(color_hex[5:7], 16)}, {opacity})"
    )


def _apply_fintech_layout(fig: go.Figure) -> go.Figure:
    """Apply fintech design system styling to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",  # transparent background
        plot_bgcolor=COLORS["bg_card"],   # white card background for plot area
        font=dict(
            family="Inter, sans-serif",
            color=COLORS["primary_text"],
            size=12,
        ),
        title_font=dict(
            family="Outfit, sans-serif",
            size=16,
            color=COLORS["primary_text"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=COLORS["muted"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=COLORS["bg_card"],
            font=dict(family="Inter, sans-serif", size=13, color=COLORS["primary_text"]),
            bordercolor=COLORS["border"],
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor=COLORS["grid"],
        zeroline=False,
        tickfont=dict(size=11, color=COLORS["muted"]),
        title_font=dict(size=12, color=COLORS["muted"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor=COLORS["grid"],
        zeroline=False,
        tickfont=dict(size=11, color=COLORS["muted"]),
        title_font=dict(size=12, color=COLORS["muted"]),
    )

    return fig


def calculate_drawdown(series: pd.Series) -> pd.Series:
    """Calculate drawdown from value series.

    Drawdown is the peak-to-trough decline during a specific period.

    Args:
        series: Time series of values (e.g., portfolio value)

    Returns:
        pd.Series: Drawdown percentages (negative values) with the same
            DatetimeIndex as the input. Values range from 0 (at peak) to
            negative percentages (in drawdown).

    Examples:
        >>> values = pd.Series([100, 110, 105, 120, 115])
        >>> dd = calculate_drawdown(values)
        >>> dd.iloc[-1]
        -4.166666666666666
    """
    cummax = series.expanding().max()
    drawdown = ((series - cummax) / cummax) * 100
    return drawdown


def create_main_dashboard(
    results: pd.DataFrame,
    all_benchmark_results: Dict[str, pd.DataFrame],
    benchmarks: List[str],
    log_scale: bool = False
) -> go.Figure:
    """Create 2x2 dashboard with all main charts.

    Creates a comprehensive dashboard showing:
    - Portfolio vs Benchmark Value (top-left)
    - Cumulative Returns (top-right)
    - Active Return (bottom-left)
    - Drawdown Over Time (bottom-right)

    Args:
        results: DataFrame with portfolio_value, portfolio_return, benchmark_value, benchmark_return
        all_benchmark_results: Dict mapping benchmark names to their result DataFrames
        benchmarks: List of benchmark ticker symbols
        log_scale: If True, use logarithmic scale for value chart only (default: False)

    Returns:
        Plotly Figure object with 2x2 subplots

    Note:
        Log scale is only applied to the Portfolio vs Benchmark Value chart (top-left)
        since it contains only positive values. Other charts remain linear as they
        can contain negative values (returns, active return, drawdown).
    """
    # Calculate metrics for charts
    active_return = (results['portfolio_return'] - results['benchmark_return']) * 100
    portfolio_drawdown = calculate_drawdown(results['portfolio_value'])

    # Create 2x2 grid of charts with improved spacing
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Portfolio vs Benchmark Value',
            'Portfolio vs Benchmark Cumulative Returns',
            'Active Return (Portfolio - Benchmark)',
            'Portfolio Drawdown Over Time'
        ),
        vertical_spacing=SUBPLOT_VERTICAL_SPACING,
        horizontal_spacing=SUBPLOT_HORIZONTAL_SPACING
    )

    # Chart 1: Portfolio vs Benchmark Value (top-left)
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['portfolio_value'],
            name='Portfolio',
            line=dict(color=PORTFOLIO_COLOR, width=PORTFOLIO_LINE_WIDTH),
            opacity=PORTFOLIO_OPACITY,
            hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add all benchmarks to the value chart
    for idx, bench_name in enumerate(benchmarks):
        bench_result = all_benchmark_results[bench_name]
        fig.add_trace(
            go.Scatter(
                x=bench_result.index,
                y=bench_result['benchmark_value'],
                name=bench_name,
                line=dict(
                    color=BENCHMARK_COLORS[idx % len(BENCHMARK_COLORS)],
                    width=BENCHMARK_LINE_WIDTH,
                    dash=BENCHMARK_DASH_STYLES[idx % len(BENCHMARK_DASH_STYLES)]
                ),
                opacity=BENCHMARK_OPACITY,
                hovertemplate=(
                    f'<b>{bench_name}</b><br>Date: %{{x}}<br>Value: $%{{y:,.2f}}<extra></extra>'
                )
            ),
            row=1, col=1
        )

    # Chart 2: Cumulative Returns (top-right)
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['portfolio_return'] * 100,
            name='Portfolio Return',
            line=dict(color=PORTFOLIO_COLOR, width=PORTFOLIO_LINE_WIDTH),
            opacity=PORTFOLIO_OPACITY,
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
                line=dict(
                    color=BENCHMARK_COLORS[idx % len(BENCHMARK_COLORS)],
                    width=BENCHMARK_LINE_WIDTH,
                    dash=BENCHMARK_DASH_STYLES[idx % len(BENCHMARK_DASH_STYLES)]
                ),
                opacity=BENCHMARK_OPACITY,
                hovertemplate=(
                    f'<b>{bench_name}</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>'
                ),
                showlegend=False
            ),
            row=1, col=2
        )

    # Chart 3: Active Return (bottom-left)
    # Split into positive and negative areas for better visual differentiation
    positive_active = active_return.copy()
    positive_active[active_return < 0] = 0
    negative_active = active_return.copy()
    negative_active[active_return >= 0] = 0

    # Add positive area (blue)
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=positive_active,
            name='Outperformance',
            fill='tozeroy',
            fillcolor=_rgba(POSITIVE_COLOR, FILL_OPACITY),
            line=dict(color=POSITIVE_COLOR, width=PORTFOLIO_LINE_WIDTH),
            opacity=PORTFOLIO_OPACITY,
            hovertemplate=(
                '<b>Outperformance</b><br>Date: %{x}<br>Difference: %{y:.2f}%<extra></extra>'
            ),
            showlegend=False
        ),
        row=2, col=1
    )

    # Add negative area (orange)
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=negative_active,
            name='Underperformance',
            fill='tozeroy',
            fillcolor=_rgba(NEGATIVE_COLOR, FILL_OPACITY),
            line=dict(color=NEGATIVE_COLOR, width=PORTFOLIO_LINE_WIDTH),
            opacity=PORTFOLIO_OPACITY,
            hovertemplate=(
                '<b>Underperformance</b><br>Date: %{x}<br>Difference: %{y:.2f}%<extra></extra>'
            ),
            showlegend=False
        ),
        row=2, col=1
    )

    fig.add_hline(y=0, line_dash="solid", line_color="black",
                  line_width=REFERENCE_LINE_WIDTH, opacity=REFERENCE_LINE_OPACITY, row=2, col=1)

    # Chart 4: Drawdown (bottom-right)
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=portfolio_drawdown,
            name='Portfolio DD',
            fill='tozeroy',
            fillcolor=_rgba(PORTFOLIO_COLOR, FILL_OPACITY),
            line=dict(color=PORTFOLIO_COLOR, width=PORTFOLIO_LINE_WIDTH),
            opacity=PORTFOLIO_OPACITY,
            hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )

    # Add all benchmarks drawdowns
    for idx, bench_name in enumerate(benchmarks):
        bench_result = all_benchmark_results[bench_name]
        bench_dd = calculate_drawdown(bench_result['benchmark_value'])

        fig.add_trace(
            go.Scatter(
                x=bench_result.index,
                y=bench_dd,
                name=f'{bench_name} DD',
                fill='tozeroy',
                fillcolor=_rgba(BENCHMARK_COLORS[idx % len(BENCHMARK_COLORS)], FILL_OPACITY),
                line=dict(
                    color=BENCHMARK_COLORS[idx % len(BENCHMARK_COLORS)],
                    width=BENCHMARK_LINE_WIDTH,
                    dash=BENCHMARK_DASH_STYLES[idx % len(BENCHMARK_DASH_STYLES)]
                ),
                opacity=BENCHMARK_OPACITY,
                hovertemplate=(
                    f'<b>{bench_name}</b><br>Date: %{{x}}<br>Drawdown: %{{y:.2f}}%<extra></extra>'
                ),
                showlegend=False
            ),
            row=2, col=2
        )

    fig.add_hline(y=0, line_dash="solid", line_color="black",
                  line_width=REFERENCE_LINE_WIDTH, opacity=REFERENCE_LINE_OPACITY, row=2, col=2)

    # Update axes labels with improved typography
    fig.update_xaxes(title_text="Date", title_font=dict(size=AXIS_TITLE_FONT_SIZE), row=1, col=1)
    fig.update_xaxes(title_text="Date", title_font=dict(size=AXIS_TITLE_FONT_SIZE), row=1, col=2)
    fig.update_xaxes(title_text="Date", title_font=dict(size=AXIS_TITLE_FONT_SIZE), row=2, col=1)
    fig.update_xaxes(title_text="Date", title_font=dict(size=AXIS_TITLE_FONT_SIZE), row=2, col=2)

    # Set y-axis labels and scale
    # Log scale for value chart (always positive)
    if log_scale:
        fig.update_yaxes(
            title_text="Value ($) - Log Scale",
            title_font=dict(size=AXIS_TITLE_FONT_SIZE),
            type="log", row=1, col=1
        )
    else:
        fig.update_yaxes(
            title_text="Value ($)",
            title_font=dict(size=AXIS_TITLE_FONT_SIZE), row=1, col=1
        )

    # Returns chart stays linear (can be negative)
    fig.update_yaxes(
        title_text="Return (%)", title_font=dict(size=AXIS_TITLE_FONT_SIZE), row=1, col=2
    )
    fig.update_yaxes(
        title_text="Active Return (%)", title_font=dict(size=AXIS_TITLE_FONT_SIZE), row=2, col=1
    )
    fig.update_yaxes(
        title_text="Drawdown (%)", title_font=dict(size=AXIS_TITLE_FONT_SIZE), row=2, col=2
    )

    # Update layout with improved visual hierarchy
    fig.update_layout(
        title=dict(
            text="Portfolio vs Benchmark Performance",
            font=dict(size=SUBPLOT_TITLE_FONT_SIZE)
        ),
        height=DASHBOARD_HEIGHT,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=LEGEND_FONT_SIZE)
        ),
        hovermode='x unified',
        template='plotly_white'
    )

    # Update subplot title annotations (preserve existing text and positioning)
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(
            size=14, color=COLORS["primary_text"], family="Outfit, sans-serif"
        )

    fig = _apply_fintech_layout(fig)
    return fig


def create_rolling_returns_chart(
    results: pd.DataFrame,
    all_benchmark_results: Dict[str, pd.DataFrame],
    benchmarks: List[str],
    windows: List[int] = None
) -> go.Figure:
    """Create rolling returns analysis chart.

    Shows portfolio and benchmark performance over different rolling windows
    to assess consistency and volatility.

    Args:
        results: DataFrame with portfolio_value
        all_benchmark_results: Dict mapping benchmark names to their result DataFrames
        benchmarks: List of benchmark ticker symbols
        windows: List of rolling window sizes in days (default: [30, 90, 180])

    Returns:
        Plotly Figure object showing rolling returns
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    # Create rolling returns chart
    fig = go.Figure()

    for window in windows:
        # Calculate rolling returns for portfolio
        portfolio_rolling = results['portfolio_value'].pct_change(window) * 100

        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=portfolio_rolling,
                name=f'Portfolio {window}D',
                line=dict(width=PORTFOLIO_LINE_WIDTH),
                opacity=PORTFOLIO_OPACITY,
                hovertemplate=(
                    f'<b>Portfolio {window}D</b><br>Date: %{{x}}<br>'
                    f'Return: %{{y:.2f}}%<extra></extra>'
                )
            )
        )

    # Add all benchmarks rolling returns for comparison
    for idx, bench_name in enumerate(benchmarks):
        bench_result = all_benchmark_results[bench_name]
        for window in windows:
            bench_rolling = bench_result['benchmark_value'].pct_change(window) * 100

            fig.add_trace(
                go.Scatter(
                    x=bench_result.index,
                    y=bench_rolling,
                    name=f'{bench_name} {window}D',
                    line=dict(
                        color=BENCHMARK_COLORS[idx % len(BENCHMARK_COLORS)],
                        width=BENCHMARK_LINE_WIDTH,
                        dash=BENCHMARK_DASH_STYLES[idx % len(BENCHMARK_DASH_STYLES)]
                    ),
                    opacity=BENCHMARK_OPACITY,
                    hovertemplate=(
                        f'<b>{bench_name} {window}D</b><br>Date: %{{x}}<br>'
                        f'Return: %{{y:.2f}}%<extra></extra>'
                    )
                )
            )

    # Add zero line with subtle styling
    fig.add_hline(y=0, line_dash="solid", line_color="black",
                  line_width=REFERENCE_LINE_WIDTH, opacity=REFERENCE_LINE_OPACITY)

    # Update layout with improved typography
    fig.update_layout(
        title=dict(
            text="Rolling Returns (30, 90, 180 Day Periods)",
            font=dict(size=SUBPLOT_TITLE_FONT_SIZE)
        ),
        xaxis_title=dict(text="Date", font=dict(size=AXIS_TITLE_FONT_SIZE)),
        yaxis_title=dict(text="Rolling Return (%)", font=dict(size=AXIS_TITLE_FONT_SIZE)),
        height=CHART_HEIGHT,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=LEGEND_FONT_SIZE)
        )
    )

    fig = _apply_fintech_layout(fig)
    return fig


def create_rolling_sharpe_chart(
    results: pd.DataFrame,
    all_benchmark_results: Dict[str, pd.DataFrame],
    benchmarks: List[str]
) -> go.Figure:
    """Create rolling 12-month Sharpe ratio chart.

    Shows how risk-adjusted performance evolves over time using a
    12-month (252 trading day) rolling window.

    Args:
        results: DataFrame with portfolio_rolling_sharpe_12m
        all_benchmark_results: Dict mapping benchmark names to their result DataFrames
        benchmarks: List of benchmark ticker symbols

    Returns:
        Plotly Figure object showing rolling Sharpe ratios
    """
    fig = go.Figure()

    # Add portfolio rolling Sharpe
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['portfolio_rolling_sharpe_12m'],
            name='Portfolio',
            line=dict(color=PORTFOLIO_COLOR, width=PORTFOLIO_LINE_WIDTH),
            opacity=PORTFOLIO_OPACITY,
            hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>12M Sharpe: %{y:.2f}<extra></extra>'
        )
    )

    # Add all benchmarks rolling Sharpe
    for idx, bench_name in enumerate(benchmarks):
        bench_result = all_benchmark_results[bench_name]

        fig.add_trace(
            go.Scatter(
                x=bench_result.index,
                y=bench_result['benchmark_rolling_sharpe_12m'],
                name=bench_name,
                line=dict(
                    color=BENCHMARK_COLORS[idx % len(BENCHMARK_COLORS)],
                    width=BENCHMARK_LINE_WIDTH,
                    dash=BENCHMARK_DASH_STYLES[idx % len(BENCHMARK_DASH_STYLES)]
                ),
                opacity=BENCHMARK_OPACITY,
                hovertemplate=(
                    f'<b>{bench_name}</b><br>Date: %{{x}}<br>12M Sharpe: %{{y:.2f}}<extra></extra>'
                )
            )
        )

    # Add zero line for reference (subtle)
    fig.add_hline(y=0, line_dash="solid", line_color="black",
                  line_width=REFERENCE_LINE_WIDTH, opacity=REFERENCE_LINE_OPACITY)

    # Add reference lines for "good" Sharpe ratios (subtle, colorblind-safe teal)
    fig.add_hline(y=1, line_dash="dash", line_color="#029E73",
                  line_width=REFERENCE_LINE_WIDTH, opacity=REFERENCE_LINE_OPACITY,
                  annotation_text="Sharpe = 1", annotation_position="right")
    fig.add_hline(y=2, line_dash="dash", line_color="#029E73",
                  line_width=REFERENCE_LINE_WIDTH, opacity=REFERENCE_LINE_OPACITY,
                  annotation_text="Sharpe = 2", annotation_position="right")

    # Update layout with improved typography
    fig.update_layout(
        title=dict(
            text="Rolling Sharpe Ratio (12M)",
            font=dict(size=SUBPLOT_TITLE_FONT_SIZE)
        ),
        xaxis_title=dict(text="Date", font=dict(size=AXIS_TITLE_FONT_SIZE)),
        yaxis_title=dict(
            text="Rolling 12-Month Sharpe Ratio", font=dict(size=AXIS_TITLE_FONT_SIZE)
        ),
        height=CHART_HEIGHT,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=LEGEND_FONT_SIZE)
        )
    )

    fig = _apply_fintech_layout(fig)
    return fig
