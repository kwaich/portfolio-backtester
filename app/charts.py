"""Chart generation functions for the ETF Backtester.

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
    PORTFOLIO_MARKER,
    BENCHMARK_MARKERS,
    POSITIVE_COLOR,
    NEGATIVE_COLOR,
    ROLLING_WINDOWS,
    DASHBOARD_HEIGHT,
    CHART_HEIGHT
)


def calculate_drawdown(series: pd.Series) -> pd.Series:
    """Calculate drawdown from value series.
    
    Drawdown is the peak-to-trough decline during a specific period.
    
    Args:
        series: Time series of values (e.g., portfolio value)
    
    Returns:
        Series of drawdown percentages
    
    Examples:
        >>> values = pd.Series([100, 110, 105, 120, 115])
        >>> dd = calculate_drawdown(values)
        >>> # Returns negative percentages showing decline from peak
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
    
    # Create 2x2 grid of charts
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
            line=dict(color=PORTFOLIO_COLOR, width=2),
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
                    width=2,
                    dash=BENCHMARK_DASH_STYLES[idx % len(BENCHMARK_DASH_STYLES)]
                ),
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
            line=dict(color=PORTFOLIO_COLOR, width=2),
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
                    width=2,
                    dash=BENCHMARK_DASH_STYLES[idx % len(BENCHMARK_DASH_STYLES)]
                ),
                hovertemplate=f'<b>{bench_name}</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>',
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
            fillcolor=f'rgba({int(POSITIVE_COLOR[1:3], 16)}, {int(POSITIVE_COLOR[3:5], 16)}, {int(POSITIVE_COLOR[5:7], 16)}, 0.3)',
            line=dict(color=POSITIVE_COLOR, width=1.5),
            hovertemplate='<b>Outperformance</b><br>Date: %{x}<br>Difference: %{y:.2f}%<extra></extra>',
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
            fillcolor=f'rgba({int(NEGATIVE_COLOR[1:3], 16)}, {int(NEGATIVE_COLOR[3:5], 16)}, {int(NEGATIVE_COLOR[5:7], 16)}, 0.3)',
            line=dict(color=NEGATIVE_COLOR, width=1.5),
            hovertemplate='<b>Underperformance</b><br>Date: %{x}<br>Difference: %{y:.2f}%<extra></extra>',
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
            line=dict(color=PORTFOLIO_COLOR, width=1.5),
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
                line=dict(
                    color=BENCHMARK_COLORS[idx % len(BENCHMARK_COLORS)],
                    width=1.5,
                    dash=BENCHMARK_DASH_STYLES[idx % len(BENCHMARK_DASH_STYLES)]
                ),
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

    # Set y-axis labels and scale
    # Log scale for value chart (always positive)
    if log_scale:
        fig.update_yaxes(title_text="Value ($) - Log Scale", type="log", row=1, col=1)
    else:
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)

    # Returns chart stays linear (can be negative)
    fig.update_yaxes(title_text="Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Active Return (%)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=DASHBOARD_HEIGHT,
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
                line=dict(width=2),
                hovertemplate=f'<b>Portfolio {window}D</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>'
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
                        width=1.5,
                        dash=BENCHMARK_DASH_STYLES[idx % len(BENCHMARK_DASH_STYLES)]
                    ),
                    hovertemplate=f'<b>{bench_name} {window}D</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>'
                )
            )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
    
    # Update layout
    fig.update_layout(
        title="Rolling Returns (30, 90, 180 Day Periods)",
        xaxis_title="Date",
        yaxis_title="Rolling Return (%)",
        height=CHART_HEIGHT,
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
            line=dict(color=PORTFOLIO_COLOR, width=2),
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
                    width=2,
                    dash=BENCHMARK_DASH_STYLES[idx % len(BENCHMARK_DASH_STYLES)]
                ),
                hovertemplate=f'<b>{bench_name}</b><br>Date: %{{x}}<br>12M Sharpe: %{{y:.2f}}<extra></extra>'
            )
        )

    # Add zero line for reference
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)

    # Add reference lines for "good" Sharpe ratios
    fig.add_hline(y=1, line_dash="dash", line_color="green", opacity=0.2,
                  annotation_text="Sharpe = 1", annotation_position="right")
    fig.add_hline(y=2, line_dash="dash", line_color="darkgreen", opacity=0.2,
                  annotation_text="Sharpe = 2", annotation_position="right")

    # Update layout
    fig.update_layout(
        title="Rolling 12-Month Sharpe Ratio",
        xaxis_title="Date",
        yaxis_title="Rolling 12-Month Sharpe Ratio",
        height=CHART_HEIGHT,
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

    return fig
