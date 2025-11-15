"""Portfolio and date range presets for the ETF Backtester.

This module provides pre-configured portfolios and date range shortcuts
to make backtesting easier for common scenarios.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List


def get_portfolio_presets() -> Dict[str, Dict[str, any]]:
    """Return dictionary of predefined portfolio configurations.
    
    Returns:
        Dictionary mapping portfolio names to their configurations.
        Each configuration contains:
        - tickers: List of ticker symbols
        - weights: List of portfolio weights
        - benchmark: Benchmark ticker symbol
    
    Examples:
        >>> presets = get_portfolio_presets()
        >>> tech = presets["Tech Giants"]
        >>> tech["tickers"]
        ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    """
    return {
        "Custom (Manual Entry)": {
            "tickers": [],
            "weights": [],
            "benchmark": "VWRA.L"
        },
        "Default UK ETFs": {
            "tickers": ["VDCP.L", "VHYD.L"],
            "weights": [0.5, 0.5],
            "benchmark": "VWRA.L"
        },
        "60/40 US Stocks/Bonds": {
            "tickers": ["VOO", "BND"],
            "weights": [0.6, 0.4],
            "benchmark": "SPY"
        },
        "Tech Giants": {
            "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "weights": [0.25, 0.25, 0.25, 0.25],
            "benchmark": "QQQ"
        },
        "Dividend Aristocrats": {
            "tickers": ["JNJ", "PG", "KO", "PEP"],
            "weights": [0.25, 0.25, 0.25, 0.25],
            "benchmark": "SPY"
        },
        "Global Diversified": {
            "tickers": ["VTI", "VXUS", "BND"],
            "weights": [0.5, 0.3, 0.2],
            "benchmark": "VT"
        }
    }


def get_date_presets() -> Dict[str, datetime]:
    """Return dictionary of predefined date ranges.
    
    Returns:
        Dictionary mapping preset names to start dates.
        End date is always today.
    
    Available presets:
        - "1Y": 1 year ago
        - "3Y": 3 years ago
        - "5Y": 5 years ago
        - "10Y": 10 years ago
        - "YTD": Start of current year
        - "Max": January 1, 2010
    
    Examples:
        >>> presets = get_date_presets()
        >>> start_date = presets["5Y"]
        >>> # Returns datetime 5 years ago from today
    """
    today = datetime.today()
    
    return {
        "1Y": today - timedelta(days=365),
        "3Y": today - timedelta(days=365 * 3),
        "5Y": today - timedelta(days=365 * 5),
        "10Y": today - timedelta(days=365 * 10),
        "YTD": datetime(today.year, 1, 1),
        "Max": datetime(2010, 1, 1)
    }


def get_portfolio_preset_names() -> List[str]:
    """Get list of all available portfolio preset names.
    
    Returns:
        List of portfolio preset names
    """
    return list(get_portfolio_presets().keys())


def get_date_preset_names() -> List[str]:
    """Get list of all available date preset names.
    
    Returns:
        List of date preset names
    """
    return list(get_date_presets().keys())
