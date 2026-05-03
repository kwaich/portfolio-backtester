"""Deterministic mock data for e2e tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_mock_prices(
    tickers: list[str],
    start: str = "2020-01-01",
    end: str = "2020-12-31",
    start_price: float = 100.0,
    end_price: float = 150.0,
) -> pd.DataFrame:
    """Return a deterministic price DataFrame with a business-day index.

    Prices ramp linearly from start_price to end_price for each ticker.
    """
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    if n < 2:
        raise ValueError(f"Date range must span at least 2 business days, got {n}")
    data: dict[str, pd.Series] = {}
    for i, ticker in enumerate(tickers):
        # Slightly different slopes per ticker to avoid identical columns
        slope = (end_price - start_price) / (n - 1) * (1 + 0.1 * i)
        prices = start_price + slope * np.arange(n)
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


def get_basic_backtest_expected() -> dict[str, str]:
    """Pre-calculated expected values for the basic backtest scenario.

    Uses AAPL + MSFT, 2020-01-01 to 2020-12-31, weights 0.6 / 0.4,
    capital $100,000.
    """
    # Computed from create_mock_prices(["AAPL", "MSFT"], start="2020-01-01",
    # end="2020-12-31", start_price=100.0, end_price=150.0) with weights 0.6/0.4
    # and capital $100,000 (buy-and-hold, no benchmark).
    return {
        "ending_value": "$152,000.00",
        "cagr": "52.00%",
    }
