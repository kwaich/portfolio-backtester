"""Wrapper entry point for e2e tests.

Injects a MockRepository with deterministic prices before starting the
real Streamlit application.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on path so 'app' and 'backtest' imports resolve
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from app.data_repository import MockRepository, set_repository
from app.main import main

# Import fixtures to build deterministic mock data
from e2e.fixtures import create_mock_prices

# Build mock prices for the tickers used in e2e tests
_mock_prices = create_mock_prices(
    tickers=["AAPL", "MSFT", "SPY", "VWRA.L", "VDCP.L", "VHYD.L"],
    start="2018-01-01",
    end="2024-12-31",
    start_price=100.0,
    end_price=150.0,
)

# Inject mock repository before Streamlit starts
set_repository(
    MockRepository(
        prices=_mock_prices,
        search_results=[
            ("AAPL", "Apple Inc."),
            ("MSFT", "Microsoft Corp."),
            ("SPY", "SPDR S&P 500 ETF Trust"),
        ],
        ticker_names={
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corp.",
            "SPY": "SPDR S&P 500 ETF Trust",
            "VWRA.L": "Vanguard FTSE All-World UCITS ETF",
            "VDCP.L": "Vanguard Dividend Appreciation ETF",
            "VHYD.L": "Vanguard High Dividend Yield ETF",
        },
    )
)

if __name__ == "__main__":
    main()
