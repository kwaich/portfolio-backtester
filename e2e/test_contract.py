"""Contract test: verify e2e fixture expected values match actual backtest engine.

This test ensures that when the backtest engine changes, the hardcoded expected
values in fixtures.py are updated to match.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np  # noqa: E402

from e2e.fixtures import create_mock_prices, get_basic_backtest_expected  # noqa: E402
from backtest import compute_metrics, summarize  # noqa: E402


def test_fixture_values_match_engine() -> None:
    """Compute metrics from mock prices and assert they match fixture expectations."""
    prices = create_mock_prices(
        tickers=["AAPL", "MSFT"],
        start="2020-01-01",
        end="2020-12-31",
        start_price=100.0,
        end_price=150.0,
    )

    weights = np.array([0.6, 0.4])
    capital = 100_000.0

    # Use SPY as benchmark (also in mock prices)
    benchmark_prices = prices["SPY"] if "SPY" in prices.columns else prices.iloc[:, 0]

    table = compute_metrics(
        prices,
        weights,
        benchmark_prices,
        capital,
        rebalance_freq=None,
        dca_amount=None,
        dca_freq=None,
    )

    results = summarize(table["portfolio_value"], capital)
    expected = get_basic_backtest_expected()

    # Format using same logic as UI
    ending_value = f"${results['ending_value']:,.2f}"
    cagr = f"{results['cagr']:.2%}"

    assert ending_value == expected["ending_value"], (
        f"Fixture ending_value mismatch: expected {expected['ending_value']}, "
        f"got {ending_value}. Update fixtures.py or investigate engine change."
    )
    assert cagr == expected["cagr"], (
        f"Fixture CAGR mismatch: expected {expected['cagr']}, "
        f"got {cagr}. Update fixtures.py or investigate engine change."
    )


if __name__ == "__main__":
    test_fixture_values_match_engine()
    print("test_contract PASSED")
