"""E2E test: basic backtest flow with two tickers."""

from __future__ import annotations

import re
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from playwright.sync_api import expect

from e2e.fixtures import get_basic_backtest_expected
from e2e.helpers import with_page


def _run_test(page) -> None:
    """Inner test logic."""
    # Wait for the sidebar form to render
    page.wait_for_selector("text=Portfolio Composition")

    # Fill ticker 1 (AAPL, weight 0.6)
    page.get_by_role("textbox", name="Ticker 1").fill("AAPL")
    page.get_by_role("spinbutton", name="Weight 1").fill("0.6")

    # Fill ticker 2 (MSFT, weight 0.4)
    page.get_by_role("textbox", name="Ticker 2").fill("MSFT")
    page.get_by_role("spinbutton", name="Weight 2").fill("0.4")

    # Fill benchmark
    page.get_by_role("textbox", name="Benchmark 1").fill("SPY")

    # Set date range — Streamlit date inputs are custom textboxes with
    # aria-label "Select a date."; we target by role and pick first/second.
    page.get_by_role("textbox", name="Select a date.").nth(0).fill("2020-01-01")
    page.get_by_role("textbox", name="Select a date.").nth(1).fill("2020-12-31")

    # Click Run Backtest
    page.get_by_role("button", name=re.compile("Run Backtest")).click()

    # Wait for results
    page.wait_for_selector("text=Summary Statistics", timeout=30000)

    # Assert expected metrics appear with exact computed values
    expected = get_basic_backtest_expected()
    expect(page.get_by_text(expected["ending_value"]).first).to_be_visible()
    expect(page.get_by_text(expected["cagr"]).first).to_be_visible()


def test_basic_backtest() -> None:
    """Run a two-ticker backtest and verify ending value and CAGR."""
    with_page(_run_test, test_name="test_basic_backtest")


if __name__ == "__main__":
    test_basic_backtest()
    print("test_basic_backtest PASSED")
