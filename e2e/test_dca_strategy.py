"""E2E test: enable DCA and assert DCA-specific results."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from playwright.sync_api import expect

from e2e.helpers import with_page


def _run_test(page) -> None:
    """Inner test logic."""
    page.wait_for_selector("text=Portfolio Composition", timeout=10000)

    # Fill tickers and weights
    ticker1 = page.get_by_role("textbox", name="Ticker 1")
    ticker1.fill("AAPL")
    ticker1.press("Tab")
    page.get_by_role("spinbutton", name="Weight 1").fill("1.0")

    # Select DCA frequency by finding the selectbox that contains the label text
    dca_select = page.locator('[data-testid="stSelectbox"]').filter(
        has_text="DCA Contribution Frequency"
    )
    dca_select.click()
    page.get_by_role("option", name="Monthly").click()

    # Click Run Backtest to submit form and trigger rerun
    page.get_by_role("button", name="Run Backtest").click()

    # Wait for the DCA amount field to appear after form rerender
    dca_amount = page.get_by_role("spinbutton", name="DCA Contribution Amount ($)")
    dca_amount.wait_for(timeout=10000)
    dca_amount.fill("1000")

    # Click Run Backtest again with DCA amount set
    page.get_by_role("button", name="Run Backtest").click()

    # Wait for results and assert DCA-specific labels
    page.wait_for_selector("text=Summary Statistics", timeout=30000)
    expect(page.get_by_text("Total Contributions").first).to_be_visible()
    expect(page.get_by_text("IRR").first).to_be_visible()


def test_dca_strategy() -> None:
    with_page(_run_test, test_name="test_dca_strategy")


if __name__ == "__main__":
    test_dca_strategy()
    print("test_dca_strategy PASSED")
