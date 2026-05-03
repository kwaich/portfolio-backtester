"""E2E test: enable rebalancing and verify it produces results."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from playwright.sync_api import expect  # noqa: E402

from e2e.helpers import with_page  # noqa: E402


def _run_test(page) -> None:
    """Inner test logic."""
    page.wait_for_selector("text=Portfolio Composition", timeout=10000)

    # Fill single ticker
    page.get_by_role("textbox", name="Ticker 1").fill("AAPL")
    page.get_by_role("spinbutton", name="Weight 1").fill("1.0")

    # Select rebalancing frequency by finding the selectbox that contains the label text
    rebalance_select = page.locator('[data-testid="stSelectbox"]').filter(
        has_text="Rebalancing Frequency"
    )
    rebalance_select.click()
    page.get_by_role("option", name="Quarterly").click()

    # Click Run Backtest
    page.get_by_role("button", name="Run Backtest").click()

    # Wait for results
    page.wait_for_selector("text=Summary Statistics", timeout=30000)

    # Assert basic metrics are visible
    expect(page.get_by_text("Ending Value").first).to_be_visible()
    expect(page.get_by_text("CAGR").first).to_be_visible()


def test_rebalancing() -> None:
    with_page(_run_test, test_name="test_rebalancing")


if __name__ == "__main__":
    test_rebalancing()
    print("test_rebalancing PASSED")
