"""E2E test: select a portfolio preset and run backtest."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from playwright.sync_api import expect

from e2e.helpers import with_page

SELECTED_PRESET = "Default UK ETFs"
EXPECTED_TICKER1 = "VDCP.L"


def _run_test(page) -> None:
    """Inner test logic."""
    page.wait_for_selector("text=Example Portfolio", timeout=10000)

    # Select preset by finding the selectbox that contains the label text
    preset_select = page.locator('[data-testid="stSelectbox"]').filter(
        has_text="Example Portfolio"
    )
    preset_select.click()
    page.get_by_role("option", name=SELECTED_PRESET).click()

    # Wait for the app to rerun and populate tickers
    ticker1 = page.get_by_role("textbox", name="Ticker 1")
    expect(ticker1).not_to_have_value("", timeout=5000)
    expect(ticker1).to_have_value(EXPECTED_TICKER1)

    # Click Run Backtest
    page.get_by_role("button", name="Run Backtest").click()

    # Wait for results and assert Ending Value is visible
    page.wait_for_selector("text=Summary Statistics", timeout=30000)
    expect(page.get_by_text("Ending Value").first).to_be_visible()


def test_preset_flow() -> None:
    with_page(_run_test, test_name="test_preset_flow")


if __name__ == "__main__":
    test_preset_flow()
    print("test_preset_flow PASSED")
