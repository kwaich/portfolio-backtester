"""E2E test: enable DCA and assert DCA-specific results."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from playwright.sync_api import sync_playwright, expect

from e2e.server import streamlit_server


def test_dca_strategy() -> None:
    with streamlit_server() as url:
        with sync_playwright() as p:
            with p.chromium.launch(headless=True) as browser:
                page = browser.new_page()
                page.set_viewport_size({"width": 1280, "height": 720})
                page.goto(url)

                page.wait_for_selector("text=Portfolio Composition", timeout=10000)

                # Fill tickers and weights
                ticker1 = page.get_by_role("textbox", name="Ticker 1")
                ticker1.fill("AAPL")
                ticker1.press("Tab")
                page.get_by_role("spinbutton", name="Weight 1").fill("1.0")

                # Select DCA frequency (Monthly)
                dca_select = page.locator("[data-testid='stSelectbox']").nth(2)
                dca_select.click()
                page.get_by_role("option", name="Monthly").click()

                # Click Run Backtest to submit form and trigger rerun
                # (DCA amount input is conditionally rendered inside the form,
                # so it only appears after the first submission)
                page.get_by_role("button", name="Run Backtest").click()

                # Wait for results from first run
                page.wait_for_selector("text=Summary Statistics", timeout=30000)

                # Now fill the DCA amount that appeared after form submission
                dca_amount = page.get_by_role("spinbutton", name="DCA Contribution Amount ($)")
                dca_amount.fill("1000")

                # Click Run Backtest again with DCA amount set
                page.get_by_role("button", name="Run Backtest").click()

                # Wait for results and assert DCA-specific label
                page.wait_for_selector("text=Summary Statistics", timeout=30000)
                expect(page.get_by_text("Total Contributions").first).to_be_visible()


if __name__ == "__main__":
    test_dca_strategy()
    print("test_dca_strategy PASSED")
