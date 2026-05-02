"""E2E test: basic backtest flow with two tickers."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Ensure e2e package is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from playwright.sync_api import sync_playwright, expect

from e2e.fixtures import get_basic_backtest_expected
from e2e.server import streamlit_server


def test_basic_backtest() -> None:
    """Run a two-ticker backtest and verify ending value and CAGR."""
    with streamlit_server() as url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.set_viewport_size({"width": 1280, "height": 720})
                page.goto(url)

                # Wait for the sidebar form to render
                page.wait_for_selector("text=Portfolio Composition")

                # Fill ticker 1 (AAPL, weight 0.6)
                # Streamlit text inputs are identified by their label
                ticker1 = page.get_by_role("textbox", name="Ticker 1")
                ticker1.fill("AAPL")

                weight1 = page.get_by_role("spinbutton", name="Weight 1")
                weight1.fill("0.6")

                # Fill ticker 2 (MSFT, weight 0.4)
                ticker2 = page.get_by_role("textbox", name="Ticker 2")
                ticker2.fill("MSFT")

                weight2 = page.get_by_role("spinbutton", name="Weight 2")
                weight2.fill("0.4")

                # Fill benchmark
                bench1 = page.get_by_role("textbox", name="Benchmark 1")
                bench1.fill("SPY")

                # Set date range
                # Streamlit date inputs share the same aria-label, so we rely on
                # DOM order (nth). This is fragile if the page adds extra date fields.
                start_date = page.get_by_role("textbox", name="Select a date.").nth(0)
                start_date.fill("2020-01-01")

                end_date = page.get_by_role("textbox", name="Select a date.").nth(1)
                end_date.fill("2020-12-31")

                # Click Run Backtest
                run_button = page.get_by_role("button", name=re.compile("Run Backtest"))
                run_button.click()

                # Wait for results
                page.wait_for_selector("text=Summary Statistics", timeout=30000)

                # Assert expected metrics appear with exact computed values
                expected = get_basic_backtest_expected()
                expect(page.get_by_text(expected["ending_value"], exact=True)).to_be_visible()
                expect(page.get_by_text(expected["cagr"], exact=True)).to_be_visible()
            finally:
                browser.close()


if __name__ == "__main__":
    test_basic_backtest()
    print("test_basic_backtest PASSED")
