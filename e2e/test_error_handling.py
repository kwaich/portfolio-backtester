"""E2E test: invalid ticker shows an error message."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from playwright.sync_api import sync_playwright, expect

from e2e.server import streamlit_server


def test_error_handling() -> None:
    with streamlit_server() as url:
        with sync_playwright() as p:
            with p.chromium.launch(headless=True) as browser:
                page = browser.new_page()
                page.set_viewport_size({"width": 1280, "height": 720})
                page.goto(url)

                page.wait_for_selector("text=Portfolio Composition", timeout=10000)

                # Fill an invalid ticker (too long, >10 chars)
                ticker1 = page.get_by_role("textbox", name="Ticker 1")
                ticker1.fill("INVALID123456")
                ticker1.press("Tab")

                # Click Run Backtest
                page.get_by_role("button", name="Run Backtest").click()

                # Wait for error message in main content area
                page.wait_for_selector("text=Ticker Validation Failed", timeout=30000)
                expect(page.get_by_text("Ticker Validation Failed")).to_be_visible()


if __name__ == "__main__":
    test_error_handling()
    print("test_error_handling PASSED")
