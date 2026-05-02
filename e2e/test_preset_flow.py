"""E2E test: select a portfolio preset and run backtest."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from playwright.sync_api import sync_playwright, expect

from e2e.server import streamlit_server


# Preset chosen for this test: "Default UK ETFs" (first non-Custom option).
# First ticker is "VDCP.L" and second is "VHYD.L".
SELECTED_PRESET = "Default UK ETFs"
EXPECTED_TICKER1 = "VDCP.L"


def test_preset_flow() -> None:
    with streamlit_server() as url:
        with sync_playwright() as p:
            with p.chromium.launch(headless=True) as browser:
                page = browser.new_page()
                page.set_viewport_size({"width": 1280, "height": 720})
                page.goto(url)

                # Wait for preset selector
                page.wait_for_selector("text=Example Portfolio", timeout=10000)

                # Open the preset dropdown and select a known preset
                preset_select = page.locator("[data-testid='stSelectbox']").first
                preset_select.click()
                page.get_by_role("option", name=SELECTED_PRESET).click()

                # Wait for the app to rerun and populate tickers
                ticker1 = page.get_by_role("textbox", name="Ticker 1")
                expect(ticker1).not_to_have_value("", timeout=5000)

                # Verify the expected ticker value for the selected preset
                expect(ticker1).to_have_value(EXPECTED_TICKER1)

                # Click Run Backtest
                run_button = page.get_by_role("button", name="Run Backtest")
                run_button.click()

                # Wait for results and assert Ending Value is visible
                page.wait_for_selector("text=Summary Statistics", timeout=30000)
                expect(page.get_by_text("Ending Value").first).to_be_visible()


if __name__ == "__main__":
    test_preset_flow()
    print("test_preset_flow PASSED")
