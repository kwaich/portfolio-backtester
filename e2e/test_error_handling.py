"""E2E test: invalid ticker shows an error message."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from playwright.sync_api import expect  # noqa: E402

from e2e.helpers import with_page  # noqa: E402


def _test_too_long_ticker(page) -> None:
    """Ticker >10 chars fails in validate_tickers."""
    page.wait_for_selector("text=Portfolio Composition", timeout=10000)
    ticker1 = page.get_by_role("textbox", name="Ticker 1")
    ticker1.fill("INVALID123456")
    ticker1.press("Tab")
    page.get_by_role("button", name="Run Backtest").click()
    page.wait_for_selector("text=Ticker Validation Failed", timeout=30000)
    expect(page.get_by_text("Ticker Validation Failed")).to_be_visible()


def _test_all_numbers_ticker(page) -> None:
    """All-numbers ticker fails in validate_tickers."""
    page.wait_for_selector("text=Portfolio Composition", timeout=10000)
    ticker1 = page.get_by_role("textbox", name="Ticker 1")
    ticker1.fill("12345")
    ticker1.press("Tab")
    page.get_by_role("button", name="Run Backtest").click()
    page.wait_for_selector("text=Ticker Validation Failed", timeout=30000)
    expect(page.get_by_text("Ticker Validation Failed")).to_be_visible()


def _test_invalid_chars_ticker(page) -> None:
    """Ticker with invalid characters fails in validate_tickers."""
    page.wait_for_selector("text=Portfolio Composition", timeout=10000)
    ticker1 = page.get_by_role("textbox", name="Ticker 1")
    ticker1.fill("AAPL!")
    ticker1.press("Tab")
    page.get_by_role("button", name="Run Backtest").click()
    page.wait_for_selector("text=Ticker Validation Failed", timeout=30000)
    expect(page.get_by_text("Ticker Validation Failed")).to_be_visible()


def test_error_handling() -> None:
    """Run all error handling sub-tests."""
    with_page(_test_too_long_ticker, test_name="test_error_too_long")
    with_page(_test_all_numbers_ticker, test_name="test_error_all_numbers")
    with_page(_test_invalid_chars_ticker, test_name="test_error_invalid_chars")


if __name__ == "__main__":
    test_error_handling()
    print("test_error_handling PASSED")
