"""Shared helpers for e2e tests."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable

from playwright.sync_api import sync_playwright, Page

# Ensure repo root is on path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from e2e.server import streamlit_server


def _screenshot_on_failure(page: Page, test_name: str) -> None:
    """Capture screenshot on test failure."""
    screenshot_dir = Path("e2e/screenshots")
    screenshot_dir.mkdir(exist_ok=True)
    path = screenshot_dir / f"{test_name}_failure.png"
    try:
        page.screenshot(path=str(path), full_page=True)
        print(f"Screenshot saved: {path}")
    except Exception as e:
        print(f"Failed to capture screenshot: {e}")


def with_page(test_fn: Callable[[Page], None], test_name: str = "test") -> None:
    """Run a test function with a fully initialized Playwright page.

    Handles Streamlit server lifecycle, browser launch, viewport setup,
    and screenshot capture on failure.
    """
    with streamlit_server() as url:
        with sync_playwright() as p:
            with p.chromium.launch(headless=True) as browser:
                page = browser.new_page()
                page.set_viewport_size({"width": 1280, "height": 720})
                page.goto(url)
                try:
                    test_fn(page)
                except Exception:
                    _screenshot_on_failure(page, test_name)
                    raise
