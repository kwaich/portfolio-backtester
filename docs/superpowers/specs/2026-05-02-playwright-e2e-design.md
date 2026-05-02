# Playwright E2E Test Suite — Design Spec

> Date: 2026-05-02
> Scope: Add minimal, standalone Playwright e2e tests for critical Streamlit UI journeys

---

## Goal

Add a small suite of end-to-end tests that launch the Streamlit app in a real browser, drive it via Playwright, and assert on visible results. The tests must run without network calls (deterministic mock data) and without the `pytest-playwright` plugin.

---

## Architecture

We create an `e2e/` directory with standalone Python scripts. Each script manages its own Streamlit subprocess lifecycle, opens a browser via `playwright.sync_api`, interacts with the UI, and asserts on page content.

Mock data is injected through an existing hook: `app.data_repository.set_repository(MockRepository(...))`. A thin wrapper entry point (`e2e/run_app.py`) adjusts `sys.path` so `app` and `backtest` modules resolve correctly when Streamlit launches from `e2e/`, sets a mock repository with deterministic prices, and calls `app.main.main()`. No production code is modified.

---

## Files

| File | Purpose |
|------|---------|
| `e2e/__init__.py` | Package marker |
| `e2e/fixtures.py` | Deterministic `pd.DataFrame` generators and pre-calculated expected metrics |
| `e2e/run_app.py` | Wrapper entry point; injects `MockRepository` and calls `app.main.main()` |
| `e2e/server.py` | `StreamlitServer` context manager: free-port selection, subprocess spawn, health-check, teardown |
| `e2e/test_basic_backtest.py` | End-to-end script: fill tickers/weights, run backtest, assert on summary metrics |
| `e2e/test_preset_flow.py` | End-to-end script: select a portfolio preset, verify auto-populated inputs, run backtest |
| `e2e/test_error_handling.py` | End-to-end script: enter invalid ticker, assert error toast appears |
| `e2e/test_dca_strategy.py` | End-to-end script: enable DCA, assert DCA-specific results appear |

---

## Mock Data Strategy

`fixtures.py` exposes a helper `create_mock_prices(tickers, start, end)` that returns a `pd.DataFrame` with a business-day `DatetimeIndex` and deterministic prices (e.g., linear ramps from 100 to 150). This makes expected CAGR, ending value, and total return easy to pre-calculate and hard-code into assertions.

Example:
```python
prices = create_mock_prices(
    tickers=["AAPL", "MSFT"],
    start="2020-01-01",
    end="2020-12-31"
)
# AAPL: 100 -> 150, MSFT: 80 -> 120
# Expected portfolio CAGR ≈ 45.0%, ending value ≈ $145,000
```

---

## Process Lifecycle (`StreamlitServer`)

A context manager handles the full lifecycle:

1. **Port selection** — bind to port 0 to get a free ephemeral port.
2. **Subprocess start** — `streamlit run e2e/run_app.py --server.port=<port> --server.headless=true`.
3. **Health check** — poll stdout for "You can now view your Streamlit app" (timeout: 30 s).
4. **Yield** — return the local URL (e.g., `http://localhost:12345`).
5. **Teardown** — `terminate()` the subprocess, then `kill()` if it does not exit within 5 s.

```python
with StreamlitServer(entrypoint="e2e/run_app.py") as url:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        # ... interact and assert
```

---

## Test Scenarios

### `test_basic_backtest.py`
1. Start server.
2. Navigate to app.
3. In sidebar, enter ticker `AAPL`, weight `0.6`.
4. Enter ticker `MSFT`, weight `0.4`.
5. Set start date to `2020-01-01` and end date to `2020-12-31` (deterministic range).
6. Click "Run Backtest".
7. Wait for "Summary Statistics" heading.
8. Assert page contains pre-calculated "Ending Value" (`$145,000.00`) and "CAGR" (`45.00%`).

### `test_preset_flow.py`
1. Start server.
2. Navigate to app.
3. Select a portfolio preset (e.g., "VDCP/VHYD vs VWRA") from the dropdown.
4. Assert the first ticker input contains a preset value (e.g., read widget value via Playwright locator).
5. Click "Run Backtest".
6. Wait for "Summary Statistics" heading and assert "Ending Value" is visible.

### `test_error_handling.py`
1. Start server.
2. Navigate to app.
3. Enter an invalid ticker (e.g., `INVALID123`).
4. Click "Run Backtest".
5. Assert the main content area contains "Ticker Validation Failed" (the `st.error()` title emitted by `_run_backtest()`).

### `test_dca_strategy.py`
1. Start server.
2. Navigate to app.
3. Enter valid tickers/weights.
4. Select a DCA frequency (e.g., "Monthly") and set a contribution amount.
5. Click "Run Backtest".
6. Assert DCA-specific labels (e.g., "Total Contributions") appear in the summary.

---

## Dependencies

Add to `requirements.txt` (or a separate `requirements-e2e.txt`):
```
playwright>=1.40.0
```

One-time setup:
```bash
playwright install chromium
```

---

## Running the Suite

```bash
# Install browsers (one-time)
playwright install chromium

# Run each test script independently
python e2e/test_basic_backtest.py
python e2e/test_preset_flow.py
python e2e/test_error_handling.py
python e2e/test_dca_strategy.py
```

No pytest runner required. Each script exits with code 0 on success or raises an assertion error on failure.

---

## Success Criteria

- [ ] All four e2e scripts pass locally in under 60 s total.
- [ ] No modifications to `app/`, `backtest.py`, or existing tests.
- [ ] Mock data is deterministic; assertions are stable across runs.
- [ ] Streamlit subprocess is always terminated, even on test failure.
- [ ] `pytest -v` still passes (existing unit tests are unaffected).

---

## Open Questions / Notes

- **CI integration:** GitHub Actions can run these with `playwright install chromium` in a separate job, but that is out of scope for this design.
- **Parallel execution:** Scripts currently run serially. If needed later, port allocation logic already supports parallel runs.
- **Screenshot on failure:** Optional enhancement: capture `page.screenshot()` in an `except` block for debugging.
