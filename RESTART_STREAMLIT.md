# How to Restart Streamlit and Clear Cache

## The Issue
Streamlit caches Python modules and may not pick up code changes automatically.
The ticker name feature is working correctly in tests but may show blank in the UI.

## Solution Steps:

### Option 1: Stop and Restart Streamlit (Recommended)
```bash
# Stop the current Streamlit process (Ctrl+C in the terminal)
# Then restart it:
streamlit run app.py
```

### Option 2: Use Streamlit's "Rerun" Button
- In the Streamlit web UI (top-right corner), click the ☰ menu
- Select "Rerun" or press 'R' on your keyboard

### Option 3: Clear Streamlit Cache
- In the Streamlit web UI, click the ☰ menu
- Select "Clear cache"
- Then click "Rerun"

### Option 4: Hard Restart with Cache Clearing
```bash
# Stop Streamlit (Ctrl+C)
# Clear Streamlit cache directory
rm -rf ~/.streamlit/cache
# Restart Streamlit
streamlit run app.py
```

## Verify the Fix

After restarting, run a backtest with the default tickers (VDCP.L, VHYD.L).
The Portfolio Composition table should now show:

| Ticker | Name                                          | Weight |
|--------|-----------------------------------------------|--------|
| VDCP.L | Vanguard USD Corporate Bond UCITS ETF        | 50.0%  |
| VHYD.L | Vanguard USD EM Government Bond UCITS ETF    | 50.0%  |

## If Still Blank

If names are still blank after restart, check:

1. **Are you using custom tickers?**
   - Only 53 curated tickers have names
   - Custom/unknown tickers show blank (expected behavior)
   - List of supported tickers: Run `python test_ticker_names.py`

2. **Check browser cache:**
   - Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
   - Or open in incognito/private window

3. **Verify code is latest:**
   ```bash
   git status
   git log --oneline -5
   ```

## Test Script

Run this to verify the backend is working:
```bash
python test_ticker_names.py
```

All tests should show ✅ (they do in our verification).
