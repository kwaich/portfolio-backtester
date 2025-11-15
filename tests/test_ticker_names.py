#!/usr/bin/env python3
"""Test script to verify ticker name functionality.

This script fetches ticker names from Yahoo Finance dynamically.
Note: May fail if Yahoo Finance API is rate-limited or blocked.
"""

import sys
sys.path.insert(0, '.')

from app.ticker_data import get_ticker_name, get_all_tickers

print("=" * 70)
print("TICKER NAME FUNCTIONALITY TEST (Live Yahoo Finance)")
print("=" * 70)
print()
print("NOTE: This test fetches live data from Yahoo Finance API.")
print("Some tests may fail if the API is rate-limited or blocked.")
print()

# Test 1: Verify get_all_tickers works (curated list still used for search)
print("Test 1: Get all tickers (curated list for search)")
print("-" * 70)
all_tickers = get_all_tickers()
print(f"Total tickers in curated list: {len(all_tickers)}")
print(f"First 5 tickers: {all_tickers[:5]}")
print()

# Test 2: Test default portfolio tickers (fetched from Yahoo Finance)
print("Test 2: Default portfolio tickers (from Yahoo Finance)")
print("-" * 70)
default_tickers = ["VDCP.L", "VHYD.L", "VWRA.L"]
for ticker in default_tickers:
    print(f"Fetching {ticker}...", end=" ", flush=True)
    name = get_ticker_name(ticker)
    status = "✅" if name else "⚠️"
    print(f"{status} -> {name if name else '(no data)'}")
print()

# Test 3: Test common tickers (fetched from Yahoo Finance)
print("Test 3: Common tickers (from Yahoo Finance)")
print("-" * 70)
common_tickers = ["AAPL", "MSFT", "GOOGL", "SPY", "VOO", "QQQ"]
for ticker in common_tickers:
    print(f"Fetching {ticker}...", end=" ", flush=True)
    name = get_ticker_name(ticker)
    status = "✅" if name else "⚠️"
    print(f"{status} -> {name if name else '(no data)'}")
print()

# Test 4: Test unknown ticker
print("Test 4: Unknown ticker handling")
print("-" * 70)
unknown = get_ticker_name("UNKNOWN_TICKER_XYZ123")
print(f"get_ticker_name('UNKNOWN_TICKER_XYZ123') = '{unknown}'")
print(f"Returns empty string: {'✅' if unknown == '' else '❌'}")
print()

# Test 5: Simulate render_portfolio_composition
print("Test 5: Simulate render_portfolio_composition()")
print("-" * 70)
import pandas as pd
import numpy as np

tickers = ["VDCP.L", "VHYD.L"]
weights = np.array([0.5, 0.5])

print("Fetching ticker names...")
ticker_names = [get_ticker_name(ticker) for ticker in tickers]

composition_data = {
    "Ticker": tickers,
    "Name": ticker_names,
    "Weight": [f"{w:.1%}" for w in weights]
}

df = pd.DataFrame(composition_data)
print(df.to_string())
print()

# Check if any names are blank
blank_names = [t for t, n in zip(tickers, ticker_names) if not n]
if blank_names:
    print(f"⚠️  WARNING: Blank names found for: {blank_names}")
    print("    This may be due to Yahoo Finance API being blocked or rate-limited.")
else:
    print("✅ All ticker names fetched successfully!")

print()
print("=" * 70)
print("NOTE: Names are fetched from Yahoo Finance in real-time.")
print("Restart Streamlit to see the updated names in the UI.")
print("=" * 70)
