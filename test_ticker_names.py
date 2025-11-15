#!/usr/bin/env python3
"""Test script to verify ticker name functionality."""

import sys
sys.path.insert(0, '.')

from app.ticker_data import get_ticker_name, get_all_tickers

print("=" * 70)
print("TICKER NAME FUNCTIONALITY TEST")
print("=" * 70)
print()

# Test 1: Verify get_all_tickers works
print("Test 1: Get all tickers")
print("-" * 70)
all_tickers = get_all_tickers()
print(f"Total tickers in curated list: {len(all_tickers)}")
print(f"First 5 tickers: {all_tickers[:5]}")
print()

# Test 2: Test default portfolio tickers
print("Test 2: Default portfolio tickers")
print("-" * 70)
default_tickers = ["VDCP.L", "VHYD.L", "VWRA.L"]
for ticker in default_tickers:
    name = get_ticker_name(ticker)
    status = "✅" if name else "❌"
    print(f"{status} {ticker:10} -> {name}")
print()

# Test 3: Test common tickers
print("Test 3: Common tickers")
print("-" * 70)
common_tickers = ["AAPL", "MSFT", "GOOGL", "SPY", "VOO", "QQQ"]
for ticker in common_tickers:
    name = get_ticker_name(ticker)
    status = "✅" if name else "❌"
    print(f"{status} {ticker:10} -> {name}")
print()

# Test 4: Test unknown ticker
print("Test 4: Unknown ticker handling")
print("-" * 70)
unknown = get_ticker_name("UNKNOWN_TICKER")
print(f"get_ticker_name('UNKNOWN_TICKER') = '{unknown}'")
print(f"Returns empty string: {'✅' if unknown == '' else '❌'}")
print()

# Test 5: Simulate render_portfolio_composition
print("Test 5: Simulate render_portfolio_composition()")
print("-" * 70)
import pandas as pd
import numpy as np

tickers = ["VDCP.L", "VHYD.L"]
weights = np.array([0.5, 0.5])

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
    print(f"❌ WARNING: Blank names found for: {blank_names}")
else:
    print("✅ All ticker names found successfully!")

print()
print("=" * 70)
print("If all tests pass, the issue is likely that Streamlit needs restart!")
print("=" * 70)

