"""Ticker data and search functionality for the ETF Backtester.

This module provides a curated list of popular tickers (ETFs and stocks)
and search functionality for ticker selection in the UI, including
lazy search from Yahoo Finance.
"""

from __future__ import annotations

import logging
from typing import List, Optional
import requests
import yfinance as yf

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    # Fallback to functools.lru_cache if streamlit not available (for testing)
    from functools import lru_cache

# Popular ETFs organized by category
POPULAR_ETFS = {
    "Global Equity": [
        ("VWRA.L", "Vanguard FTSE All-World UCITS ETF"),
        ("VT", "Vanguard Total World Stock ETF"),
        ("ACWI", "iShares MSCI ACWI ETF"),
        ("URTH", "iShares MSCI World ETF"),
    ],
    "US Equity": [
        ("SPY", "SPDR S&P 500 ETF Trust"),
        ("VOO", "Vanguard S&P 500 ETF"),
        ("VTI", "Vanguard Total Stock Market ETF"),
        ("IVV", "iShares Core S&P 500 ETF"),
        ("QQQ", "Invesco QQQ Trust (Nasdaq-100)"),
    ],
    "International Equity": [
        ("VXUS", "Vanguard Total International Stock ETF"),
        ("VEA", "Vanguard FTSE Developed Markets ETF"),
        ("VWO", "Vanguard FTSE Emerging Markets ETF"),
        ("IEFA", "iShares Core MSCI EAFE ETF"),
        ("EEM", "iShares MSCI Emerging Markets ETF"),
    ],
    "European Equity": [
        ("VEUR.L", "Vanguard FTSE Developed Europe UCITS ETF"),
        ("VGK", "Vanguard FTSE Europe ETF"),
        ("EZU", "iShares MSCI Eurozone ETF"),
    ],
    "Fixed Income": [
        ("VHYD.L", "Vanguard USD EM Government Bond UCITS ETF"),
        ("VDCP.L", "Vanguard USD Corporate Bond UCITS ETF"),
        ("AGG", "iShares Core U.S. Aggregate Bond ETF"),
        ("BND", "Vanguard Total Bond Market ETF"),
        ("TLT", "iShares 20+ Year Treasury Bond ETF"),
        ("LQD", "iShares iBoxx Investment Grade Corporate Bond ETF"),
    ],
    "Sector ETFs": [
        ("XLK", "Technology Select Sector SPDR Fund"),
        ("XLF", "Financial Select Sector SPDR Fund"),
        ("XLE", "Energy Select Sector SPDR Fund"),
        ("XLV", "Health Care Select Sector SPDR Fund"),
        ("XLI", "Industrial Select Sector SPDR Fund"),
    ],
    "Commodities & Alternatives": [
        ("GLD", "SPDR Gold Trust"),
        ("SLV", "iShares Silver Trust"),
        ("USO", "United States Oil Fund"),
        ("VNQ", "Vanguard Real Estate ETF"),
    ],
}

# Popular individual stocks
POPULAR_STOCKS = {
    "Technology": [
        ("AAPL", "Apple Inc."),
        ("MSFT", "Microsoft Corporation"),
        ("GOOGL", "Alphabet Inc. (Google) Class A"),
        ("GOOG", "Alphabet Inc. (Google) Class C"),
        ("AMZN", "Amazon.com Inc."),
        ("META", "Meta Platforms Inc. (Facebook)"),
        ("NVDA", "NVIDIA Corporation"),
        ("TSLA", "Tesla Inc."),
    ],
    "Finance": [
        ("JPM", "JPMorgan Chase & Co."),
        ("BAC", "Bank of America Corporation"),
        ("WFC", "Wells Fargo & Company"),
        ("GS", "Goldman Sachs Group Inc."),
        ("MS", "Morgan Stanley"),
    ],
    "Healthcare": [
        ("JNJ", "Johnson & Johnson"),
        ("UNH", "UnitedHealth Group Inc."),
        ("PFE", "Pfizer Inc."),
        ("ABBV", "AbbVie Inc."),
    ],
    "Consumer": [
        ("WMT", "Walmart Inc."),
        ("PG", "Procter & Gamble Co."),
        ("KO", "Coca-Cola Company"),
        ("PEP", "PepsiCo Inc."),
    ],
}


def get_all_tickers() -> List[tuple[str, str]]:
    """Get all tickers from both ETFs and stocks.

    Returns:
        List of (ticker, name) tuples

    Examples:
        >>> tickers = get_all_tickers()
        >>> len(tickers) > 0
        True
        >>> ("SPY", "SPDR S&P 500 ETF Trust") in tickers
        True
    """
    all_tickers = []

    # Add all ETFs
    for category_etfs in POPULAR_ETFS.values():
        all_tickers.extend(category_etfs)

    # Add all stocks
    for category_stocks in POPULAR_STOCKS.values():
        all_tickers.extend(category_stocks)

    # Sort by ticker symbol
    all_tickers.sort(key=lambda x: x[0])

    return all_tickers


def get_ticker_symbols() -> List[str]:
    """Get list of all ticker symbols (without names).

    Returns:
        List of ticker symbols

    Examples:
        >>> symbols = get_ticker_symbols()
        >>> "SPY" in symbols
        True
        >>> "VWRA.L" in symbols
        True
    """
    return [ticker for ticker, _ in get_all_tickers()]


def search_tickers(query: str) -> List[str]:
    """Search for tickers matching a query string.

    Args:
        query: Search query (partial ticker or name)

    Returns:
        List of matching ticker symbols

    Examples:
        >>> search_tickers("SPY")
        ['SPY']
        >>> search_tickers("vanguard")
        ['VWRA.L', 'VT', 'VOO', 'VTI', ...]
        >>> search_tickers("")
        []
    """
    if not query:
        return []

    query_lower = query.strip().lower()
    all_tickers = get_all_tickers()

    matches = []
    for ticker, name in all_tickers:
        # Match against ticker symbol or name
        if query_lower in ticker.lower() or query_lower in name.lower():
            matches.append(ticker)

    return matches


def format_ticker_option(ticker: str) -> str:
    """Format ticker for display in dropdown with name.

    Args:
        ticker: Ticker symbol

    Returns:
        Formatted string "TICKER - Name"

    Examples:
        >>> format_ticker_option("SPY")
        'SPY - SPDR S&P 500 ETF Trust'
        >>> format_ticker_option("UNKNOWN")
        'UNKNOWN'
    """
    all_tickers = get_all_tickers()
    ticker_dict = dict(all_tickers)

    if ticker in ticker_dict:
        return f"{ticker} - {ticker_dict[ticker]}"
    else:
        return ticker


def _get_ticker_name_impl(ticker: str) -> str:
    """Internal implementation of get_ticker_name.

    This is separated to allow for flexible caching strategies.
    """
    if not ticker or not ticker.strip():
        return ""

    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info

        # info might be None or empty dict
        if not info:
            logging.warning(f"No info available for ticker: {ticker}")
            return ""

        # Try to get longName first, then shortName as fallback
        name = info.get('longName') or info.get('shortName') or ""

        if name:
            logging.info(f"Fetched ticker name for {ticker}: {name}")
        else:
            logging.warning(f"No name fields available for ticker: {ticker}")

        return name

    except Exception as e:
        logging.warning(f"Error fetching ticker name for {ticker}: {e}")
        return ""


if HAS_STREAMLIT:
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_ticker_name(ticker: str) -> str:
        """Get the full name for a ticker symbol from Yahoo Finance.

        Fetches the ticker name dynamically from Yahoo Finance API.
        Results are cached for 1 hour to avoid repeated API calls.

        Args:
            ticker: Ticker symbol (e.g., "AAPL", "VWRA.L", "SPY")

        Returns:
            Full name of the ticker, or empty string if not found or error occurs

        Examples:
            >>> get_ticker_name("SPY")
            'SPDR S&P 500 ETF Trust'
            >>> get_ticker_name("AAPL")
            'Apple Inc.'
            >>> get_ticker_name("UNKNOWN")
            ''
        """
        return _get_ticker_name_impl(ticker)

    # Provide cache_clear method for compatibility with tests
    def _clear_ticker_name_cache():
        """Clear the get_ticker_name cache (Streamlit version)."""
        try:
            get_ticker_name.clear()
        except:
            pass  # Streamlit cache might not support clear() in all versions

    get_ticker_name.cache_clear = _clear_ticker_name_cache

else:
    # Testing fallback - use lru_cache
    @lru_cache(maxsize=500)
    def get_ticker_name(ticker: str) -> str:
        """Get the full name for a ticker symbol from Yahoo Finance (testing version)."""
        return _get_ticker_name_impl(ticker)


def _search_yahoo_finance_impl(query: str, limit: int = 10) -> List[tuple[str, str]]:
    """Internal implementation of Yahoo Finance search."""
    if not query or len(query) < 1:
        return []

    try:
        # Yahoo Finance autocomplete endpoint
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            "q": query,
            "quotesCount": limit,
            "newsCount": 0,
            "enableFuzzyQuery": False,
            "quotesQueryId": "tss_match_phrase_query"
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://finance.yahoo.com",
            "Origin": "https://finance.yahoo.com",
        }

        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()

        data = response.json()
        quotes = data.get("quotes", [])

        results = []
        for quote in quotes[:limit]:
            symbol = quote.get("symbol", "")
            # Get long name or short name
            name = quote.get("longname") or quote.get("shortname", "")
            if symbol and name:
                results.append((symbol, name))

        logging.info(f"Yahoo Finance search for '{query}': found {len(results)} results")
        return results

    except requests.RequestException as e:
        logging.warning(f"Yahoo Finance search failed for '{query}': {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error in Yahoo Finance search for '{query}': {e}")
        return []


if HAS_STREAMLIT:
    @st.cache_data(ttl=1800, show_spinner=False)
    def search_yahoo_finance(query: str, limit: int = 10) -> List[tuple[str, str]]:
        """Search Yahoo Finance for ticker symbols matching a query.

        This function queries Yahoo Finance's autocomplete API to find tickers
        that match the search query. Results are cached for 30 minutes to reduce API calls.

        Args:
            query: Search query (ticker symbol or company name)
            limit: Maximum number of results to return (default: 10)

        Returns:
            List of (ticker, name) tuples matching the query

        Examples:
            >>> results = search_yahoo_finance("apple")
            >>> any("AAPL" in ticker for ticker, _ in results)
            True
            >>> search_yahoo_finance("")
            []

        Note:
            - Requires internet connection
            - Results are cached for 30 minutes
            - Returns empty list on network errors or rate limiting (403)
            - Yahoo Finance may block requests - this is expected behavior
            - The app will fall back to curated ticker list if Yahoo search fails
        """
        return _search_yahoo_finance_impl(query, limit)

    # Provide cache_clear method for compatibility with tests
    def _clear_yahoo_search_cache():
        """Clear the search_yahoo_finance cache (Streamlit version)."""
        try:
            search_yahoo_finance.clear()
        except:
            pass  # Streamlit cache might not support clear() in all versions

    search_yahoo_finance.cache_clear = _clear_yahoo_search_cache

else:
    # Testing fallback - use lru_cache
    @lru_cache(maxsize=100)
    def search_yahoo_finance(query: str, limit: int = 10) -> List[tuple[str, str]]:
        """Search Yahoo Finance for ticker symbols matching a query (testing version)."""
        return _search_yahoo_finance_impl(query, limit)


def search_tickers_with_yahoo(query: str, limit: int = 10) -> List[tuple[str, str]]:
    """Search for tickers using both curated list and Yahoo Finance.

    This function first searches the curated list of popular tickers,
    then supplements with Yahoo Finance search results if needed.

    Args:
        query: Search query (partial ticker or name)
        limit: Maximum total results to return (default: 10)

    Returns:
        List of (ticker, name) tuples matching the query

    Examples:
        >>> results = search_tickers_with_yahoo("SPY")
        >>> len(results) > 0
        True
        >>> results = search_tickers_with_yahoo("apple", limit=5)
        >>> len(results) <= 5
        True
    """
    if not query:
        return []

    query_lower = query.lower()
    all_tickers = get_all_tickers()

    # First, search curated list
    curated_matches = []
    for ticker, name in all_tickers:
        if query_lower in ticker.lower() or query_lower in name.lower():
            curated_matches.append((ticker, name))

    # If we have enough matches from curated list, return those
    if len(curated_matches) >= limit:
        return curated_matches[:limit]

    # Otherwise, supplement with Yahoo Finance search
    yahoo_matches = search_yahoo_finance(query, limit=limit)

    # Combine results, removing duplicates (prefer curated)
    curated_symbols = {ticker for ticker, _ in curated_matches}
    combined = curated_matches.copy()

    for ticker, name in yahoo_matches:
        if ticker not in curated_symbols and len(combined) < limit:
            combined.append((ticker, name))

    return combined[:limit]
