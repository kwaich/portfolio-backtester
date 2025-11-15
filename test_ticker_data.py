"""Unit tests for app/ticker_data.py module.

Tests ticker data functionality including:
- Popular ticker lists
- Ticker search
- Yahoo Finance integration
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, Mock
import requests

from app.ticker_data import (
    get_all_tickers,
    get_ticker_symbols,
    search_tickers,
    format_ticker_option,
    get_ticker_name,
    search_yahoo_finance,
    search_tickers_with_yahoo,
    POPULAR_ETFS,
    POPULAR_STOCKS,
)


class TestTickerLists:
    """Tests for curated ticker lists."""

    def test_popular_etfs_structure(self):
        """Test that POPULAR_ETFS has expected structure."""
        assert isinstance(POPULAR_ETFS, dict)
        assert len(POPULAR_ETFS) > 0

        for category, tickers in POPULAR_ETFS.items():
            assert isinstance(category, str)
            assert isinstance(tickers, list)
            assert len(tickers) > 0

            for ticker, name in tickers:
                assert isinstance(ticker, str)
                assert isinstance(name, str)
                assert len(ticker) > 0
                assert len(name) > 0

    def test_popular_stocks_structure(self):
        """Test that POPULAR_STOCKS has expected structure."""
        assert isinstance(POPULAR_STOCKS, dict)
        assert len(POPULAR_STOCKS) > 0

        for category, tickers in POPULAR_STOCKS.items():
            assert isinstance(category, str)
            assert isinstance(tickers, list)
            assert len(tickers) > 0

            for ticker, name in tickers:
                assert isinstance(ticker, str)
                assert isinstance(name, str)
                assert len(ticker) > 0
                assert len(name) > 0

    def test_get_all_tickers(self):
        """Test get_all_tickers returns combined list."""
        all_tickers = get_all_tickers()

        assert isinstance(all_tickers, list)
        assert len(all_tickers) > 0

        # Check structure
        for ticker, name in all_tickers:
            assert isinstance(ticker, str)
            assert isinstance(name, str)

        # Check that list is sorted by ticker
        ticker_symbols = [t for t, _ in all_tickers]
        assert ticker_symbols == sorted(ticker_symbols)

        # Verify some known tickers are present
        ticker_dict = dict(all_tickers)
        assert "SPY" in ticker_dict
        assert "AAPL" in ticker_dict
        assert "VWRA.L" in ticker_dict

    def test_get_ticker_symbols(self):
        """Test get_ticker_symbols returns only symbols."""
        symbols = get_ticker_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) > 0

        # All elements should be strings
        assert all(isinstance(s, str) for s in symbols)

        # Should contain known tickers
        assert "SPY" in symbols
        assert "AAPL" in symbols
        assert "VWRA.L" in symbols

        # Should be sorted
        assert symbols == sorted(symbols)


class TestTickerSearch:
    """Tests for ticker search functionality."""

    def test_search_tickers_empty_query(self):
        """Test search_tickers with empty query."""
        results = search_tickers("")
        assert results == []

    def test_search_tickers_by_symbol(self):
        """Test search_tickers by ticker symbol."""
        results = search_tickers("SPY")
        assert "SPY" in results

    def test_search_tickers_by_symbol_case_insensitive(self):
        """Test search_tickers is case insensitive."""
        results_upper = search_tickers("SPY")
        results_lower = search_tickers("spy")
        assert results_upper == results_lower

    def test_search_tickers_by_name(self):
        """Test search_tickers by company/fund name."""
        results = search_tickers("vanguard")
        assert len(results) > 0

        # Should contain some vanguard tickers
        assert any("V" in ticker for ticker in results)

    def test_search_tickers_partial_match(self):
        """Test search_tickers with partial matches."""
        results = search_tickers("van")
        assert len(results) > 0

    def test_format_ticker_option_known_ticker(self):
        """Test format_ticker_option with known ticker."""
        formatted = format_ticker_option("SPY")
        assert "SPY" in formatted
        assert "SPDR S&P 500 ETF Trust" in formatted
        assert " - " in formatted

    def test_format_ticker_option_unknown_ticker(self):
        """Test format_ticker_option with unknown ticker."""
        formatted = format_ticker_option("UNKNOWN")
        assert formatted == "UNKNOWN"

    def test_get_ticker_name_known_ticker(self):
        """Test get_ticker_name with known ticker."""
        name = get_ticker_name("SPY")
        assert name == "SPDR S&P 500 ETF Trust"

    def test_get_ticker_name_unknown_ticker(self):
        """Test get_ticker_name with unknown ticker."""
        name = get_ticker_name("UNKNOWN")
        assert name == ""


class TestYahooFinanceSearch:
    """Tests for Yahoo Finance search integration."""

    def setup_method(self):
        """Clear the cache before each test."""
        search_yahoo_finance.cache_clear()

    def test_search_yahoo_finance_empty_query(self):
        """Test search_yahoo_finance with empty query."""
        results = search_yahoo_finance("")
        assert results == []

    @patch('app.ticker_data.requests.get')
    def test_search_yahoo_finance_successful(self, mock_get):
        """Test search_yahoo_finance with successful API response."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "quotes": [
                {"symbol": "AAPL", "longname": "Apple Inc."},
                {"symbol": "MSFT", "longname": "Microsoft Corporation"},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        results = search_yahoo_finance("apple")

        assert len(results) == 2
        assert ("AAPL", "Apple Inc.") in results
        assert ("MSFT", "Microsoft Corporation") in results

    @patch('app.ticker_data.requests.get')
    def test_search_yahoo_finance_with_shortname(self, mock_get):
        """Test search_yahoo_finance when only shortname is available."""
        # Mock response with shortname instead of longname
        mock_response = Mock()
        mock_response.json.return_value = {
            "quotes": [
                {"symbol": "TEST", "shortname": "Test Company"},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        results = search_yahoo_finance("test")

        assert len(results) == 1
        assert ("TEST", "Test Company") in results

    @patch('app.ticker_data.requests.get')
    def test_search_yahoo_finance_network_error(self, mock_get):
        """Test search_yahoo_finance handles network errors gracefully."""
        mock_get.side_effect = requests.RequestException("Network error")

        results = search_yahoo_finance("apple")

        # Should return empty list on error
        assert results == []

    @patch('app.ticker_data.requests.get')
    def test_search_yahoo_finance_invalid_json(self, mock_get):
        """Test search_yahoo_finance handles invalid JSON response."""
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        results = search_yahoo_finance("apple")

        # Should return empty list on error
        assert results == []

    @patch('app.ticker_data.requests.get')
    def test_search_yahoo_finance_limit(self, mock_get):
        """Test search_yahoo_finance respects limit parameter."""
        # Mock response with many results
        mock_response = Mock()
        mock_response.json.return_value = {
            "quotes": [
                {"symbol": f"TICK{i}", "longname": f"Company {i}"}
                for i in range(20)
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        results = search_yahoo_finance("test", limit=5)

        assert len(results) == 5

    @patch('app.ticker_data.requests.get')
    def test_search_yahoo_finance_missing_data(self, mock_get):
        """Test search_yahoo_finance skips quotes with missing data."""
        # Mock response with incomplete data
        mock_response = Mock()
        mock_response.json.return_value = {
            "quotes": [
                {"symbol": "AAPL", "longname": "Apple Inc."},
                {"symbol": "", "longname": "No Symbol"},  # Missing symbol
                {"symbol": "MSFT", "longname": ""},  # Missing name
                {"symbol": "GOOGL", "shortname": "Alphabet"},  # Valid
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        results = search_yahoo_finance("test")

        # Should only include quotes with both symbol and name
        assert len(results) == 2
        assert ("AAPL", "Apple Inc.") in results
        assert ("GOOGL", "Alphabet") in results

    @patch('app.ticker_data.requests.get')
    def test_search_yahoo_finance_caching(self, mock_get):
        """Test that search_yahoo_finance caches results."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "quotes": [{"symbol": "AAPL", "longname": "Apple Inc."}]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # First call
        results1 = search_yahoo_finance("apple")
        # Second call with same query
        results2 = search_yahoo_finance("apple")

        # Should only call API once due to caching
        assert mock_get.call_count == 1
        assert results1 == results2


class TestCombinedSearch:
    """Tests for combined search functionality."""

    def test_search_tickers_with_yahoo_empty_query(self):
        """Test search_tickers_with_yahoo with empty query."""
        results = search_tickers_with_yahoo("")
        assert results == []

    @patch('app.ticker_data.search_yahoo_finance')
    def test_search_tickers_with_yahoo_curated_only(self, mock_yahoo):
        """Test search when curated list has enough results."""
        # Search for SPY - should be in curated list
        results = search_tickers_with_yahoo("SPY", limit=10)

        # Should not call Yahoo Finance if curated has enough results
        # (SPY should match immediately)
        assert len(results) >= 1
        assert any(ticker == "SPY" for ticker, _ in results)

    @patch('app.ticker_data.search_yahoo_finance')
    def test_search_tickers_with_yahoo_supplements_with_api(self, mock_yahoo):
        """Test search supplements curated results with Yahoo Finance."""
        # Mock Yahoo Finance to return additional results
        mock_yahoo.return_value = [
            ("RARE1", "Rare Company 1"),
            ("RARE2", "Rare Company 2"),
        ]

        # Search for something that might not be in curated list
        results = search_tickers_with_yahoo("rare", limit=10)

        # Should call Yahoo Finance for supplemental results
        assert mock_yahoo.called

    @patch('app.ticker_data.search_yahoo_finance')
    def test_search_tickers_with_yahoo_respects_limit(self, mock_yahoo):
        """Test search respects the limit parameter."""
        mock_yahoo.return_value = [
            (f"TICK{i}", f"Company {i}")
            for i in range(20)
        ]

        results = search_tickers_with_yahoo("test", limit=5)

        assert len(results) <= 5

    @patch('app.ticker_data.search_yahoo_finance')
    def test_search_tickers_with_yahoo_removes_duplicates(self, mock_yahoo):
        """Test search removes duplicates between curated and Yahoo results."""
        # Yahoo returns ticker that's also in curated list
        mock_yahoo.return_value = [
            ("SPY", "SPDR S&P 500 ETF Trust"),
            ("NEWticker", "New Company"),
        ]

        results = search_tickers_with_yahoo("spy", limit=10)

        # Should only include SPY once
        ticker_symbols = [ticker for ticker, _ in results]
        assert ticker_symbols.count("SPY") == 1


class TestTickerDataEdgeCases:
    """Tests for edge cases and error conditions."""

    def setup_method(self):
        """Clear the cache before each test."""
        search_yahoo_finance.cache_clear()

    def test_search_tickers_special_characters(self):
        """Test search handles special characters in ticker symbols."""
        # Search for London Stock Exchange tickers (with .L suffix)
        results = search_tickers("VWRA.L")
        assert "VWRA.L" in results

    def test_get_all_tickers_no_duplicates(self):
        """Test that get_all_tickers doesn't return duplicates."""
        all_tickers = get_all_tickers()
        ticker_symbols = [ticker for ticker, _ in all_tickers]

        # Check for duplicates
        assert len(ticker_symbols) == len(set(ticker_symbols))

    def test_search_tickers_whitespace(self):
        """Test search handles whitespace in query."""
        # Search with whitespace should still work
        results = search_tickers("  SPY  ")
        assert len(results) >= 1

    @patch('app.ticker_data.requests.get')
    def test_search_yahoo_finance_timeout(self, mock_get):
        """Test search_yahoo_finance handles timeout gracefully."""
        mock_get.side_effect = requests.Timeout("Request timed out")

        results = search_yahoo_finance("apple")

        # Should return empty list on timeout
        assert results == []
