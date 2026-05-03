"""Tests for the DataRepository pattern."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from app.data_repository import (
    YahooFinanceRepository,
    MockRepository,
    get_repository,
    set_repository,
    DEFAULT_CACHE_TTL_HOURS,
)


class TestYahooFinanceRepositoryCache:
    """Test caching functionality in YahooFinanceRepository."""

    def test_get_cache_key_order_invariant(self):
        repo = YahooFinanceRepository()
        key1 = repo._get_cache_key(["AAPL", "MSFT"], "2020-01-01", "2021-01-01")
        key2 = repo._get_cache_key(["MSFT", "AAPL"], "2020-01-01", "2021-01-01")
        assert key1 == key2
        assert len(key1) == 32

    def test_get_cache_key_different_params(self):
        repo = YahooFinanceRepository()
        key1 = repo._get_cache_key(["AAPL"], "2020-01-01", "2021-01-01")
        key2 = repo._get_cache_key(["AAPL"], "2020-01-01", "2022-01-01")
        assert key1 != key2

    def test_save_and_load_cached_prices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = YahooFinanceRepository()
            cache_path = Path(tmpdir) / "test_cache"
            test_data = pd.DataFrame(
                {"AAPL": [100, 101, 102], "MSFT": [200, 201, 202]},
                index=pd.date_range("2020-01-01", periods=3),
            )

            repo.save_cached_prices(cache_path, test_data)
            loaded_data = repo.load_cached_prices(cache_path)

            assert loaded_data is not None
            pd.testing.assert_frame_equal(loaded_data, test_data, check_freq=False)
            assert cache_path.with_suffix(".parquet").exists()
            assert cache_path.with_suffix(".json").exists()

    def test_cache_expiration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = YahooFinanceRepository()
            cache_path = Path(tmpdir) / "test_cache"
            test_data = pd.DataFrame(
                {"AAPL": [100, 101, 102]},
                index=pd.date_range("2020-01-01", periods=3),
            )

            repo.save_cached_prices(cache_path, test_data)

            # Backdate metadata to simulate stale cache
            metadata_path = cache_path.with_suffix(".json")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            metadata["timestamp"] = time.time() - (25 * 3600)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            loaded = repo.load_cached_prices(cache_path, max_age_hours=24)
            assert loaded is None
            assert not cache_path.with_suffix(".parquet").exists()
            assert not cache_path.with_suffix(".json").exists()

    def test_load_nonexistent_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = YahooFinanceRepository()
            cache_path = Path(tmpdir) / "nonexistent"
            result = repo.load_cached_prices(cache_path)
            assert result is None


class TestYahooFinanceRepositoryDownload:
    """Test yfinance download integration."""

    @patch("app.data_repository.yf.download")
    def test_single_ticker_download(self, mock_yf):
        repo = YahooFinanceRepository()
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        mock_yf.return_value = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104]}, index=dates)

        result = repo.get_prices(["AAPL"], "2020-01-01", "2020-01-05", use_cache=False)

        assert list(result.columns) == ["AAPL"]
        assert len(result) == 5
        mock_yf.assert_called_once()

    @patch("app.data_repository.yf.download")
    def test_multi_ticker_download(self, mock_yf):
        repo = YahooFinanceRepository()
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        # yfinance returns a MultiIndex DataFrame for multiple tickers
        data = pd.DataFrame(
            {
                ("Adj Close", "AAPL"): [100, 101, 102, 103, 104],
                ("Adj Close", "MSFT"): [200, 201, 202, 203, 204],
            },
            index=dates,
        )
        data.columns = pd.MultiIndex.from_tuples(data.columns)
        mock_yf.return_value = data

        result = repo.get_prices(["AAPL", "MSFT"], "2020-01-01", "2020-01-05", use_cache=False)

        assert list(result.columns) == ["AAPL", "MSFT"]
        mock_yf.assert_called_once()

    @patch("app.data_repository.yf.download")
    def test_empty_data_raises_error(self, mock_yf):
        repo = YahooFinanceRepository()
        mock_yf.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="No price data returned"):
            repo.get_prices(["INVALID"], "2020-01-01", "2020-01-05", use_cache=False)

    @patch("app.data_repository.yf.download")
    def test_cache_hit_avoids_download(self, mock_yf):
        repo = YahooFinanceRepository()
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        cached = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104]}, index=dates)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = repo._get_cache_path(["AAPL"], "2020-01-01", "2020-01-05")
            # Patch cache dir to tmpdir
            with patch.object(
                repo,
                "_get_cache_path",
                return_value=Path(tmpdir) / cache_path.name,
            ):
                repo.save_cached_prices(Path(tmpdir) / cache_path.name, cached)
                result = repo.get_prices(["AAPL"], "2020-01-01", "2020-01-05", use_cache=True)

        mock_yf.assert_not_called()
        pd.testing.assert_frame_equal(result, cached, check_freq=False)


class TestYahooFinanceRepositoryNetwork:
    """Test Yahoo Finance network calls with mocked responses."""

    @patch("app.data_repository.requests.get")
    def test_search_tickers_success(self, mock_get):
        repo = YahooFinanceRepository()
        mock_get.return_value.json.return_value = {
            "quotes": [
                {"symbol": "AAPL", "longname": "Apple Inc."},
                {"symbol": "MSFT", "longname": "Microsoft Corporation"},
            ]
        }
        mock_get.return_value.raise_for_status = lambda: None

        results = repo.search_tickers("apple", limit=5)

        assert results == [("AAPL", "Apple Inc."), ("MSFT", "Microsoft Corporation")]
        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["q"] == "apple"
        assert kwargs["params"]["quotesCount"] == 5

    @patch("app.data_repository.requests.get")
    def test_search_tickers_empty_response(self, mock_get):
        repo = YahooFinanceRepository()
        mock_get.return_value.json.return_value = {"quotes": []}
        mock_get.return_value.raise_for_status = lambda: None

        results = repo.search_tickers("xyznonexistent")
        assert results == []

    @patch("app.data_repository.requests.get")
    def test_search_tickers_network_error(self, mock_get):
        repo = YahooFinanceRepository()
        mock_get.side_effect = Exception("Connection timeout")

        results = repo.search_tickers("AAPL")
        assert results == []

    @patch("app.data_repository.yf.Ticker")
    def test_get_ticker_name_success(self, mock_ticker_cls):
        repo = YahooFinanceRepository()
        mock_ticker = mock_ticker_cls.return_value
        mock_ticker.info = {"longName": "Apple Inc."}

        name = repo.get_ticker_name("AAPL")
        assert name == "Apple Inc."
        mock_ticker_cls.assert_called_once_with("AAPL")

    @patch("app.data_repository.yf.Ticker")
    def test_get_ticker_name_short_name_fallback(self, mock_ticker_cls):
        repo = YahooFinanceRepository()
        mock_ticker = mock_ticker_cls.return_value
        mock_ticker.info = {"shortName": "Apple"}

        name = repo.get_ticker_name("AAPL")
        assert name == "Apple"

    @patch("app.data_repository.yf.Ticker")
    def test_get_ticker_name_empty_info(self, mock_ticker_cls):
        repo = YahooFinanceRepository()
        mock_ticker = mock_ticker_cls.return_value
        mock_ticker.info = {}

        name = repo.get_ticker_name("AAPL")
        assert name == ""

    @patch("app.data_repository.yf.Ticker")
    def test_get_ticker_name_error(self, mock_ticker_cls):
        repo = YahooFinanceRepository()
        mock_ticker_cls.side_effect = Exception("Network error")

        name = repo.get_ticker_name("AAPL")
        assert name == ""


class TestMockRepository:
    """Test MockRepository behavior."""

    def test_get_prices_returns_provided_data(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        prices = pd.DataFrame({"AAPL": [100, 101, 102]}, index=dates)
        repo = MockRepository(prices=prices)

        result = repo.get_prices(["AAPL"], "2020-01-01", "2020-01-05")
        pd.testing.assert_frame_equal(result, prices)
        assert repo.get_prices_calls == [
            (["AAPL"], "2020-01-01", "2020-01-05", True, DEFAULT_CACHE_TTL_HOURS)
        ]

    def test_get_prices_generates_default_data(self):
        repo = MockRepository()
        result = repo.get_prices(["AAPL"], "2020-01-01", "2020-01-05")
        assert list(result.columns) == ["AAPL"]
        assert len(result) > 0

    def test_search_tickers(self):
        repo = MockRepository(search_results=[("AAPL", "Apple Inc.")])
        result = repo.search_tickers("apple")
        assert result == [("AAPL", "Apple Inc.")]
        assert repo.search_tickers_calls == [("apple", 10)]

    def test_get_ticker_name(self):
        repo = MockRepository(ticker_names={"AAPL": "Apple Inc."})
        assert repo.get_ticker_name("AAPL") == "Apple Inc."
        assert repo.get_ticker_name("UNKNOWN") == ""
        assert repo.get_ticker_name_calls == ["AAPL", "UNKNOWN"]


class TestModuleSingleton:
    """Test get_repository / set_repository module singleton."""

    def test_get_repository_returns_yahoo_finance(self):
        repo = get_repository()
        assert isinstance(repo, YahooFinanceRepository)

    def test_set_repository_changes_default(self):
        original = get_repository()
        mock = MockRepository()
        set_repository(mock)
        assert get_repository() is mock
        # Restore
        set_repository(original)
        assert get_repository() is original
