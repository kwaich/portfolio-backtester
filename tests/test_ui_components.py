"""Tests for app/ui_components.py display functions."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# display_welcome_screen
# ---------------------------------------------------------------------------

class TestDisplayWelcomeScreen:
    def test_uses_columns_for_centering(self):
        mock_col = MagicMock()
        mock_col.__enter__ = lambda s: s
        mock_col.__exit__ = MagicMock(return_value=False)

        with patch("streamlit.columns", return_value=[MagicMock(), mock_col, MagicMock()]) as mock_cols, \
             patch("streamlit.markdown") as mock_md, \
             patch("streamlit.caption"):
            from app.ui_components import display_welcome_screen
            display_welcome_screen()

        mock_cols.assert_called_once_with([1, 2, 1])

    def test_renders_title_with_emoji(self):
        mock_col = MagicMock()
        mock_col.__enter__ = lambda s: s
        mock_col.__exit__ = MagicMock(return_value=False)

        with patch("streamlit.columns", return_value=[MagicMock(), mock_col, MagicMock()]), \
             patch("streamlit.markdown") as mock_md, \
             patch("streamlit.caption"):
            from app.ui_components import display_welcome_screen
            display_welcome_screen()

        args, _ = mock_md.call_args
        assert "📈" in args[0]
        assert "Portfolio Backtester" in args[0]


# ---------------------------------------------------------------------------
# display_section_header
# ---------------------------------------------------------------------------

class TestDisplaySectionHeader:
    def test_calls_subheader_with_title(self):
        with patch("streamlit.subheader") as mock_sub:
            from app.ui_components import display_section_header
            display_section_header("Performance Overview")

        mock_sub.assert_called_once_with("Performance Overview")

    def test_passes_title_verbatim(self):
        with patch("streamlit.subheader") as mock_sub:
            from app.ui_components import display_section_header
            display_section_header("Rolling Sharpe Ratio")

        mock_sub.assert_called_once_with("Rolling Sharpe Ratio")


# ---------------------------------------------------------------------------
# display_info_bar
# ---------------------------------------------------------------------------

class TestDisplayInfoBar:
    def test_renders_caption_with_tickers_and_dates(self):
        with patch("streamlit.caption") as mock_cap:
            from app.ui_components import display_info_bar
            display_info_bar(["AAPL", "MSFT"], [0.6, 0.4], ["SPY"], "2020-01-01", "2024-12-31")

        mock_cap.assert_called_once()
        text = mock_cap.call_args[0][0]
        assert "AAPL" in text
        assert "MSFT" in text
        assert "SPY" in text
        assert "2020-01-01" in text
        assert "2024-12-31" in text

    def test_skips_render_when_no_tickers(self):
        with patch("streamlit.caption") as mock_cap:
            from app.ui_components import display_info_bar
            display_info_bar([], [0.5], ["SPY"], "2020-01-01", "2024-12-31")

        mock_cap.assert_not_called()

    def test_skips_render_when_no_weights(self):
        with patch("streamlit.caption") as mock_cap:
            from app.ui_components import display_info_bar
            display_info_bar(["AAPL"], [], ["SPY"], "2020-01-01", "2024-12-31")

        mock_cap.assert_not_called()

    def test_fallback_benchmark_dash_when_empty(self):
        with patch("streamlit.caption") as mock_cap:
            from app.ui_components import display_info_bar
            display_info_bar(["AAPL"], [1.0], [], "2020-01-01", "2024-12-31")

        text = mock_cap.call_args[0][0]
        assert "—" in text

    def test_weight_formatted_as_percentage(self):
        with patch("streamlit.caption") as mock_cap:
            from app.ui_components import display_info_bar
            display_info_bar(["AAPL"], [0.6], ["SPY"], "2020-01-01", "2024-12-31")

        text = mock_cap.call_args[0][0]
        assert "60%" in text


# ---------------------------------------------------------------------------
# display_hero_metrics_row
# ---------------------------------------------------------------------------

class TestDisplayHeroMetricsRow:
    def _make_col_mock(self):
        m = MagicMock()
        m.__enter__ = lambda s: s
        m.__exit__ = MagicMock(return_value=False)
        return m

    def test_creates_correct_number_of_columns(self):
        metrics = {"Ending Value": "$150,000", "Total Return": "50.00%", "CAGR": "8.56%"}
        cols = [self._make_col_mock() for _ in metrics]

        with patch("streamlit.columns", return_value=cols) as mock_cols, \
             patch("streamlit.metric"):
            from app.ui_components import display_hero_metrics_row
            display_hero_metrics_row(metrics)

        mock_cols.assert_called_once_with(3)

    def test_calls_metric_for_each_entry(self):
        metrics = {"Total Return": "50.00%", "CAGR": "8.56%"}
        cols = [self._make_col_mock() for _ in metrics]

        with patch("streamlit.columns", return_value=cols), \
             patch("streamlit.metric") as mock_metric:
            from app.ui_components import display_hero_metrics_row
            display_hero_metrics_row(metrics)

        assert mock_metric.call_count == 2
        mock_metric.assert_any_call("Total Return", "50.00%")
        mock_metric.assert_any_call("CAGR", "8.56%")


# ---------------------------------------------------------------------------
# display_metrics_tables
# ---------------------------------------------------------------------------

class TestDisplayMetricsTables:
    def _make_col_mock(self):
        m = MagicMock()
        m.__enter__ = lambda s: s
        m.__exit__ = MagicMock(return_value=False)
        return m

    def test_creates_two_columns(self):
        cols = [self._make_col_mock(), self._make_col_mock()]
        with patch("streamlit.columns", return_value=cols) as mock_cols, \
             patch("streamlit.caption"), \
             patch("streamlit.dataframe"):
            from app.ui_components import display_metrics_tables
            display_metrics_tables({"CAGR": "8%"}, {"Sharpe": "1.2"})

        mock_cols.assert_called_once_with(2)

    def test_calls_dataframe_twice(self):
        cols = [self._make_col_mock(), self._make_col_mock()]
        with patch("streamlit.columns", return_value=cols), \
             patch("streamlit.caption"), \
             patch("streamlit.dataframe") as mock_df:
            from app.ui_components import display_metrics_tables
            display_metrics_tables({"CAGR": "8%"}, {"Sharpe": "1.2"})

        assert mock_df.call_count == 2

    def test_dataframe_hides_index(self):
        cols = [self._make_col_mock(), self._make_col_mock()]
        with patch("streamlit.columns", return_value=cols), \
             patch("streamlit.caption"), \
             patch("streamlit.dataframe") as mock_df:
            from app.ui_components import display_metrics_tables
            display_metrics_tables({"CAGR": "8%"}, {"Sharpe": "1.2"})

        for c in mock_df.call_args_list:
            assert c.kwargs.get("hide_index") is True


# ---------------------------------------------------------------------------
# display_downloads
# ---------------------------------------------------------------------------

class TestDisplayDownloads:
    def _make_col_mock(self):
        m = MagicMock()
        m.__enter__ = lambda s: s
        m.__exit__ = MagicMock(return_value=False)
        return m

    def test_shows_subheader(self):
        with patch("streamlit.subheader") as mock_sub, \
             patch("streamlit.columns", return_value=[self._make_col_mock(), self._make_col_mock()]), \
             patch("streamlit.download_button"):
            from app.ui_components import display_downloads
            display_downloads()

        mock_sub.assert_called_once_with("Downloads")

    def test_csv_button_shown_when_data_provided(self):
        with patch("streamlit.subheader"), \
             patch("streamlit.columns", return_value=[self._make_col_mock(), self._make_col_mock()]), \
             patch("streamlit.download_button") as mock_dl:
            from app.ui_components import display_downloads
            display_downloads(csv_data=b"data,here")

        calls = [str(c) for c in mock_dl.call_args_list]
        assert any("backtest_results.csv" in c for c in calls)

    def test_chart_button_shown_when_data_provided(self):
        with patch("streamlit.subheader"), \
             patch("streamlit.columns", return_value=[self._make_col_mock(), self._make_col_mock()]), \
             patch("streamlit.download_button") as mock_dl:
            from app.ui_components import display_downloads
            display_downloads(chart_data=b"\x89PNG")

        calls = [str(c) for c in mock_dl.call_args_list]
        assert any("backtest_chart.png" in c for c in calls)

    def test_no_buttons_when_no_data(self):
        with patch("streamlit.subheader"), \
             patch("streamlit.columns", return_value=[self._make_col_mock(), self._make_col_mock()]), \
             patch("streamlit.download_button") as mock_dl:
            from app.ui_components import display_downloads
            display_downloads()

        mock_dl.assert_not_called()
