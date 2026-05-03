"""Unit tests for plot_backtest.py"""

from __future__ import annotations

import logging

import pytest

import plot_backtest


class TestPlotParseArgs:
    """Test argument parsing for plot_backtest CLI"""

    def test_verbose_flag(self):
        """Test --verbose flag parsing"""
        args = plot_backtest.parse_args(["--csv", "test.csv", "--verbose"])
        assert args.verbose is True

    def test_verbose_short_flag(self):
        """Test -v short flag parsing"""
        args = plot_backtest.parse_args(["--csv", "test.csv", "-v"])
        assert args.verbose is True

    def test_verbose_default(self):
        """Test verbose defaults to False"""
        args = plot_backtest.parse_args(["--csv", "test.csv"])
        assert args.verbose is False

    def test_csv_required(self):
        """Test --csv is required"""
        with pytest.raises(SystemExit):
            plot_backtest.parse_args([])


class TestPlotLoggingSetup:
    """Test _setup_logging function in plot_backtest"""

    def test_setup_logging_verbose(self):
        """Test _setup_logging sets DEBUG level when verbose=True"""
        original_root = logging.getLogger().level
        original_module = logging.getLogger("plot_backtest").level
        try:
            plot_backtest._setup_logging(verbose=True)
            assert logging.getLogger().level == logging.DEBUG
            assert logging.getLogger("plot_backtest").level == logging.DEBUG
        finally:
            logging.getLogger().setLevel(original_root)
            logging.getLogger("plot_backtest").setLevel(original_module)

    def test_setup_logging_default(self):
        """Test _setup_logging sets INFO level when verbose=False"""
        original_root = logging.getLogger().level
        original_module = logging.getLogger("plot_backtest").level
        try:
            plot_backtest._setup_logging(verbose=False)
            assert logging.getLogger().level == logging.INFO
            assert logging.getLogger("plot_backtest").level == logging.INFO
        finally:
            logging.getLogger().setLevel(original_root)
            logging.getLogger("plot_backtest").setLevel(original_module)
