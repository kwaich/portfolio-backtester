# ETF Backtester

A lightweight, flexible Python utility for backtesting ETF portfolio strategies against benchmarks. Download historical price data, calculate performance metrics, and visualize results with ease.

## Features

- **Portfolio Backtesting**: Test buy-and-hold strategies with customizable weights
- **Comprehensive Metrics**: Returns, CAGR, Sharpe ratio, Sortino ratio, volatility, and maximum drawdown
- **Data Caching**: Automatic caching of price data for faster iteration
- **Flexible Visualization**: Generate publication-ready charts or interactive plots
- **Easy CLI**: Simple command-line interface with sensible defaults
- **Well-Tested**: Comprehensive unit test coverage

## Quick Start

### Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

Run a backtest with default settings (VDCP.L/VHYD.L vs VWRA.L):

```bash
python backtest.py
```

### Custom Portfolio

Test a custom portfolio allocation:

```bash
python backtest.py \
  --tickers AAPL MSFT GOOGL \
  --weights 0.5 0.3 0.2 \
  --benchmark SPY \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --capital 100000 \
  --output results/my_backtest.csv
```

### Visualization

Generate charts from backtest results:

```bash
python plot_backtest.py --csv results/my_backtest.csv --output charts/my_backtest
```

This creates two PNG files:
- `my_backtest_value.png` - Portfolio vs benchmark value over time
- `my_backtest_active.png` - Active return (portfolio - benchmark)

For interactive plots, omit the `--output` parameter:

```bash
python plot_backtest.py --csv results/my_backtest.csv
```

## Command-Line Options

### backtest.py

| Option | Default | Description |
|--------|---------|-------------|
| `--tickers` | VDCP.L VHYD.L | Portfolio ticker symbols |
| `--weights` | 0.5 0.5 | Portfolio weights (auto-normalized) |
| `--benchmark` | VWRA.L | Benchmark ticker for comparison |
| `--start` | 2018-01-01 | Backtest start date (YYYY-MM-DD) |
| `--end` | Today | Backtest end date (YYYY-MM-DD) |
| `--capital` | 100000 | Initial capital |
| `--output` | None | CSV file path for detailed results |
| `--no-cache` | False | Disable data caching |

### plot_backtest.py

| Option | Description |
|--------|-------------|
| `--csv` | Path to backtest CSV output (required) |
| `--output` | Prefix for PNG files (optional, shows interactive plots if omitted) |

## Performance Metrics

The backtester calculates the following metrics:

- **Ending Value**: Final portfolio/benchmark value
- **Total Return**: Overall return over the period
- **CAGR**: Compound Annual Growth Rate
- **Volatility**: Annualized standard deviation of returns (252 trading days)
- **Sharpe Ratio**: Risk-adjusted return (assuming 0% risk-free rate)
- **Sortino Ratio**: Return relative to downside deviation only
- **Max Drawdown**: Largest peak-to-trough decline
- **Active Return**: Portfolio return minus benchmark return

## Example Output

```
======================================================================
BACKTEST RESULTS
======================================================================
Capital: $100,000.00
Time Span: 2018-01-02 → 2024-12-31
Portfolio: VDCP.L (50.0%), VHYD.L (50.0%)
Benchmark: VWRA.L
----------------------------------------------------------------------

PORTFOLIO PERFORMANCE:
  Ending Value:    $     142,538.21
  Total Return:            42.54%
  CAGR:                     5.32%
  Volatility:              12.45%
  Sharpe Ratio:             0.43
  Sortino Ratio:            0.61
  Max Drawdown:           -18.32%

BENCHMARK PERFORMANCE:
  Ending Value:    $     138,291.45
  Total Return:            38.29%
  CAGR:                     4.87%
  Volatility:              14.21%
  Sharpe Ratio:             0.34
  Sortino Ratio:            0.48
  Max Drawdown:           -22.15%

RELATIVE PERFORMANCE:
  Active Return:           +4.25%
  Active CAGR:             +0.45%
======================================================================
```

## Data Caching

Price data is automatically cached in `.cache/` to speed up repeated backtests:

- Cache files are named by MD5 hash of tickers and date range
- Cached data is reused for identical requests
- Use `--no-cache` to force fresh downloads
- Cache directory is gitignored by default

## Project Structure

```
backtester/
├── backtest.py           # Main backtesting engine
├── plot_backtest.py      # Visualization utility
├── test_backtest.py      # Unit tests
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── PROJECT_SUMMARY.md   # Additional documentation
├── CLAUDE.md           # AI assistant guide
├── .gitignore          # Git ignore rules
├── .venv/              # Virtual environment (gitignored)
├── .cache/             # Price data cache (gitignored)
├── results/            # CSV outputs (gitignored)
└── charts/             # PNG outputs (gitignored)
```

## Development

### Running Tests

```bash
# Run all tests
pytest test_backtest.py -v

# Run with coverage
pytest test_backtest.py --cov=backtest --cov-report=html

# Run specific test class
pytest test_backtest.py::TestSummarize -v
```

### Code Quality

The codebase follows these conventions:
- PEP 8 style guidelines
- Type hints for all functions
- Comprehensive docstrings
- Extensive error handling with helpful messages

## Requirements

- Python 3.9+
- numpy >= 1.24.0
- pandas >= 2.0.0
- yfinance >= 0.2.0
- matplotlib >= 3.7.0
- pytest >= 7.0.0 (for testing)

## Data Source

Historical price data is fetched from Yahoo Finance via the [yfinance](https://github.com/ranaroussi/yfinance) library. An internet connection is required for initial data downloads (cached data can be used offline).

## Limitations

- **Buy-and-hold only**: No rebalancing or dynamic strategies
- **No transaction costs**: Assumes zero trading fees
- **No taxes**: Does not model tax implications
- **Daily granularity**: Uses daily adjusted close prices
- **Yahoo Finance dependency**: Data quality depends on Yahoo Finance
- **No dividends toggle**: Always uses adjusted prices (includes dividends)

## Future Enhancements

Potential improvements (see [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for details):

- Periodic rebalancing strategies
- Transaction cost modeling
- Configuration file support (YAML/JSON)
- Multiple time period analysis
- Currency conversion for multi-currency portfolios
- Interactive dashboard (Streamlit/Dash)
- Additional data sources beyond Yahoo Finance

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest test_backtest.py -v`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is open source and available for educational and personal use.

## Troubleshooting

### "Missing data for tickers"

- Verify ticker symbols are correct for your exchange (e.g., `.L` for London)
- Check that tickers were trading during your specified date range
- Try a more recent start date

### "No overlapping data"

- Portfolio tickers and benchmark must have overlapping trading history
- Use `--start` date after all securities began trading
- Check for delisted or recently IPO'd securities

### Cache issues

- Clear cache: `rm -rf .cache/`
- Disable cache: use `--no-cache` flag
- Cache location: `.cache/` in project root

### Import errors

- Ensure virtual environment is activated: `source .venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (requires 3.9+)

## Support

For issues, questions, or suggestions:
- Open an issue in the repository
- Check existing documentation in PROJECT_SUMMARY.md
- Review unit tests for usage examples

---

**Happy Backtesting!** 📈
