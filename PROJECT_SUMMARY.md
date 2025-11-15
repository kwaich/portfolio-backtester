# Backtest Utility

This directory contains a lightweight ETF backtester (`backtest.py`) plus a
plotting helper (`plot_backtest.py`). The workflow is:

1. Create/activate the virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
2. Install deps: `python -m pip install numpy pandas yfinance matplotlib`.
3. Run `backtest.py` to pull daily data via yfinance, calculate portfolio vs.
   benchmark metrics, and optionally export a CSV.
4. Use `plot_backtest.py` to convert the CSV into PNG charts or interactive plots.

## Scripts

- `backtest.py` — CLI to download prices (defaults: VDCP.L/VHYD.L, benchmark
  VWRA.L), compute buy-and-hold values/returns, and print stats. Supports
  `--tickers`, `--weights`, `--start`, `--end`, `--capital`, `--benchmark`, and
  `--output <csv>` for time-series export. Example:

  ```bash
  source .venv/bin/activate
  python backtest.py --start 2018-01-01 --end 2024-12-31 \
      --capital 100000 --weights 0.5 0.5 --benchmark VWRA.L \
      --output results/backtest_series.csv
  ```

- `plot_backtest.py` — Reads the CSV (must include a `date` column plus the columns
  emitted by `backtest.py`) and plots portfolio vs. benchmark value plus active
  return. Use `--output charts/run` to save PNGs (`run_value.png`,
  `run_active.png`); omit `--output` to show the plots interactively.

## Notes

- All dependencies live inside `.venv`; update pip via
  `./.venv/bin/python -m pip install --upgrade pip` if desired.
- Both scripts rely on network access to fetch data from Yahoo Finance via
  yfinance.
- Export CSVs into a `results/` folder (ignored by git) to keep working copies
  out of version control.
