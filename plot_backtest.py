"""Plot helper for the portfolio backtest CSV output.

Example:

source .venv/bin/activate
python plot_backtest.py --csv results/backtest_series.csv \
    --output charts/backtest

This will save two PNGs (value + active return). If --output is omitted the
plots are shown interactively using matplotlib's default backend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot portfolio vs benchmark")
    parser.add_argument("--csv", required=True, help="CSV exported via backtest.py")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional prefix for PNGs; e.g. charts/run will create run_value.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    if "date" not in df.columns:
        raise SystemExit("CSV missing 'date' column; run backtest.py with --output")
    df = df.set_index(pd.to_datetime(df["date"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    df[["portfolio_value", "benchmark_value"]].plot(ax=ax)
    ax.set_title("Portfolio vs Benchmark Value")
    ax.set_ylabel("Value ($)")
    ax.grid(True, alpha=0.3)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    df["active_return"].plot(ax=ax2, color="purple")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Active Return")
    ax2.set_ylabel("Return")
    ax2.grid(True, alpha=0.3)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{args.output}_value.png", dpi=150, bbox_inches="tight")
        fig2.savefig(f"{args.output}_active.png", dpi=150, bbox_inches="tight")
        print(f"Saved plots to {args.output}_value.png and {args.output}_active.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
