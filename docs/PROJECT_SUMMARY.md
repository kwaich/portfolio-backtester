# Portfolio Backtester Utility – Documentation Hub

This document provides a concise roadmap for the repository. Detailed user instructions, developer workflows, and module-level notes now live in focused guides so we avoid repeating the same content in multiple places.

## Where to Start

- **User workflows, installation, CLI & UI usage** → [`README.md`](../README.md#quick-start)
- **Developer workflows and coding standards** → [`docs/DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md)
- **Testing expectations, coverage targets, and regression workflow** → [`docs/TESTING_GUIDE.md`](TESTING_GUIDE.md)
- **File-by-file deep dive** → [`docs/FILE_REFERENCE.md`](FILE_REFERENCE.md)
- **Change history** → [`docs/CHANGELOG.md`](CHANGELOG.md)

## High-Level Architecture

| Component | Location | Highlights |
|-----------|----------|------------|
| Core CLI backtester | `backtest.py` | Data download + caching, portfolio metrics, DCA/IRR handling, logging, input validation |
| Streamlit web UI | `app/` package | Modular architecture (config, presets, validation/state manager, UI components, charts, ticker search, main orchestration) |
| Visualization helper | `plot_backtest.py` | Reads CSV output, produces dashboard/individual charts, logging + validation |
| Test suite | `tests/` | 256 automated tests spanning engine, UI, ticker utilities, session state, and integration coverage (`pytest -v`) |

See `docs/FILE_REFERENCE.md` for module-by-module details.

## Key Capabilities

- Buy-and-hold backtesting with rebalancing, DCA, and smart caching
- Comprehensive metrics: CAGR, IRR, Sharpe, Sortino, volatility, drawdown, delta indicators
- Plotly-based Streamlit dashboard with presets, multiple benchmarks, rolling analyses, and accessible color palette
- Matplotlib helper to batch-generate PNG dashboards
- 256-test suite ensuring ~88% line coverage (tracked in `docs/TESTING_GUIDE.md`)

## Documentation Map

```
docs/
├── CHANGELOG.md       # Chronological feature history
├── DEVELOPER_GUIDE.md # Hands-on workflows, coding conventions, scenarios
├── FILE_REFERENCE.md  # Detailed explanation of every module/file
├── PROJECT_SUMMARY.md # This roadmap
├── TESTING_GUIDE.md   # Authoritative testing + coverage guide
└── ... (CLAUDE.md at repo root for AI assistant context)
```

Each guide owns its domain to prevent duplication:
- README owns quick start, user instructions, and CLI/Web UI walkthroughs.
- DEVELOPER_GUIDE layers on contributor workflows without repeating setup/test commands—those link back to README and TESTING_GUIDE.
- TESTING_GUIDE tracks the single source of truth for counts/coverage.
- FILE_REFERENCE documents every module once; other docs link to it.

## Recent Highlights

For full details see `docs/CHANGELOG.md`, but the latest notable work includes:
- Colorblind-accessible palettes and revamped active-return visuals across Streamlit and matplotlib outputs
- New Streamlit presets, rolling metrics, and centralized `state_manager.py`
- Expanded documentation/tests bundled with the refactors (all tracked via 256 passing tests)

## Need Something Else?

- Troubleshooting tips, CLI flags, and visualization instructions live in the README.
- Contribution checklists and manual testing workflows live in the developer guide.
- For AI/automation context, consult `CLAUDE.md`.

By funneling updates to the appropriate guide we can keep every document focused while eliminating duplicated sections. When in doubt, link rather than copy.
