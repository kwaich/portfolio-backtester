# Repository Guidelines

## Project Structure & Module Organization
`backtest.py` houses the CLI engine, portfolio math, and cache helpers, while `plot_backtest.py` renders PNG dashboards from CSV outputs. The Streamlit experience lives in `app/` (component modules) and `app.py` (entrypoint). Domain docs reside in `docs/`, CI lives in `.github/workflows/`, fixtures and generated media go to `results/` and `charts/`, and the full pytest suite sits under `tests/`. Keep heavy data in `.cache/` (auto gitignored) and commit only deterministic assets.

## Build, Test, and Development Commands
Create the virtualenv and install dependencies: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. Run a CLI backtest (`python backtest.py --tickers AAPL MSFT --weights 0.6 0.4 --benchmark SPY`; add `--verbose` for debug logging) and plot outputs with `python plot_backtest.py --csv results/sample.csv --dashboard`. Launch the Streamlit UI via `streamlit run app.py`. Execute fast feedback loops with `pytest -v`, target modules using `pytest tests/test_backtest.py -k compute_metrics`, and inspect coverage using `pytest --cov=backtest --cov=app --cov-report=term-missing`.

## Coding Style & Naming Conventions
Follow PEP 8 with <100 character lines, snake_case functions, CamelCase classes, and descriptive module-level docstrings. Annotate signatures with type hints and prefer dataclasses or TypedDicts for structured payloads. Use explicit logger instances (`logging.getLogger(__name__)`), no ad-hoc prints inside core flows. Tests should mirror the module under test and use triple-A structure with names such as `test_download_prices_handles_nan`. When adding presets or chart assets, align filenames with the ticker tuple (`charts/vdra_vs_vwra_*`).

## Testing Guidelines
Pytest is the single source of truth (386 passing cases as of `docs/TESTING_GUIDE.md`); keep overall coverage ≥85% and extend suites whenever functionality moves. Add regression tests before fixing bugs, and prefer module-specific files (`tests/test_app.py` for Streamlit behaviors, `tests/test_integration.py` for CLI-to-chart workflows, `tests/test_benchmarks.py` for performance baselines, `tests/test_properties.py` for hypothesis-driven checks). High-value additions include data-quality edge cases, cache TTL handling, and UI presets. Tests that touch filesystem artifacts should target `tmp_path` or sandboxed folders instead of writing into `results/`.

## Commit & Pull Request Guidelines
Recent history shows imperative, scope-tagged commits (`Accessibility & Visual Hierarchy Enhancements (#11)`, `docs: remove obsolete implementation...`). Keep titles under 72 characters, describe *what* and *why*, and reference issue/PR numbers when applicable. Before opening a PR, run `pytest -v` and any affected CLI/UI flows; CI validates via `.github/workflows/python-package.yml`. Attach formatted metrics or screenshots for chart/UI tweaks. PRs should summarize behavioral impact, list validation steps, and call out new configuration knobs so reviewers can verify real-money calculations safely.

## Agent & Automation Notes
If you are using AI tooling, review `CLAUDE.md` alongside this guide: it defines agent permissions, escalation rules, and safe-edit workflows specific to the portfolio backtester. Align prompts with those guardrails, log every automated change, and ensure agents defer to human reviewers for financial or compliance-sensitive updates.
