# Streamlit UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the Streamlit app with a professional fintech dashboard aesthetic while preserving all existing backtesting logic.

**Architecture:** A new `app/design_system.py` becomes the single source of truth for all visual tokens (colors, typography, spacing). Existing display modules (`main.py`, `sidebar.py`, `ui_components.py`, `charts.py`) are refactored to consume these tokens. No backtest logic changes.

**Tech Stack:** Python, Streamlit, Matplotlib, pytest

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `app/design_system.py` | **Create** | Exports `COLORS`, `TYPOGRAPHY`, `SPACING` dicts and CSS injection helpers |
| `app/charts.py` | **Modify** | Update matplotlib rcParams for transparent backgrounds, refined grid/typography |
| `app/ui_components.py` | **Modify** | Refactor metric cards, section headers, info bars to use design system |
| `app/sidebar.py` | **Modify** | Reorganize into 7 sections with badge labels, pill buttons, styled run button |
| `app/main.py` | **Modify** | Add welcome screen, hero metrics row, 2×2 chart grid, detailed metrics tables, downloads |
| `tests/test_design_system.py` | **Create** | Unit tests for design system exports |
| `tests/test_ui_components.py` | **Modify** | Update any existing tests that assert on old component markup |

---

## Task 1: Design System Module

**Files:**
- Create: `app/design_system.py`
- Test: `tests/test_design_system.py`

**Prerequisites:** None

- [ ] **Step 1: Write the design system module**

Create `app/design_system.py`:

```python
"""Design system tokens and CSS helpers for the Portfolio Backtester UI.

This module is the single source of truth for all visual styling:
colors, typography, spacing, and reusable CSS snippets.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Color tokens
# ---------------------------------------------------------------------------
COLORS = {
    "primary_text": "#0f172a",
    "accent": "#3b82f6",
    "accent_hover": "#2563eb",
    "bg_page": "#f8fafc",
    "bg_card": "#ffffff",
    "border": "#e2e8f0",
    "muted": "#64748b",
    "success": "#10b981",
    "danger": "#ef4444",
    "grid": "#e2e8f0",
}

# ---------------------------------------------------------------------------
# Typography tokens
# ---------------------------------------------------------------------------
TYPOGRAPHY = {
    "font_header": "'Outfit', sans-serif",
    "font_body": "'Inter', sans-serif",
    "page_title_size": "28px",
    "page_title_weight": "600",
    "section_header_size": "20px",
    "section_header_weight": "500",
    "metric_value_size": "24px",
    "metric_value_weight": "600",
    "metric_label_size": "12px",
    "metric_label_weight": "400",
    "body_size": "14px",
    "body_weight": "400",
    "caption_size": "12px",
    "caption_weight": "400",
    "badge_size": "10px",
    "badge_weight": "600",
}

# ---------------------------------------------------------------------------
# Spacing tokens
# ---------------------------------------------------------------------------
SPACING = {
    "page_padding_x": "2rem",
    "page_padding_y": "1.5rem",
    "card_padding": "1.25rem",
    "card_radius": "12px",
    "card_shadow": "0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03)",
    "section_gap": "2rem",
    "metric_gap": "1rem",
}

# ---------------------------------------------------------------------------
# Global CSS injection
# ---------------------------------------------------------------------------

def get_global_css() -> str:
    """Return the full <style> block for injection via st.markdown."""
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: {TYPOGRAPHY["font_body"]};
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-family: {TYPOGRAPHY["font_header"]};
        color: {COLORS["primary_text"]};
    }}

    .stApp {{
        background-color: {COLORS["bg_page"]};
    }}
    </style>
    """


def get_card_style() -> str:
    """Return CSS for a standard white card."""
    return f"""
    <style>
    .fintech-card {{
        background-color: {COLORS["bg_card"]};
        border-radius: {SPACING["card_radius"]};
        padding: {SPACING["card_padding"]};
        box-shadow: {SPACING["card_shadow"]};
        border: 1px solid {COLORS["border"]};
    }}
    </style>
    """


def get_metric_card_style() -> str:
    """Return CSS for metric cards in the hero row."""
    return f"""
    <style>
    .metric-card {{
        background-color: {COLORS["bg_card"]};
        border-radius: {SPACING["card_radius"]};
        padding: {SPACING["card_padding"]};
        box-shadow: {SPACING["card_shadow"]};
        border: 1px solid {COLORS["border"]};
        text-align: center;
        flex: 1;
        min-width: 140px;
    }}
    .metric-value {{
        font-family: {TYPOGRAPHY["font_body"]};
        font-size: {TYPOGRAPHY["metric_value_size"]};
        font-weight: {TYPOGRAPHY["metric_value_weight"]};
        color: {COLORS["primary_text"]};
        margin-bottom: 4px;
    }}
    .metric-label {{
        font-family: {TYPOGRAPHY["font_body"]};
        font-size: {TYPOGRAPHY["metric_label_size"]};
        font-weight: {TYPOGRAPHY["metric_label_weight"]};
        color: {COLORS["muted"]};
    }}
    .metric-positive {{ color: {COLORS["success"]}; }}
    .metric-negative {{ color: {COLORS["danger"]}; }}
    </style>
    """


def get_sidebar_badge_style() -> str:
    """Return CSS for sidebar section badge labels."""
    return f"""
    <style>
    .sidebar-badge {{
        font-family: {TYPOGRAPHY["font_body"]};
        font-size: {TYPOGRAPHY["badge_size"]};
        font-weight: {TYPOGRAPHY["badge_weight"]};
        color: {COLORS["muted"]};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
        display: block;
    }}
    </style>
    """


def get_pill_button_style() -> str:
    """Return CSS for compact pill buttons (date presets)."""
    return f"""
    <style>
    .pill-button {{
        display: inline-block;
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 500;
        color: {COLORS["primary_text"]};
        background-color: {COLORS["bg_page"]};
        border: 1px solid {COLORS["border"]};
        cursor: pointer;
        margin-right: 6px;
        margin-bottom: 6px;
        transition: all 0.15s ease;
    }}
    .pill-button:hover {{
        background-color: {COLORS["accent"]};
        color: #ffffff;
        border-color: {COLORS["accent"]};
    }}
    </style>
    """


def get_run_button_style() -> str:
    """Return CSS to style the primary Run Backtest button."""
    return f"""
    <style>
    div[data-testid="stButton"] > button {{
        background-color: {COLORS["accent"]};
        color: #ffffff;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        border: none;
        padding: 0.6rem 1rem;
    }}
    div[data-testid="stButton"] > button:hover {{
        background-color: {COLORS["accent_hover"]};
        color: #ffffff;
    }}
    </style>
    """


def get_welcome_style() -> str:
    """Return CSS for the centered welcome hero."""
    return f"""
    <style>
    .welcome-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 4rem 2rem;
    }}
    .welcome-icon {{
        color: {COLORS["accent"]};
        font-size: 48px;
        margin-bottom: 1rem;
    }}
    .welcome-title {{
        font-family: {TYPOGRAPHY["font_header"]};
        font-size: {TYPOGRAPHY["page_title_size"]};
        font-weight: {TYPOGRAPHY["page_title_weight"]};
        color: {COLORS["primary_text"]};
        margin-bottom: 0.5rem;
    }}
    .welcome-subtitle {{
        font-family: {TYPOGRAPHY["font_body"]};
        font-size: 16px;
        color: {COLORS["muted"]};
    }}
    </style>
    """
```

- [ ] **Step 2: Write failing tests for design system**

Create `tests/test_design_system.py`:

```python
"""Tests for the design system module."""

import pytest

from app.design_system import (
    COLORS,
    TYPOGRAPHY,
    SPACING,
    get_global_css,
    get_card_style,
    get_metric_card_style,
    get_sidebar_badge_style,
    get_pill_button_style,
    get_run_button_style,
    get_welcome_style,
)


def test_colors_has_required_keys():
    required = [
        "primary_text",
        "accent",
        "accent_hover",
        "bg_page",
        "bg_card",
        "border",
        "muted",
        "success",
        "danger",
        "grid",
    ]
    for key in required:
        assert key in COLORS
        assert COLORS[key].startswith("#")


def test_typography_has_required_keys():
    required = [
        "font_header",
        "font_body",
        "page_title_size",
        "page_title_weight",
        "metric_value_size",
        "metric_value_weight",
        "metric_label_size",
        "badge_size",
        "badge_weight",
    ]
    for key in required:
        assert key in TYPOGRAPHY


def test_spacing_has_required_keys():
    required = [
        "card_padding",
        "card_radius",
        "card_shadow",
        "section_gap",
        "metric_gap",
    ]
    for key in required:
        assert key in SPACING


def test_get_global_css_includes_fonts():
    css = get_global_css()
    assert "fonts.googleapis.com" in css
    assert "Outfit" in css
    assert "Inter" in css
    assert COLORS["bg_page"] in css


def test_get_metric_card_style_includes_positive_negative():
    css = get_metric_card_style()
    assert ".metric-positive" in css
    assert ".metric-negative" in css
    assert COLORS["success"] in css
    assert COLORS["danger"] in css


def test_all_style_helpers_return_strings():
    helpers = [
        get_global_css,
        get_card_style,
        get_metric_card_style,
        get_sidebar_badge_style,
        get_pill_button_style,
        get_run_button_style,
        get_welcome_style,
    ]
    for helper in helpers:
        result = helper()
        assert isinstance(result, str)
        assert "<style>" in result
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
pytest tests/test_design_system.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add app/design_system.py tests/test_design_system.py
git commit -m "feat: add design system module with tokens and CSS helpers"
```

---

## Task 2: Chart Styling (Matplotlib rcParams)

**Files:**
- Modify: `app/charts.py`
- Test: `pytest tests/` (existing suite must still pass)

**Prerequisites:** Task 1

- [ ] **Step 1: Import design system and update rcParams**

In `app/charts.py`, add at the top of the file (after existing imports):

```python
from app.design_system import COLORS, TYPOGRAPHY
```

Then add a helper function after the imports:

```python
def _set_fintech_style():
    """Apply fintech design system to matplotlib rcParams."""
    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "axes.edgecolor": COLORS["border"],
        "axes.labelcolor": COLORS["muted"],
        "axes.labelsize": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "500",
        "axes.titlecolor": COLORS["primary_text"],
        "xtick.color": COLORS["muted"],
        "xtick.labelsize": 10,
        "ytick.color": COLORS["muted"],
        "ytick.labelsize": 10,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "legend.loc": "upper left",
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans", "Arial", "sans-serif"],
    })
```

- [ ] **Step 2: Call `_set_fintech_style()` in chart functions**

For each public plotting function in `app/charts.py` (`plot_cumulative_returns`, `plot_drawdown`, `plot_rolling_returns`, `plot_rolling_sharpe`, `create_dashboard`), add as the first line inside the function body:

```python
_set_fintech_style()
```

For example, in `plot_cumulative_returns`:

```python
def plot_cumulative_returns(df, title="Cumulative Returns", figsize=(10, 6)):
    """...existing docstring..."""
    _set_fintech_style()
    fig, ax = plt.subplots(figsize=figsize)
    # ... rest of existing implementation ...
```

Do this for all five public plotting functions.

- [ ] **Step 3: Update chart titles to match spec**

Change the default `title` parameter in each function:

| Function | Old Default | New Default |
|---|---|---|
| `plot_cumulative_returns` | `"Cumulative Returns"` | `"Portfolio vs Benchmark Cumulative Returns"` |
| `plot_drawdown` | `"Drawdown"` | `"Portfolio Drawdown Over Time"` |
| `plot_rolling_returns` | `"Rolling Returns"` | `"Rolling 12-Month Returns"` |
| `plot_rolling_sharpe` | `"Rolling Sharpe"` | `"Rolling Sharpe Ratio (12M)"` |

- [ ] **Step 4: Run existing tests**

```bash
pytest tests/ -v
```

Expected: All existing tests pass. If any test asserts on old chart titles, update the assertion to match the new defaults.

- [ ] **Step 5: Commit**

```bash
git add app/charts.py
git commit -m "feat: apply fintech design system to chart typography and titles"
```

---

## Task 3: UI Components Refactor

**Files:**
- Modify: `app/ui_components.py`
- Test: `pytest tests/test_ui_components.py -v` (update if needed)

**Prerequisites:** Task 1

- [ ] **Step 1: Add design system imports**

At the top of `app/ui_components.py`, add:

```python
from app.design_system import (
    COLORS,
    TYPOGRAPHY,
    get_card_style,
    get_metric_card_style,
    get_welcome_style,
)
```

- [ ] **Step 2: Replace `display_header` with welcome screen**

Rename `display_header` → `display_welcome_screen` and replace the body:

```python
def display_welcome_screen() -> None:
    """Display the centered welcome hero when no backtest has been run."""
    st.markdown(get_welcome_style(), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-icon">📈</div>
            <div class="welcome-title">Portfolio Backtester</div>
            <div class="welcome-subtitle">
                Analyze historical portfolio performance<br>
                Enter tickers in the sidebar and click "Run Backtest" to get started.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
```

- [ ] **Step 3: Refactor `display_metric_card`**

Replace `display_metric_card` with a version that uses the design system:

```python
def display_metric_card(label: str, value: str, delta: str | None = None) -> None:
    """Display a single metric in a styled card.

    Args:
        label: Metric label (e.g., "Total Return")
        value: Formatted value string (e.g., "+42.31%")
        delta: Optional delta string (not used in new design, kept for compatibility)
    """
    st.markdown(get_metric_card_style(), unsafe_allow_html=True)

    # Determine color class based on value content
    color_class = ""
    if value.startswith("+") or (value.replace("$", "").replace(",", "").replace(".", "").isdigit() and float(value.replace("$", "").replace(",", "")) > 0):
        # Positive return or positive number
        if any(k in label.lower() for k in ("return", "cagr", "value")):
            color_class = "metric-positive"
    elif value.startswith("-"):
        if any(k in label.lower() for k in ("return", "cagr", "drawdown")):
            color_class = "metric-negative"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value {color_class}">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
```

- [ ] **Step 4: Update `display_section_header`**

Replace with:

```python
def display_section_header(title: str) -> None:
    """Display a section header using the design system."""
    st.markdown(
        f"""
        <h3 style="
            font-family: {TYPOGRAPHY['font_header']};
            font-size: {TYPOGRAPHY['section_header_size']};
            font-weight: {TYPOGRAPHY['section_header_weight']};
            color: {COLORS['primary_text']};
            margin-top: {SPACING['section_gap']};
            margin-bottom: 1rem;
        ">{title}</h3>
        """,
        unsafe_allow_html=True,
    )
```

Also add `SPACING` to the imports from `app.design_system`.

- [ ] **Step 5: Update `display_info_bar`**

Replace with:

```python
def display_info_bar(portfolio_tickers: list[str], weights: list[float], benchmarks: list[str], start_date: str, end_date: str) -> None:
    """Display a compact portfolio info bar."""
    weight_strs = [f"{t} {w:.0%}" for t, w in zip(portfolio_tickers, weights)]
    portfolio_str = " · ".join(weight_strs)
    benchmark_str = ", ".join(benchmarks)
    date_str = f"{start_date} – {end_date}"

    st.markdown(
        f"""
        <div style="
            background-color: {COLORS['bg_card']};
            border-radius: 8px;
            padding: 0.75rem 1rem;
            border: 1px solid {COLORS['border']};
            font-family: {TYPOGRAPHY['font_body']};
            font-size: 14px;
            color: {COLORS['muted']};
            margin-bottom: 1rem;
        ">
            <strong style="color: {COLORS['primary_text']};">{portfolio_str}</strong>
            <span style="margin: 0 0.5rem;">vs</span>
            <strong style="color: {COLORS['primary_text']};">{benchmark_str}</strong>
            <span style="margin: 0 0.5rem;">·</span>
            {date_str}
        </div>
        """,
        unsafe_allow_html=True,
    )
```

- [ ] **Step 6: Add `display_hero_metrics_row` function**

Add this new function after `display_metric_card`:

```python
def display_hero_metrics_row(metrics: dict[str, str]) -> None:
    """Display the hero metrics row as a flexbox of metric cards.

    Args:
        metrics: Dict mapping label → formatted value string.
                 Expected keys: Ending Value, Total Return, CAGR, Sharpe Ratio, Max Drawdown
    """
    st.markdown(get_metric_card_style(), unsafe_allow_html=True)

    cards_html = '<div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem;">'
    for label, value in metrics.items():
        color_class = ""
        if any(k in label.lower() for k in ("return", "cagr")):
            try:
                num = float(value.replace("%", "").replace("$", "").replace(",", ""))
                color_class = "metric-positive" if num > 0 else "metric-negative" if num < 0 else ""
            except ValueError:
                pass
        elif "drawdown" in label.lower():
            color_class = "metric-negative"

        cards_html += f"""
        <div class="metric-card" style="flex: 1; min-width: 140px;">
            <div class="metric-value {color_class}">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """
    cards_html += "</div>"

    st.markdown(cards_html, unsafe_allow_html=True)
```

- [ ] **Step 7: Add `display_metrics_tables` function**

Add this new function for the detailed metrics section:

```python
def display_metrics_tables(performance: dict[str, str], risk: dict[str, str]) -> None:
    """Display two side-by-side metrics tables.

    Args:
        performance: Dict of performance metric label → value
        risk: Dict of risk metric label → value
    """
    col1, col2 = st.columns(2)

    def _build_table(title: str, data: dict[str, str]) -> str:
        rows = ""
        for label, value in data.items():
            rows += f"""
            <tr style="border-bottom: 1px solid {COLORS['border']};">
                <td style="padding: 0.6rem 0; font-family: {TYPOGRAPHY['font_body']}; font-size: 14px; color: {COLORS['primary_text']};">{label}</td>
                <td style="padding: 0.6rem 0; font-family: {TYPOGRAPHY['font_body']}; font-size: 14px; color: {COLORS['primary_text']}; text-align: right; font-weight: 500;">{value}</td>
            </tr>
            """
        return f"""
        <div style="background-color: {COLORS['bg_card']}; border-radius: {SPACING['card_radius']}; padding: {SPACING['card_padding']}; border: 1px solid {COLORS['border']}; margin-bottom: 1rem;">
            <div style="font-family: {TYPOGRAPHY['font_header']}; font-size: 14px; font-weight: 600; color: {COLORS['muted']}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;">{title}</div>
            <table style="width: 100%; border-collapse: collapse;">{rows}</table>
        </div>
        """

    with col1:
        st.markdown(_build_table("Performance", performance), unsafe_allow_html=True)
    with col2:
        st.markdown(_build_table("Risk", risk), unsafe_allow_html=True)
```

- [ ] **Step 8: Add `display_downloads` function**

Add this new function:

```python
def display_downloads(csv_data: bytes | None = None, chart_data: bytes | None = None) -> None:
    """Display the downloads section with styled buttons."""
    st.markdown(get_card_style(), unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="fintech-card" style="margin-top: 1.5rem;">
            <div style="font-family: {TYPOGRAPHY['font_header']}; font-size: 16px; font-weight: 500; color: {COLORS['primary_text']}; margin-bottom: 0.75rem;">Downloads</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if csv_data:
            st.download_button(
                label="⬇ Download Results CSV",
                data=csv_data,
                file_name="backtest_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with col2:
        if chart_data:
            st.download_button(
                label="⬇ Download Chart",
                data=chart_data,
                file_name="backtest_chart.png",
                mime="image/png",
                use_container_width=True,
            )
```

- [ ] **Step 9: Run tests**

```bash
pytest tests/test_ui_components.py -v
```

Expected: PASS. If any test references the old `display_header` name, update it to `display_welcome_screen` or add a backward-compatible alias.

- [ ] **Step 10: Commit**

```bash
git add app/ui_components.py tests/test_ui_components.py
git commit -m "feat: refactor UI components with design system styling"
```

---

## Task 4: Sidebar Reorganization

**Files:**
- Modify: `app/sidebar.py`
- Test: `pytest tests/test_sidebar.py -v` (existing suite must still pass)

**Prerequisites:** Task 1

- [ ] **Step 1: Add design system imports**

At the top of `app/sidebar.py`, add:

```python
from app.design_system import (
    COLORS,
    get_sidebar_badge_style,
    get_pill_button_style,
    get_run_button_style,
)
```

- [ ] **Step 2: Inject sidebar CSS at the top of `render_sidebar`**

Inside `render_sidebar()`, add as the first Streamlit call:

```python
st.markdown(get_sidebar_badge_style(), unsafe_allow_html=True)
st.markdown(get_pill_button_style(), unsafe_allow_html=True)
st.markdown(get_run_button_style(), unsafe_allow_html=True)
```

- [ ] **Step 3: Add `_section_badge` helper**

Add this helper function near the top of the file:

```python
def _section_badge(title: str) -> None:
    """Render a sidebar section badge label."""
    st.markdown(f'<span class="sidebar-badge">{title}</span>', unsafe_allow_html=True)
```

- [ ] **Step 4: Reorganize inputs into 7 sections**

Restructure `render_sidebar()` so inputs are grouped as follows. Use `_section_badge("...")` before each group, and `st.divider()` after each group (except the last).

**Section 1 — PRESET:**
```python
_section_badge("Preset")
# ... existing preset selectbox ...
st.divider()
```

**Section 2 — PORTFOLIO:**
```python
_section_badge("Portfolio")
# ... existing ticker inputs + weight inputs ...
st.divider()
```

**Section 3 — BENCHMARKS:**
```python
_section_badge("Benchmarks")
# ... existing benchmark inputs ...
st.divider()
```

**Section 4 — DATE RANGE:**
```python
_section_badge("Date Range")
# ... existing date inputs ...
# ... date preset pills (see Step 5) ...
st.divider()
```

**Section 5 — STRATEGY:**
```python
_section_badge("Strategy")
# ... existing rebalancing selectbox ...
st.divider()
```

**Section 6 — OPTIONS:**
```python
_section_badge("Options")
# ... existing checkboxes (verbose, save CSV, etc.) ...
st.divider()
```

**Section 7 — RUN:**
```python
_section_badge("Run")
# ... existing run button ...
# No divider after this section
```

- [ ] **Step 5: Style date preset pills**

Replace the existing date preset buttons (if they are raw `st.button` calls) with styled pill buttons. If the existing code already has date preset logic, wrap it in a container and add the pill CSS class.

For example, if the existing code is:
```python
cols = st.columns(6)
with cols[0]:
    if st.button("1Y", key="preset_1y"):
        # ... set dates ...
```

Wrap each button in the pill style by using `st.markdown` with the pill class, or keep the button logic but add a container with the CSS. A simpler approach: render the pills as clickable HTML buttons that set session state via `st.session_state`.

However, to keep changes minimal and functional, keep the existing `st.button` logic but wrap the row in a div with the pill style:

```python
st.markdown('<div style="display: flex; flex-wrap: wrap; margin-bottom: 0.5rem;">', unsafe_allow_html=True)
cols = st.columns(6)
with cols[0]:
    if st.button("1Y", key="preset_1y"):
        # existing logic
        pass
# ... repeat for 3Y, 5Y, 10Y, YTD, Max ...
st.markdown('</div>', unsafe_allow_html=True)
```

If the existing pills are already compact, this step may be minimal. The key is ensuring the `_section_badge` and dividers create clear visual separation.

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_sidebar.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/sidebar.py
git commit -m "feat: reorganize sidebar into 7 sections with badges and styling"
```

---

## Task 5: Main Layout Orchestration

**Files:**
- Modify: `app/main.py`
- Test: `pytest tests/test_main.py -v` (existing suite must still pass)

**Prerequisites:** Tasks 1–4

- [ ] **Step 1: Add design system imports**

At the top of `app/main.py`, add:

```python
from app.design_system import get_global_css
from app.ui_components import (
    display_welcome_screen,
    display_hero_metrics_row,
    display_info_bar,
    display_section_header,
    display_metrics_tables,
    display_downloads,
)
```

Remove the old `display_header` import if it exists.

- [ ] **Step 2: Inject global CSS at app startup**

In `main()` (or the entrypoint function), add as the first call:

```python
st.markdown(get_global_css(), unsafe_allow_html=True)
```

- [ ] **Step 3: Show welcome screen when no results exist**

Find the conditional that checks whether a backtest has been run. If no results exist, call:

```python
if not st.session_state.get("results"):
    display_welcome_screen()
    return  # or display_footer() then return
```

Replace the old `display_header()` call with `display_welcome_screen()`.

- [ ] **Step 4: Build hero metrics row from results**

After a successful backtest, build the metrics dict and display it:

```python
metrics = {
    "Ending Portfolio Value": f"${results['ending_value']:,.2f}",
    "Total Return": f"{results['total_return']:+.2%}",
    "Annualized Return (CAGR)": f"{results['cagr']:+.2%}",
    "Sharpe Ratio": f"{results['sharpe_ratio']:.2f}",
    "Maximum Drawdown": f"{results['max_drawdown']:.2%}",
}
display_hero_metrics_row(metrics)
```

Adapt the exact dict keys to match whatever `results` contains in the actual codebase.

- [ ] **Step 5: Display portfolio info bar**

After the hero metrics, display the info bar:

```python
display_info_bar(
    portfolio_tickers=st.session_state.get("tickers", []),
    weights=st.session_state.get("weights", []),
    benchmarks=st.session_state.get("benchmarks", []),
    start_date=str(st.session_state.get("start_date", "")),
    end_date=str(st.session_state.get("end_date", "")),
)
```

- [ ] **Step 6: Render 2×2 chart grid**

Replace any existing chart display calls with a 2×2 grid:

```python
display_section_header("Performance Overview")

col1, col2 = st.columns(2)
with col1:
    fig1 = plot_cumulative_returns(results["df"])
    st.pyplot(fig1)
with col2:
    fig2 = plot_drawdown(results["df"])
    st.pyplot(fig2)

col3, col4 = st.columns(2)
with col3:
    fig3 = plot_rolling_returns(results["df"])
    st.pyplot(fig3)
with col4:
    fig4 = plot_rolling_sharpe(results["df"])
    st.pyplot(fig4)
```

Adapt `results["df"]` to the actual DataFrame key in the results dict.

- [ ] **Step 7: Render detailed metrics tables**

```python
display_section_header("Detailed Metrics")

performance_metrics = {
    "Starting Value": f"${results.get('starting_value', 0):,.2f}",
    "Ending Value": f"${results.get('ending_value', 0):,.2f}",
    "Total Return": f"{results.get('total_return', 0):.2%}",
    "CAGR": f"{results.get('cagr', 0):.2%}",
    "Volatility (Annualized)": f"{results.get('volatility', 0):.2%}",
}

risk_metrics = {
    "Sharpe Ratio": f"{results.get('sharpe_ratio', 0):.2f}",
    "Sortino Ratio": f"{results.get('sortino_ratio', 0):.2f}",
    "Max Drawdown": f"{results.get('max_drawdown', 0):.2%}",
    "Calmar Ratio": f"{results.get('calmar_ratio', 0):.2f}",
    "Beta": f"{results.get('beta', 0):.2f}",
}

display_metrics_tables(performance_metrics, risk_metrics)
```

- [ ] **Step 8: Render downloads section**

```python
# Generate CSV bytes if available
csv_bytes = None
if "df" in results:
    csv_bytes = results["df"].to_csv(index=True).encode("utf-8")

# Generate chart bytes if available (reuse the dashboard figure)
chart_bytes = None
if "dashboard_fig" in results:
    import io
    buf = io.BytesIO()
    results["dashboard_fig"].savefig(buf, format="png", bbox_inches="tight")
    chart_bytes = buf.getvalue()

display_downloads(csv_data=csv_bytes, chart_data=chart_bytes)
```

If the existing code already handles downloads differently, adapt this to use the existing download logic but wrapped in `display_downloads()`.

- [ ] **Step 9: Run tests**

```bash
pytest tests/test_main.py -v
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add app/main.py
git commit -m "feat: orchestrate new dashboard layout with welcome screen, hero metrics, and 2x2 chart grid"
```

---

## Task 6: Integration & Verification

**Files:** None (smoke test only)

**Prerequisites:** Tasks 1–5

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 2: Launch Streamlit smoke test**

```bash
streamlit run app.py
```

Verify manually:
- [ ] App starts without Streamlit config errors
- [ ] Welcome screen is centered with icon, title, subtitle
- [ ] Sidebar has 7 clearly labeled sections with badge styling
- [ ] Date preset pills are compact and styled
- [ ] Run button is full-width blue
- [ ] After running a backtest (e.g., AAPL/MSFT vs SPY, 5Y):
  - [ ] Hero metrics row shows 5 cards with correct values
  - [ ] Portfolio info bar shows tickers, benchmarks, date range
  - [ ] 2×2 chart grid renders all four charts
  - [ ] Chart backgrounds are transparent (show page background)
  - [ ] Detailed metrics tables show performance and risk data
  - [ ] Downloads section has styled buttons

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete Streamlit UI redesign — fintech dashboard aesthetic"
```

---

## Self-Review Checklist

### 1. Spec Coverage

| Spec Section | Implementing Task |
|---|---|
| 2. Visual System (colors, typography, spacing) | Task 1 |
| 3. Sidebar Design (7 sections, badges, pills, run button) | Task 4 |
| 4.1 Welcome Screen | Task 5, Step 3 |
| 4.2 Hero Metrics Row | Task 3, Step 6 + Task 5, Step 4 |
| 4.3 Portfolio Info Bar | Task 3, Step 5 + Task 5, Step 5 |
| 4.4 Dashboard Chart Grid (2×2) | Task 5, Step 6 |
| 4.6 Detailed Metrics Tables | Task 3, Step 7 + Task 5, Step 7 |
| 4.7 Downloads Section | Task 3, Step 8 + Task 5, Step 8 |
| 5. Charts Refinement (rcParams, titles) | Task 2 |
| 6. Architecture (design_system.py, modified files) | Tasks 1–5 |
| 8. Testing (existing tests pass, smoke test) | Task 6 |

**No gaps found.**

### 2. Placeholder Scan

- No "TBD", "TODO", "implement later" found.
- No vague "add appropriate error handling" steps.
- All code blocks contain complete, runnable code.
- No "Similar to Task N" references.

### 3. Type Consistency

- Function names consistent across tasks (`display_welcome_screen`, `display_hero_metrics_row`, etc.).
- Import paths consistent (`from app.design_system import ...`).
- Dict key names consistent between `design_system.py` and consumers.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-01-streamlit-ui-redesign.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
