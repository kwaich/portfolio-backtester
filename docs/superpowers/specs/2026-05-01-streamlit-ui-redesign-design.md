# Streamlit UI Redesign — Design Spec

**Date:** 2026-05-01  
**Status:** Approved  
**Approach:** Refined Fintech Dashboard (Approach A)

---

## 1. Overview

Redesign the Streamlit web application for the portfolio-backtester project to achieve a clean, professional, fintech-grade visual experience. The redesign is **purely presentational** — all existing backtesting logic, data caching, and metric calculations remain unchanged.

### Goals
- Elevate visual polish from "Streamlit default" to "professional financial dashboard"
- Improve information hierarchy and scannability
- Maintain the existing sidebar-centric interaction model
- Keep accessibility standards (colorblind-safe charts)

### Non-Goals
- New chart types, metrics, or backtesting strategies
- Mobile-responsive breakpoints
- Interactive chart libraries (Plotly)
- Multi-page app architecture
- Backend performance changes

---

## 2. Visual System

### Color Palette

| Token | Hex | Usage |
|---|---|---|
| `--primary-text` | `#0f172a` | Headings, body text, labels |
| `--accent` | `#3b82f6` | Primary buttons, active states, links, highlights |
| `--accent-hover` | `#2563eb` | Button hover states |
| `--bg-page` | `#f8fafc` | Main content area background |
| `--bg-card` | `#ffffff` | Cards, panels, metric containers |
| `--border` | `#e2e8f0` | Dividers, card borders, input borders |
| `--muted` | `#64748b` | Secondary text, axis labels, timestamps |
| `--success` | `#10b981` | Positive returns, green indicators |
| `--danger` | `#ef4444` | Negative returns, drawdowns, errors |

### Typography

| Element | Font | Weight | Size | Color |
|---|---|---|---|---|
| Page title | Outfit | 600 | 28px | `--primary-text` |
| Section header | Outfit | 500 | 20px | `--primary-text` |
| Metric value | Inter | 600 | 24px | `--primary-text` |
| Metric label | Inter | 400 | 12px | `--muted` |
| Body text | Inter | 400 | 14px | `--primary-text` |
| Small / caption | Inter | 400 | 12px | `--muted` |
| Sidebar section badge | Inter | 600 | 10px | `--muted` |

Load fonts via Google Fonts injected with `st.markdown(..., unsafe_allow_html=True)`.

### Spacing & Layout

- **Page padding:** `2rem` horizontal, `1.5rem` vertical on main content area
- **Card padding:** `1.25rem`
- **Card border-radius:** `12px`
- **Card shadow:** `0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03)`
- **Section gap:** `2rem` between major sections
- **Metric card gap:** `1rem`

---

## 3. Sidebar Design

### Section Reorganization

The sidebar is reorganized into clearly delineated sections with uppercase badge-style labels:

1. **PRESET** — Portfolio preset selector (dropdown)
2. **PORTFOLIO** — Ticker inputs + weight inputs
3. **BENCHMARKS** — Benchmark ticker inputs
4. **DATE RANGE** — Start / end date pickers + quick-select pill buttons
5. **STRATEGY** — Rebalancing dropdown
6. **OPTIONS** — Checkbox toggles (verbose, save CSV, etc.)
7. **RUN** — Primary CTA button

### Sidebar Styling

- **Background:** `#ffffff` (Streamlit default, keep)
- **Section badges:** `text-transform: uppercase; letter-spacing: 0.05em; font-size: 10px; font-weight: 600; color: #64748b; margin-bottom: 8px;`
- **Dividers:** `1px solid #e2e8f0` between sections
- **Date preset pills:** Compact horizontal row of pill buttons (`border-radius: 16px; padding: 4px 12px; font-size: 12px;`) for 1Y, 3Y, 5Y, 10Y, YTD, Max
- **Run button:** Full-width, `--accent` background, white text, `border-radius: 8px`, `font-weight: 600`

---

## 4. Main Content Flow

### 4.1 Welcome Screen (No results yet)

When no backtest has been run, display a centered hero:

```
┌─────────────────────────────────────────────┐
│                                             │
│     [Icon: chart line]                      │
│                                             │
│     Portfolio Backtester                    │
│     Analyze historical portfolio performance│
│                                             │
│     Enter tickers in the sidebar and click  │
│     "Run Backtest" to get started.          │
│                                             │
└─────────────────────────────────────────────┘
```

- Icon: Lucide-style line chart, `--accent` color, 48px
- Title: Outfit 600, 28px
- Subtitle: Inter 400, 16px, `--muted`
- Vertically and horizontally centered in the main content area

### 4.2 Hero Metrics Row (After backtest)

A horizontal row of 5 metric cards, each in a white card with subtle shadow:

| Card | Value Example | Label |
|---|---|---|
| Ending Value | `$14,230.50` | Ending Portfolio Value |
| Total Return | `+42.31%` | Total Return |
| CAGR | `+7.34%` | Annualized Return (CAGR) |
| Sharpe Ratio | `1.12` | Sharpe Ratio |
| Max Drawdown | `-18.45%` | Maximum Drawdown |

- Value color: `--primary-text` for most; `--success` for positive returns, `--danger` for negative returns / drawdowns
- Each card: `flex: 1`, min-width responsive
- Layout: CSS Grid or flexbox row with `gap: 1rem`

### 4.3 Portfolio Info Bar

A compact horizontal bar below metrics showing:
- Portfolio tickers with weights (e.g., "AAPL 60% · MSFT 40%")
- Benchmark(s) (e.g., "vs SPY")
- Date range (e.g., "Jan 2019 – Jan 2024")

Styling: `--bg-card`, `border-radius: 8px`, padding `0.75rem 1rem`, text `14px Inter`, `--muted` color.

### 4.4 Dashboard Chart Grid (2×2)

Four charts in a responsive 2×2 grid:

| Top-Left | Top-Right |
|---|---|
| **Cumulative Returns** (line chart) | **Drawdown** (area chart) |
| **Rolling 12-Month Returns** (line chart) | **Rolling Sharpe Ratio** (line chart) |

Each chart in a white card container with:
- Title: Outfit 500, 16px
- Transparent figure background
- Grid lines: `0.5px solid #e2e8f0`
- Axis labels: Inter 400, 11px, `--muted`

### 4.5 Rolling Returns (Full Width)

Full-width section for the 12-month rolling returns chart if not already shown in the 2×2 grid. Alternatively, this can be omitted if all four charts fit in the 2×2 grid. **Decision:** The 2×2 grid contains the four main charts; no separate full-width rolling returns section needed.

### 4.6 Detailed Metrics Table

Two side-by-side tables (or stacked on narrow viewports):

**Left — Performance Metrics:**
| Metric | Value |
|---|---|
| Starting Value | $10,000.00 |
| Ending Value | $14,230.50 |
| Total Return | 42.31% |
| CAGR | 7.34% |
| Volatility (Ann.) | 14.2% |

**Right — Risk Metrics:**
| Metric | Value |
|---|---|
| Sharpe Ratio | 1.12 |
| Sortino Ratio | 1.45 |
| Max Drawdown | -18.45% |
| Calmar Ratio | 0.40 |
| Beta | 0.95 |

Table styling:
- Header: Inter 600, 12px, uppercase, `--muted`
- Row divider: `1px solid #f1f5f9`
- Values: Inter 400, 14px, `--primary-text`
- Positive/negative coloring on return values

### 4.7 Downloads Section

A card containing:
- "Download Results CSV" button (secondary style: white bg, `--border` border, `--primary-text` text)
- "Download Chart" button (same secondary style)
- Optional: "Save to Cache" toggle

---

## 5. Charts Refinement

### Colorblind Palette (Retained)

Keep the existing Wong colorblind-safe palette for chart series. Do not change the color assignments.

### Typography & Styling Updates

Apply these via matplotlib rcParams + figure adjustments:

| Element | Spec |
|---|---|
| Figure facecolor | `'none'` (transparent) |
| Axes facecolor | `'none'` |
| Grid color | `#e2e8f0` |
| Grid linewidth | `0.5` |
| Grid linestyle | `'-'` |
| Axis label color | `#64748b` |
| Axis label size | `11` |
| Tick label color | `#64748b` |
| Tick label size | `10` |
| Title color | `#0f172a` |
| Title size | `14` |
| Title weight | `500` |
| Legend frame | `False` |
| Legend loc | `'upper left'` or `'upper right'` depending on chart |

### Chart Titles

Use descriptive, concise titles:
- "Cumulative Returns" → "Portfolio vs Benchmark Cumulative Returns"
- "Drawdown" → "Portfolio Drawdown Over Time"
- "Rolling 12-Month Returns" → "Rolling 12-Month Returns"
- "Rolling Sharpe" → "Rolling Sharpe Ratio (12M)"

---

## 6. Architecture

### New File

- `app/design_system.py` — Exports:
  - `COLORS` dict (all design tokens)
  - `TYPOGRAPHY` dict (font families, sizes, weights)
  - `SPACING` dict (padding, gap, radius values)
  - `get_global_css()` → returns the full `<style>` block string for injection
  - `get_card_style()`, `get_metric_card_style()`, etc. for reusable CSS snippets

### Modified Files

| File | Changes |
|---|---|
| `app/sidebar.py` | Reorganize into 7 sections with badge labels. Style date preset pills. Style Run button. |
| `app/ui_components.py` | Update `display_metric_card()`, `display_section_header()`, `display_info_bar()` to use design system tokens. |
| `app/main.py` | Add welcome screen. Add hero metrics row. Add 2×2 chart grid layout. Add detailed metrics tables. Add downloads section. |
| `app/charts.py` | Update rcParams to match chart typography spec. Ensure transparent backgrounds. |

### Unchanged Files

- `backtest.py` — computation engine
- `plot_backtest.py` — chart generation logic (only styling params change)
- `app/config.py` — preset configs
- `app/state_manager.py` — session state
- `app/validation.py` — input validation
- `app/results.py` — results parsing
- `app/utils.py` — utility functions

### Data Flow

```
User Input → sidebar.py → validation.py → backtest.py → results.py
                                                      ↓
                                              charts.py / ui_components.py
                                                      ↓
                                                   main.py (display)
```

No data flow changes. Only the display layer is affected.

---

## 7. Scope

### In Scope
- Visual redesign: colors, typography, spacing, layout
- Sidebar reorganization and section styling
- Metric card components and hero layout
- Chart typography and grid refinement
- Welcome / empty state screen
- Download button styling
- 2×2 chart dashboard grid
- `app/design_system.py` creation

### Out of Scope
- New chart types or financial metrics
- New backtesting strategies
- Mobile-responsive breakpoints
- Interactive chart libraries (Plotly, Altair)
- Real-time data streaming
- Backend performance optimization
- Multi-page app architecture
- Changes to test logic or backtest calculations

---

## 8. Testing Strategy

1. **Existing test suite:** Run `pytest -v` before and after changes. All existing tests must pass without modification.
2. **Smoke test:** Launch the Streamlit app (`streamlit run app.py`) and verify it starts without configuration errors.
3. **Visual verification:** Run a sample backtest (AAPL/MSFT vs SPY, 5Y) and verify:
   - Welcome screen displays before backtest
   - Hero metrics row appears after backtest
   - 2×2 chart grid renders correctly
   - Tables and downloads section are visible
   - Sidebar sections are clearly delineated
   - No console errors or layout breakage

---

## 9. Success Criteria

- [ ] App launches without errors
- [ ] All existing pytest tests pass
- [ ] Welcome screen is centered and styled
- [ ] Hero metrics row displays 5 cards with correct styling
- [ ] 2×2 chart grid renders all four charts
- [ ] Sidebar has 7 clearly labeled sections
- [ ] Date preset pills are styled as compact pills
- [ ] Run button is full-width and visually prominent
- [ ] Charts use transparent backgrounds and updated typography
- [ ] Detailed metrics tables are styled and readable
- [ ] Download buttons are styled consistently
- [ ] Overall visual impression is "professional fintech dashboard" rather than "default Streamlit"

---

## 10. Open Questions / Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Use Approach A (Refined Fintech Dashboard) | ✅ Approved |
| 2 | Keep Wong colorblind palette for charts | ✅ Approved |
| 3 | Outfit + Inter font combination | ✅ Approved |
| 4 | 2×2 chart grid contains all four main charts | ✅ Approved |
| 5 | No mobile-responsive breakpoints in this phase | ✅ Approved |
| 6 | No interactive chart libraries (stay with matplotlib) | ✅ Approved |
