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
