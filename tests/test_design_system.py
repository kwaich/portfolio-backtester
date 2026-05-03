"""Tests for the design system module."""

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
