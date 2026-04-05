"""Shared utilities, colours, and Bokeh figure factory for all Panel pages."""
from __future__ import annotations

from bokeh.plotting import figure as _bokeh_figure
from bokeh.models import HoverTool

# ── Health colour palette ────────────────────────────────────────────

GREEN = "#4caf50"
YELLOW = "#ff9800"
RED = "#f44336"
BLUE = "#2196f3"
GREY = "#9e9e9e"

BRAND_RED = "#c62828"

# Chart line colours (10-colour cycle)
PALETTE = [
    "#42a5f5", "#66bb6a", "#ffa726", "#ef5350", "#ab47bc",
    "#26c6da", "#ff7043", "#78909c", "#ec407a", "#9ccc65",
]


def health_color(value: float, low: float, high: float) -> str:
    if low <= value <= high:
        return GREEN
    margin = (high - low) * 0.5
    if (low - margin) <= value <= (high + margin):
        return YELLOW
    return RED


def grad_health_color(grad_norm: float) -> str:
    if grad_norm < 1e-7:
        return RED
    if grad_norm > 1e3:
        return RED
    if grad_norm < 1e-4:
        return YELLOW
    if grad_norm > 1e2:
        return YELLOW
    return GREEN


def rank_health_color(effective_rank: float, max_rank: float) -> str:
    if max_rank <= 0:
        return GREY
    ratio = effective_rank / max_rank
    if ratio > 0.5:
        return GREEN
    if ratio > 0.2:
        return YELLOW
    return RED


def short_layer_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) > 3:
        parts = parts[-3:]
    return ".".join(parts)


def latest_step(entries: list[dict], key: str = "step") -> int:
    if not entries:
        return 0
    value = entries[-1].get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def latest_entries(entries: list[dict], key: str = "step") -> list[dict]:
    if not entries:
        return []
    step = entries[-1].get(key)
    latest: list[dict] = []
    for entry in reversed(entries):
        if entry.get(key) != step:
            break
        latest.append(entry)
    latest.reverse()
    return latest


def make_figure(title: str = "", x_label: str = "", y_label: str = "",
                **kwargs) -> _bokeh_figure:
    """Create a dark-themed Bokeh figure with sensible defaults."""
    from bokeh.models import WheelZoomTool, CustomAction, CustomJS

    defaults = dict(
        sizing_mode="stretch_width",
        height=300,
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    defaults.update(kwargs)
    p = _bokeh_figure(title=title, x_axis_label=x_label,
                      y_axis_label=y_label, **defaults)
    p.background_fill_color = "#1a1a2e"
    p.border_fill_color = "#16213e"
    p.outline_line_color = "#333"
    p.grid.grid_line_color = "#2a2a4a"
    p.grid.grid_line_alpha = 0.6
    p.xaxis.axis_label_text_color = "#aaa"
    p.yaxis.axis_label_text_color = "#aaa"
    p.xaxis.major_label_text_color = "#888"
    p.yaxis.major_label_text_color = "#888"
    p.title.text_color = "#ddd"
    p.title.text_font_size = "11pt"
    p.xaxis.axis_line_color = "#444"
    p.yaxis.axis_line_color = "#444"
    p.xaxis.major_tick_line_color = "#444"
    p.yaxis.major_tick_line_color = "#444"
    p.xaxis.minor_tick_line_color = None
    p.yaxis.minor_tick_line_color = None

    # Wheel-zoom active by default so scroll immediately zooms the chart
    wzt = p.select_one(WheelZoomTool)
    if wzt:
        p.toolbar.active_scroll = wzt

    # Fullscreen toggle (toolbar button)
    p.add_tools(CustomAction(
        description="Toggle fullscreen",
        callback=CustomJS(code="""
            var el = cb_obj.origin.el || document.querySelector('.bk-Canvas');
            if (!el) return;
            el = el.closest('.bk-Figure') || el.closest('.bk-plot-layout') || el;
            if (document.fullscreenElement) { document.exitFullscreen(); }
            else { el.requestFullscreen().catch(function(){}); }
        """),
    ))
    return p


def make_hbar_figure(title: str = "", y_range: list[str] | None = None,
                     **kwargs) -> _bokeh_figure:
    """Horizontal bar figure with categorical y-axis."""
    from bokeh.models import FactorRange
    yr = FactorRange(*y_range) if y_range else FactorRange()
    return make_figure(title=title, y_range=yr, **kwargs)


def style_legend(fig: _bokeh_figure) -> None:
    """Apply dark-theme styling to a figure's legend."""
    if fig.legend:
        fig.legend.click_policy = "hide"
        fig.legend.label_text_color = "#ccc"
        fig.legend.label_text_font_size = "9pt"
        fig.legend.background_fill_color = "#1a1a2e"
        fig.legend.background_fill_alpha = 0.8
        fig.legend.border_line_color = "#444"


def fmt_bytes(b: int | float) -> str:
    if b < 1024:
        return f"{b:.0f} B"
    if b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    if b < 1024 ** 3:
        return f"{b / 1024 ** 2:.1f} MB"
    return f"{b / 1024 ** 3:.2f} GB"


def fmt_num(n: int | float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(int(n))
