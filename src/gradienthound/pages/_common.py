"""Shared utilities, CSS, and helpers for all Streamlit pages."""
from __future__ import annotations

import streamlit as st

# ── Health colour palette ─────────────────────────────────────────────

GREEN = "#4caf50"
YELLOW = "#ff9800"
RED = "#f44336"
BLUE = "#2196f3"
GREY = "#9e9e9e"

# GradientHound brand
BRAND_RED = "#c62828"

SHARED_CSS = """
<style>
.block-container { padding-top: 1.5rem; }
.gh-title { color: #c62828; margin: 0; }
.health-green { color: #4caf50; font-weight: bold; }
.health-yellow { color: #ff9800; font-weight: bold; }
.health-red { color: #f44336; font-weight: bold; }
</style>
"""


def inject_css() -> None:
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


def get_ipc():
    """Retrieve the IPCChannel from session state."""
    return st.session_state.get("ipc")


def health_color(value: float, low: float, high: float) -> str:
    """Return a health hex colour: green if in [low, high], yellow if close, red otherwise."""
    if low <= value <= high:
        return GREEN
    margin = (high - low) * 0.5
    if (low - margin) <= value <= (high + margin):
        return YELLOW
    return RED


def grad_health_color(grad_norm: float) -> str:
    """Colour-code a gradient norm: green=healthy, yellow=weak, red=vanishing/exploding."""
    if grad_norm < 1e-7:
        return RED  # vanishing
    if grad_norm > 1e3:
        return RED  # exploding
    if grad_norm < 1e-4:
        return YELLOW  # weak
    if grad_norm > 1e2:
        return YELLOW  # large
    return GREEN


def rank_health_color(effective_rank: float, max_rank: float) -> str:
    """Colour-code effective rank utilization."""
    if max_rank <= 0:
        return GREY
    ratio = effective_rank / max_rank
    if ratio > 0.5:
        return GREEN
    if ratio > 0.2:
        return YELLOW
    return RED


def short_layer_name(name: str) -> str:
    """Shorten a parameter name for display."""
    parts = name.split(".")
    if len(parts) > 3:
        parts = parts[-3:]
    return ".".join(parts)
