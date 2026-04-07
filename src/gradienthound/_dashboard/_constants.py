"""Shared constants for the GradientHound dashboard."""
from __future__ import annotations

# ── Module type → node color (for Cytoscape graph) ───────────────────

MODULE_COLORS = {
    "conv": "#f0d4d8",
    "linear": "#d8ecde",
    "norm": "#faf0e0",
    "activation": "#fae0d0",
    "pool": "#e4d6ec",
    "dropout": "#f0eaea",
    "embedding": "#e0d4e8",
    "default": "#faf6f6",
}

# Colors for checkpoint series in comparison charts
SERIES_COLORS = [
    "#375a7f", "#00bc8c", "#3498db", "#f39c12", "#e74c3c",
    "#9b59b6", "#1abc9c", "#e67e22", "#2ecc71", "#fd7e14",
]

# ── Page definitions ───────────────────────────────────────────────────

PAGES = {
    "/":               ("Dashboard",      "Model overview, weights, gradients, and health"),
    "/gradient-flow":  ("Gradient Flow",  "Gradient magnitudes across layers"),
    "/live":           ("Live Training",  "Activations, weights, optimizer state, predictions, attention"),
    "/metrics":        ("Metrics",        "Run metrics from Weights & Biases"),
    "/checkpoints":    ("Checkpoints",    "Compare model checkpoints"),
    "/weightwatcher":  ("WeightWatcher",  "Spectral analysis deep-dive"),
    "/on-demand":      ("On-Demand",      "Weight heatmaps, CKA similarity, network state"),
    "/raw-data":       ("Raw Data",       "Browse raw captured IPC data"),
}

# ── Plotly light template ────────────────────────────────────────────

PLOTLY_TEMPLATE = "plotly_white"

# ── Cytoscape stylesheet ────────────────────────────────────────────

CYTO_STYLE = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "background-color": "#faf6f6",
            "color": "#1a1012",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "11px",
            "font-weight": "600",
            "font-family": "-apple-system, sans-serif",
            "width": 140,
            "height": "32px",
            "padding": "12px",
            "shape": "roundrectangle",
            "border-width": "1.5px",
            "border-color": "#555",
            "text-wrap": "wrap",
            "text-max-width": "140px",
        },
    },
    {
        "selector": "node:selected",
        "style": {
            "border-width": "3px",
            "border-color": "#375a7f",
        },
    },
    {
        "selector": ".placeholder",
        "style": {
            "shape": "diamond",
            "background-color": "#444",
            "border-color": "#375a7f",
            "color": "#fff",
            "border-style": "dashed",
        },
    },
    {
        "selector": "edge",
        "style": {
            "width": 1.5,
            "line-color": "#555",
            "target-arrow-color": "#777",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.8,
            "curve-style": "bezier",
        },
    },
    *[
        {"selector": f".mod-{cat}", "style": {"background-color": color}}
        for cat, color in MODULE_COLORS.items()
    ],
    {
        "selector": ".standalone",
        "style": {
            "border-style": "dashed",
            "border-color": "#777",
            "opacity": 0.75,
        },
    },
    {
        "selector": ".edge-detached",
        "style": {
            "line-style": "dashed",
            "line-color": "#666",
            "target-arrow-color": "#666",
            "opacity": 0.6,
        },
    },
]

# ── Health assessment colors ─────────────────────────────────────────

HEALTH_COLORS = {
    "healthy": "#28a745",
    "warning": "#ffc107",
    "critical": "#dc3545",
    "neutral": "#6c757d",
}

HEALTH_SORT = {"critical": 0, "warning": 1, "healthy": 2, "neutral": 3}
