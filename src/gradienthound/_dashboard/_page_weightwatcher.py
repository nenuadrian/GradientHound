"""WeightWatcher spectral analysis deep-dive page."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html, dcc

from ._helpers import short_layer, plotly_layout


_WW_METRIC_OPTIONS = [
    {"label": "Alpha (power-law exponent)", "value": "alpha"},
    {"label": "Alpha Weighted (shape+scale)", "value": "alpha_weighted"},
    {"label": "MP Softrank (1.0=random)", "value": "mp_softrank"},
    {"label": "Spikes (above MP edge)", "value": "num_spikes"},
    {"label": "log10(lambda_max)", "value": "log_spectral_norm"},
    {"label": "MP Edge (lambda+)", "value": "lambda_plus"},
]


def weightwatcher_page(snapshots: list[dict] | None):
    if not snapshots:
        return dbc.Container([
            html.H2("WeightWatcher", className="mt-3 mb-1"),
            html.P(
                "Spectral analysis requires processed checkpoints. "
                "Go to the Checkpoints page and click Process first.",
                className="text-muted mb-4",
            ),
            dbc.Card(dbc.CardBody([
                html.H5("How it works", className="card-title"),
                html.P(
                    "WeightWatcher fits a power-law to the eigenvalue spectral density (ESD) "
                    "of each weight matrix. Well-trained layers typically have alpha in [2, 6]. "
                    "The MP softrank measures how far a layer is from random initialisation.",
                    className="text-muted",
                ),
            ])),
        ], fluid=True)

    checkpoint_names = [snap["name"] for snap in snapshots]

    # Collect layers that have spectral data
    spectral_layers: list[str] = []
    seen: set[str] = set()
    for snap in snapshots:
        for stat in snap.get("weight_stats", []):
            layer = stat.get("layer", "")
            if layer not in seen and isinstance(stat.get("alpha"), (int, float)):
                spectral_layers.append(layer)
                seen.add(layer)

    if not spectral_layers:
        return dbc.Container([
            html.H2("WeightWatcher", className="mt-3 mb-1"),
            html.P("Checkpoints processed but no spectral metrics found.", className="text-muted mb-4"),
            dbc.Alert(
                "Install the 'powerlaw' package to enable spectral analysis: pip install powerlaw",
                color="warning",
            ),
        ], fluid=True)

    layer_options = [{"label": short_layer(l), "value": l} for l in spectral_layers]
    ckpt_options = [{"label": n, "value": n} for n in checkpoint_names]

    return dbc.Container([
        html.H2("WeightWatcher", className="mt-3 mb-1"),
        html.P(
            f"Spectral quality analysis across {len(checkpoint_names)} checkpoints "
            f"and {len(spectral_layers)} layers.",
            className="text-muted mb-4",
        ),

        # ── Controls ──────────────────────────────────────────────────
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Metric", html_for="ww-global-metric-select"),
                    dcc.Dropdown(
                        id="ww-global-metric-select",
                        options=_WW_METRIC_OPTIONS,
                        value="alpha_weighted",
                        clearable=False,
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Label("Checkpoint", html_for="ww-checkpoint-select"),
                    dcc.Dropdown(
                        id="ww-checkpoint-select",
                        options=ckpt_options,
                        value=checkpoint_names[-1],
                        clearable=False,
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Label("Layer", html_for="ww-layer-select"),
                    dcc.Dropdown(
                        id="ww-layer-select",
                        options=layer_options,
                        value=spectral_layers[0],
                        clearable=False,
                    ),
                ], md=4),
            ], className="g-3"),
        ]), className="mb-3"),

        # ── Global metric views ───────────────────────────────────────
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Metric Heatmap", className="card-title"),
                html.P(
                    "Selected metric across all layers and checkpoints. "
                    "Red dashed line marks the selected checkpoint.",
                    className="text-muted",
                ),
                dcc.Graph(id="ww-metric-heatmap"),
            ])), md=7),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Metric Trend", className="card-title"),
                html.P(
                    "Mean, min, and max of the selected metric across checkpoints.",
                    className="text-muted",
                ),
                dcc.Graph(id="ww-metric-trend"),
            ])), md=5),
        ], className="g-3 mb-3"),

        # ── Layer-level views ─────────────────────────────────────────
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Layer Metrics Across Checkpoints", className="card-title"),
                html.P(
                    "All spectral metrics for the selected layer over time.",
                    className="text-muted",
                ),
                dcc.Graph(id="ww-layer-multi-metric"),
            ])), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Eigenvalue Spectral Density", className="card-title"),
                html.P(
                    "ESD of the selected layer. Bold line = selected checkpoint.",
                    className="text-muted",
                ),
                dcc.Graph(id="ww-layer-esd"),
            ])), md=6),
        ], className="g-3 mb-3"),

        # ── Leaderboard ───────────────────────────────────────────────
        dbc.Card(dbc.CardBody([
            html.H5("Layer Leaderboard", className="card-title"),
            html.P(
                "Layers ranked by the selected metric at the selected checkpoint.",
                className="text-muted",
            ),
            html.Div(id="ww-leaderboard-wrap"),
        ]), className="mb-3"),
    ], fluid=True)
