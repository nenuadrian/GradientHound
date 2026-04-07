"""On-demand analysis page: weight heatmaps, CKA similarity, network state."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html, dcc


def on_demand_page(model_names: list[str] | None = None, has_ipc: bool = False):
    if not has_ipc:
        return dbc.Container([
            html.H2("On-Demand Analysis", className="mt-3 mb-1"),
            html.P(
                "On-demand analysis requires an active IPC channel. "
                "Pass --data-dir while training is running.",
                className="text-muted mb-4",
            ),
            dbc.Card(dbc.CardBody([
                html.H5("Available analyses", className="card-title"),
                html.Ul([
                    html.Li([html.Strong("Weight Heatmap"), " \u2014 Visualise any 2D weight matrix"]),
                    html.Li([html.Strong("CKA Similarity"), " \u2014 Linear CKA between all weight layers"]),
                    html.Li([html.Strong("Network State"), " \u2014 Full parameter dump (models < 1M params)"]),
                ]),
            ])),
        ], fluid=True)

    model_options = [{"label": n, "value": n} for n in (model_names or [])]

    return dbc.Container([
        html.H2("On-Demand Analysis", className="mt-3 mb-1"),
        html.P(
            "Trigger computationally expensive analyses that execute on the training process. "
            "Results appear when the training loop calls gradienthound.step().",
            className="text-muted mb-4",
        ),

        # ── Model selector (shared) ──────────────────────────────────
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Model"),
                    dcc.Dropdown(
                        id="od-model-select",
                        options=model_options,
                        value=model_options[0]["value"] if model_options else None,
                        clearable=False,
                        placeholder="Select model",
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Label("Layer (for heatmap)"),
                    dcc.Dropdown(id="od-heatmap-layer", placeholder="Select a 2D layer"),
                ], md=8),
            ], className="g-3"),
        ]), className="mb-3"),

        # ── Weight Heatmap ────────────────────────────────────────────
        dbc.Card(dbc.CardBody([
            html.H5("Weight Heatmap", className="card-title"),
            html.P(
                "Visualise the weight matrix of any 2D layer as a heatmap. "
                "Large matrices are downsampled to 128x128.",
                className="text-muted",
            ),
            dbc.Button(
                "Compute Heatmap", id="od-heatmap-btn", color="primary",
                className="me-2",
            ),
            html.Span(id="od-heatmap-status", className="text-muted"),
            dcc.Graph(id="od-heatmap-chart", style={"display": "none"}),
        ]), className="mb-3"),

        # ── CKA Similarity Matrix ─────────────────────────────────────
        dbc.Card(dbc.CardBody([
            html.H5("CKA Similarity Matrix", className="card-title"),
            html.P(
                "Linear Centered Kernel Alignment (CKA) between all 2D weight layers. "
                "Shows how similar different layers' learned representations are.",
                className="text-muted",
            ),
            dbc.Button(
                "Compute CKA", id="od-cka-btn", color="primary",
                className="me-2",
            ),
            html.Span(id="od-cka-status", className="text-muted"),
            dcc.Graph(id="od-cka-chart", style={"display": "none"}),
        ]), className="mb-3"),

        # ── Network State Dump ────────────────────────────────────────
        dbc.Card(dbc.CardBody([
            html.H5("Network State Dump", className="card-title"),
            html.P(
                "Download all parameter values as a table. "
                "Only available for models with fewer than 1M parameters.",
                className="text-muted",
            ),
            dbc.Button(
                "Dump Network State", id="od-state-btn", color="primary",
                className="me-2",
            ),
            html.Span(id="od-state-status", className="text-muted"),
            html.Div(id="od-state-result", className="mt-3"),
        ]), className="mb-3"),

        # Polling interval for response checks
        dcc.Interval(id="od-poll", interval=2000, n_intervals=0, disabled=True),
        dcc.Store(id="od-pending-requests", data={}),
    ], fluid=True)
