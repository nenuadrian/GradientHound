"""Raw data viewer page: browse and export captured IPC data."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html, dcc


_DATA_TYPE_OPTIONS = [
    {"label": "Gradient Stats", "value": "gradient_stats"},
    {"label": "Weight Stats", "value": "weight_stats"},
    {"label": "Activation Stats", "value": "activation_stats"},
    {"label": "Optimizer State", "value": "optimizer_state"},
    {"label": "Metrics (W&B)", "value": "metrics"},
    {"label": "Predictions", "value": "predictions"},
    {"label": "Attention", "value": "attention"},
]


def raw_data_page(has_ipc: bool = False, data_dir: str | None = None):
    if not has_ipc:
        return dbc.Container([
            html.H2("Raw Data", className="mt-3 mb-1"),
            html.P(
                "No IPC data directory configured. Pass --data-dir to browse captured data.",
                className="text-muted mb-4",
            ),
        ], fluid=True)

    return dbc.Container([
        html.H2("Raw Data", className="mt-3 mb-1"),
        html.P(
            "Browse raw captured data from the IPC channel. "
            "Select a data type and view the most recent records.",
            className="text-muted mb-4",
        ),

        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Data Type"),
                    dcc.Dropdown(
                        id="raw-data-type",
                        options=_DATA_TYPE_OPTIONS,
                        value="gradient_stats",
                        clearable=False,
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Label("Max Records"),
                    dcc.Dropdown(
                        id="raw-data-limit",
                        options=[
                            {"label": "50", "value": 50},
                            {"label": "200", "value": 200},
                            {"label": "500", "value": 500},
                            {"label": "All", "value": 0},
                        ],
                        value=50,
                        clearable=False,
                    ),
                ], md=2),
                dbc.Col([
                    dbc.Label("\u00a0"),
                    dbc.Button(
                        "Refresh", id="raw-data-refresh-btn", color="primary",
                        className="d-block",
                    ),
                ], md=2),
                dbc.Col([
                    html.Div(id="raw-data-count", className="mt-4 text-muted"),
                ], md=4),
            ], className="g-3"),
        ]), className="mb-3"),

        # ── Data table ────────────────────────────────────────────────
        html.Div(id="raw-data-table"),

        # ── Data directory info ───────────────────────────────────────
        dbc.Card(dbc.CardBody([
            html.H6("Data Directory", className="card-title"),
            html.Code(data_dir or "N/A", className="d-block"),
        ]), className="mb-3 mt-3"),
    ], fluid=True)
