"""Live training data page: activations, weight stats, optimizer state, predictions, attention."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html, dcc


def live_page(has_ipc: bool = False):
    if not has_ipc:
        return dbc.Container([
            html.H2("Live Training", className="mt-3 mb-1"),
            html.P(
                "No IPC data directory configured. Pass --data-dir to load live training captures.",
                className="text-muted mb-4",
            ),
            dbc.Card(dbc.CardBody([
                html.H5("How to use", className="card-title"),
                html.P([
                    "During training, call ",
                    html.Code("gradienthound.watch(model, log_activations=True)"),
                    " and ",
                    html.Code("gradienthound.step()"),
                    " each iteration.",
                ], className="text-muted"),
                html.Code(
                    "python -m gradienthound --data-dir /tmp/gh_run",
                    className="d-block p-2 bg-dark rounded",
                ),
            ])),
        ], fluid=True)

    return dbc.Container([
        html.H2("Live Training", className="mt-3 mb-1"),
        html.P("Real-time training data from the IPC channel.", className="text-muted mb-4"),
        dcc.Interval(id="live-refresh", interval=3000, n_intervals=0),

        dbc.Tabs([
            # ── Activations tab ─────────────────────────────────────
            dbc.Tab(label="Activations", children=[
                dbc.Card(dbc.CardBody([
                    html.H5("Activation Statistics", className="card-title mt-2"),
                    html.P(
                        "Per-layer activation mean, std, min, max, and zero fraction "
                        "captured during forward passes.",
                        className="text-muted",
                    ),
                    html.Div(id="live-act-summary", className="small text-muted mb-2"),
                    dcc.Graph(id="live-act-chart"),
                    html.Div(id="live-act-table"),
                ]), className="mt-2 mb-3"),
            ]),

            # ── Weight Stats tab ─────────────────────────────────────
            dbc.Tab(label="Weight Stats", children=[
                dbc.Card(dbc.CardBody([
                    html.H5("Live Weight Evolution", className="card-title mt-2"),
                    html.P(
                        "Per-layer weight norms and statistics captured periodically during training.",
                        className="text-muted",
                    ),
                    html.Div(id="live-weight-summary", className="small text-muted mb-2"),
                    dcc.Graph(id="live-weight-chart"),
                    html.Div(id="live-weight-table"),
                ]), className="mt-2 mb-3"),
            ]),

            # ── Optimizer State tab ──────────────────────────────────
            dbc.Tab(label="Optimizer State", children=[
                dbc.Card(dbc.CardBody([
                    html.H5("Optimizer State Evolution", className="card-title mt-2"),
                    html.P(
                        "First/second moment norms, effective learning rate, and warmup "
                        "progress from the optimizer's internal buffers.",
                        className="text-muted",
                    ),
                    html.Div(id="live-optim-summary", className="small text-muted mb-2"),
                    dcc.Graph(id="live-optim-chart"),
                    html.Div(id="live-optim-table"),
                ]), className="mt-2 mb-3"),
            ]),

            # ── Predictions tab ──────────────────────────────────────
            dbc.Tab(label="Predictions", children=[
                dbc.Card(dbc.CardBody([
                    html.H5("Prediction Calibration", className="card-title mt-2"),
                    html.P(
                        "Predicted vs actual values logged via gradienthound.log_predictions().",
                        className="text-muted",
                    ),
                    html.Div(id="live-pred-summary", className="small text-muted mb-2"),
                    dcc.Graph(id="live-pred-scatter"),
                    html.Div(id="live-pred-table"),
                ]), className="mt-2 mb-3"),
            ]),

            # ── Attention tab ────────────────────────────────────────
            dbc.Tab(label="Attention", children=[
                dbc.Card(dbc.CardBody([
                    html.H5("Attention Patterns", className="card-title mt-2"),
                    html.P(
                        "Attention weight matrices logged via gradienthound.log_attention().",
                        className="text-muted",
                    ),
                    html.Div(id="live-attn-summary", className="small text-muted mb-2"),
                    dcc.Dropdown(id="live-attn-select", placeholder="Select attention head", className="mb-2"),
                    dcc.Graph(id="live-attn-heatmap"),
                ]), className="mt-2 mb-3"),
            ]),
        ], className="mb-3"),
    ], fluid=True)
