"""Gradient flow page layout."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html, dcc


def gradient_flow_page(model_names: list[str] | None = None):
    model_options = [
        {"label": name, "value": name}
        for name in (model_names or [])
    ]
    default_model = model_options[0]["value"] if model_options else None

    return dbc.Container([
        html.H2("Gradient Flow", className="mt-3 mb-1"),
        html.P(
            "Classic gradient flow across layers using average and maximum absolute gradients.",
            className="text-muted mb-4",
        ),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Model", html_for="gradflow-model-select"),
                    dcc.Dropdown(
                        id="gradflow-model-select",
                        options=model_options,
                        value=default_model,
                        clearable=False,
                        placeholder="Select a model",
                    ),
                ], md=5),
                dbc.Col([
                    dbc.Label("Window (steps)", html_for="gradflow-window"),
                    dcc.Slider(
                        id="gradflow-window",
                        min=1,
                        max=100,
                        step=1,
                        value=10,
                        marks={1: "1", 10: "10", 25: "25", 50: "50", 100: "100"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], md=5),
                dbc.Col([
                    dbc.Label("Options"),
                    dbc.Checklist(
                        id="gradflow-hide-bias",
                        options=[{"label": "Hide bias params", "value": "hide_bias"}],
                        value=["hide_bias"],
                        switch=True,
                    ),
                ], md=2),
            ], className="g-3 mb-2"),
            html.Div(id="gradflow-summary", className="small text-muted mb-2"),
            dcc.Graph(id="gradflow-chart"),
            dcc.Interval(id="gradflow-refresh", interval=2000, n_intervals=0),
        ]), className="mb-3"),
    ], fluid=True)
