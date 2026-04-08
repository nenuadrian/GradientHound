"""Distributions page: histograms, statistics, kurtosis, and entropy."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html, dcc

from ._helpers import (
    plotly_layout, short_layer, column_summary, summary_chart,
    compute_distribution_stats_table, compute_scalar_metric_tables,
    render_checkpoint_change_table,
)


def distributions_page(model_data: dict | None, snapshots: list | None = None):
    import plotly.graph_objects as go

    children = [
        html.H2("Distributions", className="mt-3 mb-1"),
        html.P("Weight distribution histograms, statistics, kurtosis, and entropy evolution.",
                className="text-muted mb-4"),
    ]

    if not snapshots:
        children.append(dbc.Alert(
            "Load checkpoints to see distribution analysis.",
            color="info",
        ))
        return dbc.Container(children, fluid=True)

    # Collect all layers across checkpoints
    all_layers = []
    seen = set()
    for s in snapshots:
        for st in s["weight_stats"]:
            if st["layer"] not in seen:
                all_layers.append(st["layer"])
                seen.add(st["layer"])
    default_layer = all_layers[0] if all_layers else ""

    # ── Layer Histogram Comparison ───────────────────────────────────
    children.append(dbc.Card(dbc.CardBody([
        html.H5("Layer Histogram Comparison", className="card-title"),
        html.P("Select a layer to compare weight distributions across checkpoints.",
                className="text-muted"),
        dcc.Dropdown(id="ckpt-layer-select",
                     options=[{"label": short_layer(l), "value": l} for l in all_layers],
                     value=default_layer, className="mb-2"),
        dcc.Graph(id="ckpt-histogram"),
    ]), className="mb-3"))

    # ── Distribution Statistics ──────────────────────────────────────
    stat_ckpt_names, stat_layers, stat_table = compute_distribution_stats_table(snapshots, max_layers=50)

    if stat_layers:
        stat_slider_marks = {i: str(i + 1) for i in range(len(stat_ckpt_names))}

        children.append(dbc.Card(dbc.CardBody([
            html.H5("Distribution Statistics", className="card-title"),
            dbc.RadioItems(
                id="dist-stats-mode",
                options=[
                    {"label": "Full table", "value": "full"},
                    {"label": "Single checkpoint", "value": "single"},
                ],
                value="full",
                inline=True,
                className="mb-2",
            ),
            html.Div([
                dcc.Slider(
                    id="dist-stats-slider",
                    min=0,
                    max=len(stat_ckpt_names) - 1,
                    step=1,
                    value=0,
                    marks=stat_slider_marks,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Div(
                    [
                        html.Span("Selected checkpoint: ", className="text-muted"),
                        html.Span(stat_ckpt_names[0], id="dist-stats-slider-label", className="fw-semibold"),
                    ],
                    className="mt-2 small",
                ),
            ], id="dist-stats-slider-wrap", style={"display": "none"}, className="mb-2"),
            html.Div(
                id="dist-stats-table-wrap",
                children=render_checkpoint_change_table(
                    checkpoint_names=stat_ckpt_names,
                    all_layers=stat_layers,
                    values_table=stat_table,
                    mode="full",
                    selected_idx=0,
                    formatter=lambda v: html.Code(v),
                ),
            ),
        ]), className="mb-3"))

    # ── Kurtosis & Sparsity Evolution Charts ─────────────────────────
    evo_metrics = {
        "kurtosis": {"label": "Kurtosis", "color": "#e74c3c"},
        "near_zero_pct": {"label": "Near-Zero %", "color": "#9b59b6"},
        "condition_number": {"label": "Condition Number", "color": "#1abc9c"},
    }
    evo_ckpt_names, evo_layers, evo_tables = compute_scalar_metric_tables(
        snapshots, list(evo_metrics.keys()), max_layers=100,
    )
    if evo_layers and evo_tables and len(evo_ckpt_names) >= 2:
        evo_fig = go.Figure()
        for mkey, meta in evo_metrics.items():
            if mkey not in evo_tables:
                continue
            m_means, _, _ = column_summary(evo_tables[mkey], len(evo_ckpt_names))
            if mkey == "condition_number":
                import math
                m_means = [math.log10(v) if v and v > 0 else None for v in m_means]
                label = "log10(Condition)"
            else:
                label = meta["label"]
            evo_fig.add_trace(go.Scatter(
                x=evo_ckpt_names, y=m_means, mode="lines+markers",
                name=label, line={"color": meta["color"]},
            ))
        evo_fig.update_layout(
            **plotly_layout(title="Distribution Metric Evolution"),
            height=340, xaxis_title="Checkpoint", yaxis_title="Mean Value",
        )
        children.append(dbc.Card(dbc.CardBody([
            html.H5("Distribution Metric Evolution", className="card-title"),
            html.P(
                "Mean kurtosis, near-zero weight fraction, and condition number "
                "across all layers per checkpoint.",
                className="text-muted",
            ),
            dcc.Graph(figure=evo_fig),
        ]), className="mb-3"))

    # ── Weight Entropy Evolution ─────────────────────────────────────
    entropy_ckpt_names, entropy_layers, entropy_tables = compute_scalar_metric_tables(
        snapshots, ["weight_entropy"], max_layers=100,
    )
    if entropy_layers and entropy_tables and "weight_entropy" in entropy_tables and len(entropy_ckpt_names) >= 2:
        e_means, e_mins, e_maxs = column_summary(entropy_tables["weight_entropy"], len(entropy_ckpt_names))
        entropy_chart = summary_chart(
            entropy_ckpt_names, e_means, e_mins, e_maxs,
            "Weight Distribution Entropy", "Shannon Entropy",
        )
        children.append(dbc.Card(dbc.CardBody([
            html.H5("Weight Entropy Evolution", className="card-title"),
            html.P(
                "Shannon entropy of the weight distribution histogram. "
                "Drops indicate distribution collapse; increases indicate diffusion. "
                "More informative than mean/std alone.",
                className="text-muted",
            ),
            dcc.Graph(figure=entropy_chart),
        ]), className="mb-3"))

    return dbc.Container(children, fluid=True)
