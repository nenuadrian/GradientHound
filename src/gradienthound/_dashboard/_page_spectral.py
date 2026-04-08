"""Spectral page: effective rank, SVD, ESD, WeightWatcher metrics, spectral gaps."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html, dcc

from ._helpers import (
    plotly_layout, short_layer, column_summary, summary_chart,
    compute_effective_rank_table, compute_scalar_metric_tables,
    compute_spectral_gap_table,
    render_checkpoint_change_table,
)


def spectral_page(model_data: dict | None, snapshots: list | None = None):
    import plotly.graph_objects as go

    children = [
        html.H2("Spectral Analysis", className="mt-3 mb-1"),
        html.P("Effective rank, singular value spectra, ESD, and WeightWatcher metrics.",
                className="text-muted mb-4"),
    ]

    if not snapshots:
        children.append(dbc.Alert(
            "Load checkpoints to see spectral analysis.",
            color="info",
        ))
        return dbc.Container(children, fluid=True)

    snap = snapshots[-1]
    stats = snap["weight_stats"]

    # ── Effective Rank Evolution ──────────────────────────────────────
    checkpoint_names, svd_layers, svd_rank_table = compute_effective_rank_table(snapshots)

    if svd_layers:
        rank_slider_marks = {i: str(i + 1) for i in range(len(checkpoint_names))}

        children.append(dbc.Card(dbc.CardBody([
            html.H5("Effective Rank Evolution", className="card-title"),
            html.P("Entropy-based effective rank for 2D weight matrices across checkpoints.",
                    className="text-muted"),
            dbc.RadioItems(
                id="ckpt-svd-rank-mode",
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
                    id="ckpt-svd-rank-slider",
                    min=0,
                    max=len(checkpoint_names) - 1,
                    step=1,
                    value=0,
                    marks=rank_slider_marks,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Div(
                    [
                        html.Span("Selected checkpoint: ", className="text-muted"),
                        html.Span(checkpoint_names[0], id="ckpt-svd-rank-slider-label", className="fw-semibold"),
                    ],
                    className="mt-2 small",
                ),
            ], id="ckpt-svd-rank-slider-wrap", style={"display": "none"}, className="mb-2"),
            html.Div(
                id="ckpt-svd-rank-table-wrap",
                children=render_checkpoint_change_table(
                    checkpoint_names=checkpoint_names,
                    all_layers=svd_layers,
                    values_table=svd_rank_table,
                    mode="full",
                    selected_idx=0,
                    formatter=lambda v: f"{v:.4g}",
                ),
            ),
        ]), className="mb-3"))

        # ── Effective Rank Chart ─────────────────────────────────────
        if len(checkpoint_names) >= 2:
            r_means, r_mins, r_maxs = column_summary(svd_rank_table, len(checkpoint_names))
            rank_chart = summary_chart(
                checkpoint_names, r_means, r_mins, r_maxs,
                "Effective Rank Summary", "Effective Rank",
            )
            children.append(dbc.Card(dbc.CardBody([
                html.H5("Effective Rank Trend", className="card-title"),
                html.P("Aggregate effective rank across all 2D weight layers per checkpoint.",
                        className="text-muted"),
                dcc.Graph(figure=rank_chart),
            ]), className="mb-3"))

        # ── Singular Value Spectrum ──────────────────────────────────
        children.append(dbc.Card(dbc.CardBody([
            html.H5("Singular Value Spectrum", className="card-title"),
            html.P("Select a layer to compare SVD spectra across checkpoints.", className="text-muted"),
            dcc.Dropdown(id="ckpt-svd-layer-select",
                         options=[{"label": short_layer(l), "value": l} for l in svd_layers],
                         value=svd_layers[0] if svd_layers else None, className="mb-2"),
            dcc.Graph(id="ckpt-svd-spectrum"),
        ]), className="mb-3"))

    # ── WeightWatcher Metrics (curated) ──────────────────────────────
    ww_metric_meta = {
        "alpha": {"label": "Alpha", "fmt": lambda v: f"{v:.3f}"},
        "alpha_weighted": {"label": "Alpha Weighted", "fmt": lambda v: f"{v:.3f}"},
        "mp_softrank": {"label": "MP Softrank", "fmt": lambda v: f"{v:.3f}"},
        "num_spikes": {"label": "Spikes", "fmt": lambda v: f"{v:.0f}"},
        "log_spectral_norm": {"label": "log10(lambda_max)", "fmt": lambda v: f"{v:.3f}"},
        "lambda_plus": {"label": "MP Edge (lambda+)", "fmt": lambda v: f"{v:.3g}"},
    }
    ww_checkpoint_names, ww_layers, ww_tables = compute_scalar_metric_tables(
        snapshots,
        list(ww_metric_meta.keys()),
        max_layers=50,
    )

    if ww_layers and ww_tables:
        ww_options = [
            {"label": ww_metric_meta[key]["label"], "value": key}
            for key in ww_metric_meta
            if key in ww_tables
        ]
        default_ww_metric = ww_options[0]["value"]
        ww_slider_marks = {i: str(i + 1) for i in range(len(ww_checkpoint_names))}
        children.append(dbc.Card(dbc.CardBody([
            html.H5("WeightWatcher Metrics", className="card-title"),
            html.P(
                "Checkpoint spectral quality metrics. Default view is curated; "
                "raw ESD remains in the detailed spectral section below.",
                className="text-muted",
            ),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id="ww-metric-select",
                    options=ww_options,
                    value=default_ww_metric,
                    clearable=False,
                ), md=5),
                dbc.Col(dbc.RadioItems(
                    id="ww-metric-mode",
                    options=[
                        {"label": "Full table", "value": "full"},
                        {"label": "Single checkpoint", "value": "single"},
                    ],
                    value="full",
                    inline=True,
                ), md=7),
            ], className="g-2 mb-2"),
            html.Div([
                dcc.Slider(
                    id="ww-metric-slider",
                    min=0,
                    max=len(ww_checkpoint_names) - 1,
                    step=1,
                    value=0,
                    marks=ww_slider_marks,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Div(
                    [
                        html.Span("Selected checkpoint: ", className="text-muted"),
                        html.Span(ww_checkpoint_names[0], id="ww-metric-slider-label", className="fw-semibold"),
                    ],
                    className="mt-2 small",
                ),
            ], id="ww-metric-slider-wrap", style={"display": "none"}, className="mb-2"),
            html.Div(
                id="ww-metric-table-wrap",
                children=render_checkpoint_change_table(
                    checkpoint_names=ww_checkpoint_names,
                    all_layers=ww_layers,
                    values_table=ww_tables[default_ww_metric],
                    mode="full",
                    selected_idx=0,
                    formatter=ww_metric_meta[default_ww_metric]["fmt"],
                ),
            ),
        ]), className="mb-3"))

    # ── Spectral Gap Ratios ──────────────────────────────────────────
    gap_ckpt_names, gap_layers, gap_table = compute_spectral_gap_table(snapshots, max_layers=50)
    if gap_layers and len(gap_ckpt_names) >= 2:
        gap_means, gap_mins, gap_maxs = column_summary(gap_table, len(gap_ckpt_names))
        gap_chart = summary_chart(
            gap_ckpt_names, gap_means, gap_mins, gap_maxs,
            "Spectral Gap (sigma1/sigma2) Evolution", "Gap Ratio",
        )
        gap_slider_marks = {i: str(i + 1) for i in range(len(gap_ckpt_names))}
        children.append(dbc.Card(dbc.CardBody([
            html.H5("Spectral Gap Ratios", className="card-title"),
            html.P(
                "Ratio of consecutive singular values (sigma1/sigma2). "
                "A growing gap signals rank-1 collapse; in attention layers, "
                "gaps relate to effective head count.",
                className="text-muted",
            ),
            dcc.Graph(figure=gap_chart),
            dbc.RadioItems(
                id="spectral-gap-mode",
                options=[
                    {"label": "Full table", "value": "full"},
                    {"label": "Single checkpoint", "value": "single"},
                ],
                value="full",
                inline=True,
                className="mb-2 mt-3",
            ),
            html.Div([
                dcc.Slider(
                    id="spectral-gap-slider",
                    min=0,
                    max=len(gap_ckpt_names) - 1,
                    step=1,
                    value=0,
                    marks=gap_slider_marks,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Div(
                    [
                        html.Span("Selected checkpoint: ", className="text-muted"),
                        html.Span(gap_ckpt_names[0], id="spectral-gap-slider-label", className="fw-semibold"),
                    ],
                    className="mt-2 small",
                ),
            ], id="spectral-gap-slider-wrap", style={"display": "none"}, className="mb-2"),
            html.Div(
                id="spectral-gap-table-wrap",
                children=render_checkpoint_change_table(
                    checkpoint_names=gap_ckpt_names,
                    all_layers=gap_layers,
                    values_table=gap_table,
                    mode="full",
                    selected_idx=0,
                    formatter=lambda v: f"{v:.2f}",
                ),
            ),
        ]), className="mb-3"))

        # Per-layer spectral gap detail chart
        children.append(dbc.Card(dbc.CardBody([
            html.H5("Spectral Gap Detail", className="card-title"),
            html.P(
                "Select a layer to see top-5 spectral gap ratios across checkpoints.",
                className="text-muted",
            ),
            dcc.Dropdown(
                id="spectral-gap-layer-select",
                options=[{"label": short_layer(l), "value": l} for l in gap_layers],
                value=gap_layers[0] if gap_layers else None,
                className="mb-2",
            ),
            dcc.Graph(id="spectral-gap-detail-chart"),
        ]), className="mb-3"))

    # ── Singular Value Turnover ──────────────────────────────────────
    if len(snapshots) >= 2:
        svt_metric_meta = {
            "sv_turnover_rate": {"label": "SV Turnover Rate"},
            "principal_direction_stability": {"label": "Principal Direction Stability"},
        }
        svt_ckpt_names, svt_layers, svt_tables = compute_scalar_metric_tables(
            snapshots, list(svt_metric_meta.keys()), max_layers=100,
        )
        if svt_layers and svt_tables:
            svt_chart = go.Figure()
            svt_colors = {"sv_turnover_rate": "#e74c3c", "principal_direction_stability": "#375a7f"}
            for mkey, meta in svt_metric_meta.items():
                if mkey not in svt_tables:
                    continue
                m_means, _, _ = column_summary(svt_tables[mkey], len(svt_ckpt_names))
                svt_chart.add_trace(go.Scatter(
                    x=svt_ckpt_names, y=m_means, mode="lines+markers",
                    name=meta["label"],
                    line={"color": svt_colors.get(mkey, "#375a7f")},
                ))
            svt_chart.update_layout(
                **plotly_layout(title="Singular Value Dynamics"),
                height=360,
                xaxis_title="Checkpoint",
                yaxis_title="Rate / Stability",
            )

            children.append(dbc.Card(dbc.CardBody([
                html.H5("Singular Value Turnover", className="card-title"),
                html.P(
                    "Turnover rate: fraction of top-k singular vectors that don't match "
                    "the previous checkpoint (high = new directions emerging). "
                    "Principal stability: how stable the leading singular vector is.",
                    className="text-muted",
                ),
                dcc.Graph(figure=svt_chart),
            ]), className="mb-3"))

    # ── Spectral Analysis (ESD) ──────────────────────────────────────
    esd_layers = [
        s["layer"] for s in stats
        if s.get("esd") and not s["layer"].endswith(".bias")
    ]
    if esd_layers:
        children.append(dbc.Card(dbc.CardBody([
            html.H5("Spectral Analysis (ESD)", className="card-title"),
            html.P(
                "Eigenvalue Spectral Density of weight matrices. "
                "The power-law tail exponent (alpha) indicates layer quality: "
                "well-trained layers typically have alpha in [2, 6].",
                className="text-muted",
            ),
            dcc.Dropdown(
                id="esd-layer-select",
                options=[{"label": short_layer(l), "value": l} for l in esd_layers],
                value=esd_layers[0],
                className="mb-2",
            ),
            dcc.Graph(id="esd-chart"),
        ]), className="mb-3"))

    return dbc.Container(children, fluid=True)
