"""Training Dynamics page: norm change, drift, velocity, convergence, phases."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html, dcc

from ._helpers import (
    plotly_layout, short_layer, column_summary, summary_chart,
    compute_checkpoint_change_tables, compute_scalar_metric_tables,
    compute_norm_velocity_table, compute_convergence_scores,
    detect_training_phases,
    render_checkpoint_change_table,
)


def dynamics_page(model_data: dict | None, snapshots: list | None = None):
    import plotly.graph_objects as go

    children = [
        html.H2("Training Dynamics", className="mt-3 mb-1"),
        html.P("Norm changes, directional drift, velocity, convergence, and phase detection.",
                className="text-muted mb-4"),
    ]

    if not snapshots or len(snapshots) < 2:
        children.append(dbc.Alert(
            "Load at least 2 checkpoints to see training dynamics.",
            color="info",
        ))
        return dbc.Container(children, fluid=True)

    # ── Norm Change tables ───────────────────────────────────────────
    ckpt_names, change_layers, diff_table, rel_table = compute_checkpoint_change_tables(snapshots)
    diff_slider_marks = {i: str(i + 1) for i in range(len(ckpt_names))}

    children.append(dbc.Card(dbc.CardBody([
        html.H5("Norm Change per Layer", className="card-title"),
        html.P("Absolute change in L2 norm between consecutive checkpoints.", className="text-muted"),
        dbc.RadioItems(
            id="grad-diff-mode",
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
                id="grad-diff-slider",
                min=0,
                max=len(ckpt_names) - 1,
                step=1,
                value=0,
                marks=diff_slider_marks,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(
                [
                    html.Span("Selected checkpoint: ", className="text-muted"),
                    html.Span(ckpt_names[0], id="grad-diff-slider-label", className="fw-semibold"),
                ],
                className="mt-2 small",
            ),
        ], id="grad-diff-slider-wrap", style={"display": "none"}, className="mb-2"),
        html.Div(
            id="grad-diff-table-wrap",
            children=render_checkpoint_change_table(
                checkpoint_names=ckpt_names,
                all_layers=change_layers,
                values_table=diff_table,
                mode="full",
                selected_idx=0,
                formatter=lambda v: f"{v:.6g}",
            ),
        ),
    ]), className="mb-3"))

    children.append(dbc.Card(dbc.CardBody([
        html.H5("Relative Norm Change", className="card-title"),
        html.P("Percentage change in L2 norm relative to the previous checkpoint.", className="text-muted"),
        dbc.RadioItems(
            id="grad-rel-mode",
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
                id="grad-rel-slider",
                min=0,
                max=len(ckpt_names) - 1,
                step=1,
                value=0,
                marks=diff_slider_marks,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(
                [
                    html.Span("Selected checkpoint: ", className="text-muted"),
                    html.Span(ckpt_names[0], id="grad-rel-slider-label", className="fw-semibold"),
                ],
                className="mt-2 small",
            ),
        ], id="grad-rel-slider-wrap", style={"display": "none"}, className="mb-2"),
        html.Div(
            id="grad-rel-table-wrap",
            children=render_checkpoint_change_table(
                checkpoint_names=ckpt_names,
                all_layers=change_layers,
                values_table=rel_table,
                mode="full",
                selected_idx=0,
                formatter=lambda v: f"{v:.2f}%",
            ),
        ),
    ]), className="mb-3"))

    # ── Directional Drift Metrics ────────────────────────────────────
    drift_metric_meta = {
        "drift_cosine_prev": {"label": "Cosine vs Prev", "fmt": lambda v: f"{v:.4f}"},
        "drift_subspace_overlap_prev": {"label": "Subspace Overlap vs Prev", "fmt": lambda v: f"{v:.4f}"},
        "drift_cka_prev": {"label": "CKA vs Prev", "fmt": lambda v: f"{v:.4f}"},
    }
    drift_checkpoint_names, drift_layers, drift_tables = compute_scalar_metric_tables(
        snapshots,
        list(drift_metric_meta.keys()),
        max_layers=50,
    )
    if drift_layers and drift_tables:
        drift_options = [
            {"label": drift_metric_meta[key]["label"], "value": key}
            for key in drift_metric_meta
            if key in drift_tables
        ]
        default_drift_metric = drift_options[0]["value"]
        drift_slider_marks = {i: str(i + 1) for i in range(len(drift_checkpoint_names))}
        children.append(dbc.Card(dbc.CardBody([
            html.H5("Directional Drift", className="card-title"),
            html.P(
                "Checkpoint-to-checkpoint alignment metrics for each layer.",
                className="text-muted",
            ),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id="drift-metric-select",
                    options=drift_options,
                    value=default_drift_metric,
                    clearable=False,
                ), md=5),
                dbc.Col(dbc.RadioItems(
                    id="drift-metric-mode",
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
                    id="drift-metric-slider",
                    min=0,
                    max=len(drift_checkpoint_names) - 1,
                    step=1,
                    value=0,
                    marks=drift_slider_marks,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Div(
                    [
                        html.Span("Selected checkpoint: ", className="text-muted"),
                        html.Span(
                            drift_checkpoint_names[0],
                            id="drift-metric-slider-label",
                            className="fw-semibold",
                        ),
                    ],
                    className="mt-2 small",
                ),
            ], id="drift-metric-slider-wrap", style={"display": "none"}, className="mb-2"),
            html.Div(
                id="drift-metric-table-wrap",
                children=render_checkpoint_change_table(
                    checkpoint_names=drift_checkpoint_names,
                    all_layers=drift_layers,
                    values_table=drift_tables[default_drift_metric],
                    mode="full",
                    selected_idx=0,
                    formatter=drift_metric_meta[default_drift_metric]["fmt"],
                ),
            ),
        ]), className="mb-3"))

        # ── Drift Summary Chart ──────────────────────────────────────
        if len(drift_checkpoint_names) >= 2:
            drift_chart = go.Figure()
            drift_colors = {"drift_cosine_prev": "#375a7f", "drift_subspace_overlap_prev": "#e67e22", "drift_cka_prev": "#00bc8c"}
            for dkey, dtable in drift_tables.items():
                d_means, _, _ = column_summary(dtable, len(drift_checkpoint_names))
                drift_chart.add_trace(go.Scatter(
                    x=drift_checkpoint_names, y=d_means, mode="lines+markers",
                    name=drift_metric_meta[dkey]["label"],
                    line={"color": drift_colors.get(dkey, "#375a7f")},
                ))
            drift_chart.update_layout(
                **plotly_layout(title="Directional Drift Summary"),
                height=340, xaxis_title="Checkpoint", yaxis_title="Mean Alignment",
            )
            children.append(dbc.Card(dbc.CardBody([
                html.H5("Drift Summary Chart", className="card-title"),
                html.P("Mean alignment across all layers per checkpoint.", className="text-muted"),
                dcc.Graph(figure=drift_chart),
            ]), className="mb-3"))

    # ── Norm Velocity & Acceleration ─────────────────────────────────
    vel_ckpt_names, vel_layers, vel_table, acc_table = compute_norm_velocity_table(snapshots, max_layers=100)
    if vel_layers:
        vel_means, vel_mins, vel_maxs = column_summary(vel_table, len(vel_ckpt_names))
        vel_chart = go.Figure()
        vel_chart.add_trace(go.Scatter(
            x=vel_ckpt_names, y=vel_means, mode="lines+markers",
            name="mean velocity", line={"color": "#375a7f", "width": 2.5},
        ))
        vel_chart.add_trace(go.Scatter(
            x=vel_ckpt_names, y=vel_maxs, mode="lines",
            name="max", line={"color": "#e67e22", "dash": "dot"},
        ))
        vel_chart.add_trace(go.Scatter(
            x=vel_ckpt_names, y=vel_mins, mode="lines",
            name="min", line={"color": "#00bc8c", "dash": "dot"},
        ))

        if len(snapshots) >= 3:
            acc_means, _, _ = column_summary(acc_table, len(vel_ckpt_names))
            vel_chart.add_trace(go.Scatter(
                x=vel_ckpt_names, y=acc_means, mode="lines+markers",
                name="mean acceleration", line={"color": "#e74c3c", "width": 2},
                yaxis="y2",
            ))
            vel_chart.update_layout(yaxis2={
                "title": "Acceleration", "overlaying": "y", "side": "right",
                "showgrid": False,
            })

        vel_chart.update_layout(
            **plotly_layout(title="Norm Velocity & Acceleration"),
            height=360,
            xaxis_title="Checkpoint", yaxis_title="Velocity (norm delta)",
        )
        vel_chart.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        children.append(dbc.Card(dbc.CardBody([
            html.H5("Norm Velocity & Acceleration", className="card-title"),
            html.P(
                "First derivative (velocity) and second derivative (acceleration) "
                "of L2 norm across checkpoints. Negative acceleration with positive "
                "velocity indicates convergence.",
                className="text-muted",
            ),
            dcc.Graph(figure=vel_chart),
        ]), className="mb-3"))

        # Acceleration heatmap (convergence/divergence visual)
        if len(snapshots) >= 3 and any(
            v is not None for row in acc_table for v in row
        ):
            acc_signs: list[list[float | None]] = []
            for row in acc_table:
                sign_row = []
                for v in row:
                    if v is None:
                        sign_row.append(None)
                    elif abs(v) < 1e-10:
                        sign_row.append(0.0)
                    else:
                        sign_row.append(1.0 if v > 0 else -1.0)
                acc_signs.append(sign_row)

            acc_heatmap = go.Figure(go.Heatmap(
                z=acc_signs,
                x=vel_ckpt_names,
                y=[short_layer(l) for l in vel_layers],
                colorscale=[
                    [0, "#00bc8c"],
                    [0.5, "#f8f9fa"],
                    [1, "#e74c3c"],
                ],
                zmin=-1, zmax=1,
                colorbar={"title": "Accel Sign", "tickvals": [-1, 0, 1],
                          "ticktext": ["Converging", "Stable", "Diverging"]},
                hovertemplate="Layer: %{y}<br>Checkpoint: %{x}<br>Sign: %{z}<extra></extra>",
            ))
            acc_heatmap.update_layout(
                **plotly_layout(title="Acceleration Direction (Convergence Map)"),
                height=max(300, len(vel_layers) * 14 + 100),
                xaxis_title="Checkpoint",
            )
            children.append(dbc.Card(dbc.CardBody([
                html.H5("Convergence Map", className="card-title"),
                html.P(
                    "Green = decelerating (converging), Red = accelerating (diverging). "
                    "Derived from second derivative of L2 norm.",
                    className="text-muted",
                ),
                dcc.Graph(figure=acc_heatmap),
            ]), className="mb-3"))

    # ── Layer-wise Convergence Score ──────────────────────────────────
    conv_ckpt_names, conv_layers, conv_table = compute_convergence_scores(snapshots, max_layers=100)
    if conv_layers and any(v is not None for row in conv_table for v in row):
        conv_means, conv_mins, conv_maxs = column_summary(conv_table, len(conv_ckpt_names))
        conv_summary = summary_chart(
            conv_ckpt_names, conv_means, conv_mins, conv_maxs,
            "Convergence Score Evolution", "Score (0-100)",
        )

        conv_heatmap = go.Figure(go.Heatmap(
            z=conv_table,
            x=conv_ckpt_names,
            y=[short_layer(l) for l in conv_layers],
            colorscale=[
                [0, "#e74c3c"],
                [0.5, "#f39c12"],
                [1, "#00bc8c"],
            ],
            zmin=0, zmax=100,
            colorbar={"title": "Score"},
            hovertemplate="Layer: %{y}<br>Checkpoint: %{x}<br>Score: %{z:.1f}<extra></extra>",
        ))
        conv_heatmap.update_layout(
            **plotly_layout(title="Layer Convergence Heatmap"),
            height=max(300, len(conv_layers) * 14 + 100),
            xaxis_title="Checkpoint",
        )

        children.append(dbc.Card(dbc.CardBody([
            html.H5("Convergence Score", className="card-title"),
            html.P(
                "Composite score (0-100) per layer combining cosine stability, "
                "rank stability, kurtosis stability, norm velocity, and spectral "
                "structure. Higher = more converged.",
                className="text-muted",
            ),
            dcc.Graph(figure=conv_summary),
            dcc.Graph(figure=conv_heatmap),
        ]), className="mb-3"))

    # ── Training Phase Detection ─────────────────────────────────────
    if len(snapshots) >= 3:
        phases = detect_training_phases(snapshots)
        if phases:
            phase_colors = {"learning": "#00bc8c", "plateau": "#f39c12", "instability": "#e74c3c"}
            phase_rows = []
            for p in phases:
                color = phase_colors.get(p["phase"], "#6c757d")
                phase_rows.append(html.Tr([
                    html.Td(dbc.Badge(
                        p["phase"].title(),
                        style={"backgroundColor": color},
                    )),
                    html.Td(p["start_name"]),
                    html.Td(p["end_name"]),
                    html.Td(f"{p['intensity']:.4g}"),
                ]))

            children.append(dbc.Card(dbc.CardBody([
                html.H5("Training Phases", className="card-title"),
                html.P(
                    "Automatically detected training phases based on aggregate "
                    "norm change intensity and cross-layer variance.",
                    className="text-muted",
                ),
                html.Div([
                    html.Span([
                        html.Span(
                            "",
                            style={
                                "display": "inline-block", "width": "14px",
                                "height": "14px", "borderRadius": "3px",
                                "backgroundColor": color, "marginRight": "6px",
                                "verticalAlign": "middle",
                            },
                        ),
                        html.Span(label.title(), style={"marginRight": "16px"}),
                    ])
                    for label, color in phase_colors.items()
                ], className="mb-3"),
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Phase"), html.Th("From"), html.Th("To"),
                        html.Th("Mean Intensity"),
                    ])),
                    html.Tbody(phase_rows),
                ], bordered=True, hover=True, responsive=True, size="sm"),
            ]), className="mb-3"))

    # ── Update Ratio & Delta Norm ────────────────────────────────────
    ur_metric_meta = {
        "update_ratio_prev": {"label": "Update Ratio (||delta||/||W||)", "fmt": lambda v: f"{v:.4g}"},
        "delta_norm": {"label": "Delta Norm (||W_t - W_{t-1}||)", "fmt": lambda v: f"{v:.4g}"},
    }
    ur_ckpt_names, ur_layers, ur_tables = compute_scalar_metric_tables(
        snapshots, list(ur_metric_meta.keys()), max_layers=50,
    )
    if ur_layers and ur_tables:
        if "update_ratio_prev" in ur_tables and len(ur_ckpt_names) >= 2:
            ur_means, ur_mins, ur_maxs = column_summary(ur_tables["update_ratio_prev"], len(ur_ckpt_names))
            ur_chart = summary_chart(
                ur_ckpt_names, ur_means, ur_mins, ur_maxs,
                "True Update Ratio Evolution", "||delta|| / ||W||",
            )
            children.append(dbc.Card(dbc.CardBody([
                html.H5("True Update Ratio", className="card-title"),
                html.P(
                    "||W_t - W_{t-1}|| / ||W_{t-1}|| per layer. Unlike the existing "
                    "relative norm change, this captures rotational updates that preserve "
                    "norm magnitude. Layers with very low ratios may be frozen.",
                    className="text-muted",
                ),
                dcc.Graph(figure=ur_chart),
            ]), className="mb-3"))

    # ── Delta Direction Consistency ──────────────────────────────────
    if len(snapshots) >= 3:
        dd_ckpt_names, dd_layers, dd_tables = compute_scalar_metric_tables(
            snapshots, ["delta_direction_consistency"], max_layers=100,
        )
        if dd_layers and dd_tables and "delta_direction_consistency" in dd_tables:
            dd_table = dd_tables["delta_direction_consistency"]
            dd_means, _, _ = column_summary(dd_table, len(dd_ckpt_names))
            dd_chart = summary_chart(
                dd_ckpt_names, dd_means, [None] * len(dd_ckpt_names), [None] * len(dd_ckpt_names),
                "Delta Direction Consistency", "Cosine Similarity",
            )
            dd_chart.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

            dd_heatmap = go.Figure(go.Heatmap(
                z=dd_table,
                x=dd_ckpt_names,
                y=[short_layer(l) for l in dd_layers],
                colorscale=[
                    [0, "#e74c3c"],
                    [0.5, "#f8f9fa"],
                    [1, "#375a7f"],
                ],
                zmin=-1, zmax=1,
                colorbar={"title": "Cos Sim",
                          "tickvals": [-1, 0, 1],
                          "ticktext": ["Oscillating", "Orthogonal", "Consistent"]},
                hovertemplate="Layer: %{y}<br>Checkpoint: %{x}<br>Cos: %{z:.3f}<extra></extra>",
            ))
            dd_heatmap.update_layout(
                **plotly_layout(title="Delta Direction Consistency"),
                height=max(300, len(dd_layers) * 14 + 100),
                xaxis_title="Checkpoint",
            )

            children.append(dbc.Card(dbc.CardBody([
                html.H5("Delta Direction Consistency", className="card-title"),
                html.P(
                    "Cosine similarity between consecutive weight updates (delta_t vs delta_{t-1}). "
                    "Blue = updates move in the same direction (smooth convergence). "
                    "Red = updates oscillate (LR too high or saddle points).",
                    className="text-muted",
                ),
                dcc.Graph(figure=dd_chart),
                dcc.Graph(figure=dd_heatmap),
            ]), className="mb-3"))

    # ── Initialization Distance ──────────────────────────────────────
    init_metric_meta = {
        "init_dist_relative": {"label": "Relative Distance from Init"},
        "init_dist_cosine": {"label": "Cosine vs Init"},
    }
    init_ckpt_names, init_layers, init_tables = compute_scalar_metric_tables(
        snapshots, list(init_metric_meta.keys()), max_layers=100,
    )
    if init_layers and init_tables:
        init_chart = go.Figure()
        init_colors = {"init_dist_relative": "#375a7f", "init_dist_cosine": "#e67e22"}
        for mkey, meta in init_metric_meta.items():
            if mkey not in init_tables:
                continue
            m_means, _, _ = column_summary(init_tables[mkey], len(init_ckpt_names))
            fig_kwargs = {}
            if mkey == "init_dist_cosine":
                fig_kwargs["yaxis"] = "y2"
            init_chart.add_trace(go.Scatter(
                x=init_ckpt_names, y=m_means, mode="lines+markers",
                name=meta["label"],
                line={"color": init_colors.get(mkey, "#375a7f")},
                **fig_kwargs,
            ))
        init_chart.update_layout(
            **plotly_layout(title="Initialization Distance Evolution"),
            height=360,
            xaxis_title="Checkpoint",
            yaxis_title="Relative L2 Distance",
            yaxis2={"title": "Cosine vs Init", "overlaying": "y", "side": "right",
                    "showgrid": False},
        )

        children.append(dbc.Card(dbc.CardBody([
            html.H5("Initialization Distance", className="card-title"),
            html.P(
                "How far each layer has moved from the first checkpoint. "
                "Layers with small distance may be undertrained; very large distances "
                "may indicate overfitting. Cosine measures rotational change.",
                className="text-muted",
            ),
            dcc.Graph(figure=init_chart),
        ]), className="mb-3"))

    # ── Cross-Layer Update Correlation ───────────────────────────────
    latest_corr = snapshots[-1].get("delta_correlation_matrix")
    if latest_corr and isinstance(latest_corr, dict):
        corr_layers = latest_corr.get("layers", [])
        corr_matrix = latest_corr.get("matrix", [])
        if corr_layers and corr_matrix:
            corr_fig = go.Figure(go.Heatmap(
                z=corr_matrix,
                x=[short_layer(l) for l in corr_layers],
                y=[short_layer(l) for l in corr_layers],
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                colorbar={"title": "Correlation"},
                hovertemplate="Row: %{y}<br>Col: %{x}<br>Corr: %{z:.3f}<extra></extra>",
            ))
            corr_fig.update_layout(
                **plotly_layout(title=f"Cross-Layer Update Correlation \u2014 {snapshots[-1]['name']}"),
                height=max(400, len(corr_layers) * 14 + 120),
                xaxis_tickangle=-45,
            )
            children.append(dbc.Card(dbc.CardBody([
                html.H5("Cross-Layer Update Correlation", className="card-title"),
                html.P(
                    "Pairwise Pearson correlation of weight deltas between 2D layers. "
                    "High correlation means layers receive similar gradient signals. "
                    "Anti-correlation between paired layers (e.g., Q and K) is expected.",
                    className="text-muted",
                ),
                dcc.Graph(figure=corr_fig),
            ]), className="mb-3"))

    return dbc.Container(children, fluid=True)
