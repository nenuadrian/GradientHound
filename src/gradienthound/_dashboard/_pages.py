"""Page layout functions for the dashboard."""
from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import html, dcc

from ._constants import (
    MODULE_COLORS, SERIES_COLORS, PAGES,
    CYTO_STYLE, HEALTH_COLORS, HEALTH_SORT,
)
from ._helpers import (
    plotly_layout, fmt_num, module_category, short_layer,
    placeholder_page, compute_checkpoint_change_tables,
    render_checkpoint_change_table, compute_effective_rank_table,
    compute_distribution_stats_table,
    compute_optimizer_summary_table, compute_optimizer_evolution_table,
)
from ._graph import build_fx_elements, build_module_tree_elements
from ._health import weight_health, build_health_elements


# ── Dashboard page ───────────────────────────────────────────────────

def dashboard_page(model_data: dict, snapshots: list | None = None):
    import plotly.graph_objects as go

    mt = model_data.get("module_tree", {})
    modules = mt.get("modules", [])
    leaf_modules = [m for m in modules if m.get("is_leaf")]
    fx = model_data.get("fx_graph", {})
    fx_nodes = fx.get("nodes", [])
    ops = [n for n in fx_nodes if n["op"] == "call_function"]

    inputs = model_data.get("inputs", [])
    outputs = model_data.get("outputs", [])
    input_desc = ", ".join(
        f"{i.get('name', '?')}: {i.get('dtype', '?')}{i.get('shape', '')}" for i in inputs
    ) or "unknown"
    output_desc = ", ".join(
        f"{o.get('dtype', '?')}{o.get('shape', '')}" for o in outputs
    ) or "unknown"

    params = model_data.get("parameters", {})
    buffer_count = sum(1 for p in params.values() if p.get("is_buffer"))

    # ── 1. Stat cards ────────────────────────────────────────────────
    stat_items = [
        ("Parameters", fmt_num(model_data.get("total_params", 0))),
        ("Trainable", fmt_num(model_data.get("trainable_params", 0))),
        ("Layers", str(len(leaf_modules))),
        ("FX Ops", str(len(ops))),
        ("Buffers", str(buffer_count)),
    ]
    if snapshots:
        stat_items.append(("Checkpoints", str(len(snapshots))))

        # Health counts from latest snapshot
        latest_stats = snapshots[-1]["weight_stats"]
        counts = {"healthy": 0, "warning": 0, "critical": 0, "neutral": 0}
        for stat in latest_stats:
            state, _ = weight_health(stat)
            counts[state] = counts.get(state, 0) + 1
        for label in ("critical", "warning", "healthy"):
            if counts[label] > 0:
                stat_items.append((label.title(), str(counts[label])))

    children = [
        html.H2(model_data.get("model_name", "Model"), className="mt-3 mb-1"),
        html.P(model_data.get("model_class", ""), className="text-muted mb-4"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Small(label, className="text-uppercase text-muted"),
                html.H4(
                    value, className="mb-0",
                    style={"color": HEALTH_COLORS.get(label.lower(), "")} if label.lower() in HEALTH_COLORS else {},
                ),
            ])), width="auto") for label, value in stat_items
        ], className="g-3 mb-4"),

        # ── 2. I/O ──────────────────────────────────────────────────
        dbc.Card(dbc.CardBody([
            html.H5("I/O", className="card-title"),
            html.Code(f"Input:  {input_desc}", className="d-block mb-1"),
            html.Code(f"Output: {output_desc}"),
        ]), className="mb-3"),
    ]

    # ── 3. Architecture graph (health-colored when snapshots, plain otherwise)
    if snapshots:
        latest_stats = snapshots[-1]["weight_stats"]
        health_elements = build_health_elements(model_data, latest_stats)
        if health_elements:
            health_style = CYTO_STYLE + [
                {"selector": f".health-{state}", "style": {
                    "background-color": color,
                    "border-color": color,
                    "border-width": "2.5px" if state in ("critical", "warning") else "1.5px",
                }}
                for state, color in HEALTH_COLORS.items()
            ]
            children.append(dbc.Card(dbc.CardBody([
                html.H5("Architecture Health Map", className="card-title"),
                html.P(
                    "Module health derived from weight statistics. "
                    "Worst parameter determines module color.",
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
                    for label, color in HEALTH_COLORS.items()
                    if label != "neutral"
                ], className="mb-3"),
                cyto.Cytoscape(
                    id="gh-cyto-health",
                    elements=health_elements,
                    layout={
                        "name": "dagre", "rankDir": "TB",
                        "spacingFactor": 1.4, "nodeSep": 40, "rankSep": 60,
                    },
                    stylesheet=health_style,
                    style={"width": "100%", "height": "420px", "borderRadius": "8px"},
                    boxSelectionEnabled=False,
                    userZoomingEnabled=True,
                    userPanningEnabled=True,
                ),
                html.Div(id="ns-layer-detail", className="mt-3"),
            ]), className="mb-3"))

        # ── 4. Top Anomalies (checkpoint transition events) ─────────
        ranked_events: list[dict] = []
        for snap in snapshots[1:]:
            curr_name = snap.get("name", "?")
            compared_to = (snap.get("anomaly_summary") or {}).get("compared_to", snapshots[0].get("name", "?"))
            for event in snap.get("anomalies", []):
                row = dict(event)
                row["checkpoint"] = curr_name
                row["compared_to"] = compared_to
                ranked_events.append(row)

        ranked_events.sort(key=lambda e: e.get("score", 0.0), reverse=True)
        top_events = ranked_events[:20]

        if top_events:
            def _event_badge(event_type: str):
                if event_type == "rank_collapse":
                    return dbc.Badge("Rank Collapse", color="danger")
                if event_type == "kurtosis_spike":
                    return dbc.Badge("Kurtosis Spike", color="warning", text_color="dark")
                if event_type == "norm_jump_outlier":
                    return dbc.Badge("Norm Jump Outlier", color="info")
                return dbc.Badge(event_type, color="secondary")

            rows = []
            for event in top_events:
                rows.append(html.Tr([
                    html.Td(_event_badge(event.get("type", "unknown"))),
                    html.Td(html.Code(short_layer(event.get("layer", "?")))),
                    html.Td(event.get("checkpoint", "?")),
                    html.Td(event.get("compared_to", "?"), className="text-muted"),
                    html.Td(f"{event.get('score', 0.0):.2f}", className="fw-semibold"),
                    html.Td(event.get("message", ""), className="text-muted", style={"fontSize": "0.85em"}),
                ]))

            children.append(dbc.Card(dbc.CardBody([
                html.H5("Top Anomalies", className="card-title"),
                html.P(
                    "Ranked suspicious checkpoint transitions: effective-rank collapse, "
                    "kurtosis spikes, and norm-jump outliers.",
                    className="text-muted",
                ),
                html.Div(
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Type"), html.Th("Layer"), html.Th("Checkpoint"),
                            html.Th("Compared To"), html.Th("Score"), html.Th("Details"),
                        ])),
                        html.Tbody(rows),
                    ], bordered=True, hover=True, responsive=True, size="sm"),
                    style={"maxHeight": "480px", "overflowY": "auto"},
                ),
            ]), className="mb-3"))
    else:
        arch_elements = build_module_tree_elements(model_data)
        if arch_elements:
            children.append(dbc.Card(dbc.CardBody([
                html.H5(f"Architecture ({len(leaf_modules)} layers)", className="card-title"),
                html.P("Module hierarchy. Scroll to zoom, drag to pan.", className="text-muted"),
                cyto.Cytoscape(
                    id="gh-cyto-arch",
                    elements=arch_elements,
                    layout={"name": "dagre", "rankDir": "TB", "spacingFactor": 1.4, "nodeSep": 40, "rankSep": 60},
                    stylesheet=CYTO_STYLE,
                    style={"width": "100%", "height": "420px", "borderRadius": "8px"},
                    boxSelectionEnabled=False, userZoomingEnabled=True, userPanningEnabled=True,
                ),
                html.Div(id="gh-arch-detail", className="mt-3",
                          children=[html.P("Click a layer to inspect it.", className="text-muted")]),
            ]), className="mb-3"))

    # ── Sections 5–12: only when snapshots are available ─────────────
    if snapshots:
        snap = snapshots[-1]
        stats = snap["weight_stats"]
        snap_name = snap["name"]

        # Collect all layers across checkpoints
        all_layers = []
        seen = set()
        for s in snapshots:
            for st in s["weight_stats"]:
                if st["layer"] not in seen:
                    all_layers.append(st["layer"])
                    seen.add(st["layer"])
        default_layer = all_layers[0] if all_layers else ""

        # ── 4. L2 Norm per Layer (health-colored, single chart) ──────
        weight_layers = [s for s in stats if not s["layer"].endswith(".bias")]
        norm_fig = go.Figure()
        colors = [HEALTH_COLORS[weight_health(s)[0]] for s in weight_layers]
        norm_fig.add_trace(go.Bar(
            x=[short_layer(s["layer"]) for s in weight_layers],
            y=[s.get("norm_l2", 0) for s in weight_layers],
            marker_color=colors,
        ))
        norm_fig.update_layout(
            **plotly_layout(title=f"L2 Norm per Layer \u2014 {snap_name}"),
            xaxis_tickangle=-45, height=380,
        )
        children.append(dbc.Card(dbc.CardBody([
            html.H5("L2 Norm per Layer", className="card-title"),
            html.P("Weight norms colored by health state.", className="text-muted"),
            dcc.Graph(figure=norm_fig),
        ]), className="mb-3"))

        # ── 5. Layer Histogram Comparison ────────────────────────────
        children.append(dbc.Card(dbc.CardBody([
            html.H5("Layer Histogram Comparison", className="card-title"),
            html.P("Select a layer to compare weight distributions across checkpoints.",
                    className="text-muted"),
            dcc.Dropdown(id="ckpt-layer-select",
                         options=[{"label": short_layer(l), "value": l} for l in all_layers],
                         value=default_layer, className="mb-2"),
            dcc.Graph(id="ckpt-histogram"),
        ]), className="mb-3"))

        # ── 6. Effective Rank Evolution ──────────────────────────────
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

            # ── 7. Singular Value Spectrum ────────────────────────────
            children.append(dbc.Card(dbc.CardBody([
                html.H5("Singular Value Spectrum", className="card-title"),
                html.P("Select a layer to compare SVD spectra across checkpoints.", className="text-muted"),
                dcc.Dropdown(id="ckpt-svd-layer-select",
                             options=[{"label": short_layer(l), "value": l} for l in svd_layers],
                             value=svd_layers[0] if svd_layers else None, className="mb-2"),
                dcc.Graph(id="ckpt-svd-spectrum"),
            ]), className="mb-3"))

        # ── 8 & 9. Norm Change tables (only if ≥2 checkpoints) ──────
        if len(snapshots) >= 2:
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

        # ── 10. Distribution Statistics ──────────────────────────────
        stat_ckpt_names, stat_layers, stat_table = compute_distribution_stats_table(snapshots, max_layers=50)
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

        # ── Optimizer metrics (when present in checkpoint state) ───
        children.extend(_optimizer_state_cards(snapshots))

        # ── 11. Parameter Health table ───────────────────────────────
        health_rows: list[dict] = []
        for stat in stats:
            state, reason = weight_health(stat)
            shape_str = "x".join(str(s) for s in stat.get("shape", []))
            kind = "bias" if stat["layer"].endswith(".bias") else "weight"
            eff_rank = stat.get("effective_rank")
            max_rank = stat.get("max_rank")
            rank_str = (
                f"{100 * eff_rank / max(max_rank, 1):.0f}%"
                if eff_rank is not None and max_rank else "-"
            )
            health_rows.append({
                "layer": stat["layer"],
                "kind": kind,
                "shape": shape_str,
                "state": state,
                "reason": reason,
                "norm_l2": stat.get("norm_l2", 0),
                "near_zero_pct": stat.get("near_zero_pct"),
                "rank_util": rank_str,
                "condition": stat.get("condition_number"),
                "kurtosis": stat.get("kurtosis"),
                "mean": stat.get("mean", 0),
                "std": stat.get("std", 0),
            })

        health_rows.sort(key=lambda r: (HEALTH_SORT.get(r["state"], 3), r["layer"]))

        def _badge(state):
            return dbc.Badge(
                state.title(),
                color=(
                    "danger" if state == "critical"
                    else "warning" if state == "warning"
                    else "success" if state == "healthy"
                    else "secondary"
                ),
            )

        def _metric(val, fmt=".4g"):
            if val is None:
                return "-"
            return f"{val:{fmt}}"

        table_rows = []
        for r in health_rows:
            table_rows.append(html.Tr([
                html.Td(html.Code(short_layer(r["layer"]))),
                html.Td(r["kind"]),
                html.Td(html.Code(r["shape"])),
                html.Td(_badge(r["state"])),
                html.Td(r["reason"], className="text-muted", style={"fontSize": "0.85em"}),
                html.Td(_metric(r["norm_l2"])),
                html.Td(_metric(r["near_zero_pct"], ".1f")),
                html.Td(r["rank_util"]),
                html.Td(_metric(r["condition"], ".1e")),
                html.Td(_metric(r["kurtosis"], ".1f")),
            ], id={"type": "ns-row", "index": r["layer"]}))

        health_table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Layer"), html.Th("Kind"), html.Th("Shape"),
                html.Th("Health"), html.Th("Reason"),
                html.Th("L2 Norm"), html.Th("Zero%"),
                html.Th("Rank"), html.Th("Cond#"), html.Th("Kurt"),
            ])),
            html.Tbody(table_rows),
        ], bordered=True, hover=True, responsive=True, size="sm")

        children.append(dbc.Card(dbc.CardBody([
            html.H5(f"Parameter Health ({len(health_rows)} entries)", className="card-title"),
            html.Div(health_table, style={"maxHeight": "600px", "overflowY": "auto"}),
        ]), className="mb-3"))

    # ── 12. Computation Graph ────────────────────────────────────────
    fx = model_data.get("fx_graph")
    if fx and fx.get("nodes"):
        fx_elements = build_fx_elements(model_data)
        if fx_elements:
            children.append(dbc.Card(dbc.CardBody([
                html.H5(f"Computation Graph ({len(ops)} ops)", className="card-title"),
                html.P("ATen-level ops from torch.export. Click a node to inspect.",
                        className="text-muted"),
                cyto.Cytoscape(
                    id="gh-cyto",
                    elements=fx_elements,
                    layout={"name": "dagre", "rankDir": "TB", "spacingFactor": 1.2, "nodeSep": 30, "rankSep": 50},
                    stylesheet=CYTO_STYLE,
                    style={"width": "100%", "height": "520px", "borderRadius": "8px"},
                    boxSelectionEnabled=False, userZoomingEnabled=True, userPanningEnabled=True,
                ),
                html.Div(id="gh-node-detail", className="mt-3",
                          children=[html.P("Click a node to inspect it.", className="text-muted")]),
            ]), className="mb-3"))

    # ── 13. Module tree table ────────────────────────────────────────
    if leaf_modules:
        rows = []
        for m in leaf_modules:
            cat = module_category(m["type"])
            color = MODULE_COLORS.get(cat, MODULE_COLORS["default"])
            attrs = m.get("attributes", {})
            attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items()) if attrs else ""
            rows.append(html.Tr([
                html.Td(m["path"]),
                html.Td(dbc.Badge(m["type"], style={"backgroundColor": color, "color": "#222"})),
                html.Td(fmt_num(m.get("params", 0))),
                html.Td(html.Code(attr_str) if attr_str else ""),
            ]))
        children.append(dbc.Card(dbc.CardBody([
            html.H5(f"Module Tree ({len(leaf_modules)} layers)", className="card-title"),
            dbc.Table([
                html.Thead(html.Tr([html.Th("Path"), html.Th("Type"), html.Th("Params"), html.Th("Attributes")])),
                html.Tbody(rows),
            ], bordered=True, hover=True, responsive=True, size="sm", className="mb-0"),
        ]), className="mb-3"))

    # ── 14. Parameters table ─────────────────────────────────────────
    if params:
        p_rows = []
        for name, meta in params.items():
            if meta.get("is_buffer"):
                continue
            p_rows.append(html.Tr([
                html.Td(name),
                html.Td(html.Code("x".join(str(s) for s in meta.get("shape", [])))),
                html.Td(meta.get("dtype", "")),
                html.Td(fmt_num(meta.get("numel", 0))),
            ]))
        children.append(dbc.Card(dbc.CardBody([
            html.H5("Parameters", className="card-title"),
            dbc.Table([
                html.Thead(html.Tr([html.Th("Name"), html.Th("Shape"), html.Th("Dtype"), html.Th("Elements")])),
                html.Tbody(p_rows),
            ], bordered=True, hover=True, responsive=True, size="sm", className="mb-0"),
        ]), className="mb-3"))

    return dbc.Container(children, fluid=True)


def landing_page_empty():
    return dbc.Container([
        html.H2("GradientHound", className="mt-3 mb-1"),
        html.P("No model loaded. Use --model path/to/model.gh.json to load an export.",
                className="text-muted mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5(title, className="card-title"),
                html.P(desc, className="text-muted mb-0"),
            ])), md=4, className="mb-3")
            for path, (title, desc) in PAGES.items() if path != "/"
        ]),
    ], fluid=True)


# ── Checkpoints page ─────────────────────────────────────────────────

def checkpoints_page(ckpt_paths: list[str], snapshots: list[dict] | None):
    children = [
        html.H2("Checkpoints", className="mt-3 mb-1"),
        html.P(f"{len(ckpt_paths)} checkpoint files loaded", className="text-muted mb-4"),
    ]

    file_rows = []
    for i, path in enumerate(ckpt_paths):
        p = Path(path)
        size = ""
        if p.exists():
            sz = p.stat().st_size
            if sz >= 1_073_741_824:
                size = f"{sz / 1_073_741_824:.1f} GB"
            elif sz >= 1_048_576:
                size = f"{sz / 1_048_576:.1f} MB"
            elif sz >= 1_024:
                size = f"{sz / 1_024:.1f} KB"
            else:
                size = f"{sz} B"
        status = ""
        if snapshots:
            snap = next((s for s in snapshots if s["path"] == path), None)
            if snap:
                status = f"{len(snap['weight_stats'])} params"
        file_rows.append(html.Tr([
            html.Td(str(i + 1)),
            html.Td(p.name),
            html.Td(size),
            html.Td(dbc.Badge(status, color="success") if status else "\u2014"),
            html.Td(html.Code(str(path), style={"fontSize": "0.8em"})),
        ]))

    children.append(dbc.Card(dbc.CardBody([
        html.H5("Checkpoint Files", className="card-title"),
        dbc.Table([
            html.Thead(html.Tr([html.Th("#"), html.Th("File"), html.Th("Size"), html.Th("Status"), html.Th("Path")])),
            html.Tbody(file_rows),
        ], bordered=True, hover=True, responsive=True, size="sm"),
    ]), className="mb-3"))

    if snapshots:
        n_params = len({s["layer"] for snap in snapshots for s in snap["weight_stats"]})
        children.append(dbc.Alert([
            html.Strong(f"Processed \u2014 {len(snapshots)} checkpoints, {n_params} parameters. "),
            "Head back to the Dashboard to explore weight analysis, health charts, and optimizer metrics.",
        ], color="success", className="mb-3"))
    else:
        children.append(dbc.Card(dbc.CardBody([
            html.H5("Process", className="card-title"),
            html.P(
                "Load all checkpoints and compute per-parameter weight statistics "
                "(norms, distributions, SVD, kurtosis). This may take a moment for large models.",
                className="text-muted",
            ),
            dbc.Button("Process Checkpoints", id="ckpt-process-btn", color="primary", n_clicks=0),
            html.Div(id="ckpt-status", className="mt-2 text-muted"),
        ]), className="mb-3"))

    return dbc.Container(children, fluid=True)


def _fmt_bytes(n: int) -> str:
    if n >= 1_073_741_824:
        return f"{n / 1_073_741_824:.1f} GB"
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    if n >= 1_024:
        return f"{n / 1_024:.1f} KB"
    return f"{n} B"


def _optimizer_state_cards(snapshots: list[dict]) -> list:
    """Build optimizer analysis cards for the checkpoints page."""
    has_any = any(snap.get("optimizer_states") for snap in snapshots)
    if not has_any:
        return []

    cards: list = [
        html.H3("Optimizer States", className="mt-4 mb-2"),
        html.P(
            "Optimizer state dicts detected in checkpoint files. "
            "Statistics are computed from the saved first/second moment buffers.",
            className="text-muted mb-3",
        ),
    ]

    # ── Summary table ────────────────────────────────────────────────
    ckpt_names, opt_rows = compute_optimizer_summary_table(snapshots)

    if opt_rows:
        header = [html.Th("Optimizer"), html.Th("Type")] + [
            html.Th(name) for name in ckpt_names
        ]
        body_rows = []
        for orow in opt_rows:
            cells = [
                html.Td(html.Strong(orow["name"])),
                html.Td(dbc.Badge(orow["type"], color="info")),
            ]
            for cell in orow["cells"]:
                if cell is None:
                    cells.append(html.Td("\u2014", className="text-muted"))
                else:
                    n_groups = len(cell.get("groups", []))
                    mem = _fmt_bytes(cell.get("total_state_bytes", 0))
                    n_tensors = cell.get("n_state_tensors", 0)
                    cells.append(html.Td([
                        html.Div(f"{n_groups} group{'s' if n_groups != 1 else ''}, {n_tensors} tensors"),
                        html.Small(f"State memory: {mem}", className="text-muted"),
                    ]))
            body_rows.append(html.Tr(cells))

        cards.append(dbc.Card(dbc.CardBody([
            html.H5("Overview", className="card-title"),
            html.Div(
                dbc.Table([
                    html.Thead(html.Tr(header)),
                    html.Tbody(body_rows),
                ], bordered=True, hover=True, responsive=True, size="sm"),
                style={"overflowX": "auto"},
            ),
        ]), className="mb-3"))

    # ── Hyperparameter comparison table ──────────────────────────────
    ckpt_names, evo_rows = compute_optimizer_evolution_table(snapshots)

    if evo_rows:
        # Hyperparameters
        hp_header = [html.Th("Optimizer"), html.Th("Group")] + [
            html.Th(name) for name in ckpt_names
        ]
        hp_body: list = []
        for erow in evo_rows:
            cells = [
                html.Td(erow["optimizer"]),
                html.Td(str(erow["group_index"])),
            ]
            for cell in erow["cells"]:
                if cell is None:
                    cells.append(html.Td("\u2014", className="text-muted"))
                else:
                    parts = []
                    lr = cell.get("lr")
                    if lr is not None:
                        parts.append(f"lr={lr:.2e}")
                    betas = cell.get("betas")
                    if betas:
                        parts.append(f"betas={betas}")
                    wd = cell.get("weight_decay")
                    if wd is not None and wd > 0:
                        parts.append(f"wd={wd:.2e}")
                    eps = cell.get("eps")
                    if eps is not None:
                        parts.append(f"eps={eps:.0e}")
                    mom = cell.get("momentum")
                    if mom is not None and mom > 0:
                        parts.append(f"mom={mom}")
                    cells.append(html.Td(html.Code(", ".join(parts) if parts else "\u2014")))
            hp_body.append(html.Tr(cells))

        cards.append(dbc.Card(dbc.CardBody([
            html.H5("Hyperparameters Across Checkpoints", className="card-title"),
            html.Div(
                dbc.Table([
                    html.Thead(html.Tr(hp_header)),
                    html.Tbody(hp_body),
                ], bordered=True, hover=True, responsive=True, size="sm"),
                style={"overflowX": "auto"},
            ),
        ]), className="mb-3"))

        # ── State statistics table ───────────────────────────────────
        st_header = [html.Th("Optimizer"), html.Th("Group")] + [
            html.Th(name) for name in ckpt_names
        ]
        st_body: list = []
        for erow in evo_rows:
            cells = [
                html.Td(erow["optimizer"]),
                html.Td(str(erow["group_index"])),
            ]
            for cell in erow["cells"]:
                if cell is None:
                    cells.append(html.Td("\u2014", className="text-muted"))
                else:
                    lines: list = []
                    step = cell.get("step")
                    if step is not None and step > 0:
                        lines.append(html.Div(f"Step: {step:,}"))
                    ean = cell.get("exp_avg_norm_mean")
                    if ean is not None:
                        lines.append(html.Div(
                            f"1st moment norm: {ean:.4g} "
                            f"(max {cell.get('exp_avg_norm_max', 0):.4g})"
                        ))
                    esm = cell.get("exp_avg_sq_mean")
                    if esm is not None:
                        lines.append(html.Div(f"2nd moment mean: {esm:.4g}"))
                    elr = cell.get("effective_lr")
                    if elr is not None:
                        lines.append(html.Div([
                            "Effective LR: ",
                            html.Strong(f"{elr:.4g}"),
                        ]))
                    bc = cell.get("bias_correction2")
                    if bc is not None:
                        wp = cell.get("warmup_pct", 0)
                        lines.append(html.Div(
                            f"Bias corr: {bc:.6f} ({wp:.1f}% warmed up)"
                        ))
                    mnm = cell.get("momentum_norm_mean")
                    if mnm is not None:
                        lines.append(html.Div(
                            f"Momentum norm: {mnm:.4g} "
                            f"(max {cell.get('momentum_norm_max', 0):.4g})"
                        ))
                    if not lines:
                        lines.append(html.Div("No state yet", className="text-muted"))
                    cells.append(html.Td(lines))
            st_body.append(html.Tr(cells))

        cards.append(dbc.Card(dbc.CardBody([
            html.H5("Optimizer State Statistics", className="card-title"),
            html.P(
                "Per-group statistics from the optimizer's internal buffers. "
                "Effective LR is estimated as lr / (sqrt(mean_v) + eps).",
                className="text-muted mb-2",
            ),
            html.Div(
                dbc.Table([
                    html.Thead(html.Tr(st_header)),
                    html.Tbody(st_body),
                ], bordered=True, hover=True, responsive=True, size="sm"),
                style={"maxHeight": "600px", "overflowY": "auto", "overflowX": "auto"},
            ),
        ]), className="mb-3"))

    return cards


def checkpoints_page_empty():
    return dbc.Container([
        html.H2("Checkpoints", className="mt-3 mb-1"),
        html.P("No checkpoints loaded.", className="text-muted mb-4"),
        dbc.Card(dbc.CardBody([
            html.H5("How to use", className="card-title"),
            html.P("Pass checkpoint files via the CLI to compare them:", className="text-muted"),
            html.Code("python -m gradienthound --checkpoints epoch1.pt epoch5.pt epoch10.pt",
                       className="d-block p-2 bg-dark rounded"),
        ])),
    ], fluid=True)
