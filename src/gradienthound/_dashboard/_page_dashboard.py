"""Dashboard and landing page layouts."""
from __future__ import annotations

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import html, dcc

from ._constants import (
    MODULE_COLORS, CYTO_STYLE, HEALTH_COLORS, HEALTH_SORT, PAGES,
)
from ._helpers import (
    plotly_layout, fmt_num, module_category, short_layer,
    compute_checkpoint_change_tables,
    render_checkpoint_change_table, compute_effective_rank_table,
    compute_distribution_stats_table, compute_scalar_metric_tables,
    compute_spectral_gap_table, compute_spectral_gap_ratios,
    compute_norm_velocity_table, compute_convergence_scores,
    detect_training_phases,
)
from ._graph import build_fx_elements, build_module_tree_elements
from ._health import weight_health, build_health_elements
from ._page_checkpoints import _optimizer_state_cards


def _fmt_bytes(n: int | float) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if n != int(n) else f"{int(n)} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _split_model_data_for_submodel(
    model_data: dict, stats: list[dict], sub_model: str,
) -> tuple[dict, list[dict]]:
    """Extract model_data and weight stats for a single sub-model.

    Returns a model_data dict with only modules/params under the sub-model
    prefix, and the corresponding weight stat entries.
    """
    mt = model_data.get("module_tree", {})
    # Filter modules belonging to this sub-model
    sm_modules = [
        m for m in mt.get("modules", [])
        if m["path"].startswith(sub_model + ".")
    ]
    # Find the root of this sub-model's subtree
    sm_root_path = f"{sub_model}.{sub_model}"
    sm_root = next((m for m in sm_modules if m["path"] == sm_root_path), None)
    sm_model_data = dict(model_data)
    sm_model_data["module_tree"] = {
        "name": sm_root_path if sm_root else sub_model,
        "modules": sm_modules,
    }
    # Filter stats to this sub-model
    sm_stats = [s for s in stats if s["layer"].startswith(sub_model + ".")]
    return sm_model_data, sm_stats


def _build_live_analysis_sections(live: dict, model_data: dict) -> list:
    """Build dashboard sections for live-model analyses (FLOPs, activations, pruning)."""
    import plotly.graph_objects as go

    sections: list = []
    if not live:
        return sections

    sub_models = model_data.get("sub_models", [])
    # Color palette for sub-models
    _SM_COLORS = ["#375a7f", "#e67e22", "#00bc8c", "#e74c3c", "#9b59b6",
                  "#3498db", "#1abc9c", "#f39c12", "#c0392b", "#8e44ad"]

    def _sub_model_of(module_name: str) -> str | None:
        """Return the sub-model prefix a module belongs to, or None."""
        for sm in sub_models:
            if module_name.startswith(sm + ".") or module_name == sm:
                return sm
        return None

    def _sm_color(sm: str | None) -> str:
        if sm is None or not sub_models:
            return _SM_COLORS[0]
        idx = sub_models.index(sm) if sm in sub_models else 0
        return _SM_COLORS[idx % len(_SM_COLORS)]

    # ── FLOPs breakdown ──────────────────────────────────────────────
    flops_data = live.get("flops")
    if flops_data:
        by_module = flops_data.get("by_module", {})
        by_operator = flops_data.get("by_operator", {})
        unsupported = flops_data.get("unsupported_ops", {})

        flops_children: list = []

        # Per-module bar chart (colored by sub-model when multiple)
        if by_module:
            sorted_mods = sorted(by_module.items(), key=lambda kv: kv[1], reverse=True)
            mod_names = [short_layer(m) for m, _ in sorted_mods]
            mod_flops = [f for _, f in sorted_mods]
            bar_colors = [_sm_color(_sub_model_of(m)) for m, _ in sorted_mods]

            fig = go.Figure(go.Bar(
                x=mod_names, y=mod_flops,
                marker_color=bar_colors,
                hovertemplate="%{x}<br>%{y:,.0f} FLOPs<extra></extra>",
            ))
            fig.update_layout(
                **plotly_layout(title="FLOPs per Module"),
                height=340,
                xaxis_title="Module",
                yaxis_title="FLOPs",
                xaxis_tickangle=-45,
            )
            flops_children.append(dcc.Graph(figure=fig, config={"displayModeBar": False}))

        # Per-sub-model FLOPs summary (only when multiple models)
        if sub_models and by_module:
            sm_flops: dict[str, int] = {sm: 0 for sm in sub_models}
            for mod, count in by_module.items():
                sm = _sub_model_of(mod)
                if sm:
                    sm_flops[sm] += count
            sm_rows = []
            total = flops_data.get("total_flops", 1)
            for sm in sub_models:
                pct = sm_flops[sm] / max(total, 1) * 100
                sm_rows.append(html.Tr([
                    html.Td([
                        html.Span("", style={
                            "display": "inline-block", "width": "12px", "height": "12px",
                            "borderRadius": "2px", "backgroundColor": _sm_color(sm),
                            "marginRight": "8px", "verticalAlign": "middle",
                        }),
                        html.Strong(sm),
                    ]),
                    html.Td(fmt_num(sm_flops[sm])),
                    html.Td(f"{pct:.1f}%"),
                ]))
            flops_children.append(
                html.H6("Per Network", className="mt-3 mb-2"),
            )
            flops_children.append(
                dbc.Table([
                    html.Thead(html.Tr([html.Th("Network"), html.Th("FLOPs"), html.Th("% Total")])),
                    html.Tbody(sm_rows),
                ], bordered=True, hover=True, size="sm"),
            )

        # Operator + unsupported summary
        op_items: list = []
        if by_operator:
            for op, count in sorted(by_operator.items(), key=lambda kv: kv[1], reverse=True):
                op_items.append(html.Tr([
                    html.Td(html.Code(op)),
                    html.Td(f"{count:,}"),
                ]))
        if unsupported:
            for op, count in unsupported.items():
                op_items.append(html.Tr([
                    html.Td([html.Code(op), " ", dbc.Badge("unsupported", color="warning", className="ms-1")]),
                    html.Td(f"{count}×"),
                ]))
        if op_items:
            flops_children.append(
                dbc.Row([
                    dbc.Col([
                        html.H6("By Operator", className="mt-3 mb-2"),
                        dbc.Table([
                            html.Thead(html.Tr([html.Th("Operator"), html.Th("FLOPs / Count")])),
                            html.Tbody(op_items),
                        ], bordered=True, hover=True, size="sm"),
                    ], md=6),
                ]),
            )

        sections.append(dbc.Card(dbc.CardBody([
            html.H5([
                "FLOPs Analysis",
                dbc.Badge(fmt_num(flops_data.get("total_flops", 0)), color="primary", className="ms-2"),
            ], className="card-title"),
            html.P(
                "Forward-pass floating-point operations per module (fvcore).",
                className="text-muted",
            ),
            *flops_children,
        ]), className="mb-3"))

    # ── Activation memory ────────────────────────────────────────────
    activations_data = live.get("activations")
    if activations_data:
        sorted_acts = sorted(
            activations_data.items(),
            key=lambda kv: kv[1].get("bytes", 0),
            reverse=True,
        )

        # Bar chart of activation memory per module
        act_names = [short_layer(name) for name, _ in sorted_acts]
        act_bytes = [info.get("bytes", 0) for _, info in sorted_acts]
        act_colors = [_sm_color(_sub_model_of(name)) for name, _ in sorted_acts]

        fig = go.Figure(go.Bar(
            x=act_names, y=act_bytes,
            marker_color=act_colors,
            hovertemplate="%{x}<br>%{y:,.0f} bytes<extra></extra>",
        ))
        fig.update_layout(
            **plotly_layout(title="Activation Memory per Module"),
            height=340,
            xaxis_title="Module",
            yaxis_title="Bytes",
            xaxis_tickangle=-45,
        )

        total_bytes = sum(act_bytes)

        # Detail table
        act_rows = []
        for name, info in sorted_acts:
            shape = info.get("shape", [])
            shape_str = "×".join(str(s) for s in shape) if shape else ""
            pct = info.get("bytes", 0) / max(total_bytes, 1) * 100
            act_rows.append(html.Tr([
                html.Td(html.Code(short_layer(name))),
                html.Td(html.Code(shape_str)),
                html.Td(info.get("dtype", "")),
                html.Td(f"{info.get('numel', 0):,}"),
                html.Td(_fmt_bytes(info.get("bytes", 0))),
                html.Td(f"{pct:.1f}%"),
            ]))

        sections.append(dbc.Card(dbc.CardBody([
            html.H5([
                "Activation Memory",
                dbc.Badge(_fmt_bytes(total_bytes), color="success", className="ms-2"),
            ], className="card-title"),
            html.P(
                "Per-module output tensor sizes from a single forward pass. "
                "Peak training memory includes activations for all layers simultaneously.",
                className="text-muted",
            ),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.Div(
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Module"), html.Th("Shape"), html.Th("Dtype"),
                        html.Th("Elements"), html.Th("Size"), html.Th("% Total"),
                    ])),
                    html.Tbody(act_rows),
                ], bordered=True, hover=True, responsive=True, size="sm"),
                style={"maxHeight": "400px", "overflowY": "auto"},
            ),
        ]), className="mb-3"))

    # ── Pruning dependency groups ────────────────────────────────────
    pruning_groups = live.get("pruning_groups")
    if pruning_groups:
        group_cards: list = []
        # Sort by n_prunable descending for visual priority
        sorted_groups = sorted(
            enumerate(pruning_groups),
            key=lambda ig: ig[1].get("n_prunable", 0),
            reverse=True,
        )
        for idx, group in sorted_groups:
            coupled = group.get("coupled_layers", [])
            n_prunable = group.get("n_prunable", 0)
            sub_model = group.get("sub_model")
            # Filter to named modules (skip internal _ElementWiseOp etc.)
            named_layers = [l for l in coupled if not l.startswith("_")]

            items = [
                html.Code(l, className="d-block", style={"fontSize": "0.85em"})
                for l in (named_layers or coupled)
            ]
            title_parts = [f"Group {idx}"]
            if sub_model:
                title_parts.append(dbc.Badge(
                    sub_model, className="ms-2",
                    style={"backgroundColor": _sm_color(sub_model)},
                ))
            title_parts.append(dbc.Badge(f"{n_prunable} ch", color="info", className="ms-1"))
            group_cards.append(dbc.Col(dbc.Card(dbc.CardBody([
                html.H6(title_parts, className="card-title mb-2"),
                *items,
                html.Small(
                    f"{len(coupled)} coupled ops total",
                    className="text-muted d-block mt-2",
                ) if len(coupled) > len(named_layers) else None,
            ]), color="light"), md=4, className="mb-2"))

        sections.append(dbc.Card(dbc.CardBody([
            html.H5([
                "Pruning Groups",
                dbc.Badge(f"{len(pruning_groups)} groups", color="info", className="ms-2"),
            ], className="card-title"),
            html.P(
                "Coupled layer groups from Torch-Pruning dependency analysis. "
                "Layers within a group must be pruned together to maintain consistency.",
                className="text-muted",
            ),
            dbc.Row(group_cards),
        ]), className="mb-3"))

    return sections


def _column_summary(table: list[list[float | None]], n_cols: int):
    """Return (means, mins, maxs) lists for each column of a metric table."""
    means: list[float | None] = []
    mins: list[float | None] = []
    maxs: list[float | None] = []
    for ci in range(n_cols):
        col = [row[ci] for row in table if ci < len(row) and row[ci] is not None]
        if col:
            means.append(sum(col) / len(col))
            mins.append(min(col))
            maxs.append(max(col))
        else:
            means.append(None)
            mins.append(None)
            maxs.append(None)
    return means, mins, maxs


def _summary_chart(checkpoint_names, means, mins, maxs, title, y_label):
    """Build a mean/min/max summary line chart."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=checkpoint_names, y=means, mode="lines+markers",
        name="mean", line={"color": "#375a7f", "width": 2.5},
    ))
    if any(v is not None for v in maxs):
        fig.add_trace(go.Scatter(
            x=checkpoint_names, y=maxs, mode="lines",
            name="max", line={"color": "#e67e22", "dash": "dot"},
        ))
    if any(v is not None for v in mins):
        fig.add_trace(go.Scatter(
            x=checkpoint_names, y=mins, mode="lines",
            name="min", line={"color": "#00bc8c", "dash": "dot"},
        ))
    fig.update_layout(
        **plotly_layout(title=title), height=320,
        xaxis_title="Checkpoint", yaxis_title=y_label,
    )
    return fig


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
    live = model_data.get("live_analysis", {})

    # ── 1. Stat cards ────────────────────────────────────────────────
    stat_items = [
        ("Parameters", fmt_num(model_data.get("total_params", 0))),
        ("Trainable", fmt_num(model_data.get("trainable_params", 0))),
        ("Layers", str(len(leaf_modules))),
        ("FX Ops", str(len(ops))),
        ("Buffers", str(buffer_count)),
    ]
    flops_data = live.get("flops")
    if flops_data:
        stat_items.append(("FLOPs", fmt_num(flops_data.get("total_flops", 0))))
    activations_data = live.get("activations")
    if activations_data:
        total_act_bytes = sum(a.get("bytes", 0) for a in activations_data.values())
        stat_items.append(("Act. Memory", _fmt_bytes(total_act_bytes)))
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

    # ── 2b. Live analysis sections (FLOPs, activations, pruning) ────
    children.extend(_build_live_analysis_sections(live, model_data))

    # ── 3. Architecture graph (health-colored when snapshots, plain otherwise)
    if snapshots:
        latest_stats = snapshots[-1]["weight_stats"]
        health_style = CYTO_STYLE + [
            {"selector": f".health-{state}", "style": {
                "background-color": color,
                "border-color": color,
                "border-width": "2.5px" if state in ("critical", "warning") else "1.5px",
            }}
            for state, color in HEALTH_COLORS.items()
        ]

        sub_models = model_data.get("sub_models")
        if sub_models:
            # Multi-model: render each sub-model as a separate graph
            health_graphs: list = []
            for sm_idx, sm_name in enumerate(sub_models):
                sm_model_data, sm_stats = _split_model_data_for_submodel(
                    model_data, latest_stats, sm_name,
                )
                sm_elements = build_health_elements(sm_model_data, sm_stats)
                if sm_elements:
                    health_graphs.append(html.Div([
                        html.H6(sm_name, className="mt-2 mb-1"),
                        cyto.Cytoscape(
                            id=f"gh-cyto-health-{sm_idx}",
                            elements=sm_elements,
                            layout={
                                "name": "dagre", "rankDir": "LR",
                                "spacingFactor": 1.3, "nodeSep": 30, "rankSep": 50,
                            },
                            stylesheet=health_style,
                            style={"width": "100%", "height": "200px", "borderRadius": "8px"},
                            boxSelectionEnabled=False,
                            userZoomingEnabled=True,
                            userPanningEnabled=True,
                        ),
                    ]))
            if health_graphs:
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
                    *health_graphs,
                    html.Div(id="ns-layer-detail", className="mt-3"),
                ]), className="mb-3"))
        else:
            # Single model
            health_elements = build_health_elements(model_data, latest_stats)
            if health_elements:
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

        # ── 4b. Norm Evolution Across Checkpoints ────────────────────
        if len(snapshots) >= 2:
            ckpt_names_norm = [s["name"] for s in snapshots]
            # Build norm table: layers x checkpoints
            weight_layer_names = [s["layer"] for s in weight_layers]
            norm_table: list[list[float | None]] = []
            for layer in weight_layer_names:
                row: list[float | None] = []
                for sn in snapshots:
                    stat = next((s for s in sn["weight_stats"] if s["layer"] == layer), None)
                    row.append(stat.get("norm_l2") if stat else None)
                norm_table.append(row)

            n_means, n_mins, n_maxs = _column_summary(norm_table, len(ckpt_names_norm))
            norm_evo_fig = _summary_chart(
                ckpt_names_norm, n_means, n_mins, n_maxs,
                "L2 Norm Evolution", "L2 Norm",
            )
            children.append(dbc.Card(dbc.CardBody([
                html.H5("Norm Evolution Across Checkpoints", className="card-title"),
                html.P("Mean, min, and max L2 norm across all weight layers per checkpoint.",
                        className="text-muted"),
                dcc.Graph(figure=norm_evo_fig),
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

            # ── 6b. Effective Rank Chart ─────────────────────────────
            if len(checkpoint_names) >= 2:
                r_means, r_mins, r_maxs = _column_summary(svd_rank_table, len(checkpoint_names))
                rank_chart = _summary_chart(
                    checkpoint_names, r_means, r_mins, r_maxs,
                    "Effective Rank Summary", "Effective Rank",
                )
                children.append(dbc.Card(dbc.CardBody([
                    html.H5("Effective Rank Trend", className="card-title"),
                    html.P("Aggregate effective rank across all 2D weight layers per checkpoint.",
                            className="text-muted"),
                    dcc.Graph(figure=rank_chart),
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

        # ── 10.1 Kurtosis & Sparsity Evolution Charts ──────────────
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
                m_means, _, _ = _column_summary(evo_tables[mkey], len(evo_ckpt_names))
                if mkey == "condition_number":
                    # Log scale for condition numbers
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

        # ── 10.2 Weight Entropy Evolution ──────────────────────────
        entropy_ckpt_names, entropy_layers, entropy_tables = compute_scalar_metric_tables(
            snapshots, ["weight_entropy"], max_layers=100,
        )
        if entropy_layers and entropy_tables and "weight_entropy" in entropy_tables and len(entropy_ckpt_names) >= 2:
            e_means, e_mins, e_maxs = _column_summary(entropy_tables["weight_entropy"], len(entropy_ckpt_names))
            entropy_chart = _summary_chart(
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

        # ── 10a. WeightWatcher Metrics (curated) ───────────────────
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

        # ── 10c. Directional Drift Metrics ──────────────────────────
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

            # ── 10d. Drift Summary Chart ────────────────────────────
            if len(drift_checkpoint_names) >= 2:
                drift_chart = go.Figure()
                drift_colors = {"drift_cosine_prev": "#375a7f", "drift_subspace_overlap_prev": "#e67e22", "drift_cka_prev": "#00bc8c"}
                for dkey, dtable in drift_tables.items():
                    d_means, _, _ = _column_summary(dtable, len(drift_checkpoint_names))
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

        # ── 10e. Spectral Gap Ratios ──────────────────────────────
        gap_ckpt_names, gap_layers, gap_table = compute_spectral_gap_table(snapshots, max_layers=50)
        if gap_layers and len(gap_ckpt_names) >= 2:
            gap_means, gap_mins, gap_maxs = _column_summary(gap_table, len(gap_ckpt_names))
            gap_chart = _summary_chart(
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

        # ── 10f. Norm Velocity & Acceleration ──────────────────────
        if len(snapshots) >= 2:
            vel_ckpt_names, vel_layers, vel_table, acc_table = compute_norm_velocity_table(snapshots, max_layers=100)
            if vel_layers:
                vel_means, vel_mins, vel_maxs = _column_summary(vel_table, len(vel_ckpt_names))
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
                    acc_means, _, _ = _column_summary(acc_table, len(vel_ckpt_names))
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
                # Add zero line
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
                    # Build sign heatmap: +1 = diverging, -1 = converging, 0 = stable
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
                            [0, "#00bc8c"],      # green = converging (negative acc)
                            [0.5, "#f8f9fa"],     # neutral
                            [1, "#e74c3c"],       # red = diverging (positive acc)
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

        # ── 10g. Layer-wise Convergence Score ──────────────────────
        if len(snapshots) >= 2:
            conv_ckpt_names, conv_layers, conv_table = compute_convergence_scores(snapshots, max_layers=100)
            if conv_layers and any(v is not None for row in conv_table for v in row):
                conv_means, conv_mins, conv_maxs = _column_summary(conv_table, len(conv_ckpt_names))
                conv_summary = _summary_chart(
                    conv_ckpt_names, conv_means, conv_mins, conv_maxs,
                    "Convergence Score Evolution", "Score (0-100)",
                )

                conv_heatmap = go.Figure(go.Heatmap(
                    z=conv_table,
                    x=conv_ckpt_names,
                    y=[short_layer(l) for l in conv_layers],
                    colorscale=[
                        [0, "#e74c3c"],       # red = unstable
                        [0.5, "#f39c12"],      # yellow = transitioning
                        [1, "#00bc8c"],        # green = converged
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

        # ── 10h. Training Phase Detection ──────────────────────────
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

        # ── 10i. Update Ratio & Delta Norm ────────────────────────
        if len(snapshots) >= 2:
            ur_metric_meta = {
                "update_ratio_prev": {"label": "Update Ratio (||delta||/||W||)", "fmt": lambda v: f"{v:.4g}"},
                "delta_norm": {"label": "Delta Norm (||W_t - W_{t-1}||)", "fmt": lambda v: f"{v:.4g}"},
            }
            ur_ckpt_names, ur_layers, ur_tables = compute_scalar_metric_tables(
                snapshots, list(ur_metric_meta.keys()), max_layers=50,
            )
            if ur_layers and ur_tables:
                # Summary chart of update ratio
                if "update_ratio_prev" in ur_tables and len(ur_ckpt_names) >= 2:
                    ur_means, ur_mins, ur_maxs = _column_summary(ur_tables["update_ratio_prev"], len(ur_ckpt_names))
                    ur_chart = _summary_chart(
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

        # ── 10j. Delta Direction Consistency ───────────────────────
        if len(snapshots) >= 3:
            dd_ckpt_names, dd_layers, dd_tables = compute_scalar_metric_tables(
                snapshots, ["delta_direction_consistency"], max_layers=100,
            )
            if dd_layers and dd_tables and "delta_direction_consistency" in dd_tables:
                dd_table = dd_tables["delta_direction_consistency"]
                dd_means, _, _ = _column_summary(dd_table, len(dd_ckpt_names))
                dd_chart = _summary_chart(
                    dd_ckpt_names, dd_means, [None] * len(dd_ckpt_names), [None] * len(dd_ckpt_names),
                    "Delta Direction Consistency", "Cosine Similarity",
                )
                dd_chart.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

                dd_heatmap = go.Figure(go.Heatmap(
                    z=dd_table,
                    x=dd_ckpt_names,
                    y=[short_layer(l) for l in dd_layers],
                    colorscale=[
                        [0, "#e74c3c"],       # red = oscillating
                        [0.5, "#f8f9fa"],      # neutral
                        [1, "#375a7f"],        # blue = consistent
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

        # ── 10k. Initialization Distance ───────────────────────────
        if len(snapshots) >= 2:
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
                    m_means, _, _ = _column_summary(init_tables[mkey], len(init_ckpt_names))
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

        # ── 10l. Cross-Layer Update Correlation ────────────────────
        if len(snapshots) >= 2:
            # Use the latest transition's correlation matrix
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

        # ── 10m. Singular Value Turnover ───────────────────────────
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
                    m_means, _, _ = _column_summary(svt_tables[mkey], len(svt_ckpt_names))
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

        # ── 10b. Spectral Analysis (ESD) ────────────────────────────
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
                "alpha": stat.get("alpha"),
                "mp_softrank": stat.get("mp_softrank"),
                "num_spikes": stat.get("num_spikes"),
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

        # Check if any layer has spectral metrics
        has_spectral = any(r["alpha"] is not None for r in health_rows)

        table_rows = []
        for r in health_rows:
            cells = [
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
            ]
            if has_spectral:
                cells.append(html.Td(_metric(r["alpha"], ".2f")))
                cells.append(html.Td(_metric(r["mp_softrank"], ".2f")))
                cells.append(html.Td(str(r["num_spikes"]) if r["num_spikes"] is not None else "-"))
            table_rows.append(html.Tr(cells, id={"type": "ns-row", "index": r["layer"]}))

        header_cells = [
            html.Th("Layer"), html.Th("Kind"), html.Th("Shape"),
            html.Th("Health"), html.Th("Reason"),
            html.Th("L2 Norm"), html.Th("Zero%"),
            html.Th("Rank"), html.Th("Cond#"), html.Th("Kurt"),
        ]
        if has_spectral:
            header_cells.extend([
                html.Th("Alpha", title="Power-law exponent of eigenvalue spectral density"),
                html.Th("MP Soft", title="Marchenko-Pastur softrank (1.0 = random)"),
                html.Th("Spikes", title="Eigenvalues above MP bulk edge"),
            ])

        health_table = dbc.Table([
            html.Thead(html.Tr(header_cells)),
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
