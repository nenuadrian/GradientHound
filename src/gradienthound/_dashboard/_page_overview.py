"""Overview page: model summary, health map, and stat cards."""
from __future__ import annotations

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import html, dcc

from ._constants import (
    MODULE_COLORS, CYTO_STYLE, HEALTH_COLORS, PAGES,
)
from ._helpers import (
    fmt_num, fmt_bytes, module_category, short_layer,
    split_model_data_for_submodel,
)
from ._graph import build_module_tree_elements
from ._health import weight_health, build_health_elements


def overview_page(model_data: dict, snapshots: list | None = None):
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

    # ── Stat cards ───────────────────────────────────────────────────
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
        stat_items.append(("Act. Memory", fmt_bytes(total_act_bytes)))
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

        # ── I/O ──────────────────────────────────────────────────────
        dbc.Card(dbc.CardBody([
            html.H5("I/O", className="card-title"),
            html.Code(f"Input:  {input_desc}", className="d-block mb-1"),
            html.Code(f"Output: {output_desc}"),
        ]), className="mb-3"),
    ]

    # ── Architecture graph (health-colored when snapshots, plain otherwise)
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
            health_graphs: list = []
            for sm_idx, sm_name in enumerate(sub_models):
                sm_model_data, sm_stats = split_model_data_for_submodel(
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
