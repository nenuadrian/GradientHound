"""Architecture page: FX graph, module tree, parameters, and live analysis."""
from __future__ import annotations

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import html, dcc

from ._constants import MODULE_COLORS, CYTO_STYLE
from ._helpers import plotly_layout, fmt_num, fmt_bytes, module_category, short_layer
from ._graph import build_fx_elements


def _build_live_analysis_sections(live: dict, model_data: dict) -> list:
    """Build sections for live-model analyses (FLOPs, activations, pruning)."""
    import plotly.graph_objects as go

    sections: list = []
    if not live:
        return sections

    sub_models = model_data.get("sub_models", [])
    _SM_COLORS = ["#375a7f", "#e67e22", "#00bc8c", "#e74c3c", "#9b59b6",
                  "#3498db", "#1abc9c", "#f39c12", "#c0392b", "#8e44ad"]

    def _sub_model_of(module_name: str) -> str | None:
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
                    html.Td(f"{count}\u00d7"),
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

        act_rows = []
        for name, info in sorted_acts:
            shape = info.get("shape", [])
            shape_str = "\u00d7".join(str(s) for s in shape) if shape else ""
            pct = info.get("bytes", 0) / max(total_bytes, 1) * 100
            act_rows.append(html.Tr([
                html.Td(html.Code(short_layer(name))),
                html.Td(html.Code(shape_str)),
                html.Td(info.get("dtype", "")),
                html.Td(f"{info.get('numel', 0):,}"),
                html.Td(fmt_bytes(info.get("bytes", 0))),
                html.Td(f"{pct:.1f}%"),
            ]))

        sections.append(dbc.Card(dbc.CardBody([
            html.H5([
                "Activation Memory",
                dbc.Badge(fmt_bytes(total_bytes), color="success", className="ms-2"),
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
        sorted_groups = sorted(
            enumerate(pruning_groups),
            key=lambda ig: ig[1].get("n_prunable", 0),
            reverse=True,
        )
        for idx, group in sorted_groups:
            coupled = group.get("coupled_layers", [])
            n_prunable = group.get("n_prunable", 0)
            sub_model = group.get("sub_model")
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


def architecture_page(model_data: dict, snapshots: list | None = None):
    mt = model_data.get("module_tree", {})
    modules = mt.get("modules", [])
    leaf_modules = [m for m in modules if m.get("is_leaf")]
    fx = model_data.get("fx_graph", {})
    fx_nodes = fx.get("nodes", [])
    ops = [n for n in fx_nodes if n["op"] == "call_function"]
    params = model_data.get("parameters", {})
    live = model_data.get("live_analysis", {})

    children = [
        html.H2("Architecture", className="mt-3 mb-1"),
        html.P("Model structure, computation graph, and resource analysis.", className="text-muted mb-4"),
    ]

    # ── PyTorch model repr ──────────────────────────────────────────
    pytorch_repr = mt.get("pytorch_repr", "")
    if pytorch_repr:
        children.append(dbc.Card(dbc.CardBody([
            html.H5("Model Summary", className="card-title"),
            html.P("Textual representation of the model (equivalent to print(model) in PyTorch).",
                    className="text-muted"),
            html.Pre(
                pytorch_repr,
                style={
                    "backgroundColor": "#1a1a2e",
                    "color": "#e0e0e0",
                    "padding": "16px",
                    "borderRadius": "6px",
                    "fontSize": "0.82em",
                    "maxHeight": "500px",
                    "overflowY": "auto",
                    "whiteSpace": "pre",
                    "fontFamily": "SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                },
            ),
        ]), className="mb-3"))

    # ── Live analysis sections (FLOPs, activations, pruning) ─────────
    children.extend(_build_live_analysis_sections(live, model_data))

    # ── Computation Graph ────────────────────────────────────────────
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

    # ── Module tree table ────────────────────────────────────────────
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

    # ── Parameters table ─────────────────────────────────────────────
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
