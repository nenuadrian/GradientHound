"""GradientHound standalone Dash dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import Dash, html, dcc, callback, Output, Input, State, no_update

cyto.load_extra_layouts()

# ── Module type → node color (for Cytoscape graph) ───────────────────

_MODULE_COLORS = {
    "conv": "#f0d4d8",
    "linear": "#d8ecde",
    "norm": "#faf0e0",
    "activation": "#fae0d0",
    "pool": "#e4d6ec",
    "dropout": "#f0eaea",
    "embedding": "#e0d4e8",
    "default": "#faf6f6",
}

# Colors for checkpoint series in comparison charts
_SERIES_COLORS = [
    "#375a7f", "#00bc8c", "#3498db", "#f39c12", "#e74c3c",
    "#9b59b6", "#1abc9c", "#e67e22", "#2ecc71", "#fd7e14",
]

# ── Page definitions ───────────────────────────────────────────────────

_PAGES = {
    "/":              ("Dashboard",     "Model overview and architecture"),
    "/checkpoints":   ("Checkpoints",   "Compare model checkpoints"),
    "/metrics":       ("Metrics",       "Live metric charts from training"),
    "/weights":       ("Weights",       "Weight distributions, SVD, heatmaps"),
    "/gradients":     ("Gradients",     "Gradient flow, cosine similarity, dead neurons"),
    "/training":      ("Training",      "Predictions, CKA, attention patterns"),
    "/optimizers":    ("Optimizers",    "Optimizer config, param groups, state buffers"),
    "/network-state": ("Network State", "Layer-by-layer parameter values"),
}

# ── Plotly dark template ─────────────────────────────────────────────

_PLOTLY_TEMPLATE = "plotly_dark"


def _plotly_layout(**overrides) -> dict:
    base = {"template": _PLOTLY_TEMPLATE, "margin": {"l": 60, "r": 20, "t": 40, "b": 60}}
    base.update(overrides)
    return base


# ── Cytoscape stylesheet ────────────────────────────────────────────

_CYTO_STYLE = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "background-color": "#faf6f6",
            "color": "#1a1012",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "11px",
            "font-weight": "600",
            "font-family": "-apple-system, sans-serif",
            "width": 140,
            "height": "32px",
            "padding": "12px",
            "shape": "roundrectangle",
            "border-width": "1.5px",
            "border-color": "#555",
            "text-wrap": "wrap",
            "text-max-width": "140px",
        },
    },
    {
        "selector": "node:selected",
        "style": {
            "border-width": "3px",
            "border-color": "#375a7f",
        },
    },
    {
        "selector": ".placeholder",
        "style": {
            "shape": "diamond",
            "background-color": "#444",
            "border-color": "#375a7f",
            "color": "#fff",
            "border-style": "dashed",
        },
    },
    {
        "selector": "edge",
        "style": {
            "width": 1.5,
            "line-color": "#555",
            "target-arrow-color": "#777",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.8,
            "curve-style": "bezier",
        },
    },
    *[
        {"selector": f".mod-{cat}", "style": {"background-color": color}}
        for cat, color in _MODULE_COLORS.items()
    ],
]


# ── Helpers ──────────────────────────────────────────────────────────

def _fmt_num(n: int | float) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _module_category(type_name: str) -> str:
    lower = type_name.lower()
    for key in ("conv", "linear", "norm", "pool", "dropout", "embed"):
        if key in lower:
            return key
    if any(k in lower for k in ("relu", "gelu", "silu", "sigmoid", "tanh", "softmax", "activation")):
        return "activation"
    return "default"


def _short_target(target: str) -> str:
    return target.replace("torch._ops.", "").replace(".default", "").replace(".Tensor", "")


def _short_layer(name: str, max_parts: int = 3) -> str:
    parts = name.split(".")
    if len(parts) > max_parts:
        parts = parts[-max_parts:]
    return ".".join(parts)


def _placeholder_page(title: str, description: str):
    return dbc.Container([
        html.H2(title, className="mt-3 mb-1"),
        html.P(description, className="text-muted mb-4"),
        dbc.Card(dbc.CardBody([
            html.H5("Coming soon", className="card-title"),
            html.P("This page will be available once data is loaded.", className="text-muted"),
        ])),
    ], fluid=True)


# ── Build Cytoscape elements from model data ─────────────────────────

def _build_fx_elements(model_data: dict) -> list[dict]:
    fx = model_data["fx_graph"]
    fx_nodes = fx.get("nodes", [])
    edges = fx.get("edges", [])
    sig = model_data.get("graph_signature", {})

    elements: list[dict] = []
    node_ids: set[str] = set()

    skip_nodes = set()
    for n in fx_nodes:
        if n["op"] == "placeholder" and n["target"] not in sig.get("user_inputs", []):
            skip_nodes.add(n["name"])
    for n in fx_nodes:
        if n["op"] == "call_function":
            visible_args = [a for a in n.get("args", []) if a not in skip_nodes]
            if not visible_args and n.get("args"):
                skip_nodes.add(n["name"])

    for n in fx_nodes:
        if n["name"] in skip_nodes or n["op"] == "output":
            continue
        node_id = n["name"]
        node_ids.add(node_id)
        mod_path = n.get("nn_module") or ""
        mod_type = n.get("nn_module_type") or ""
        shape = n.get("output_shape")
        shape_str = "x".join(str(s) for s in shape) if shape else ""

        if n["op"] == "placeholder":
            label = n["name"]
            if shape_str:
                label += f"\n{shape_str}"
            elements.append({"data": {"id": node_id, "label": label, **n}, "classes": "placeholder"})
        elif n["op"] == "call_function":
            short = _short_target(n.get("target", ""))
            label = short
            if mod_path:
                label += f"\n[{mod_path}]"
            elif shape_str:
                label += f"\n{shape_str}"
            cat = _module_category(mod_type) if mod_type else _module_category(short)
            elements.append({"data": {"id": node_id, "label": label, **n}, "classes": f"mod-{cat}"})

    for e in edges:
        if e["from"] in skip_nodes or e["to"] in skip_nodes:
            continue
        if e["from"] in node_ids and e["to"] in node_ids:
            elements.append({"data": {"source": e["from"], "target": e["to"]}})

    return elements


def _build_module_tree_elements(model_data: dict) -> list[dict]:
    mt = model_data.get("module_tree", {})
    modules = mt.get("modules", [])
    if not modules:
        return []

    elements: list[dict] = []
    leaf_paths = []

    for m in modules:
        if not m.get("is_leaf"):
            continue
        path = m["path"]
        cat = _module_category(m["type"])
        label = f"{m['type']}\n[{path}]"
        if m.get("params"):
            label += f"\n{_fmt_num(m['params'])}"
        elements.append({
            "data": {"id": path, "label": label,
                     "op": "module", "nn_module": path,
                     "nn_module_type": m.get("type_full", m["type"]),
                     "output_shape": None, "output_dtype": None,
                     "target": m["type"], "args": [],
                     "params": m.get("params", 0)},
            "classes": f"mod-{cat}",
        })
        leaf_paths.append(path)

    for i in range(len(leaf_paths) - 1):
        elements.append({"data": {"source": leaf_paths[i], "target": leaf_paths[i + 1]}})

    return elements


# ── Dashboard page ───────────────────────────────────────────────────

def _dashboard_page(model_data: dict, snapshots: list | None = None):
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

    stat_items = [
        ("Parameters", _fmt_num(model_data.get("total_params", 0))),
        ("Trainable", _fmt_num(model_data.get("trainable_params", 0))),
        ("Layers", str(len(leaf_modules))),
        ("FX Ops", str(len(ops))),
        ("Buffers", str(buffer_count)),
    ]
    if snapshots:
        stat_items.append(("Checkpoints", str(len(snapshots))))

    children = [
        html.H2(model_data.get("model_name", "Model"), className="mt-3 mb-1"),
        html.P(model_data.get("model_class", ""), className="text-muted mb-4"),

        # Stat cards
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Small(label, className="text-uppercase text-muted"),
                html.H4(value, className="mb-0"),
            ])), width="auto") for label, value in stat_items
        ], className="g-3 mb-4"),

        # I/O
        dbc.Card(dbc.CardBody([
            html.H5("I/O", className="card-title"),
            html.Code(f"Input:  {input_desc}", className="d-block mb-1"),
            html.Code(f"Output: {output_desc}"),
        ]), className="mb-3"),
    ]

    # Architecture graph
    arch_elements = _build_module_tree_elements(model_data)
    if arch_elements:
        children.append(dbc.Card(dbc.CardBody([
            html.H5(f"Architecture ({len(leaf_modules)} layers)", className="card-title"),
            html.P("Module hierarchy. Scroll to zoom, drag to pan.", className="text-muted"),
            cyto.Cytoscape(
                id="gh-cyto-arch",
                elements=arch_elements,
                layout={"name": "dagre", "rankDir": "TB", "spacingFactor": 1.4, "nodeSep": 40, "rankSep": 60},
                stylesheet=_CYTO_STYLE,
                style={"width": "100%", "height": "420px", "borderRadius": "8px"},
                boxSelectionEnabled=False, userZoomingEnabled=True, userPanningEnabled=True,
            ),
            html.Div(id="gh-arch-detail", className="mt-3",
                      children=[html.P("Click a layer to inspect it.", className="text-muted")]),
        ]), className="mb-3"))

    # FX computation graph
    fx = model_data.get("fx_graph")
    if fx and fx.get("nodes"):
        fx_elements = _build_fx_elements(model_data)
        if fx_elements:
            children.append(dbc.Card(dbc.CardBody([
                html.H5(f"Computation Graph ({len(ops)} ops)", className="card-title"),
                html.P("ATen-level ops from torch.export. Click a node to inspect.",
                        className="text-muted"),
                cyto.Cytoscape(
                    id="gh-cyto",
                    elements=fx_elements,
                    layout={"name": "dagre", "rankDir": "TB", "spacingFactor": 1.2, "nodeSep": 30, "rankSep": 50},
                    stylesheet=_CYTO_STYLE,
                    style={"width": "100%", "height": "520px", "borderRadius": "8px"},
                    boxSelectionEnabled=False, userZoomingEnabled=True, userPanningEnabled=True,
                ),
                html.Div(id="gh-node-detail", className="mt-3",
                          children=[html.P("Click a node to inspect it.", className="text-muted")]),
            ]), className="mb-3"))

    # Checkpoint summary
    if snapshots:
        children.append(_ckpt_summary_table(snapshots))

    # Module tree table
    if leaf_modules:
        rows = []
        for m in leaf_modules:
            cat = _module_category(m["type"])
            color = _MODULE_COLORS.get(cat, _MODULE_COLORS["default"])
            attrs = m.get("attributes", {})
            attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items()) if attrs else ""
            rows.append(html.Tr([
                html.Td(m["path"]),
                html.Td(dbc.Badge(m["type"], style={"backgroundColor": color, "color": "#222"})),
                html.Td(_fmt_num(m.get("params", 0))),
                html.Td(html.Code(attr_str) if attr_str else ""),
            ]))
        children.append(dbc.Card(dbc.CardBody([
            html.H5(f"Module Tree ({len(leaf_modules)} layers)", className="card-title"),
            dbc.Table([
                html.Thead(html.Tr([html.Th("Path"), html.Th("Type"), html.Th("Params"), html.Th("Attributes")])),
                html.Tbody(rows),
            ], bordered=True, hover=True, responsive=True, size="sm", className="mb-0"),
        ]), className="mb-3"))

    # Parameters table
    if params:
        p_rows = []
        for name, meta in params.items():
            if meta.get("is_buffer"):
                continue
            p_rows.append(html.Tr([
                html.Td(name),
                html.Td(html.Code("x".join(str(s) for s in meta.get("shape", [])))),
                html.Td(meta.get("dtype", "")),
                html.Td(_fmt_num(meta.get("numel", 0))),
            ]))
        children.append(dbc.Card(dbc.CardBody([
            html.H5("Parameters", className="card-title"),
            dbc.Table([
                html.Thead(html.Tr([html.Th("Name"), html.Th("Shape"), html.Th("Dtype"), html.Th("Elements")])),
                html.Tbody(p_rows),
            ], bordered=True, hover=True, responsive=True, size="sm", className="mb-0"),
        ]), className="mb-3"))

    return dbc.Container(children, fluid=True)


def _landing_page_empty():
    return dbc.Container([
        html.H2("GradientHound", className="mt-3 mb-1"),
        html.P("No model loaded. Use --model path/to/model.gh.json to load an export.",
                className="text-muted mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5(title, className="card-title"),
                html.P(desc, className="text-muted mb-0"),
            ])), md=4, className="mb-3")
            for path, (title, desc) in _PAGES.items() if path != "/"
        ]),
    ], fluid=True)


# ── Checkpoint comparison summary ────────────────────────────────────

def _ckpt_summary_table(snapshots: list[dict]):
    all_layers: list[str] = []
    seen = set()
    for snap in snapshots:
        for s in snap["weight_stats"]:
            layer = s["layer"]
            if layer not in seen:
                all_layers.append(layer)
                seen.add(layer)

    snap_lookup = {}
    for snap in snapshots:
        snap_lookup[snap["name"]] = {s["layer"]: s for s in snap["weight_stats"]}

    header = [html.Th("Layer")] + [html.Th(snap["name"]) for snap in snapshots]
    rows = []
    for layer in all_layers[:50]:
        cells = [html.Td(html.Code(_short_layer(layer)))]
        for snap in snapshots:
            stats = snap_lookup[snap["name"]].get(layer)
            if stats:
                cells.append(html.Td(html.Code(
                    f"L2={stats['norm_l2']:.4g}  \u03bc={stats['mean']:.3g}  \u03c3={stats['std']:.3g}"
                )))
            else:
                cells.append(html.Td("\u2014"))
        rows.append(html.Tr(cells))

    return dbc.Card(dbc.CardBody([
        html.H5(
            f"Checkpoint Comparison ({len(all_layers)} parameters, {len(snapshots)} checkpoints)",
            className="card-title",
        ),
        html.Div(
            dbc.Table([
                html.Thead(html.Tr(header)),
                html.Tbody(rows),
            ], bordered=True, hover=True, responsive=True, size="sm"),
            style={"maxHeight": "500px", "overflowY": "auto"},
        ),
    ]), className="mb-3")


# ── Checkpoints page ─────────────────────────────────────────────────

def _checkpoints_page(ckpt_paths: list[str], snapshots: list[dict] | None):
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
            "Visit the Weights or Gradients pages to explore comparison charts.",
        ], color="success", className="mb-3"))
        children.append(_ckpt_summary_table(snapshots))
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


def _checkpoints_page_empty():
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


# ── Weights page ─────────────────────────────────────────────────────

def _weights_page_checkpoints(snapshots: list[dict]):
    import plotly.graph_objects as go

    if not snapshots:
        return _placeholder_page("Weights", "Process checkpoints to view weight analysis.")

    all_layers = []
    seen = set()
    for snap in snapshots:
        for s in snap["weight_stats"]:
            if s["layer"] not in seen:
                all_layers.append(s["layer"])
                seen.add(s["layer"])

    default_layer = all_layers[0] if all_layers else ""

    # Norm evolution
    norm_fig = go.Figure()
    for i, snap in enumerate(snapshots):
        layer_names = [_short_layer(s["layer"]) for s in snap["weight_stats"]]
        norms = [s["norm_l2"] for s in snap["weight_stats"]]
        norm_fig.add_trace(go.Bar(
            x=layer_names, y=norms, name=snap["name"],
            marker_color=_SERIES_COLORS[i % len(_SERIES_COLORS)],
        ))
    norm_fig.update_layout(**_plotly_layout(title="L2 Norm per Layer"), barmode="group",
                           xaxis_tickangle=-45, height=400)

    # SVD effective rank
    svd_layers = []
    for snap in snapshots:
        for s in snap["weight_stats"]:
            if "effective_rank" in s and s["layer"] not in svd_layers:
                svd_layers.append(s["layer"])

    svd_fig = go.Figure()
    if svd_layers:
        snap_names = [s["name"] for s in snapshots]
        for j, layer in enumerate(svd_layers):
            ranks = []
            for snap in snapshots:
                stat = next((s for s in snap["weight_stats"] if s["layer"] == layer), None)
                ranks.append(stat["effective_rank"] if stat and "effective_rank" in stat else None)
            svd_fig.add_trace(go.Scatter(
                x=snap_names, y=ranks, mode="lines+markers", name=_short_layer(layer),
                line={"color": _SERIES_COLORS[j % len(_SERIES_COLORS)]},
            ))
        svd_fig.update_layout(**_plotly_layout(title="Effective Rank across Checkpoints"), height=400)

    # Stats table
    stat_rows = []
    for layer in all_layers[:50]:
        cells = [html.Td(html.Code(_short_layer(layer)))]
        for snap in snapshots:
            stat = next((s for s in snap["weight_stats"] if s["layer"] == layer), None)
            if stat:
                cells.append(html.Td(html.Code(
                    f"\u03bc={stat['mean']:.4g}  \u03c3={stat['std']:.4g}  k={stat.get('kurtosis', 0):.2f}"
                )))
            else:
                cells.append(html.Td("\u2014"))
        stat_rows.append(html.Tr(cells))
    stat_header = [html.Th("Layer")] + [html.Th(s["name"]) for s in snapshots]

    children = [
        html.H2("Weights", className="mt-3 mb-1"),
        html.P(f"Weight analysis across {len(snapshots)} checkpoints", className="text-muted mb-4"),

        # Histogram
        dbc.Card(dbc.CardBody([
            html.H5("Layer Histogram Comparison", className="card-title"),
            html.P("Select a layer to compare weight distributions across checkpoints.",
                    className="text-muted"),
            dcc.Dropdown(id="ckpt-layer-select",
                         options=[{"label": _short_layer(l), "value": l} for l in all_layers],
                         value=default_layer, className="mb-2"),
            dcc.Graph(id="ckpt-histogram"),
        ]), className="mb-3"),

        # Norms
        dbc.Card(dbc.CardBody([
            html.H5("L2 Norm per Layer", className="card-title"),
            dcc.Graph(figure=norm_fig),
        ]), className="mb-3"),
    ]

    if svd_layers:
        children.append(dbc.Card(dbc.CardBody([
            html.H5("Effective Rank Evolution", className="card-title"),
            html.P("Entropy-based effective rank for 2D weight matrices across checkpoints.",
                    className="text-muted"),
            dcc.Graph(figure=svd_fig),
        ]), className="mb-3"))

        children.append(dbc.Card(dbc.CardBody([
            html.H5("Singular Value Spectrum", className="card-title"),
            html.P("Select a layer to compare SVD spectra across checkpoints.", className="text-muted"),
            dcc.Dropdown(id="ckpt-svd-layer-select",
                         options=[{"label": _short_layer(l), "value": l} for l in svd_layers],
                         value=svd_layers[0] if svd_layers else None, className="mb-2"),
            dcc.Graph(id="ckpt-svd-spectrum"),
        ]), className="mb-3"))

    children.append(dbc.Card(dbc.CardBody([
        html.H5("Distribution Statistics", className="card-title"),
        html.Div(
            dbc.Table([
                html.Thead(html.Tr(stat_header)),
                html.Tbody(stat_rows),
            ], bordered=True, hover=True, responsive=True, size="sm"),
            style={"maxHeight": "500px", "overflowY": "auto"},
        ),
    ]), className="mb-3"))

    return dbc.Container(children, fluid=True)


# ── Gradients (weight-change proxy) page ─────────────────────────────

def _gradients_page_checkpoints(snapshots: list[dict]):
    import plotly.graph_objects as go

    if len(snapshots) < 2:
        return dbc.Container([
            html.H2("Gradients", className="mt-3 mb-1"),
            html.P("Weight change analysis requires at least 2 checkpoints.", className="text-muted"),
        ], fluid=True)

    all_layers = [s["layer"] for s in snapshots[0]["weight_stats"]]
    lookup = {}
    for snap in snapshots:
        lookup[snap["name"]] = {s["layer"]: s for s in snap["weight_stats"]}

    diff_fig = go.Figure()
    for i in range(1, len(snapshots)):
        prev_name = snapshots[i - 1]["name"]
        curr_name = snapshots[i]["name"]
        layers_short, diff_norms = [], []
        for layer in all_layers:
            prev_stat = lookup[prev_name].get(layer)
            curr_stat = lookup[curr_name].get(layer)
            if prev_stat and curr_stat:
                diff_norms.append(abs(curr_stat["norm_l2"] - prev_stat["norm_l2"]))
                layers_short.append(_short_layer(layer))
        diff_fig.add_trace(go.Bar(
            x=layers_short, y=diff_norms,
            name=f"{prev_name} \u2192 {curr_name}",
            marker_color=_SERIES_COLORS[(i - 1) % len(_SERIES_COLORS)],
        ))
    diff_fig.update_layout(**_plotly_layout(title="L2 Norm Change between Consecutive Checkpoints"),
                           barmode="group", xaxis_tickangle=-45, height=400)

    rel_fig = go.Figure()
    for i in range(1, len(snapshots)):
        prev_name = snapshots[i - 1]["name"]
        curr_name = snapshots[i]["name"]
        layers_short, rel_changes = [], []
        for layer in all_layers:
            prev_stat = lookup[prev_name].get(layer)
            curr_stat = lookup[curr_name].get(layer)
            if prev_stat and curr_stat and prev_stat["norm_l2"] > 1e-12:
                rel_changes.append(abs(curr_stat["norm_l2"] - prev_stat["norm_l2"]) / prev_stat["norm_l2"] * 100)
                layers_short.append(_short_layer(layer))
        rel_fig.add_trace(go.Bar(
            x=layers_short, y=rel_changes,
            name=f"{prev_name} \u2192 {curr_name}",
            marker_color=_SERIES_COLORS[(i - 1) % len(_SERIES_COLORS)],
        ))
    rel_fig.update_layout(**_plotly_layout(title="Relative Norm Change (%)"),
                          barmode="group", xaxis_tickangle=-45, yaxis_title="%", height=400)

    return dbc.Container([
        html.H2("Gradients", className="mt-3 mb-1"),
        html.P("Weight change analysis between consecutive checkpoints (gradient data requires live training).",
                className="text-muted mb-4"),
        dbc.Card(dbc.CardBody([
            html.H5("Norm Change per Layer", className="card-title"),
            html.P("Absolute change in L2 norm between consecutive checkpoints.", className="text-muted"),
            dcc.Graph(figure=diff_fig),
        ]), className="mb-3"),
        dbc.Card(dbc.CardBody([
            html.H5("Relative Norm Change", className="card-title"),
            html.P("Percentage change in L2 norm relative to the earlier checkpoint.", className="text-muted"),
            dcc.Graph(figure=rel_fig),
        ]), className="mb-3"),
    ], fluid=True)


# ── Node detail panel ────────────────────────────────────────────────

def _node_detail_panel(node_data: dict):
    if not node_data:
        return html.P("Select a node.", className="text-muted")

    items = []

    def _item(key, val):
        if val is None or val == "":
            return
        if isinstance(val, list):
            val = "x".join(str(s) for s in val) if all(isinstance(v, (int, float)) for v in val) else str(val)
        items.append(dbc.ListGroupItem([
            html.Strong(key + ": "),
            html.Code(str(val)),
        ]))

    _item("Name", node_data.get("id"))
    _item("Op", node_data.get("op"))
    target = node_data.get("target", "")
    if target:
        _item("Target", _short_target(target))
    _item("Module", node_data.get("nn_module"))
    _item("Module Type", (node_data.get("nn_module_type") or "").split(".")[-1])
    _item("Output Shape", node_data.get("output_shape"))
    _item("Output Dtype", node_data.get("output_dtype"))
    _item("Source Fn", node_data.get("source_fn"))
    args = node_data.get("args", [])
    if args:
        _item("Inputs", ", ".join(str(a) for a in args))

    return dbc.Card(dbc.CardBody([
        html.H6("Node Details", className="card-title"),
        dbc.ListGroup(items, flush=True),
    ]))


# ── App factory ──────────────────────────────────────────────────────

def create_app(
    data_dir: str | None = None,
    model_path: str | None = None,
    checkpoint_paths: list[str] | None = None,
    loader_path: str | None = None,
) -> Dash:
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        title="GradientHound",
        external_stylesheets=[dbc.themes.DARKLY],
    )

    # Load model export
    model_data: dict | None = None
    if model_path:
        p = Path(model_path)
        if p.is_dir():
            gh_files = sorted(p.glob("*.gh.json"))
            if gh_files:
                with open(gh_files[0]) as f:
                    model_data = json.load(f)
        elif p.exists():
            with open(p) as f:
                model_data = json.load(f)

    # IPC channel
    ipc = None
    if data_dir:
        from gradienthound.ipc import IPCChannel
        ipc = IPCChannel(directory=data_dir)

    # Checkpoint state
    ckpt_state: dict = {
        "paths": checkpoint_paths or [],
        "loader_path": loader_path,
        "processed": False,
        "snapshots": [],
    }
    has_checkpoints = bool(ckpt_state["paths"])

    # Navbar
    nav_links = [
        dbc.NavLink(title, href=path, id=f"nav-{path}", active="exact")
        for path, (title, _) in _PAGES.items()
    ]
    navbar = dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("GradientHound", href="/", className="fw-bold"),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                dbc.Nav(nav_links, className="ms-auto", navbar=True),
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ], fluid=True),
        color="dark",
        dark=True,
        sticky="top",
        className="mb-3",
    )

    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="ckpt-store", data=None),
        navbar,
        dbc.Container(html.Div(id="gh-content"), fluid=True, className="px-4"),
    ])

    # ── Navbar toggle (mobile) ──────────────────────────────────────

    @callback(
        Output("navbar-collapse", "is_open"),
        Input("navbar-toggler", "n_clicks"),
        State("navbar-collapse", "is_open"),
    )
    def _toggle_navbar(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open

    # ── Process checkpoints ──────────────────────────────────────────

    if has_checkpoints:
        @callback(
            Output("ckpt-store", "data"),
            Output("ckpt-status", "children"),
            Output("ckpt-process-btn", "disabled"),
            Input("ckpt-process-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def _process_ckpts(n_clicks):
            if not n_clicks:
                return no_update, no_update, no_update
            from gradienthound.checkpoint import process_checkpoints
            try:
                snapshots = process_checkpoints(
                    ckpt_state["paths"], loader_path=ckpt_state["loader_path"],
                )
                ckpt_state["snapshots"] = snapshots
                ckpt_state["processed"] = True
                n_params = len({s["layer"] for snap in snapshots for s in snap["weight_stats"]})
                return {"ready": True}, f"Done \u2014 {len(snapshots)} checkpoints, {n_params} params", True
            except Exception as exc:
                return no_update, f"Error: {exc}", False

    # ── Routing ──────────────────────────────────────────────────────

    @callback(
        Output("gh-content", "children"),
        Input("url", "pathname"),
        Input("ckpt-store", "data"),
    )
    def _route(pathname, ckpt_data):
        snapshots = ckpt_state["snapshots"] if ckpt_state["processed"] else None

        if pathname is None or pathname == "/":
            if model_data:
                return _dashboard_page(model_data, snapshots=snapshots)
            return _landing_page_empty()

        if pathname == "/checkpoints":
            if has_checkpoints:
                return _checkpoints_page(ckpt_state["paths"], snapshots)
            return _checkpoints_page_empty()

        if pathname == "/weights" and snapshots:
            return _weights_page_checkpoints(snapshots)

        if pathname == "/gradients" and snapshots:
            return _gradients_page_checkpoints(snapshots)

        if pathname in _PAGES:
            title, desc = _PAGES[pathname]
            return _placeholder_page(title, desc)

        return _placeholder_page("Not Found", f"No page at {pathname}")

    # ── Node click detail panels ─────────────────────────────────────

    @callback(
        Output("gh-node-detail", "children"),
        Input("gh-cyto", "tapNodeData"),
        prevent_initial_call=True,
    )
    def _on_fx_node_click(data):
        if data is None:
            return no_update
        return _node_detail_panel(data)

    @callback(
        Output("gh-arch-detail", "children"),
        Input("gh-cyto-arch", "tapNodeData"),
        prevent_initial_call=True,
    )
    def _on_arch_node_click(data):
        if data is None:
            return no_update
        return _node_detail_panel(data)

    # ── Histogram callback ───────────────────────────────────────────

    @callback(
        Output("ckpt-histogram", "figure"),
        Input("ckpt-layer-select", "value"),
        prevent_initial_call=True,
    )
    def _update_histogram(selected_layer):
        import plotly.graph_objects as go
        snapshots = ckpt_state.get("snapshots", [])
        fig = go.Figure()
        for i, snap in enumerate(snapshots):
            stat = next((s for s in snap["weight_stats"] if s["layer"] == selected_layer), None)
            if stat and "hist_counts" in stat:
                fig.add_trace(go.Bar(
                    x=stat["hist_centers"], y=stat["hist_counts"], name=snap["name"],
                    marker_color=_SERIES_COLORS[i % len(_SERIES_COLORS)], opacity=0.65,
                ))
        fig.update_layout(**_plotly_layout(title=f"Weight Distribution \u2014 {_short_layer(selected_layer)}"),
                          barmode="overlay", height=360)
        return fig

    # ── SVD spectrum callback ────────────────────────────────────────

    @callback(
        Output("ckpt-svd-spectrum", "figure"),
        Input("ckpt-svd-layer-select", "value"),
        prevent_initial_call=True,
    )
    def _update_svd_spectrum(selected_layer):
        import plotly.graph_objects as go
        snapshots = ckpt_state.get("snapshots", [])
        fig = go.Figure()
        for i, snap in enumerate(snapshots):
            stat = next((s for s in snap["weight_stats"] if s["layer"] == selected_layer), None)
            if stat and "singular_values" in stat:
                svs = stat["singular_values"]
                fig.add_trace(go.Scatter(
                    x=list(range(len(svs))), y=svs, mode="lines", name=snap["name"],
                    line={"color": _SERIES_COLORS[i % len(_SERIES_COLORS)]},
                ))
        fig.update_layout(**_plotly_layout(title=f"Singular Values \u2014 {_short_layer(selected_layer)}"),
                          xaxis_title="Index", yaxis_title="Singular Value", yaxis_type="log", height=360)
        return fig

    return app
