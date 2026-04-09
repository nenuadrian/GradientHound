"""Embeddings page: t-SNE / PCA projections of layer weight statistics."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html, dcc

from ._helpers import (
    plotly_layout, short_layer, compute_layer_embeddings,
    _EMBED_FEATURE_KEYS,
)
from ._constants import SERIES_COLORS


def embeddings_page(model_data: dict | None, snapshots: list | None = None):
    import plotly.graph_objects as go

    children = [
        html.H2("Embeddings", className="mt-3 mb-1"),
        html.P(
            "t-SNE and PCA projections of per-layer weight statistics. "
            "Each point is one layer; proximity means similar weight characteristics.",
            className="text-muted mb-4",
        ),
    ]

    if not snapshots or len(snapshots) < 1:
        children.append(dbc.Alert(
            "Load at least 1 checkpoint to see layer embeddings.",
            color="info",
        ))
        return dbc.Container(children, fluid=True)

    # Check which optional reducers are available
    try:
        import sklearn  # noqa: F401
        has_sklearn = True
    except ImportError:
        has_sklearn = False

    try:
        import umap  # noqa: F401
        has_umap = True
    except ImportError:
        has_umap = False

    # ── Controls ────────────────────────────────────────────────────
    method_options = [{"label": "PCA", "value": "pca"}]
    default_method = "pca"
    if has_umap:
        method_options.insert(0, {"label": "UMAP", "value": "umap"})
        default_method = "umap"
    if has_sklearn:
        method_options.insert(0, {"label": "t-SNE", "value": "tsne"})
        default_method = "tsne"

    controls = dbc.Row([
        dbc.Col([
            dbc.Label("Method", html_for="embed-method", className="small fw-semibold"),
            dcc.Dropdown(
                id="embed-method",
                options=method_options,
                value=default_method,
                clearable=False,
            ),
        ], md=3),
        dbc.Col([
            dbc.Label(
                "Perplexity / Neighbors",
                html_for="embed-perplexity",
                className="small fw-semibold",
            ),
            dcc.Slider(
                id="embed-perplexity",
                min=2, max=50, step=1, value=8,
                marks={2: "2", 10: "10", 20: "20", 30: "30", 50: "50"},
                tooltip={"placement": "bottom", "always_visible": False},
                disabled=not (has_sklearn or has_umap),
            ),
        ], md=5),
        dbc.Col([
            dbc.Label("Color by", html_for="embed-color-by", className="small fw-semibold"),
            dcc.Dropdown(
                id="embed-color-by",
                options=[
                    {"label": "Checkpoint", "value": "checkpoint"},
                    {"label": "Layer type", "value": "layer_type"},
                    {"label": "Depth (position)", "value": "depth"},
                ],
                value="checkpoint" if len(snapshots) > 1 else "layer_type",
                clearable=False,
            ),
        ], md=4),
    ], className="g-2 mb-3")

    children.append(dbc.Card(dbc.CardBody([
        html.H5("Projection Settings", className="card-title"),
        controls,
    ]), className="mb-3"))

    missing = []
    if not has_sklearn:
        missing.append("scikit-learn (t-SNE)")
    if not has_umap:
        missing.append("umap-learn (UMAP)")
    if missing:
        children.append(dbc.Alert(
            [
                html.Strong("Optional packages not installed: "),
                ", ".join(missing) + ". ",
                "Using PCA fallback. Install with: ",
                html.Code(f"pip install {' '.join(p.split()[0] for p in missing)}"),
            ],
            color="warning",
            className="mb-3",
        ))

    # ── Compute initial embedding ───────────────────────────────────
    children.append(html.Div(id="embed-chart-wrap"))

    # Pre-compute the default chart so the page isn't blank on load
    initial = _build_embedding_chart(
        snapshots,
        method=default_method,
        perplexity=8.0,
        color_by="checkpoint" if len(snapshots) > 1 else "layer_type",
    )
    if initial is not None:
        children[-1] = html.Div(
            dbc.Card(dbc.CardBody(dcc.Graph(figure=initial, id="embed-scatter")), className="mb-3"),
            id="embed-chart-wrap",
        )
    else:
        children[-1] = html.Div(
            dbc.Alert(
                "Not enough layers with complete statistics for embedding. "
                "Need at least 4 layers with norm, std, kurtosis, etc.",
                color="warning",
            ),
            id="embed-chart-wrap",
        )

    # ── Feature distribution summary ────────────────────────────────
    if snapshots:
        latest = snapshots[-1]
        stats = latest.get("weight_stats", [])
        feature_avail = {}
        for key in _EMBED_FEATURE_KEYS:
            count = sum(1 for s in stats if s.get(key) is not None)
            feature_avail[key] = count

        feat_rows = []
        for key, count in feature_avail.items():
            color = "success" if count > 0 else "secondary"
            feat_rows.append(html.Tr([
                html.Td(html.Code(key)),
                html.Td(dbc.Badge(f"{count}/{len(stats)}", color=color)),
            ]))

        children.append(dbc.Card(dbc.CardBody([
            html.H5("Feature Availability", className="card-title"),
            html.P(
                "Statistics used as dimensions for embedding. "
                "Layers missing any feature are excluded from the projection.",
                className="text-muted",
            ),
            dbc.Table([
                html.Thead(html.Tr([html.Th("Feature"), html.Th("Layers with data")])),
                html.Tbody(feat_rows),
            ], bordered=True, hover=True, responsive=True, size="sm"),
        ]), className="mb-3"))

    return dbc.Container(children, fluid=True)


def _classify_layer_type(name: str) -> str:
    """Classify a layer name into a broad type category."""
    lower = name.lower()
    for key in ("conv", "linear", "norm", "pool", "dropout", "embed", "attention", "mlp"):
        if key in lower:
            return key
    if any(k in lower for k in ("relu", "gelu", "silu", "sigmoid", "tanh")):
        return "activation"
    if "bias" in lower:
        return "bias"
    if "weight" in lower:
        return "weight"
    return "other"


def _build_embedding_chart(
    snapshots: list[dict],
    *,
    method: str = "tsne",
    perplexity: float = 8.0,
    color_by: str = "checkpoint",
):
    """Build the Plotly scatter figure for layer embeddings."""
    import plotly.graph_objects as go

    points = compute_layer_embeddings(
        snapshots, method=method, perplexity=perplexity,
    )
    if not points:
        return None

    method_label = {"tsne": "t-SNE", "umap": "UMAP", "pca": "PCA"}.get(method, method)

    if color_by == "checkpoint":
        # Group by checkpoint name
        groups: dict[str, list] = {}
        for p in points:
            groups.setdefault(p["checkpoint"], []).append(p)

        fig = go.Figure()
        for idx, (ckpt_name, pts) in enumerate(groups.items()):
            color = SERIES_COLORS[idx % len(SERIES_COLORS)]
            fig.add_trace(go.Scatter(
                x=[p["x"] for p in pts],
                y=[p["y"] for p in pts],
                mode="markers",
                name=ckpt_name,
                marker={"size": 8, "color": color, "opacity": 0.8},
                text=[short_layer(p["layer"]) for p in pts],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"Checkpoint: {ckpt_name}<br>"
                    f"{method_label}-1: %{{x:.3f}}<br>"
                    f"{method_label}-2: %{{y:.3f}}"
                    "<extra></extra>"
                ),
            ))

        # Draw trajectory arrows if multiple checkpoints
        if len(groups) > 1:
            ckpt_order = [s["name"] for s in snapshots]
            # Build layer -> ordered positions
            layer_positions: dict[str, list[tuple[float, float, str]]] = {}
            for p in points:
                layer_positions.setdefault(p["layer"], []).append(
                    (p["x"], p["y"], p["checkpoint"])
                )
            for layer, positions in layer_positions.items():
                if len(positions) < 2:
                    continue
                # Sort by checkpoint order
                pos_sorted = sorted(
                    positions,
                    key=lambda t: ckpt_order.index(t[2]) if t[2] in ckpt_order else 999,
                )
                for i in range(len(pos_sorted) - 1):
                    x0, y0, _ = pos_sorted[i]
                    x1, y1, _ = pos_sorted[i + 1]
                    fig.add_annotation(
                        x=x1, y=y1, ax=x0, ay=y0,
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True,
                        arrowhead=2, arrowsize=1, arrowwidth=1,
                        arrowcolor="rgba(100,100,100,0.3)",
                    )

    elif color_by == "layer_type":
        type_colors = {
            "conv": "#f0d4d8", "linear": "#28a745", "norm": "#ffc107",
            "attention": "#3498db", "mlp": "#e67e22", "embed": "#9b59b6",
            "pool": "#1abc9c", "dropout": "#95a5a6", "activation": "#e74c3c",
            "bias": "#6c757d", "weight": "#375a7f", "other": "#adb5bd",
        }
        groups = {}
        for p in points:
            lt = _classify_layer_type(p["layer"])
            groups.setdefault(lt, []).append(p)

        fig = go.Figure()
        for lt, pts in groups.items():
            fig.add_trace(go.Scatter(
                x=[p["x"] for p in pts],
                y=[p["y"] for p in pts],
                mode="markers",
                name=lt,
                marker={"size": 8, "color": type_colors.get(lt, "#adb5bd"), "opacity": 0.8},
                text=[f"{short_layer(p['layer'])} ({p['checkpoint']})" for p in pts],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"Type: {lt}<br>"
                    f"{method_label}-1: %{{x:.3f}}<br>"
                    f"{method_label}-2: %{{y:.3f}}"
                    "<extra></extra>"
                ),
            ))

    elif color_by == "depth":
        # Color by position index within the model
        # Collect unique layer names in order from first snapshot
        all_layers = []
        seen = set()
        for snap in snapshots:
            for s in snap.get("weight_stats", []):
                name = s.get("layer", "")
                if name not in seen:
                    all_layers.append(name)
                    seen.add(name)
        depth_map = {name: i for i, name in enumerate(all_layers)}
        max_depth = max(depth_map.values()) if depth_map else 1

        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        colors = [depth_map.get(p["layer"], 0) / max(max_depth, 1) for p in points]
        texts = [f"{short_layer(p['layer'])} ({p['checkpoint']})" for p in points]

        fig = go.Figure(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker={
                "size": 8,
                "color": [depth_map.get(p["layer"], 0) for p in points],
                "colorscale": "Viridis",
                "colorbar": {"title": "Depth"},
                "opacity": 0.8,
            },
            text=texts,
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{method_label}-1: %{{x:.3f}}<br>"
                f"{method_label}-2: %{{y:.3f}}"
                "<extra></extra>"
            ),
        ))
    else:
        return None

    fig.update_layout(
        **plotly_layout(title=f"Layer Embedding ({method_label})"),
        height=560,
        xaxis_title=f"{method_label}-1",
        yaxis_title=f"{method_label}-2",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig
