from __future__ import annotations

import hashlib
from dataclasses import dataclass
from html import escape
from typing import Any

import graphviz
import torch.nn as nn


# Attributes worth showing in node labels
_ATTR_KEYS = [
    "in_channels", "out_channels", "kernel_size", "stride", "padding",
    "in_features", "out_features", "num_features",
    "eps", "momentum", "p", "inplace",
    "normalized_shape", "num_heads", "dropout",
    "output_size", "num_embeddings", "embedding_dim",
]

# Color palette by module category
_COLORS: dict[str, str] = {
    "conv": "#f0d4d8",
    "linear": "#d8ecde",
    "norm": "#faf0e0",
    "activation": "#fae0d0",
    "pool": "#e4d6ec",
    "dropout": "#f0eaea",
    "embedding": "#e0d4e8",
    "default": "#faf6f6",
}

_CONTAINER_BG = "#faf6f6"
_CONTAINER_BORDER = "#d4b8bc"
_CONTAINER_LABEL = "#5e4a4e"
_CARD_BORDER = "#d0b4b8"
_SEQUENTIAL_EDGE = "#8b5c64"
_PARALLEL_EDGE = "#8a7a7c"
_DETACHED_EDGE = "#9a8a8c"


@dataclass(frozen=True)
class _RenderEndpoints:
    entry_id: str
    exit_id: str
    cluster_id: str | None = None


def _module_category(type_name: str) -> str:
    lower = type_name.lower()
    if "conv" in lower:
        return "conv"
    if "linear" in lower:
        return "linear"
    if "norm" in lower or "layernorm" in lower:
        return "norm"
    if any(k in lower for k in ("relu", "gelu", "silu", "sigmoid", "tanh", "softmax", "activation")):
        return "activation"
    if "pool" in lower:
        return "pool"
    if "dropout" in lower:
        return "dropout"
    if "embed" in lower:
        return "embedding"
    return "default"


def _get_module_attrs(module: nn.Module) -> dict[str, Any]:
    attrs = {}
    for key in _ATTR_KEYS:
        if hasattr(module, key):
            val = getattr(module, key)
            if isinstance(val, (int, float, bool, str, tuple, list)):
                attrs[key] = val
    return attrs


def _count_own_params(module: nn.Module) -> int:
    """Count parameters directly owned by this module (not children)."""
    child_params = set()
    for child in module.children():
        for p in child.parameters():
            child_params.add(id(p))
    return sum(p.numel() for p in module.parameters() if id(p) not in child_params)


def _is_container(module: nn.Module) -> bool:
    type_name = type(module).__name__
    return type_name in ("Sequential", "ModuleList", "ModuleDict") or (
        list(module.children()) and _count_own_params(module) == 0
    )


def extract_model_graph(name: str, model: nn.Module) -> dict:
    """Extract a JSON-serializable graph description from a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    modules = []
    try:
        pytorch_repr = str(model)
    except Exception:
        pytorch_repr = "<unable to render model representation>"

    for path, mod in model.named_modules():
        if path == "":
            path = name

        children = []
        for child_name, _ in mod.named_children():
            child_path = f"{path}.{child_name}" if path != name else child_name
            children.append(child_path)

        modules.append({
            "path": path,
            "type": type(mod).__name__,
            "type_full": f"{type(mod).__module__}.{type(mod).__qualname__}",
            "params": _count_own_params(mod),
            "is_leaf": len(children) == 0,
            "is_container": _is_container(mod),
            "depth": path.count(".") + (0 if path == name else 1),
            "attributes": _get_module_attrs(mod),
            "children": children,
        })

    return {
        "name": name,
        "class": type(model).__name__,
        "total_params": total_params,
        "modules": modules,
        "pytorch_repr": pytorch_repr,
    }


def _format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def render_graphviz(
    graph_data: dict,
    overlays: dict[str, dict[str, Any]] | None = None,
) -> graphviz.Digraph:
    """Build a Graphviz Digraph from the extracted model graph data."""
    dot = graphviz.Digraph(
        name=graph_data["name"],
        graph_attr={
            "rankdir": "TB",
            "fontname": "Helvetica",
            "bgcolor": "transparent",
            "pad": "0.35",
            "compound": "true",
            "nodesep": "0.35",
            "ranksep": "0.65",
            "splines": "spline",
            "newrank": "true",
        },
        node_attr={
            "fontname": "Helvetica",
            "shape": "plain",
        },
        edge_attr={
            "arrowsize": "0.7",
            "color": _PARALLEL_EDGE,
            "penwidth": "1.3",
        },
    )

    modules = graph_data.get("modules", [])
    modules_by_path = {m["path"]: m for m in modules}
    root_path = graph_data["name"]
    root = modules_by_path.get(root_path)
    overlays = overlays or {}

    if root is None:
        dot.node(
            "__missing_root__",
            label=f"{graph_data.get('class', 'Model')}\\n(graph unavailable)",
            fillcolor="#FDEDEC",
            shape="box",
            style="filled,rounded",
        )
        return dot

    # Keep rendering robust for very large models by capping visible nodes.
    visible_paths = set(_select_visible_paths(root_path, modules_by_path, max_nodes=400))
    io_cache: dict[str, tuple[str | None, str | None]] = {}
    subtree_cache: dict[str, int] = {}

    _add_root_node(
        dot=dot,
        graph_data=graph_data,
        root_path=root_path,
        root=root,
        modules_by_path=modules_by_path,
        io_cache=io_cache,
        overlay=overlays.get(root_path),
    )

    root_children = [c for c in root.get("children", []) if c in visible_paths]
    rendered_children = [
        _render_module(
            dot=dot,
            parent=dot,
            path=child,
            modules_by_path=modules_by_path,
            visible_paths=visible_paths,
            io_cache=io_cache,
            subtree_cache=subtree_cache,
            overlays=overlays,
        )
        for child in root_children
    ]
    _connect_root_children(dot, root_path, root, root_children, rendered_children, modules_by_path)

    return dot


def _select_visible_paths(
    root_path: str,
    modules_by_path: dict[str, dict],
    max_nodes: int,
) -> list[str]:
    """Return a BFS-ordered list of module paths up to max_nodes."""
    order: list[str] = []
    queue = [root_path]
    seen: set[str] = set()

    while queue and len(order) < max_nodes:
        current = queue.pop(0)
        if current in seen:
            continue
        mod = modules_by_path.get(current)
        if mod is None:
            continue

        seen.add(current)
        order.append(current)

        for child in mod.get("children", []):
            if child not in seen and child in modules_by_path:
                queue.append(child)

    return order


def _render_module(
    *,
    dot: graphviz.Digraph,
    parent: graphviz.Digraph,
    path: str,
    modules_by_path: dict[str, dict],
    visible_paths: set[str],
    io_cache: dict[str, tuple[str | None, str | None]],
    subtree_cache: dict[str, int],
    overlays: dict[str, dict[str, Any]],
) -> _RenderEndpoints:
    """Render a module or container recursively and return its flow endpoints."""
    mod = modules_by_path[path]
    visible_children = [c for c in mod.get("children", []) if c in visible_paths]
    overlay = overlays.get(path)

    if not visible_children:
        _add_module_node(
            parent=parent,
            path=path,
            mod=mod,
            modules_by_path=modules_by_path,
            io_cache=io_cache,
            subtree_cache=subtree_cache,
            overlay=overlay,
        )
        return _RenderEndpoints(entry_id=path, exit_id=path)

    cluster_id = _cluster_id(path)
    entry_id = _anchor_id(path, "in")
    exit_id = _anchor_id(path, "out")

    with parent.subgraph(name=cluster_id) as cluster:
        _style_container_cluster(
            cluster=cluster,
            path=path,
            mod=mod,
            visible_children=visible_children,
            modules_by_path=modules_by_path,
            io_cache=io_cache,
            subtree_cache=subtree_cache,
            overlay=overlay,
        )
        _add_anchor_node(cluster, entry_id)
        _add_anchor_node(cluster, exit_id)

        rendered_children = [
            _render_module(
                dot=dot,
                parent=cluster,
                path=child,
                modules_by_path=modules_by_path,
                visible_paths=visible_paths,
                io_cache=io_cache,
                subtree_cache=subtree_cache,
                overlays=overlays,
            )
            for child in visible_children
        ]

    _connect_container_children(dot, mod, visible_children, rendered_children, entry_id, exit_id, modules_by_path)
    return _RenderEndpoints(entry_id=entry_id, exit_id=exit_id, cluster_id=cluster_id)


def _add_root_node(
    *,
    dot: graphviz.Digraph,
    graph_data: dict,
    root_path: str,
    root: dict,
    modules_by_path: dict[str, dict],
    io_cache: dict[str, tuple[str | None, str | None]],
    overlay: dict[str, Any] | None,
) -> None:
    """Render the root model node as a compact summary card."""
    in_dim, out_dim = _infer_node_io(root_path, modules_by_path, io_cache)
    title = graph_data.get("name", root_path)
    subtitle = graph_data.get("class")
    if subtitle == title:
        subtitle = None

    primary_parts = [
        f"{len(root.get('children', []))} top-level blocks",
        f"{_format_params(graph_data.get('total_params', 0))} params",
    ]
    flow = _io_summary(in_dim, out_dim)
    if flow:
        primary_parts.append(flow)

    body_lines = [
        " | ".join(primary_parts),
        f"{max(len(graph_data.get('modules', [])) - 1, 0)} modules tracked",
    ]
    accent = "#f0d4d8"
    border = _CARD_BORDER
    border_width = "1"
    tooltip = _tooltip_text(
        [
            title,
            f"class={graph_data.get('class')}" if graph_data.get("class") else None,
            f"total_params={graph_data.get('total_params', 0):,}",
        ]
    )

    accent, border, tooltip, body_lines, border_width = _apply_overlay_style(
        accent=accent,
        border=border,
        tooltip=tooltip,
        body_lines=body_lines,
        border_width=border_width,
        overlay=overlay,
    )

    dot.node(
        root_path,
        label=_card_label(
            title=title,
            subtitle=subtitle,
            body_lines=body_lines,
            accent=accent,
            border=border,
            border_width=border_width,
        ),
        tooltip=tooltip,
    )


def _add_module_node(
    *,
    parent: graphviz.Digraph,
    path: str,
    mod: dict,
    modules_by_path: dict[str, dict],
    io_cache: dict[str, tuple[str | None, str | None]],
    subtree_cache: dict[str, int],
    overlay: dict[str, Any] | None,
) -> None:
    """Add a compact card for a module or collapsed container."""
    title, subtitle = _node_heading(path, mod)
    in_dim, out_dim = _infer_node_io(path, modules_by_path, io_cache)

    if mod.get("is_container"):
        total_params = _count_subtree_params(path, modules_by_path, subtree_cache)
        flow = _io_summary(in_dim, out_dim)
        body_lines = [
            f"{len(mod.get('children', []))} child modules",
            f"{_format_params(total_params)} subtree params",
        ]
        if flow:
            body_lines[-1] = f"{body_lines[-1]} | {flow}"

        accent = _CONTAINER_BG
        border = _CONTAINER_BORDER
        border_width = "1"
        tooltip = _module_tooltip(path, mod, in_dim, out_dim)
        accent, border, tooltip, body_lines, border_width = _apply_overlay_style(
            accent=accent,
            border=border,
            tooltip=tooltip,
            body_lines=body_lines,
            border_width=border_width,
            overlay=overlay,
        )

        parent.node(
            path,
            label=_card_label(
                title=title,
                subtitle=subtitle or "Container",
                body_lines=body_lines,
                accent=accent,
                border=border,
                border_width=border_width,
            ),
            tooltip=tooltip,
        )
        return

    attr_summary = _attribute_summary(mod.get("attributes", {}))
    flow = _io_summary(in_dim, out_dim)
    params_text = f"{_format_params(mod['params'])} params"

    if flow and attr_summary:
        body_lines = [flow, f"{attr_summary} | {params_text}"]
    elif flow:
        body_lines = [flow, params_text]
    elif attr_summary:
        body_lines = [attr_summary, params_text]
    else:
        body_lines = [params_text]

    accent = _COLORS.get(_module_category(mod["type"]), _COLORS["default"])
    border = _CARD_BORDER
    border_width = "1"
    tooltip = _module_tooltip(path, mod, in_dim, out_dim)
    accent, border, tooltip, body_lines, border_width = _apply_overlay_style(
        accent=accent,
        border=border,
        tooltip=tooltip,
        body_lines=body_lines,
        border_width=border_width,
        overlay=overlay,
    )

    parent.node(
        path,
        label=_card_label(
            title=title,
            subtitle=subtitle,
            body_lines=body_lines,
            accent=accent,
            border=border,
            border_width=border_width,
        ),
        tooltip=tooltip,
    )


def _node_kind(mod: dict, children: list[str]) -> str:
    """Return a human-readable module kind for labeling."""
    if not children:
        return "module"
    return "sequential" if _is_sequential_container(mod, children) else "parallel"


def _infer_node_io(
    path: str,
    modules_by_path: dict[str, dict],
    cache: dict[str, tuple[str | None, str | None]],
) -> tuple[str | None, str | None]:
    """Infer (in_dim, out_dim) for a node from attrs and/or children."""
    if path in cache:
        return cache[path]

    mod = modules_by_path.get(path)
    if mod is None:
        return (None, None)

    leaf_in, leaf_out = _leaf_io_from_attrs(mod.get("attributes", {}))
    if not mod.get("is_container") or not mod.get("children"):
        cache[path] = (leaf_in, leaf_out)
        return cache[path]

    children = [c for c in mod.get("children", []) if c in modules_by_path]
    child_ios = [_infer_node_io(c, modules_by_path, cache) for c in children]

    if _is_sequential_container(mod, children):
        in_dim = leaf_in
        out_dim = leaf_out
        for ci, _ in child_ios:
            if ci:
                in_dim = ci
                break
        for _, co in reversed(child_ios):
            if co:
                out_dim = co
                break
        cache[path] = (in_dim, out_dim)
        return cache[path]

    # Parallel container: summarize heterogeneous branches.
    in_candidates = [ci for ci, _ in child_ios if ci]
    out_candidates = [co for _, co in child_ios if co]
    in_dim = leaf_in or _summarize_dims(in_candidates)
    out_dim = leaf_out or _summarize_dims(out_candidates)
    cache[path] = (in_dim, out_dim)
    return cache[path]


def _leaf_io_from_attrs(attrs: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract input/output dimensions for common layer attribute patterns."""
    if "in_features" in attrs or "out_features" in attrs:
        in_dim = str(attrs.get("in_features")) if "in_features" in attrs else None
        out_dim = str(attrs.get("out_features")) if "out_features" in attrs else None
        return (in_dim, out_dim)

    if "in_channels" in attrs or "out_channels" in attrs:
        in_dim = str(attrs.get("in_channels")) if "in_channels" in attrs else None
        out_dim = str(attrs.get("out_channels")) if "out_channels" in attrs else None
        return (in_dim, out_dim)

    if "num_embeddings" in attrs and "embedding_dim" in attrs:
        return (str(attrs["num_embeddings"]), str(attrs["embedding_dim"]))

    if "num_features" in attrs:
        d = str(attrs["num_features"])
        return (d, d)

    if "normalized_shape" in attrs:
        d = _format_dim(attrs["normalized_shape"])
        return (d, d)

    if "output_size" in attrs:
        return (None, _format_dim(attrs["output_size"]))

    return (None, None)


def _format_dim(value: Any) -> str:
    """Format scalar/tuple/list dimensions into a compact string."""
    if isinstance(value, (tuple, list)):
        return "x".join(str(v) for v in value)
    return str(value)


def _summarize_dims(values: list[str]) -> str | None:
    """Summarize branch dims for parallel containers."""
    if not values:
        return None
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0]
    if len(unique) <= 3:
        return " | ".join(unique)
    return "multi"


def _style_container_cluster(
    *,
    cluster: graphviz.Digraph,
    path: str,
    mod: dict,
    visible_children: list[str],
    modules_by_path: dict[str, dict],
    io_cache: dict[str, tuple[str | None, str | None]],
    subtree_cache: dict[str, int],
    overlay: dict[str, Any] | None,
) -> None:
    """Apply Graphviz cluster styling for container modules."""
    short_name = path.split(".")[-1]
    container_type = _node_kind(mod, visible_children).title()
    total_params = _format_params(_count_subtree_params(path, modules_by_path, subtree_cache))
    in_dim, out_dim = _infer_node_io(path, modules_by_path, io_cache)
    flow = _io_summary(in_dim, out_dim)

    label_parts = [
        container_type,
        f"{len(mod.get('children', []))} children",
        f"{total_params} params",
    ]
    if flow:
        label_parts.append(flow)

    color = _CONTAINER_BORDER
    fillcolor = _CONTAINER_BG
    fontcolor = _CONTAINER_LABEL
    style = "rounded,filled"
    penwidth = "1.2"
    tooltip = _module_tooltip(path, mod, in_dim, out_dim)
    if overlay:
        color = overlay.get("color", color)
        fillcolor = overlay.get("fillcolor", fillcolor)
        fontcolor = overlay.get("fontcolor", fontcolor)
        style = overlay.get("style", style)
        tooltip = overlay.get("tooltip", tooltip)
        if overlay.get("penwidth") is not None:
            penwidth = str(overlay["penwidth"])

    cluster.attr(
        label=f"{short_name} | {' | '.join(label_parts)}",
        labelloc="t",
        labeljust="l",
        fontname="Helvetica",
        fontsize="11",
        fontcolor=fontcolor,
        color=color,
        fillcolor=fillcolor,
        style=style,
        penwidth=penwidth,
        margin="18",
        tooltip=tooltip,
    )


def _add_anchor_node(container: graphviz.Digraph, node_id: str) -> None:
    """Create an invisible point used as a stable edge anchor inside a cluster."""
    container.node(
        node_id,
        label="",
        shape="point",
        width="0.01",
        height="0.01",
        style="invis",
    )


def _is_standalone_module(mod: dict) -> bool:
    """Detect utility modules not on the main data path.

    Standalone modules are non-container leaves with no dimensional IO
    attributes and zero learnable parameters (e.g. RunningNorm, Dropout).
    """
    if mod.get("children"):
        return False
    if mod.get("params", 0) > 0:
        return False
    attrs = mod.get("attributes", {})
    return not any(
        k in attrs
        for k in ("in_features", "out_features", "in_channels", "out_channels",
                   "num_embeddings", "embedding_dim", "num_features")
    )


def _partition_children(
    children: list[str],
    rendered_children: list[_RenderEndpoints],
    modules_by_path: dict[str, dict],
) -> tuple[list[tuple[str, _RenderEndpoints]], list[tuple[str, _RenderEndpoints]]]:
    """Split (child_path, endpoints) pairs into (branches, standalone)."""
    branches: list[tuple[str, _RenderEndpoints]] = []
    standalone: list[tuple[str, _RenderEndpoints]] = []
    for child_path, ep in zip(children, rendered_children):
        mod = modules_by_path.get(child_path)
        if mod is not None and _is_standalone_module(mod):
            standalone.append((child_path, ep))
        else:
            branches.append((child_path, ep))
    return branches, standalone


def _add_fork_join_node(graph: graphviz.Digraph, node_id: str) -> None:
    """Add a small filled circle used as a visual fork/join indicator."""
    graph.node(
        node_id,
        label="",
        shape="circle",
        width="0.12",
        height="0.12",
        fixedsize="true",
        style="filled",
        fillcolor=_PARALLEL_EDGE,
        color=_PARALLEL_EDGE,
    )


def _connect_root_children(
    dot: graphviz.Digraph,
    root_path: str,
    root: dict,
    root_children: list[str],
    rendered_children: list[_RenderEndpoints],
    modules_by_path: dict[str, dict],
) -> None:
    """Connect the root summary card to its first-level modules."""
    if not root_children or not rendered_children:
        return

    if _is_sequential_container(root, root_children):
        _connect_node_to_item(dot, root_path, rendered_children[0], color=_SEQUENTIAL_EDGE)
        for left, right in zip(rendered_children, rendered_children[1:]):
            _connect_items(dot, left, right, color=_SEQUENTIAL_EDGE)
        return

    # Mixed / parallel layout: separate utility modules from data-flow branches.
    branches, standalone = _partition_children(
        root_children, rendered_children, modules_by_path,
    )

    if len(branches) >= 2:
        # Visual fork/join for parallel branches.
        fork_id = _anchor_id(root_path, "fork")
        join_id = _anchor_id(root_path, "join")
        _add_fork_join_node(dot, fork_id)
        _add_fork_join_node(dot, join_id)
        dot.edge(root_path, fork_id, color=_SEQUENTIAL_EDGE, arrowhead="none")
        for _, ep in branches:
            _connect_node_to_item(dot, fork_id, ep, color=_PARALLEL_EDGE)
            _connect_item_to_node(dot, ep, join_id, color=_PARALLEL_EDGE)
        # Same-rank hint for branch entries.
        with dot.subgraph() as s:
            s.attr(rank="same")
            for _, ep in branches:
                s.node(ep.entry_id)
    elif branches:
        _connect_node_to_item(dot, root_path, branches[0][1], color=_SEQUENTIAL_EDGE)

    # Standalone / utility modules shown with dashed edges off to the side.
    for _, ep in standalone:
        dot.edge(
            root_path, ep.entry_id,
            style="dashed", color=_DETACHED_EDGE, arrowsize="0.5",
        )


def _connect_container_children(
    dot: graphviz.Digraph,
    mod: dict,
    children: list[str],
    rendered_children: list[_RenderEndpoints],
    entry_id: str,
    exit_id: str,
    modules_by_path: dict[str, dict],
) -> None:
    """Connect internal container flow using anchors and cluster-aware edges."""
    if not rendered_children:
        return

    if _is_sequential_container(mod, children):
        _connect_node_to_item(dot, entry_id, rendered_children[0], color=_SEQUENTIAL_EDGE)
        for left, right in zip(rendered_children, rendered_children[1:]):
            _connect_items(dot, left, right, color=_SEQUENTIAL_EDGE)
        _connect_item_to_node(dot, rendered_children[-1], exit_id, color=_SEQUENTIAL_EDGE)
        return

    # Parallel container: separate branches from standalone utility modules.
    branches, standalone = _partition_children(
        children, rendered_children, modules_by_path,
    )

    for _, ep in branches:
        _connect_node_to_item(dot, entry_id, ep, color=_PARALLEL_EDGE)
        _connect_item_to_node(dot, ep, exit_id, color=_PARALLEL_EDGE)

    # Same-rank hint for parallel branch entries.
    if len(branches) >= 2:
        with dot.subgraph() as s:
            s.attr(rank="same")
            for _, ep in branches:
                s.node(ep.entry_id)

    # Standalone utility modules get dashed edges, no exit connection.
    for _, ep in standalone:
        dot.edge(
            entry_id, ep.entry_id,
            style="dashed", color=_DETACHED_EDGE, arrowsize="0.5",
        )


def _connect_items(
    dot: graphviz.Digraph,
    left: _RenderEndpoints,
    right: _RenderEndpoints,
    *,
    color: str,
) -> None:
    """Connect two rendered modules, respecting cluster boundaries."""
    attrs = {"color": color}
    if left.cluster_id:
        attrs["ltail"] = left.cluster_id
    if right.cluster_id:
        attrs["lhead"] = right.cluster_id
    dot.edge(left.exit_id, right.entry_id, **attrs)


def _connect_node_to_item(
    dot: graphviz.Digraph,
    node_id: str,
    item: _RenderEndpoints,
    *,
    color: str,
) -> None:
    """Connect a concrete node to a rendered item."""
    attrs = {"color": color}
    if item.cluster_id:
        attrs["lhead"] = item.cluster_id
    dot.edge(node_id, item.entry_id, **attrs)


def _connect_item_to_node(
    dot: graphviz.Digraph,
    item: _RenderEndpoints,
    node_id: str,
    *,
    color: str,
) -> None:
    """Connect a rendered item to a concrete node."""
    attrs = {"color": color}
    if item.cluster_id:
        attrs["ltail"] = item.cluster_id
    dot.edge(item.exit_id, node_id, **attrs)


def _is_sequential_container(mod: dict, children: list[str]) -> bool:
    """Decide whether direct children should be shown as an ordered chain."""
    if len(children) < 2:
        return False

    mod_type = mod.get("type", "")
    if mod_type in {"Sequential", "ModuleList"}:
        return True
    if mod_type == "ModuleDict":
        return False

    # Fallback heuristic for custom ordered containers with numeric child names.
    return _children_look_indexed(children)


def _children_look_indexed(children: list[str]) -> bool:
    """Heuristic: many ordered containers use numeric child names."""
    if not children:
        return False
    names = [c.split(".")[-1] for c in children]
    numeric = sum(1 for n in names if n.isdigit())
    return numeric >= max(2, len(names) // 2)


def _count_subtree_params(
    path: str,
    modules_by_path: dict[str, dict],
    cache: dict[str, int],
) -> int:
    """Return parameter count for a module including all descendants."""
    if path in cache:
        return cache[path]

    mod = modules_by_path[path]
    total = mod["params"]
    for child in mod.get("children", []):
        if child in modules_by_path:
            total += _count_subtree_params(child, modules_by_path, cache)

    cache[path] = total
    return total


def _node_heading(path: str, mod: dict) -> tuple[str, str | None]:
    """Return the title/subtitle pair for a node card."""
    short_name = path.split(".")[-1]
    type_name = mod["type"]

    if short_name.isdigit():
        return type_name, f"Layer {short_name}"

    if short_name == type_name:
        return short_name, None

    return short_name, type_name


def _attribute_summary(attrs: dict[str, Any]) -> str | None:
    """Return a short attribute summary suitable for a node body."""
    parts: list[str] = []

    if "kernel_size" in attrs:
        parts.append(f"k {_format_dim(attrs['kernel_size'])}")
    if "stride" in attrs and not _matches_any(attrs["stride"], 1, (1, 1)):
        parts.append(f"s {_format_dim(attrs['stride'])}")
    if "padding" in attrs and not _matches_any(attrs["padding"], 0, (0, 0)):
        parts.append(f"p {_format_dim(attrs['padding'])}")
    if "num_features" in attrs:
        parts.append(f"{attrs['num_features']} features")
    if "normalized_shape" in attrs:
        parts.append(f"shape {_format_dim(attrs['normalized_shape'])}")
    if "num_heads" in attrs:
        parts.append(f"{attrs['num_heads']} heads")
    if "output_size" in attrs:
        parts.append(f"out {_format_dim(attrs['output_size'])}")
    if "p" in attrs:
        parts.append(f"drop {attrs['p']}")
    if "inplace" in attrs and attrs["inplace"]:
        parts.append("inplace")
    if "num_embeddings" in attrs and "embedding_dim" in attrs:
        parts.append(f"vocab {attrs['num_embeddings']}")
        parts.append(f"dim {attrs['embedding_dim']}")

    if not parts:
        return None
    return " | ".join(parts[:3])


def _io_summary(in_dim: str | None, out_dim: str | None) -> str | None:
    """Format an in/out summary, suppressing noisy heterogeneous values."""
    if in_dim and out_dim and _is_clean_dim(in_dim) and _is_clean_dim(out_dim):
        return f"{in_dim} -> {out_dim}"
    if in_dim and _is_clean_dim(in_dim):
        return f"in {in_dim}"
    if out_dim and _is_clean_dim(out_dim):
        return f"out {out_dim}"
    return None


def _is_clean_dim(value: str) -> bool:
    """Return whether a dimension summary is concise enough to display inline."""
    return value != "multi" and " | " not in value


def _matches_any(value: Any, *candidates: Any) -> bool:
    """Return whether a value equals any candidate without requiring hashing."""
    return any(value == candidate for candidate in candidates)


def _card_label(
    *,
    title: str,
    subtitle: str | None,
    body_lines: list[str],
    accent: str,
    border: str,
    border_width: str,
) -> str:
    """Build an HTML-like Graphviz label for a compact module card."""
    subtitle_html = ""
    if subtitle:
        subtitle_html = (
            f'<BR/><FONT POINT-SIZE="9" COLOR="#5e4a4e">{escape(subtitle)}</FONT>'
        )

    body_parts = []
    for idx, line in enumerate(line for line in body_lines if line):
        if idx == 0:
            body_parts.append(f'<FONT POINT-SIZE="10">{escape(line)}</FONT>')
        else:
            body_parts.append(f'<FONT POINT-SIZE="9" COLOR="#6e5a5e">{escape(line)}</FONT>')

    body_html = "<BR/>".join(body_parts)

    return (
        f"<<TABLE BORDER=\"{border_width}\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\" "
        f"COLOR=\"{border}\" BGCOLOR=\"#faf6f6\">"
        f"<TR><TD BGCOLOR=\"{accent}\" ALIGN=\"LEFT\" CELLPADDING=\"7\">"
        f"<FONT POINT-SIZE=\"12\"><B>{escape(title)}</B></FONT>{subtitle_html}"
        "</TD></TR>"
        f"<TR><TD ALIGN=\"LEFT\" CELLPADDING=\"7\">{body_html}</TD></TR>"
        "</TABLE>>"
    )


def _module_tooltip(
    path: str,
    mod: dict,
    in_dim: str | None,
    out_dim: str | None,
) -> str:
    """Return hover text with the full module context."""
    parts = [
        path,
        mod.get("type_full") or mod["type"],
        f"params={mod['params']:,}",
    ]

    flow = _io_summary(in_dim, out_dim)
    if flow:
        parts.append(flow)

    attrs = mod.get("attributes", {})
    if attrs:
        parts.append(", ".join(f"{k}={_format_dim(v)}" for k, v in attrs.items()))

    return _tooltip_text(parts)


def _tooltip_text(parts: list[str | None]) -> str:
    """Join tooltip parts while skipping empty values."""
    return " | ".join(part for part in parts if part)


def _apply_overlay_style(
    *,
    accent: str,
    border: str,
    tooltip: str,
    body_lines: list[str],
    border_width: str,
    overlay: dict[str, Any] | None,
) -> tuple[str, str, str, list[str], str]:
    """Apply optional overlay styling and extra lines to a card."""
    if not overlay:
        return accent, border, tooltip, body_lines, border_width

    accent = overlay.get("fillcolor", accent)
    border = overlay.get("color", border)
    tooltip = overlay.get("tooltip", tooltip)
    if overlay.get("penwidth") is not None:
        border_width = str(max(1, round(float(overlay["penwidth"]))))
    extra_lines = [str(line) for line in overlay.get("extra_lines", []) if line]
    return accent, border, tooltip, [*body_lines, *extra_lines], border_width


def _cluster_id(path: str) -> str:
    """Return a Graphviz-safe cluster identifier."""
    return f"cluster_{_stable_token(path)}"


def _anchor_id(path: str, kind: str) -> str:
    """Return a stable synthetic node identifier."""
    return f"__gh_{kind}_{_stable_token(path)}"


def _stable_token(value: str) -> str:
    """Return a short stable token for synthetic Graphviz ids."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
