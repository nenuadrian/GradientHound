from __future__ import annotations

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
    "conv": "#D4E6F1",
    "linear": "#D5F5E3",
    "norm": "#FEF9E7",
    "activation": "#FDEBD0",
    "pool": "#E8DAEF",
    "dropout": "#F2F3F4",
    "embedding": "#D6EAF8",
    "default": "#FFFFFF",
}


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


def _format_attrs(attrs: dict[str, Any]) -> str:
    """Format attributes into a compact label string."""
    parts = []
    if "in_channels" in attrs and "out_channels" in attrs:
        ks = attrs.get("kernel_size", "")
        if isinstance(ks, (tuple, list)):
            ks = "x".join(str(k) for k in ks)
        parts.append(f"{attrs['in_channels']}->{attrs['out_channels']}, k={ks}")
    elif "in_features" in attrs and "out_features" in attrs:
        parts.append(f"{attrs['in_features']}->{attrs['out_features']}")
    elif "num_features" in attrs:
        parts.append(f"features={attrs['num_features']}")
    if "p" in attrs:
        parts.append(f"p={attrs['p']}")
    if "output_size" in attrs:
        parts.append(f"out={attrs['output_size']}")
    return ", ".join(parts)


def _format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def render_graphviz(graph_data: dict) -> graphviz.Digraph:
    """Build a Graphviz Digraph from the extracted model graph data."""
    dot = graphviz.Digraph(
        name=graph_data["name"],
        graph_attr={
            "rankdir": "TB",
            "fontname": "Helvetica",
            "bgcolor": "transparent",
            "pad": "0.5",
        },
        node_attr={
            "fontname": "Helvetica",
            "fontsize": "10",
            "shape": "box",
            "style": "filled,rounded",
            "penwidth": "1.2",
        },
        edge_attr={
            "arrowsize": "0.7",
            "color": "#666666",
        },
    )

    modules = graph_data.get("modules", [])
    modules_by_path = {m["path"]: m for m in modules}
    root_path = graph_data["name"]
    root = modules_by_path.get(root_path)

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
    visible_paths = _select_visible_paths(root_path, modules_by_path, max_nodes=400)
    io_cache: dict[str, tuple[str | None, str | None]] = {}

    for path in visible_paths:
        mod = modules_by_path[path]
        _add_module_node(dot, path, mod, root_path, modules_by_path, io_cache)

    # Connect edges with structural semantics:
    # sequential containers are chained, parallel containers fan out.
    _add_structure_edges(dot, visible_paths, modules_by_path)

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


def _add_module_node(
    dot: graphviz.Digraph,
    path: str,
    mod: dict,
    root_path: str,
    modules_by_path: dict[str, dict],
    io_cache: dict[str, tuple[str | None, str | None]],
) -> None:
    """Add a styled node for a module path."""
    short_name = path if path == root_path else path.split(".")[-1]
    children = mod.get("children", [])
    kind = _node_kind(mod, children)
    in_dim, out_dim = _infer_node_io(path, modules_by_path, io_cache)

    label_parts = [
        f"In: {in_dim or '-'}",
        f"{short_name} | {kind}",
        f"Params: {mod['params']:,}",
        f"Out: {out_dim or '-'}",
    ]
    label = "\\n".join(label_parts)

    if path == root_path:
        dot.node(path, label=label, fillcolor="#D6EAF8", shape="box", style="filled,rounded")
        return

    if mod["is_container"]:
        dot.node(
            path,
            label=label,
            fillcolor="#F8F9FA",
            color="#9AA0A6",
            shape="box",
            style="filled,rounded,dashed",
        )
        return

    color = _COLORS.get(_module_category(mod["type"]), _COLORS["default"])
    attr_str = _format_attrs(mod.get("attributes", {}))
    final_label = label
    if attr_str:
        final_label = f"{label}\\n{attr_str}"
    dot.node(path, label=final_label, fillcolor=color)


def _node_kind(mod: dict, children: list[str]) -> str:
    """Return a human-readable module kind for labeling."""
    if mod.get("is_container"):
        return "sequential" if _is_sequential_container(mod, children) else "parallel"
    return "module"


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


def _add_structure_edges(
    dot: graphviz.Digraph,
    visible_paths: list[str],
    modules_by_path: dict[str, dict],
) -> None:
    """Connect children as sequential chains or parallel fan-outs."""
    visible = set(visible_paths)

    for path in visible_paths:
        mod = modules_by_path[path]
        children = [c for c in mod.get("children", []) if c in visible]
        if not children:
            continue

        if _is_sequential_container(mod, children):
            # Sequential: parent -> first child, then child_i -> child_{i+1}
            dot.edge(path, children[0], color="#4E79A7")
            for i in range(len(children) - 1):
                dot.edge(children[i], children[i + 1], color="#4E79A7")
            continue

        # Parallel: parent fans out to all direct children.
        for child in children:
            dot.edge(path, child, color="#7A7A7A")


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


