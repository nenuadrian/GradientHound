"""Cytoscape graph element builders (FX graph + module tree)."""
from __future__ import annotations

from ._constants import MODULE_COLORS
from ._helpers import fmt_num, module_category, short_target


def build_fx_elements(model_data: dict) -> list[dict]:
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
            short = short_target(n.get("target", ""))
            label = short
            if mod_path:
                label += f"\n[{mod_path}]"
            elif shape_str:
                label += f"\n{shape_str}"
            cat = module_category(mod_type) if mod_type else module_category(short)
            elements.append({"data": {"id": node_id, "label": label, **n}, "classes": f"mod-{cat}"})

    for e in edges:
        if e["from"] in skip_nodes or e["to"] in skip_nodes:
            continue
        if e["from"] in node_ids and e["to"] in node_ids:
            elements.append({"data": {"source": e["from"], "target": e["to"]}})

    return elements


def _is_sequential_tree(mod: dict, children: list[str]) -> bool:
    """Check if children should be displayed as a sequential chain."""
    if len(children) < 2:
        return False
    mod_type = mod.get("type", "")
    if mod_type in {"Sequential", "ModuleList"}:
        return True
    if mod_type == "ModuleDict":
        return False
    names = [c.split(".")[-1] for c in children]
    numeric = sum(1 for n in names if n.isdigit())
    return numeric >= max(2, len(names) // 2)


def _is_standalone_leaf(mod: dict) -> bool:
    """Detect utility modules not on the main data path."""
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


def _tree_first_leaf(path: str, modules_by_path: dict) -> str | None:
    """Return the first leaf in a subtree (DFS left)."""
    mod = modules_by_path.get(path)
    if mod is None:
        return None
    children = [c for c in mod.get("children", []) if c in modules_by_path]
    if not children:
        return path
    return _tree_first_leaf(children[0], modules_by_path)


def _tree_last_leaf(path: str, modules_by_path: dict) -> str | None:
    """Return the last leaf in a subtree (DFS right)."""
    mod = modules_by_path.get(path)
    if mod is None:
        return None
    children = [c for c in mod.get("children", []) if c in modules_by_path]
    if not children:
        return path
    return _tree_last_leaf(children[-1], modules_by_path)


def _collect_tree_edges(
    root_path: str,
    modules_by_path: dict,
    leaf_set: set[str],
) -> list[tuple[str, str, str]]:
    """Walk the module tree and return (src, tgt, edge_class) triples."""
    edges: list[tuple[str, str, str]] = []

    def visit(path: str) -> None:
        mod = modules_by_path.get(path)
        if mod is None:
            return
        children = [c for c in mod.get("children", []) if c in modules_by_path]
        if not children:
            return

        for child in children:
            visit(child)

        if _is_sequential_tree(mod, children):
            for i in range(len(children) - 1):
                src = _tree_last_leaf(children[i], modules_by_path)
                tgt = _tree_first_leaf(children[i + 1], modules_by_path)
                if src and tgt and src in leaf_set and tgt in leaf_set:
                    edges.append((src, tgt, ""))

    visit(root_path)
    return edges


def build_module_tree_elements(model_data: dict) -> list[dict]:
    mt = model_data.get("module_tree", {})
    modules = mt.get("modules", [])
    if not modules:
        return []

    modules_by_path = {m["path"]: m for m in modules}
    root_path = mt.get("name", "")

    elements: list[dict] = []
    leaf_set: set[str] = set()

    for m in modules:
        if not m.get("is_leaf"):
            continue
        path = m["path"]
        leaf_set.add(path)
        cat = module_category(m["type"])
        label = f"{m['type']}\n[{path}]"
        if m.get("params"):
            label += f"\n{fmt_num(m['params'])}"
        elements.append({
            "data": {"id": path, "label": label,
                     "op": "module", "nn_module": path,
                     "nn_module_type": m.get("type_full", m["type"]),
                     "output_shape": None, "output_dtype": None,
                     "target": m["type"], "args": [],
                     "params": m.get("params", 0)},
            "classes": f"mod-{cat}",
        })

    edges = _collect_tree_edges(root_path, modules_by_path, leaf_set)

    connected: set[str] = set()
    for src, tgt, _cls in edges:
        connected.add(src)
        connected.add(tgt)
    for el in elements:
        if "source" in el["data"]:
            continue
        path = el["data"]["id"]
        if path not in connected and _is_standalone_leaf(modules_by_path.get(path, {})):
            el["classes"] += " standalone"

    for src, tgt, cls in edges:
        edge_data: dict = {"source": src, "target": tgt}
        elements.append({"data": edge_data, "classes": cls} if cls else {"data": edge_data})

    return elements
