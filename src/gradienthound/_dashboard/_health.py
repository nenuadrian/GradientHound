"""Weight and module health assessment."""
from __future__ import annotations

from ._constants import HEALTH_SORT
from ._graph import (
    _collect_tree_edges,
    _is_standalone_leaf,
    build_module_tree_elements,
)
from ._helpers import fmt_num, module_category


def weight_health(stat: dict) -> tuple[str, str]:
    """Assess health from a checkpoint weight_stats entry.

    Returns ``(state, reason)`` where state is one of
    *healthy*, *warning*, *critical*, or *neutral*.
    """
    near_zero = stat.get("near_zero_pct")
    eff_rank = stat.get("effective_rank")
    max_rank = stat.get("max_rank")
    cond = stat.get("condition_number")
    kurtosis = stat.get("kurtosis")

    if near_zero is not None and near_zero >= 99:
        return "critical", "Weights almost entirely near zero"
    if near_zero is not None and near_zero >= 95:
        return "warning", "High fraction of near-zero weights"

    if eff_rank is not None and max_rank:
        rank_pct = 100.0 * eff_rank / max(max_rank, 1)
        if rank_pct < 10:
            return "critical", f"Rank collapsed ({rank_pct:.0f}%)"
        if rank_pct < 25:
            return "warning", f"Low rank utilisation ({rank_pct:.0f}%)"

    if cond is not None and cond > 1e6:
        return "critical", f"Extreme conditioning ({cond:.1e})"
    if cond is not None and cond > 1e4:
        return "warning", f"Poor conditioning ({cond:.1e})"

    if kurtosis is not None and abs(kurtosis) > 50:
        return "warning", f"Extreme kurtosis ({kurtosis:.1f})"

    if stat.get("norm_l2") is not None:
        return "healthy", "Looks healthy"

    return "neutral", "No data"


def module_health_from_params(
    param_stats: list[dict],
) -> tuple[str, str]:
    """Aggregate parameter-level health to a single module health.

    Worst parameter wins.
    """
    worst_state = "neutral"
    worst_reason = "No data"
    for stat in param_stats:
        state, reason = weight_health(stat)
        if HEALTH_SORT.get(state, 3) < HEALTH_SORT.get(worst_state, 3):
            worst_state = state
            worst_reason = reason
    return worst_state, worst_reason


def build_health_elements(
    model_data: dict,
    stats: list[dict],
) -> list[dict]:
    """Build Cytoscape elements with health-colored nodes from checkpoint stats."""
    mt = model_data.get("module_tree", {})
    modules = mt.get("modules", [])
    if not modules:
        return []

    modules_by_path = {m["path"]: m for m in modules}
    root_path = mt.get("name", "")

    # Map parameter stats to their parent module path
    module_stats: dict[str, list[dict]] = {}
    for stat in stats:
        param_name = stat["layer"]
        if "." in param_name:
            mod_path = param_name.rsplit(".", 1)[0]
        else:
            mod_path = root_path
        module_stats.setdefault(mod_path, []).append(stat)

    elements: list[dict] = []
    leaf_set: set[str] = set()

    for m in modules:
        if not m.get("is_leaf"):
            continue
        path = m["path"]
        leaf_set.add(path)
        cat = module_category(m["type"])

        param_stats_list = module_stats.get(path, [])
        state, reason = module_health_from_params(param_stats_list)

        label = f"{m['type']}\n[{path}]"
        if m.get("params"):
            label += f"\n{fmt_num(m['params'])}"
        if state != "neutral":
            label += f"\n{state.upper()}"

        elements.append({
            "data": {
                "id": path, "label": label,
                "op": "module", "nn_module": path,
                "nn_module_type": m.get("type_full", m["type"]),
                "output_shape": None, "output_dtype": None,
                "target": m["type"], "args": [],
                "params": m.get("params", 0),
            },
            "classes": f"mod-{cat} health-{state}",
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
