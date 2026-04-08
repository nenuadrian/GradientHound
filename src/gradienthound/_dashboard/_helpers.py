"""Shared helper / utility functions for the dashboard."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html, dcc

from ._constants import PLOTLY_TEMPLATE


def plotly_layout(**overrides) -> dict:
    base = {"template": PLOTLY_TEMPLATE, "margin": {"l": 60, "r": 20, "t": 40, "b": 60}}
    base.update(overrides)
    return base


def fmt_num(n: int | float) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def module_category(type_name: str) -> str:
    lower = type_name.lower()
    for key in ("conv", "linear", "norm", "pool", "dropout", "embed"):
        if key in lower:
            return key
    if any(k in lower for k in ("relu", "gelu", "silu", "sigmoid", "tanh", "softmax", "activation")):
        return "activation"
    return "default"


def short_target(target: str) -> str:
    return target.replace("torch._ops.", "").replace(".default", "").replace(".Tensor", "")


def short_layer(name: str, max_parts: int = 3) -> str:
    parts = name.split(".")
    if len(parts) > max_parts:
        parts = parts[-max_parts:]
    return ".".join(parts)


def placeholder_page(title: str, description: str):
    return dbc.Container([
        html.H2(title, className="mt-3 mb-1"),
        html.P(description, className="text-muted mb-4"),
        dbc.Card(dbc.CardBody([
            html.H5("Coming soon", className="card-title"),
            html.P("This page will be available once data is loaded.", className="text-muted"),
        ])),
    ], fluid=True)


def node_detail_panel(node_data: dict):
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
        _item("Target", short_target(target))
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


def fmt_bytes(n: int | float) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if n != int(n) else f"{int(n)} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def split_model_data_for_submodel(
    model_data: dict, stats: list[dict], sub_model: str,
) -> tuple[dict, list[dict]]:
    """Extract model_data and weight stats for a single sub-model."""
    mt = model_data.get("module_tree", {})
    sm_modules = [
        m for m in mt.get("modules", [])
        if m["path"].startswith(sub_model + ".")
    ]
    sm_root_path = f"{sub_model}.{sub_model}"
    sm_root = next((m for m in sm_modules if m["path"] == sm_root_path), None)
    sm_model_data = dict(model_data)
    sm_model_data["module_tree"] = {
        "name": sm_root_path if sm_root else sub_model,
        "modules": sm_modules,
    }
    sm_stats = [s for s in stats if s["layer"].startswith(sub_model + ".")]
    return sm_model_data, sm_stats


def column_summary(table: list[list[float | None]], n_cols: int):
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


def summary_chart(checkpoint_names, means, mins, maxs, title, y_label):
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


_MAX_CHART_POINTS = 5000


def lttb_downsample(xs: list, ys: list, threshold: int) -> tuple[list, list]:
    """Largest-Triangle-Three-Buckets downsampling."""
    n = len(xs)
    if n <= threshold:
        return xs, ys
    out_x, out_y = [xs[0]], [ys[0]]
    bucket_size = (n - 2) / (threshold - 2)
    a_idx = 0
    for i in range(1, threshold - 1):
        b_start = int((i - 1) * bucket_size) + 1
        b_end = int(i * bucket_size) + 1
        c_start = int(i * bucket_size) + 1
        c_end = int((i + 1) * bucket_size) + 1
        if c_end > n - 1:
            c_end = n - 1
        avg_x = sum(xs[c_start:c_end + 1]) / max(1, c_end - c_start + 1)
        avg_y = sum(ys[c_start:c_end + 1]) / max(1, c_end - c_start + 1)
        best_idx = b_start
        best_area = -1.0
        for j in range(b_start, min(b_end + 1, n)):
            area = abs(
                (xs[a_idx] - avg_x) * (ys[j] - ys[a_idx])
                - (xs[a_idx] - xs[j]) * (avg_y - ys[a_idx])
            )
            if area > best_area:
                best_area = area
                best_idx = j
        out_x.append(xs[best_idx])
        out_y.append(ys[best_idx])
        a_idx = best_idx
    out_x.append(xs[-1])
    out_y.append(ys[-1])
    return out_x, out_y


def compute_checkpoint_change_tables(
    snapshots: list[dict],
) -> tuple[list[str], list[str], list[list[float | None]], list[list[float | None]]]:
    """Build per-layer absolute and relative L2 norm change tables.

    Returned tables are shaped ``layers x checkpoints``. The first checkpoint
    column is ``None`` because there is no prior checkpoint for comparison.
    """
    checkpoint_names = [snap["name"] for snap in snapshots]
    all_layers = [s["layer"] for s in snapshots[0]["weight_stats"]]

    lookup = {
        snap["name"]: {s["layer"]: s for s in snap["weight_stats"]}
        for snap in snapshots
    }

    diff_table: list[list[float | None]] = []
    rel_table: list[list[float | None]] = []

    for layer in all_layers:
        diff_row: list[float | None] = [None]
        rel_row: list[float | None] = [None]

        for i in range(1, len(snapshots)):
            prev_name = snapshots[i - 1]["name"]
            curr_name = snapshots[i]["name"]
            prev_stat = lookup[prev_name].get(layer)
            curr_stat = lookup[curr_name].get(layer)

            if prev_stat and curr_stat:
                delta = abs(curr_stat["norm_l2"] - prev_stat["norm_l2"])
                diff_row.append(delta)
                if prev_stat["norm_l2"] > 1e-12:
                    rel_row.append(delta / prev_stat["norm_l2"] * 100)
                else:
                    rel_row.append(None)
            else:
                diff_row.append(None)
                rel_row.append(None)

        diff_table.append(diff_row)
        rel_table.append(rel_row)

    return checkpoint_names, all_layers, diff_table, rel_table


def compute_effective_rank_table(
    snapshots: list[dict],
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Build per-layer effective rank values shaped ``layers x checkpoints``."""
    checkpoint_names = [snap["name"] for snap in snapshots]

    all_layers = []
    seen = set()
    for snap in snapshots:
        for stat in snap["weight_stats"]:
            if "effective_rank" in stat and stat["layer"] not in seen:
                all_layers.append(stat["layer"])
                seen.add(stat["layer"])

    rank_table: list[list[float | None]] = []
    for layer in all_layers:
        row: list[float | None] = []
        for snap in snapshots:
            stat = next((s for s in snap["weight_stats"] if s["layer"] == layer), None)
            row.append(stat.get("effective_rank") if stat and "effective_rank" in stat else None)
        rank_table.append(row)

    return checkpoint_names, all_layers, rank_table


def compute_distribution_stats_table(
    snapshots: list[dict],
    max_layers: int = 50,
) -> tuple[list[str], list[str], list[list[str | None]]]:
    """Build per-layer distribution summary strings shaped ``layers x checkpoints``."""
    checkpoint_names = [snap["name"] for snap in snapshots]
    all_layers = []
    seen = set()
    for snap in snapshots:
        for stat in snap["weight_stats"]:
            layer = stat["layer"]
            if layer not in seen:
                all_layers.append(layer)
                seen.add(layer)

    selected_layers = all_layers[:max_layers]
    table: list[list[str | None]] = []

    for layer in selected_layers:
        row: list[str | None] = []
        for snap in snapshots:
            stat = next((s for s in snap["weight_stats"] if s["layer"] == layer), None)
            if stat:
                row.append(
                    f"mu={stat['mean']:.4g}  sigma={stat['std']:.4g}  k={stat.get('kurtosis', 0):.2f}"
                )
            else:
                row.append(None)
        table.append(row)

    return checkpoint_names, selected_layers, table


def compute_scalar_metric_tables(
    snapshots: list[dict],
    metric_keys: list[str],
    *,
    max_layers: int = 50,
) -> tuple[list[str], list[str], dict[str, list[list[float | None]]]]:
    """Build per-layer scalar metric tables shaped ``layers x checkpoints``.

    Returns ``(checkpoint_names, selected_layers, metric_tables)`` where
    ``metric_tables[key]`` is a table of numeric values for the given metric.
    Layers are included if they contain at least one requested metric value.
    """
    checkpoint_names = [snap["name"] for snap in snapshots]
    if not snapshots or not metric_keys:
        return checkpoint_names, [], {}

    lookup = {
        snap["name"]: {s["layer"]: s for s in snap.get("weight_stats", [])}
        for snap in snapshots
    }

    all_layers: list[str] = []
    seen: set[str] = set()
    for snap in snapshots:
        for stat in snap.get("weight_stats", []):
            layer = stat.get("layer")
            if not isinstance(layer, str) or layer in seen:
                continue
            if any(isinstance(stat.get(k), (int, float)) for k in metric_keys):
                all_layers.append(layer)
                seen.add(layer)

    selected_layers = all_layers[:max_layers]
    metric_tables: dict[str, list[list[float | None]]] = {k: [] for k in metric_keys}

    for layer in selected_layers:
        for metric_key in metric_keys:
            row: list[float | None] = []
            for snap in snapshots:
                stat = lookup[snap["name"]].get(layer)
                val = stat.get(metric_key) if stat else None
                row.append(float(val) if isinstance(val, (int, float)) else None)
            metric_tables[metric_key].append(row)

    # Drop metrics with no data across all selected layers/checkpoints.
    metric_tables = {
        key: table
        for key, table in metric_tables.items()
        if any(v is not None for row in table for v in row)
    }

    return checkpoint_names, selected_layers, metric_tables


def compute_optimizer_summary_table(
    snapshots: list[dict],
) -> tuple[list[str], list[dict]]:
    """Build a summary of optimizer states found across checkpoints.

    Returns ``(checkpoint_names, optimizer_rows)`` where each optimizer_row is
    a dict with ``name``, ``type``, and a list of per-checkpoint cell dicts.
    """
    checkpoint_names = [snap["name"] for snap in snapshots]

    # Collect all optimizer names across checkpoints (preserve order)
    all_opt_names: list[str] = []
    seen: set[str] = set()
    for snap in snapshots:
        for opt in snap.get("optimizer_states", []):
            if opt["name"] not in seen:
                all_opt_names.append(opt["name"])
                seen.add(opt["name"])

    rows: list[dict] = []
    for opt_name in all_opt_names:
        cells: list[dict | None] = []
        opt_type = "?"
        for snap in snapshots:
            match = next(
                (o for o in snap.get("optimizer_states", []) if o["name"] == opt_name),
                None,
            )
            if match:
                opt_type = match["type"]
                cells.append(match)
            else:
                cells.append(None)
        rows.append({"name": opt_name, "type": opt_type, "cells": cells})

    return checkpoint_names, rows


def compute_optimizer_evolution_table(
    snapshots: list[dict],
) -> tuple[list[str], list[dict]]:
    """Build per-group metric evolution across checkpoints.

    Returns ``(checkpoint_names, group_rows)`` where each group_row has
    ``optimizer``, ``group_index``, and per-checkpoint metric dicts.
    """
    checkpoint_names = [snap["name"] for snap in snapshots]

    # Collect all (optimizer_name, group_index) pairs
    all_keys: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for snap in snapshots:
        for opt in snap.get("optimizer_states", []):
            for g in opt.get("groups", []):
                key = (opt["name"], g["group_index"])
                if key not in seen:
                    all_keys.append(key)
                    seen.add(key)

    rows: list[dict] = []
    for opt_name, gi in all_keys:
        cells: list[dict | None] = []
        for snap in snapshots:
            match = next(
                (o for o in snap.get("optimizer_states", []) if o["name"] == opt_name),
                None,
            )
            if match:
                grp = next(
                    (g for g in match.get("groups", []) if g["group_index"] == gi),
                    None,
                )
                cells.append(grp)
            else:
                cells.append(None)
        rows.append({
            "optimizer": opt_name,
            "group_index": gi,
            "cells": cells,
        })

    return checkpoint_names, rows


def compute_spectral_gap_table(
    snapshots: list[dict],
    *,
    top_k: int = 5,
    max_layers: int = 50,
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Build per-layer spectral gap ratio (sigma_1/sigma_2) table.

    Returns ``(checkpoint_names, layers, gap_table)`` shaped ``layers x checkpoints``.
    """
    checkpoint_names = [snap["name"] for snap in snapshots]
    all_layers: list[str] = []
    seen: set[str] = set()
    for snap in snapshots:
        for stat in snap["weight_stats"]:
            layer = stat["layer"]
            if layer not in seen and "singular_values" in stat:
                svs = stat["singular_values"]
                if isinstance(svs, list) and len(svs) >= 2:
                    all_layers.append(layer)
                    seen.add(layer)

    selected_layers = all_layers[:max_layers]
    gap_table: list[list[float | None]] = []

    for layer in selected_layers:
        row: list[float | None] = []
        for snap in snapshots:
            stat = next((s for s in snap["weight_stats"] if s["layer"] == layer), None)
            if stat and "singular_values" in stat:
                svs = stat["singular_values"]
                if isinstance(svs, list) and len(svs) >= 2:
                    row.append(svs[0] / max(svs[1], 1e-12))
                else:
                    row.append(None)
            else:
                row.append(None)
        gap_table.append(row)

    return checkpoint_names, selected_layers, gap_table


def compute_spectral_gap_ratios(
    singular_values: list[float],
    top_k: int = 5,
) -> list[float]:
    """Compute sigma_i / sigma_{i+1} for the top-k singular values."""
    n = min(top_k, len(singular_values) - 1)
    return [
        singular_values[i] / max(singular_values[i + 1], 1e-12)
        for i in range(n)
    ]


def compute_norm_velocity_table(
    snapshots: list[dict],
    max_layers: int = 50,
) -> tuple[list[str], list[str], list[list[float | None]], list[list[float | None]]]:
    """Build per-layer norm velocity and acceleration tables.

    Returns ``(checkpoint_names, layers, velocity_table, acceleration_table)``
    shaped ``layers x checkpoints``. First column of velocity is None (no
    predecessor), first two columns of acceleration are None.
    """
    checkpoint_names = [snap["name"] for snap in snapshots]
    all_layers: list[str] = []
    seen: set[str] = set()
    for snap in snapshots:
        for stat in snap["weight_stats"]:
            layer = stat["layer"]
            if layer not in seen:
                all_layers.append(layer)
                seen.add(layer)

    selected_layers = all_layers[:max_layers]
    lookup = {
        snap["name"]: {s["layer"]: s for s in snap["weight_stats"]}
        for snap in snapshots
    }

    velocity_table: list[list[float | None]] = []
    acceleration_table: list[list[float | None]] = []

    for layer in selected_layers:
        norms: list[float | None] = []
        for snap in snapshots:
            stat = lookup[snap["name"]].get(layer)
            norms.append(stat.get("norm_l2") if stat else None)

        vel_row: list[float | None] = [None]
        for i in range(1, len(norms)):
            if norms[i] is not None and norms[i - 1] is not None:
                vel_row.append(norms[i] - norms[i - 1])
            else:
                vel_row.append(None)

        acc_row: list[float | None] = [None, None] if len(norms) >= 2 else [None]
        for i in range(2, len(vel_row)):
            if vel_row[i] is not None and vel_row[i - 1] is not None:
                acc_row.append(vel_row[i] - vel_row[i - 1])
            else:
                acc_row.append(None)

        velocity_table.append(vel_row)
        acceleration_table.append(acc_row)

    return checkpoint_names, selected_layers, velocity_table, acceleration_table


def compute_convergence_scores(
    snapshots: list[dict],
    max_layers: int = 50,
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Compute a composite convergence score (0-100) per layer per checkpoint.

    Higher = more converged/stable. Combines cosine stability, rank stability,
    kurtosis stability, norm velocity, and mp_softrank trend.

    Returns ``(checkpoint_names, layers, score_table)`` shaped ``layers x checkpoints``.
    First checkpoint column is None.
    """
    checkpoint_names = [snap["name"] for snap in snapshots]
    if len(snapshots) < 2:
        return checkpoint_names, [], []

    all_layers: list[str] = []
    seen: set[str] = set()
    for snap in snapshots:
        for stat in snap["weight_stats"]:
            layer = stat["layer"]
            if layer not in seen:
                all_layers.append(layer)
                seen.add(layer)

    selected_layers = all_layers[:max_layers]
    lookup = {
        snap["name"]: {s["layer"]: s for s in snap["weight_stats"]}
        for snap in snapshots
    }

    score_table: list[list[float | None]] = []

    for layer in selected_layers:
        row: list[float | None] = [None]
        for i in range(1, len(snapshots)):
            curr = lookup[snapshots[i]["name"]].get(layer)
            prev = lookup[snapshots[i - 1]["name"]].get(layer)
            if not curr or not prev:
                row.append(None)
                continue

            components: list[float] = []

            # 1. Cosine stability (drift_cosine_prev: 1.0 = identical, already [−1, 1])
            cosine = curr.get("drift_cosine_prev")
            if cosine is not None:
                components.append(max(0.0, (cosine + 1.0) / 2.0))  # map [-1,1] -> [0,1]

            # 2. Rank stability: ratio of effective ranks close to 1.0
            er_curr = curr.get("effective_rank")
            er_prev = prev.get("effective_rank")
            if er_curr is not None and er_prev is not None and er_prev > 0:
                ratio = min(er_curr, er_prev) / max(er_curr, er_prev)
                components.append(ratio)

            # 3. Kurtosis stability: small absolute delta = stable
            k_curr = curr.get("kurtosis")
            k_prev = prev.get("kurtosis")
            if k_curr is not None and k_prev is not None:
                k_delta = abs(k_curr - k_prev)
                components.append(max(0.0, 1.0 - k_delta / 10.0))

            # 4. Norm velocity: small relative change = stable
            n_curr = curr.get("norm_l2", 0)
            n_prev = prev.get("norm_l2", 0)
            if n_prev > 1e-12:
                rel_change = abs(n_curr - n_prev) / n_prev
                components.append(max(0.0, 1.0 - min(rel_change * 10, 1.0)))

            # 5. MP softrank trend: decreasing = learning structure
            mp_curr = curr.get("mp_softrank")
            mp_prev = prev.get("mp_softrank")
            if mp_curr is not None and mp_prev is not None:
                # Lower mp_softrank = more structured = better
                components.append(max(0.0, 1.0 - mp_curr))

            if components:
                score = sum(components) / len(components) * 100
                row.append(round(score, 1))
            else:
                row.append(None)

        score_table.append(row)

    return checkpoint_names, selected_layers, score_table


def detect_training_phases(
    snapshots: list[dict],
) -> list[dict]:
    """Detect training phases from checkpoint transitions.

    Returns a list of phase dicts with:
    - ``start_idx``, ``end_idx``: checkpoint indices
    - ``start_name``, ``end_name``: checkpoint names
    - ``phase``: "learning", "plateau", or "instability"
    - ``intensity``: mean relative norm change in this phase
    """
    if len(snapshots) < 3:
        return []

    # Compute mean relative norm change per transition
    intensities: list[float] = []
    for i in range(1, len(snapshots)):
        rel_changes: list[float] = []
        for stat_curr in snapshots[i]["weight_stats"]:
            layer = stat_curr["layer"]
            stat_prev = next(
                (s for s in snapshots[i - 1]["weight_stats"] if s["layer"] == layer),
                None,
            )
            if stat_prev:
                n_prev = stat_prev.get("norm_l2", 0)
                n_curr = stat_curr.get("norm_l2", 0)
                if n_prev > 1e-12:
                    rel_changes.append(abs(n_curr - n_prev) / n_prev)
        intensities.append(sum(rel_changes) / max(len(rel_changes), 1))

    if not intensities:
        return []

    # Compute thresholds using percentiles
    sorted_i = sorted(intensities)
    n = len(sorted_i)
    p25 = sorted_i[max(0, n // 4)]
    p75 = sorted_i[min(n - 1, 3 * n // 4)]

    # Also compute per-transition variance across layers (for instability detection)
    variances: list[float] = []
    for i in range(1, len(snapshots)):
        rel_changes: list[float] = []
        for stat_curr in snapshots[i]["weight_stats"]:
            layer = stat_curr["layer"]
            stat_prev = next(
                (s for s in snapshots[i - 1]["weight_stats"] if s["layer"] == layer),
                None,
            )
            if stat_prev:
                n_prev = stat_prev.get("norm_l2", 0)
                n_curr = stat_curr.get("norm_l2", 0)
                if n_prev > 1e-12:
                    rel_changes.append(abs(n_curr - n_prev) / n_prev)
        if len(rel_changes) >= 2:
            mean_rc = sum(rel_changes) / len(rel_changes)
            var = sum((x - mean_rc) ** 2 for x in rel_changes) / len(rel_changes)
            variances.append(var)
        else:
            variances.append(0.0)

    var_sorted = sorted(variances)
    var_p75 = var_sorted[min(len(var_sorted) - 1, 3 * len(var_sorted) // 4)]

    # Classify each transition
    labels: list[str] = []
    for j, intensity in enumerate(intensities):
        if variances[j] > var_p75 * 2 and intensity > p25:
            labels.append("instability")
        elif intensity <= p25:
            labels.append("plateau")
        else:
            labels.append("learning")

    # Merge consecutive same-phase labels into phase segments
    phases: list[dict] = []
    if not labels:
        return []

    current_phase = labels[0]
    start = 0
    for j in range(1, len(labels)):
        if labels[j] != current_phase:
            phase_intensities = intensities[start:j]
            phases.append({
                "start_idx": start,
                "end_idx": j,
                "start_name": snapshots[start]["name"],
                "end_name": snapshots[j]["name"],
                "phase": current_phase,
                "intensity": sum(phase_intensities) / max(len(phase_intensities), 1),
            })
            current_phase = labels[j]
            start = j

    # Final segment
    phase_intensities = intensities[start:]
    phases.append({
        "start_idx": start,
        "end_idx": len(labels),
        "start_name": snapshots[start]["name"],
        "end_name": snapshots[len(labels)]["name"],
        "phase": current_phase,
        "intensity": sum(phase_intensities) / max(len(phase_intensities), 1),
    })

    return phases


def render_checkpoint_change_table(
    checkpoint_names: list[str],
    all_layers: list[str],
    values_table: list[list[float | None]],
    mode: str,
    selected_idx: int,
    formatter,
):
    if mode == "single":
        idx = max(0, min(selected_idx, len(checkpoint_names) - 1))
        header = [html.Th("Layer"), html.Th(checkpoint_names[idx])]
        rows = []
        for layer, row_values in zip(all_layers, values_table):
            val = row_values[idx]
            rows.append(html.Tr([
                html.Td(short_layer(layer), className="text-nowrap"),
                html.Td(formatter(val) if val is not None else "-", className="text-muted" if val is None else ""),
            ]))
    else:
        header = [html.Th("Layer")] + [html.Th(name) for name in checkpoint_names]
        rows = []
        for layer, row_values in zip(all_layers, values_table):
            row_cells = [html.Td(short_layer(layer), className="text-nowrap")]
            for val in row_values:
                row_cells.append(html.Td(formatter(val) if val is not None else "-", className="text-muted" if val is None else ""))
            rows.append(html.Tr(row_cells))

    return html.Div(
        dbc.Table([
            html.Thead(html.Tr(header)),
            html.Tbody(rows),
        ], bordered=True, hover=True, responsive=True, size="sm"),
        style={"maxHeight": "500px", "overflowY": "auto"},
    )
