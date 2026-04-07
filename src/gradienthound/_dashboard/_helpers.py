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
