"""Architecture page -- model graph visualisation and live health overlays."""
from __future__ import annotations

import math
from typing import Any

import panel as pn

from gradienthound.graph import render_graphviz

_OVERLAY_OPTIONS = [
    "Structure",
    "Overall",
    "Gradient Flow",
    "Activation Health",
    "Weight Structure",
]

_STATE_FILL = {
    "neutral": "#1e1416",
    "healthy": "#1a2e22",
    "warning": "#4b3720",
    "critical": "#4a1820",
}

_BORDER_BY_SIGNAL = {
    "neutral": "#9c8088",
    "gradient": "#8bb8d4",
    "activation": "#e8a87c",
    "weight": "#b08ddb",
}

_STATE_SCORES = {
    "neutral": 0,
    "healthy": 1,
    "warning": 2,
    "critical": 3,
}


def create(ipc):
    """Build the architecture page.  Returns ``(layout, update_fn)``."""
    model_select = pn.widgets.Select(name="Model", options=[], width=320)
    overlay_select = pn.widgets.Select(
        name="Overlay",
        options=_OVERLAY_OPTIONS,
        value="Structure",
        width=220,
    )

    summary_pane = pn.Row(sizing_mode="stretch_width")
    health_meta = pn.pane.Markdown("")
    health_warning = pn.pane.Alert("", alert_type="info", visible=False)
    legend_pane = pn.pane.HTML("", visible=False, sizing_mode="stretch_width")

    graph_pane = pn.pane.HTML("", sizing_mode="stretch_width", height=620)
    repr_pane = pn.pane.Str("", sizing_mode="stretch_width")
    details_pane = pn.pane.DataFrame(sizing_mode="stretch_width")

    info_pane = pn.pane.Alert(
        "Waiting for models to be registered...",
        alert_type="info",
    )

    graph_card = pn.Card(
        graph_pane,
        title="Architecture Diagram / Health Map",
        collapsed=False,
        sizing_mode="stretch_width",
    )
    repr_card = pn.Card(
        repr_pane,
        title="PyTorch Module Dump",
        collapsed=True,
        sizing_mode="stretch_width",
    )
    details_card = pn.Card(
        details_pane,
        title="Module Details",
        collapsed=True,
        sizing_mode="stretch_width",
    )

    content = pn.Column(
        summary_pane,
        health_meta,
        health_warning,
        legend_pane,
        graph_card,
        repr_card,
        details_card,
        visible=False,
        sizing_mode="stretch_width",
    )

    layout = pn.Column(
        pn.pane.Markdown("## Model Architecture"),
        pn.Row(model_select, overlay_select, sizing_mode="stretch_width"),
        info_pane,
        content,
        sizing_mode="stretch_width",
    )

    state: dict[str, Any] = {"last_render_key": None}

    def _render_model(
        model_name: str,
        graph_data: dict[str, Any],
        overlay_mode: str,
        snapshot: dict[str, Any],
    ) -> None:
        summary_pane.clear()

        summary_pane.extend([
            pn.indicators.Number(
                name="Class",
                value=0,
                format=graph_data["class"],
                font_size="18pt",
                title_size="10pt",
            ),
            pn.indicators.Number(
                name="Parameters",
                value=graph_data["total_params"],
                format="{value:,.0f}",
                font_size="18pt",
                title_size="10pt",
            ),
            pn.indicators.Number(
                name="Modules",
                value=len(graph_data["modules"]),
                font_size="18pt",
                title_size="10pt",
            ),
        ])

        health_bundle: dict[str, Any] | None = None
        if overlay_mode != "Structure":
            health_bundle = _build_health_bundle(model_name, graph_data, snapshot, overlay_mode)
            summary_pane.extend([
                pn.indicators.Number(
                    name="Live Step",
                    value=health_bundle["live_step"],
                    format="{value:,.0f}",
                    font_size="18pt",
                    title_size="10pt",
                ),
                pn.indicators.Number(
                    name="Healthy",
                    value=health_bundle["counts"]["healthy"],
                    font_size="18pt",
                    title_size="10pt",
                ),
                pn.indicators.Number(
                    name="Warning",
                    value=health_bundle["counts"]["warning"],
                    font_size="18pt",
                    title_size="10pt",
                ),
                pn.indicators.Number(
                    name="Critical",
                    value=health_bundle["counts"]["critical"],
                    font_size="18pt",
                    title_size="10pt",
                ),
            ])

        try:
            dot = render_graphviz(
                graph_data,
                overlays=health_bundle["overlays"] if health_bundle else None,
            )
            svg_bytes = dot.pipe(format="svg")
            svg_str = svg_bytes.decode("utf-8")
            if svg_str.startswith("<?xml"):
                svg_str = svg_str[svg_str.index("?>") + 2:].strip()
            svg_str = svg_str.replace("<svg ", '<svg style="width:100%;height:100%;" ', 1)
            graph_pane.object = _zoom_html(svg_str)
        except Exception:
            graph_pane.object = "<p>Could not render architecture diagram.</p>"

        repr_pane.object = graph_data.get("pytorch_repr", "") or "N/A"
        details_pane.object = _build_details_table(graph_data, health_bundle, overlay_mode)

        if health_bundle:
            health_meta.object = _freshness_markdown(snapshot)
            health_warning.object = health_bundle["warning"]
            health_warning.visible = bool(health_bundle["warning"])
            legend_pane.object = _legend_html(overlay_mode)
            legend_pane.visible = True
        else:
            health_meta.object = "_Containers are grouped visually. Hover nodes for full attributes._"
            health_warning.visible = False
            legend_pane.visible = False

    def update() -> None:
        models = ipc.read_models()
        if not models:
            info_pane.visible = True
            content.visible = False
            return

        info_pane.visible = False
        content.visible = True

        names = list(models.keys())
        if model_select.options != names:
            model_select.options = names
            if model_select.value not in names:
                model_select.value = names[0]

        selected = model_select.value
        if not selected:
            return

        graph_data = models[selected]
        overlay_mode = overlay_select.value
        snapshot = _collect_snapshot(ipc, selected)

        render_key = (
            selected,
            overlay_mode,
            graph_data.get("total_params", 0),
            len(graph_data.get("modules", [])),
            snapshot["steps"]["gradient"],
            snapshot["steps"]["weight"],
            snapshot["steps"]["activation"],
        )
        if render_key == state["last_render_key"]:
            return

        state["last_render_key"] = render_key
        _render_model(selected, graph_data, overlay_mode, snapshot)

    def _force_refresh(event) -> None:
        state["last_render_key"] = None
        update()

    model_select.param.watch(_force_refresh, "value")
    overlay_select.param.watch(_force_refresh, "value")

    return layout, update


def _collect_snapshot(ipc, model_name: str) -> dict[str, Any]:
    grad_entries = _latest_entries_for_model(ipc.read_gradient_stats(), model_name)
    weight_entries = _latest_entries_for_model(ipc.read_weight_stats(), model_name)
    activation_entries = _latest_entries_for_model(ipc.read_activation_stats(), model_name)

    return {
        "gradient": grad_entries,
        "weight": weight_entries,
        "activation": activation_entries,
        "steps": {
            "gradient": _latest_step_for_model(grad_entries),
            "weight": _latest_step_for_model(weight_entries),
            "activation": _latest_step_for_model(activation_entries),
        },
    }


def _latest_entries_for_model(entries: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    model_entries = [entry for entry in entries if entry.get("model") == model_name]
    if not model_entries:
        return []

    step = model_entries[-1].get("step")
    latest: list[dict[str, Any]] = []
    for entry in reversed(model_entries):
        if entry.get("step") != step:
            break
        latest.append(entry)
    latest.reverse()
    return latest


def _latest_step_for_model(entries: list[dict[str, Any]]) -> int:
    if not entries:
        return 0
    try:
        return int(entries[-1].get("step", 0))
    except (TypeError, ValueError):
        return 0


def _build_health_bundle(
    model_name: str,
    graph_data: dict[str, Any],
    snapshot: dict[str, Any],
    overlay_mode: str,
) -> dict[str, Any]:
    module_metrics = _build_module_metrics(model_name, graph_data, snapshot)
    overlays: dict[str, dict[str, Any]] = {}

    counts = {"healthy": 0, "warning": 0, "critical": 0}
    for mod in graph_data["modules"]:
        metrics = module_metrics.get(mod["path"], {})
        view = _health_view(metrics, overlay_mode)

        if mod.get("is_leaf") and view["state"] in counts:
            counts[view["state"]] += 1

        extra_lines = _label_lines(metrics, overlay_mode)
        overlay = {
            "fillcolor": _STATE_FILL[view["state"]],
            "color": _BORDER_BY_SIGNAL[view["signal"]],
            "penwidth": 2.8 if view["score"] >= 2 else 1.8 if view["score"] == 1 else 1.2,
            "tooltip": _tooltip_text(mod, metrics, view),
        }
        if extra_lines:
            overlay["extra_lines"] = extra_lines
        overlays[mod["path"]] = overlay

    live_step = max(snapshot["steps"].values()) if any(snapshot["steps"].values()) else 0
    return {
        "module_metrics": module_metrics,
        "overlays": overlays,
        "counts": counts,
        "live_step": live_step,
        "warning": _warning_text(snapshot, overlay_mode),
    }


def _build_module_metrics(
    model_name: str,
    graph_data: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    modules_by_path = {mod["path"]: mod for mod in graph_data["modules"]}
    direct = {path: _empty_accumulator() for path in modules_by_path}

    for entry in snapshot["gradient"]:
        path = _module_path_from_param(entry.get("layer", ""), model_name)
        if path not in direct:
            continue
        acc = direct[path]
        grad_norm = float(entry.get("grad_norm", 0) or 0.0)
        weight_norm = float(entry.get("weight_norm", 0) or 0.0)
        acc["grad_norm_sq"] += grad_norm * grad_norm
        acc["weight_norm_sq"] += weight_norm * weight_norm
        if "dead_grad_pct" in entry:
            acc["dead_grad_sum"] += float(entry["dead_grad_pct"])
            acc["dead_grad_count"] += 1
        if "near_zero_weight_pct" in entry:
            acc["near_zero_weight_sum"] += float(entry["near_zero_weight_pct"])
            acc["near_zero_weight_count"] += 1
        if "cosine_sim" in entry:
            acc["cosine_sum"] += float(entry["cosine_sim"])
            acc["cosine_count"] += 1

    for entry in snapshot["weight"]:
        path = _module_path_from_param(entry.get("layer", ""), model_name)
        if path not in direct:
            continue
        acc = direct[path]
        if "near_zero_pct" in entry:
            acc["near_zero_weight_sum"] += float(entry["near_zero_pct"])
            acc["near_zero_weight_count"] += 1
        if "effective_rank" in entry and entry.get("max_rank", 0):
            acc["rank_util_sum"] += float(entry["effective_rank"]) / max(float(entry["max_rank"]), 1.0)
            acc["rank_util_count"] += 1
        if "condition_number" in entry:
            acc["condition_number_max"] = max(
                acc["condition_number_max"],
                float(entry["condition_number"]),
            )
            acc["condition_number_count"] += 1

    for entry in snapshot["activation"]:
        path = entry.get("layer") or model_name
        if path not in direct:
            continue
        acc = direct[path]
        if "zero_fraction" in entry:
            acc["activation_zero_sum"] += float(entry["zero_fraction"]) * 100.0
            acc["activation_zero_count"] += 1

    aggregated: dict[str, dict[str, Any]] = {}

    def visit(path: str) -> dict[str, float]:
        acc = dict(direct[path])
        for child in modules_by_path[path].get("children", []):
            if child not in modules_by_path:
                continue
            child_acc = visit(child)
            _merge_accumulators(acc, child_acc)
        aggregated[path] = _finalize_accumulator(acc)
        return acc

    visit(graph_data["name"])
    return aggregated


def _empty_accumulator() -> dict[str, float]:
    return {
        "grad_norm_sq": 0.0,
        "weight_norm_sq": 0.0,
        "dead_grad_sum": 0.0,
        "dead_grad_count": 0.0,
        "near_zero_weight_sum": 0.0,
        "near_zero_weight_count": 0.0,
        "activation_zero_sum": 0.0,
        "activation_zero_count": 0.0,
        "rank_util_sum": 0.0,
        "rank_util_count": 0.0,
        "condition_number_max": 0.0,
        "condition_number_count": 0.0,
        "cosine_sum": 0.0,
        "cosine_count": 0.0,
    }


def _merge_accumulators(into: dict[str, float], other: dict[str, float]) -> None:
    for key, value in other.items():
        if key == "condition_number_max":
            into[key] = max(into[key], value)
        else:
            into[key] += value


def _finalize_accumulator(acc: dict[str, float]) -> dict[str, float | None]:
    grad_norm = math.sqrt(acc["grad_norm_sq"]) if acc["grad_norm_sq"] > 0 else None
    weight_norm = math.sqrt(acc["weight_norm_sq"]) if acc["weight_norm_sq"] > 0 else None

    return {
        "grad_norm": grad_norm,
        "weight_norm": weight_norm,
        "grad_to_weight": (
            grad_norm / max(weight_norm, 1e-12)
            if grad_norm is not None and weight_norm is not None and weight_norm > 0
            else None
        ),
        "dead_grad_pct": _mean_or_none(acc["dead_grad_sum"], acc["dead_grad_count"]),
        "near_zero_weight_pct": _mean_or_none(
            acc["near_zero_weight_sum"],
            acc["near_zero_weight_count"],
        ),
        "activation_zero_pct": _mean_or_none(
            acc["activation_zero_sum"],
            acc["activation_zero_count"],
        ),
        "rank_util_pct": (
            100.0 * acc["rank_util_sum"] / acc["rank_util_count"]
            if acc["rank_util_count"] > 0
            else None
        ),
        "condition_number": (
            acc["condition_number_max"]
            if acc["condition_number_count"] > 0
            else None
        ),
        "cosine_sim": _mean_or_none(acc["cosine_sum"], acc["cosine_count"]),
    }


def _mean_or_none(total: float, count: float) -> float | None:
    if count <= 0:
        return None
    return total / count


def _module_path_from_param(param_name: str, model_name: str) -> str:
    if not param_name:
        return model_name
    if "." not in param_name:
        return model_name
    return param_name.rsplit(".", 1)[0]


def _health_view(metrics: dict[str, Any], overlay_mode: str) -> dict[str, Any]:
    if overlay_mode == "Gradient Flow":
        state, reason = _gradient_state(metrics)
        return {
            "state": state,
            "signal": "gradient" if state != "neutral" else "neutral",
            "reason": reason,
            "score": _STATE_SCORES[state],
        }
    if overlay_mode == "Activation Health":
        state, reason = _activation_state(metrics)
        return {
            "state": state,
            "signal": "activation" if state != "neutral" else "neutral",
            "reason": reason,
            "score": _STATE_SCORES[state],
        }
    if overlay_mode == "Weight Structure":
        state, reason = _weight_state(metrics)
        return {
            "state": state,
            "signal": "weight" if state != "neutral" else "neutral",
            "reason": reason,
            "score": _STATE_SCORES[state],
        }

    gradient = _gradient_state(metrics)
    activation = _activation_state(metrics)
    weight = _weight_state(metrics)
    ranked = [
        ("gradient", gradient),
        ("activation", activation),
        ("weight", weight),
    ]
    best_signal, (best_state, best_reason) = max(
        ranked,
        key=lambda item: _STATE_SCORES[item[1][0]],
    )

    if _STATE_SCORES[best_state] == 0:
        return {"state": "neutral", "signal": "neutral", "reason": "No live health data yet.", "score": 0}
    if _STATE_SCORES[best_state] == 1:
        best_reason = "Observed signals look healthy."

    return {
        "state": best_state,
        "signal": best_signal,
        "reason": best_reason,
        "score": _STATE_SCORES[best_state],
    }


def _gradient_state(metrics: dict[str, Any]) -> tuple[str, str]:
    grad_norm = metrics.get("grad_norm")
    dead_grad_pct = metrics.get("dead_grad_pct")

    if grad_norm is None:
        return ("neutral", "No gradient statistics yet.")
    if grad_norm < 1e-7:
        return ("critical", "Gradient norm is vanishing.")
    if grad_norm > 1e3:
        return ("critical", "Gradient norm is exploding.")
    if dead_grad_pct is not None and dead_grad_pct >= 98:
        return ("critical", "Almost all gradients are dead.")
    if grad_norm < 1e-4 or grad_norm > 1e2:
        return ("warning", "Gradient norm is drifting out of the healthy band.")
    if dead_grad_pct is not None and dead_grad_pct >= 85:
        return ("warning", "Large fraction of gradients are near zero.")
    return ("healthy", "Gradient flow looks healthy.")


def _activation_state(metrics: dict[str, Any]) -> tuple[str, str]:
    zero_pct = metrics.get("activation_zero_pct")
    if zero_pct is None:
        return ("neutral", "No activation statistics yet.")
    if zero_pct >= 97:
        return ("critical", "Activations are almost entirely zero.")
    if zero_pct >= 85:
        return ("warning", "Activations are very sparse.")
    return ("healthy", "Activation sparsity looks healthy.")


def _weight_state(metrics: dict[str, Any]) -> tuple[str, str]:
    rank_util_pct = metrics.get("rank_util_pct")
    condition_number = metrics.get("condition_number")
    near_zero_weight_pct = metrics.get("near_zero_weight_pct")

    observed = False
    state = "healthy"
    reason = "Weight structure looks healthy."

    if rank_util_pct is not None:
        observed = True
        if rank_util_pct < 10:
            return ("critical", "Effective rank has collapsed.")
        if rank_util_pct < 25:
            state = "warning"
            reason = "Effective rank is getting compressed."

    if condition_number is not None:
        observed = True
        if condition_number > 1e6:
            return ("critical", "Matrix conditioning is extremely poor.")
        if condition_number > 1e4 and _STATE_SCORES[state] < _STATE_SCORES["warning"]:
            state = "warning"
            reason = "Matrix conditioning is deteriorating."

    if near_zero_weight_pct is not None:
        observed = True
        if near_zero_weight_pct >= 99:
            return ("critical", "Weights are almost entirely near zero.")
        if near_zero_weight_pct >= 95 and _STATE_SCORES[state] < _STATE_SCORES["warning"]:
            state = "warning"
            reason = "High fraction of weights are near zero."

    if not observed:
        return ("neutral", "No weight-structure statistics yet.")
    return (state, reason)


def _label_lines(metrics: dict[str, Any], overlay_mode: str) -> list[str]:
    if overlay_mode == "Gradient Flow":
        parts = [
            _metric_part("g", metrics.get("grad_norm")),
            _metric_part("g/w", metrics.get("grad_to_weight")),
        ]
    elif overlay_mode == "Activation Health":
        parts = [_pct_part("zeros", metrics.get("activation_zero_pct"))]
    elif overlay_mode == "Weight Structure":
        parts = [
            _pct_part("rank", metrics.get("rank_util_pct")),
            _metric_part("cond", metrics.get("condition_number")),
        ]
    else:
        parts = [
            _metric_part("g", metrics.get("grad_norm")),
            _pct_part("z", metrics.get("activation_zero_pct")),
            _pct_part("r", metrics.get("rank_util_pct")),
        ]

    compact = [part for part in parts if part]
    return [" | ".join(compact)] if compact else []


def _metric_part(label: str, value: float | None) -> str:
    if value is None:
        return ""
    return f"{label}={_format_metric(value)}"


def _pct_part(label: str, value: float | None) -> str:
    if value is None:
        return ""
    return f"{label}={value:.0f}%"


def _format_metric(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.1e}"


def _tooltip_text(mod: dict[str, Any], metrics: dict[str, Any], view: dict[str, Any]) -> str:
    lines = [
        f"{mod['path']} ({mod['type']})",
        f"Health: {view['state'].title()}",
        view["reason"],
    ]

    if metrics.get("grad_norm") is not None:
        lines.append(f"Gradient L2: {_format_metric(metrics['grad_norm'])}")
    if metrics.get("grad_to_weight") is not None:
        lines.append(f"Grad/Weight: {_format_metric(metrics['grad_to_weight'])}")
    if metrics.get("dead_grad_pct") is not None:
        lines.append(f"Dead gradients: {metrics['dead_grad_pct']:.1f}%")
    if metrics.get("activation_zero_pct") is not None:
        lines.append(f"Zero activations: {metrics['activation_zero_pct']:.1f}%")
    if metrics.get("rank_util_pct") is not None:
        lines.append(f"Rank utilization: {metrics['rank_util_pct']:.1f}%")
    if metrics.get("condition_number") is not None:
        lines.append(f"Condition #: {_format_metric(metrics['condition_number'])}")

    return "\n".join(lines)


def _build_details_table(
    graph_data: dict[str, Any],
    health_bundle: dict[str, Any] | None,
    overlay_mode: str,
):
    import pandas as pd

    rows = []
    if not health_bundle or overlay_mode == "Structure":
        for mod in graph_data["modules"]:
            if not mod["is_leaf"]:
                continue
            attr_str = ", ".join(f"{k}={v}" for k, v in mod["attributes"].items())
            rows.append({
                "Path": mod["path"],
                "Type": mod["type"],
                "Parameters": mod["params"],
                "Attributes": attr_str,
            })
        return pd.DataFrame(rows)

    module_metrics = health_bundle["module_metrics"]
    for mod in graph_data["modules"]:
        if not mod["is_leaf"]:
            continue
        metrics = module_metrics.get(mod["path"], {})
        view = _health_view(metrics, overlay_mode)
        rows.append({
            "Path": mod["path"],
            "Type": mod["type"],
            "Parameters": mod["params"],
            "Health": view["state"].title(),
            "Primary Signal": view["signal"].title() if view["signal"] != "neutral" else "-",
            "Grad L2": _display_metric(metrics.get("grad_norm")),
            "Grad/Weight": _display_metric(metrics.get("grad_to_weight")),
            "Dead Grad %": _display_percent(metrics.get("dead_grad_pct")),
            "Zero Act %": _display_percent(metrics.get("activation_zero_pct")),
            "Rank Util %": _display_percent(metrics.get("rank_util_pct")),
            "Condition #": _display_metric(metrics.get("condition_number")),
        })

    rows.sort(
        key=lambda row: (
            -_STATE_SCORES[row["Health"].lower()],
            row["Path"],
        ),
    )
    return pd.DataFrame(rows)


def _display_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return _format_metric(value)


def _display_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def _warning_text(snapshot: dict[str, Any], overlay_mode: str) -> str:
    if overlay_mode == "Structure":
        return ""

    missing = []
    if overlay_mode in {"Overall", "Gradient Flow"} and not snapshot["gradient"]:
        missing.append("gradient stats (`gh.watch(...)` + `gh.step()`) ")
    if overlay_mode in {"Overall", "Activation Health"} and not snapshot["activation"]:
        missing.append("activation stats (`gh.watch(..., log_activations=True)`) ")
    if overlay_mode in {"Overall", "Weight Structure"} and not snapshot["weight"]:
        missing.append("weight snapshots (captured every `weight_every` steps)")

    if not missing:
        return ""
    if len(missing) == 3:
        return "Waiting for gradients, activations, and weight snapshots before the health map is fully populated."
    return "This overlay is partial because it is missing " + ", ".join(item.strip() for item in missing) + "."


def _freshness_markdown(snapshot: dict[str, Any]) -> str:
    parts = []
    for label, key in [
        ("Gradients", "gradient"),
        ("Activations", "activation"),
        ("Weights", "weight"),
    ]:
        step = snapshot["steps"][key]
        parts.append(f"{label}: **step {step}**" if step else f"{label}: *waiting*")
    return " | ".join(parts)


def _legend_html(overlay_mode: str) -> str:
    if overlay_mode == "Structure":
        return ""

    if overlay_mode == "Overall":
        border_text = "Border color shows the dominant issue family: blue = gradients, orange = activations, purple = weight structure."
    elif overlay_mode == "Gradient Flow":
        border_text = "Border color tracks gradient-flow diagnostics."
    elif overlay_mode == "Activation Health":
        border_text = "Border color tracks activation sparsity diagnostics."
    else:
        border_text = "Border color tracks weight-structure diagnostics."

    return f"""
    <div style="display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin:4px 0 10px 0;">
      <span style="display:inline-flex;align-items:center;gap:8px;">
        <span style="width:16px;height:16px;background:{_STATE_FILL['healthy']};border:2px solid #9c8088;border-radius:4px;"></span>
        Healthy
      </span>
      <span style="display:inline-flex;align-items:center;gap:8px;">
        <span style="width:16px;height:16px;background:{_STATE_FILL['warning']};border:2px solid #9c8088;border-radius:4px;"></span>
        Warning
      </span>
      <span style="display:inline-flex;align-items:center;gap:8px;">
        <span style="width:16px;height:16px;background:{_STATE_FILL['critical']};border:2px solid #9c8088;border-radius:4px;"></span>
        Critical
      </span>
      <span style="display:inline-flex;align-items:center;gap:8px;">
        <span style="width:16px;height:16px;background:{_STATE_FILL['neutral']};border:2px solid #9c8088;border-radius:4px;"></span>
        No live data
      </span>
    </div>
    <div style="color:#b09a9e;font-size:12px;">{border_text}</div>
    """


def _zoom_html(svg_str: str) -> str:
    # Panel's HTML pane injects content via innerHTML, which does not
    # execute <script> tags. Inline handlers keep pan/zoom working.
    _apply = (
        "var c=this.nodeType?this:this.closest('[data-s]');"
        "var v=c.querySelector('svg');"
        "v.style.transform='translate('+c.dataset.px+'px,'+c.dataset.py+'px) scale('+c.dataset.s+')';"
        "v.style.transformOrigin='0 0';"
    )

    def _btn_zoom(factor: float) -> str:
        return (
            "var c=this.closest('[data-s]');"
            "var r=c.getBoundingClientRect();"
            "var cx=r.width/2,cy=r.height/2;"
            f"var prev=parseFloat(c.dataset.s),s=Math.max(0.1,Math.min(10,prev*{factor}));"
            "c.dataset.s=s;"
            "var px=parseFloat(c.dataset.px),py=parseFloat(c.dataset.py);"
            "c.dataset.px=cx-(cx-px)*(s/prev);"
            "c.dataset.py=cy-(cy-py)*(s/prev);"
            "var v=c.querySelector('svg');"
            "v.style.transform='translate('+c.dataset.px+'px,'+c.dataset.py+'px) scale('+s+')';"
            "v.style.transformOrigin='0 0';"
        )

    _reset = (
        "var c=this.closest('[data-s]');"
        "c.dataset.s=1;c.dataset.px=0;c.dataset.py=0;"
        "var v=c.querySelector('svg');"
        "v.style.transform='';v.style.transformOrigin='';"
    )

    return f"""
    <div style="width:100%;height:600px;border:1px solid rgba(178,132,140,0.2);
                border-radius:16px;overflow:hidden;position:relative;
                background:radial-gradient(circle at top, rgba(180,60,70,0.06), transparent 30%), #0f0a0a;cursor:grab;
                box-shadow:inset 0 1px 0 rgba(255,255,255,0.03);"
         data-s="1" data-px="0" data-py="0" data-drag="0"
         onwheel="event.preventDefault();
           var c=this,rect=c.getBoundingClientRect();
           var mx=event.clientX-rect.left,my=event.clientY-rect.top;
           var prev=parseFloat(c.dataset.s);
           var s=Math.max(0.1,Math.min(10,prev*(event.deltaY>0?0.9:1.1)));
           c.dataset.s=s;
           var px=parseFloat(c.dataset.px),py=parseFloat(c.dataset.py);
           c.dataset.px=mx-(mx-px)*(s/prev);
           c.dataset.py=my-(my-py)*(s/prev);
           {_apply}"
         onmousedown="this.dataset.drag=1;
           this.dataset.sx=event.clientX-parseFloat(this.dataset.px);
           this.dataset.sy=event.clientY-parseFloat(this.dataset.py);
           this.style.cursor='grabbing';"
         onmousemove="if(this.dataset.drag!=1)return;
           this.dataset.px=event.clientX-parseFloat(this.dataset.sx);
           this.dataset.py=event.clientY-parseFloat(this.dataset.sy);
           {_apply}"
         onmouseup="this.dataset.drag=0;this.style.cursor='grab';"
         onmouseleave="this.dataset.drag=0;this.style.cursor='grab';"
    >
      {svg_str}
      <div style="position:absolute;bottom:8px;right:8px;display:flex;gap:4px;">
        <button onclick="{_btn_zoom(1.3)}" style="
          width:32px;height:32px;border:none;border-radius:10px;
          background:rgba(26,14,16,0.92);color:#f0e6e8;font-size:18px;cursor:pointer;
          box-shadow:0 8px 20px rgba(0,0,0,0.22);">+</button>
        <button onclick="{_btn_zoom(1 / 1.3)}" style="
          width:32px;height:32px;border:none;border-radius:10px;
          background:rgba(26,14,16,0.92);color:#f0e6e8;font-size:18px;cursor:pointer;
          box-shadow:0 8px 20px rgba(0,0,0,0.22);">&minus;</button>
        <button onclick="{_reset}" style="
          height:32px;border:none;border-radius:10px;padding:0 10px;
          background:linear-gradient(135deg, #d4707a 0%, #c2525e 100%);
          color:#0f0a0a;font-size:12px;font-weight:700;cursor:pointer;
          box-shadow:0 8px 20px rgba(194,82,94,0.18);">Reset</button>
      </div>
    </div>
    """
