"""Metrics page -- live metric charts with auto-refresh."""
from __future__ import annotations

from collections import defaultdict

import panel as pn
from bokeh.models import ColumnDataSource, HoverTool

from ._common import make_figure, PALETTE, latest_entries, latest_step


def create(ipc):
    """Build the metrics page. Returns ``(layout, update_fn)``."""

    step_ind = pn.indicators.Number(
        name="Step", value=0, format="{value:,.0f}",
        font_size="28pt", title_size="10pt",
    )
    models_ind = pn.indicators.Number(
        name="Models", value=0,
        font_size="28pt", title_size="10pt",
    )
    params_ind = pn.indicators.Number(
        name="Parameters", value=0, format="{value:,.0f}",
        font_size="28pt", title_size="10pt",
    )
    metrics_ind = pn.indicators.Number(
        name="Metrics", value=0,
        font_size="28pt", title_size="10pt",
    )

    health_pane = pn.pane.Alert("", alert_type="warning", visible=False)
    info_pane = pn.pane.Alert("Waiting for metrics...", alert_type="info")
    charts_col = pn.Column(sizing_mode="stretch_width")

    state: dict = {
        "sources": {},
        "series": {},
        "last_len": 0,
        "categories": {},
        "cat_grids": {},
    }

    layout = pn.Column(
        pn.pane.Markdown("## Live Metrics"),
        pn.pane.Markdown(
            "This is the only page that refreshes automatically in the background.",
        ),
        pn.Row(step_ind, models_ind, params_ind, metrics_ind,
               sizing_mode="stretch_width"),
        health_pane,
        info_pane,
        charts_col,
        sizing_mode="stretch_width",
    )

    def _categorise(keys):
        buckets: dict[str, list[str]] = defaultdict(list)
        for k in keys:
            prefix, sep, _ = k.partition("/")
            cat = prefix.strip() if sep and prefix.strip() else "Other"
            buckets[cat].append(k)
        return dict(buckets)

    def _make_chart(key, steps, vals, idx):
        src = ColumnDataSource(data={"step": steps, "value": vals})
        fig = make_figure(title=key, x_label="step", height=220)
        fig.line("step", "value", source=src,
                 color=PALETTE[idx % len(PALETTE)], line_width=2)
        fig.add_tools(HoverTool(
            tooltips=[("step", "@step"), (key, "@value{0.0000}")]))
        state["sources"][key] = src
        return pn.pane.Bokeh(fig, sizing_mode="stretch_width", min_width=300)

    def update():
        models = ipc.read_models()
        metrics = ipc.read_metrics()
        grad_stats = ipc.read_gradient_stats()

        total_params = sum(m.get("total_params", 0) for m in models.values())
        num_steps = latest_step(metrics, key="_step")
        num_grad = latest_step(grad_stats)

        step_ind.value = max(num_steps, num_grad)
        models_ind.value = len(models)
        params_ind.value = total_params
        metrics_ind.value = len(state["series"])

        if grad_stats:
            latest = latest_entries(grad_stats)
            vanishing = sum(1 for e in latest if e.get("grad_norm", 1) < 1e-7)
            exploding = sum(1 for e in latest if e.get("grad_norm", 0) > 1e3)
            parts = []
            if vanishing:
                parts.append(f"{vanishing} layers with vanishing gradients")
            if exploding:
                parts.append(f"{exploding} layers with exploding gradients")
            if parts:
                health_pane.object = " | ".join(parts)
                health_pane.alert_type = "danger" if exploding else "warning"
                health_pane.visible = True
            else:
                health_pane.visible = False

        if not metrics:
            info_pane.visible = True
            return
        info_pane.visible = False

        if len(metrics) < state["last_len"]:
            charts_col.clear()
            state["sources"].clear()
            state["series"].clear()
            state["categories"].clear()
            state["cat_grids"].clear()
            state["last_len"] = 0

        if len(metrics) == state["last_len"]:
            return

        new_entries = metrics[state["last_len"]:]
        state["last_len"] = len(metrics)
        touched_keys: set[str] = set()
        new_keys: list[str] = []

        for entry in new_entries:
            step = entry.get("_step", 0)
            for key, value in entry.items():
                if key.startswith("_"):
                    continue
                if key not in state["series"]:
                    state["series"][key] = {"step": [], "value": []}
                    new_keys.append(key)
                series = state["series"][key]
                series["step"].append(step)
                series["value"].append(value)
                touched_keys.add(key)

        metrics_ind.value = len(state["series"])

        for key in touched_keys:
            if key in state["sources"]:
                state["sources"][key].data = state["series"][key]

        if not new_keys:
            return

        new_categories = _categorise(sorted(new_keys))
        global_idx = len(state["sources"])

        for cat_name, cat_keys in new_categories.items():
            charts = []
            for key in cat_keys:
                series = state["series"][key]
                charts.append(
                    _make_chart(key, series["step"], series["value"], global_idx),
                )
                global_idx += 1

            if cat_name in state["cat_grids"]:
                grid = state["cat_grids"][cat_name]
                for chart in charts:
                    grid.append(chart)
            else:
                grid = pn.GridBox(*charts, ncols=2, sizing_mode="stretch_width")
                card = pn.Card(
                    grid,
                    title=cat_name,
                    collapsed=True,
                    sizing_mode="stretch_width",
                )
                state["categories"][cat_name] = card
                state["cat_grids"][cat_name] = grid
                charts_col.append(card)

    return layout, update
