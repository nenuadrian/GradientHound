"""Gradients page -- gradient flow, cosine similarity, update ratios, noise."""
from __future__ import annotations

import panel as pn
from bokeh.models import ColumnDataSource, HoverTool

from ._common import (
    make_figure, make_hbar_figure, short_layer_name, short_layer_names,
    style_legend, grad_health_color, GREEN, YELLOW, RED, BLUE, PALETTE,
    latest_entries, latest_step,
)


def create(ipc):
    """Build the gradients page.  Returns ``(layout, update_fn)``."""

    info_pane = pn.pane.Alert(
        "Waiting for gradient data (captured via `gh.watch` + `gh.step`)...",
        alert_type="info",
    )
    step_label = pn.pane.Markdown("")

    # ── Gradient Flow ────────────────────────────────────────────────
    flow_src = ColumnDataSource(data={"layer": [], "norm": [], "color": []})
    flow_fig = make_hbar_figure(title="Gradient Flow (per-layer L2 norms)",
                                 y_range=[], height=300)
    flow_fig.hbar(y="layer", right="norm", source=flow_src, height=0.7,
                  color="color")
    flow_summary = pn.Row(sizing_mode="stretch_width")
    flow_button = pn.widgets.Button(name="Compute", button_type="primary")

    def _on_flow_click(event):
        stats = ipc.read_gradient_stats()
        if not stats:
            return
        latest = latest_entries(stats)
        layers = short_layer_names([e["layer"] for e in latest])
        norms = [e.get("grad_norm", 0) for e in latest]
        colors = [grad_health_color(n) for n in norms]
        flow_src.data = {"layer": layers, "norm": norms, "color": colors}
        flow_fig.y_range.factors = list(reversed(layers))
        healthy = sum(1 for c in colors if c == GREEN)
        weak = sum(1 for c in colors if c == YELLOW)
        critical = sum(1 for c in colors if c == RED)
        flow_summary.clear()
        flow_summary.extend([
            pn.indicators.Number(name="Healthy", value=healthy,
                                 font_size="18pt", title_size="9pt"),
            pn.indicators.Number(name="Weak", value=weak,
                                 font_size="18pt", title_size="9pt"),
            pn.indicators.Number(name="Critical", value=critical,
                                 font_size="18pt", title_size="9pt"),
        ])

    flow_button.on_click(_on_flow_click)

    flow_card = pn.Card(
        flow_button,
        pn.pane.Bokeh(flow_fig, sizing_mode="stretch_width"),
        flow_summary,
        title="Gradient Flow", collapsed=True, sizing_mode="stretch_width",
    )

    # ── Cosine Similarity ────────────────────────────────────────────
    cos_src = ColumnDataSource(data={"layer": [], "cosine": [], "color": []})
    cos_fig = make_hbar_figure(title="Cosine Similarity (vs previous step)",
                                y_range=[], height=300)
    cos_fig.hbar(y="layer", right="cosine", source=cos_src, height=0.7,
                 color="color")
    cos_summary = pn.Row(sizing_mode="stretch_width")
    cos_button = pn.widgets.Button(name="Compute", button_type="primary")

    def _on_cos_click(event):
        stats = ipc.read_gradient_stats()
        if not stats:
            return
        latest = latest_entries(stats)
        cos_entries = [e for e in latest if "cosine_sim" in e]
        if not cos_entries:
            return
        cl = short_layer_names([e["layer"] for e in cos_entries])
        cv = [e["cosine_sim"] for e in cos_entries]
        cc = []
        stable = noisy = osc = 0
        for v in cv:
            if v > 0.5:
                cc.append(GREEN); stable += 1
            elif v >= 0:
                cc.append(YELLOW); noisy += 1
            else:
                cc.append(RED); osc += 1
        cos_src.data = {"layer": cl, "cosine": cv, "color": cc}
        cos_fig.y_range.factors = list(reversed(cl))
        cos_summary.clear()
        cos_summary.extend([
            pn.indicators.Number(name="Stable (>0.5)", value=stable,
                                 font_size="18pt", title_size="9pt"),
            pn.indicators.Number(name="Noisy (0-0.5)", value=noisy,
                                 font_size="18pt", title_size="9pt"),
            pn.indicators.Number(name="Oscillating (<0)", value=osc,
                                 font_size="18pt", title_size="9pt"),
        ])

    cos_button.on_click(_on_cos_click)

    cos_card = pn.Card(
        cos_button,
        pn.pane.Bokeh(cos_fig, sizing_mode="stretch_width"),
        cos_summary,
        title="Gradient Cosine Similarity", collapsed=True,
        sizing_mode="stretch_width",
    )

    # ── Update-to-Weight Ratio ───────────────────────────────────────
    ratio_src = ColumnDataSource(data={"layer": [], "ratio": []})
    ratio_fig = make_hbar_figure(title="Update-to-Weight Ratio",
                                  y_range=[], height=300)
    ratio_fig.hbar(y="layer", right="ratio", source=ratio_src, height=0.7,
                   color="#8bb8d4")
    ratio_caption = pn.pane.Markdown("")
    ratio_button = pn.widgets.Button(name="Compute", button_type="primary")

    def _on_ratio_click(event):
        stats = ipc.read_gradient_stats()
        if not stats:
            return
        latest = latest_entries(stats)
        optimizers = ipc.read_optimizers()
        lr = 1e-3
        for oi in optimizers.values():
            for pg in oi.get("param_groups", []):
                if "lr" in pg:
                    lr = pg["lr"]
                    break
            if lr != 1e-3:
                break
        rl = short_layer_names([e["layer"] for e in latest])
        rv = [(lr * e.get("grad_norm", 0)) / max(e.get("weight_norm", 1e-20), 1e-20)
              for e in latest]
        ratio_src.data = {"layer": rl, "ratio": rv}
        ratio_fig.y_range.factors = list(reversed(rl))
        ratio_caption.object = f"*Using lr={lr:.1e}. Healthy range: [1e-4, 1e-2]*"

    ratio_button.on_click(_on_ratio_click)

    ratio_card = pn.Card(
        ratio_button,
        pn.pane.Bokeh(ratio_fig, sizing_mode="stretch_width"),
        ratio_caption,
        title="Update-to-Weight Ratio", collapsed=True,
        sizing_mode="stretch_width",
    )

    # ── Dead Neurons ─────────────────────────────────────────────────
    from bokeh.transform import dodge
    dead_src = ColumnDataSource(data={"layer": [], "dead": [], "near_zero": []})
    dead_fig = make_hbar_figure(title="Dead Neurons", y_range=[], height=300)
    dead_fig.hbar(y=dodge("layer", -0.15, range=dead_fig.y_range),
                  right="dead", source=dead_src, height=0.25,
                  color="#e85d6f", legend_label="Dead Grad %")
    dead_fig.hbar(y=dodge("layer", 0.15, range=dead_fig.y_range),
                  right="near_zero", source=dead_src, height=0.25,
                  color="#e8a87c", legend_label="Near-Zero Weight %")
    style_legend(dead_fig)
    dead_button = pn.widgets.Button(name="Compute", button_type="primary")

    def _on_dead_click(event):
        stats = ipc.read_gradient_stats()
        if not stats:
            return
        latest = latest_entries(stats)
        dl = short_layer_names([e["layer"] for e in latest])
        dead_src.data = {
            "layer": dl,
            "dead": [e.get("dead_grad_pct", 0) for e in latest],
            "near_zero": [e.get("near_zero_weight_pct", 0) for e in latest],
        }
        dead_fig.y_range.factors = list(reversed(dl))

    dead_button.on_click(_on_dead_click)

    dead_card = pn.Card(
        dead_button,
        pn.pane.Bokeh(dead_fig, sizing_mode="stretch_width"),
        title="Dead Neurons", collapsed=True, sizing_mode="stretch_width",
    )

    # ── Noise Scale ──────────────────────────────────────────────────
    noise_src = ColumnDataSource(data={"layer": [], "noise": []})
    noise_fig = make_hbar_figure(title="Gradient Noise Scale", y_range=[],
                                  height=300)
    noise_fig.hbar(y="layer", right="noise", source=noise_src, height=0.7,
                   color="#b09a9e")
    noise_button = pn.widgets.Button(name="Compute", button_type="primary")

    def _on_noise_click(event):
        stats = ipc.read_gradient_stats()
        if not stats:
            return
        latest = latest_entries(stats)
        nl = short_layer_names([e["layer"] for e in latest])
        noise_src.data = {
            "layer": nl,
            "noise": [e.get("grad_noise_scale", 0) for e in latest],
        }
        noise_fig.y_range.factors = list(reversed(nl))

    noise_button.on_click(_on_noise_click)

    noise_card = pn.Card(
        noise_button,
        pn.pane.Bokeh(noise_fig, sizing_mode="stretch_width"),
        pn.pane.Markdown("*Var(g) / E[g]^2 -- high values = noisy gradients*"),
        title="Gradient Noise Scale", collapsed=True,
        sizing_mode="stretch_width",
    )

    # ── Norms Over Time ──────────────────────────────────────────────
    norm_time_select = pn.widgets.MultiChoice(
        name="Layers", options=[], max_items=8, width=600,
    )
    norm_time_src = ColumnDataSource(data={"xs": [], "ys": [], "colors": [],
                                           "labels": []})
    norm_time_fig = make_figure(title="Gradient Norms Over Time",
                                x_label="step", height=300)
    norm_time_fig.multi_line("xs", "ys", source=norm_time_src,
                             line_color="colors", line_width=2)
    norm_time_card = pn.Card(
        norm_time_select,
        pn.pane.Bokeh(norm_time_fig, sizing_mode="stretch_width"),
        title="Gradient Norms Over Time", collapsed=True,
        sizing_mode="stretch_width",
    )

    # ── Cosine Over Time ─────────────────────────────────────────────
    cos_time_select = pn.widgets.MultiChoice(
        name="Layers", options=[], max_items=8, width=600,
    )
    cos_time_src = ColumnDataSource(data={"xs": [], "ys": [], "colors": [],
                                          "labels": []})
    cos_time_fig = make_figure(title="Cosine Similarity Over Time",
                                x_label="step", height=300)
    cos_time_fig.multi_line("xs", "ys", source=cos_time_src,
                            line_color="colors", line_width=2)
    cos_time_card = pn.Card(
        cos_time_select,
        pn.pane.Bokeh(cos_time_fig, sizing_mode="stretch_width"),
        title="Cosine Similarity Over Time", collapsed=True,
        sizing_mode="stretch_width",
    )

    # ── layout ───────────────────────────────────────────────────────
    content = pn.Column(
        step_label,
        flow_card, cos_card, ratio_card, dead_card,
        noise_card, norm_time_card, cos_time_card,
        visible=False, sizing_mode="stretch_width",
    )

    layout = pn.Column(
        pn.pane.Markdown("## Gradient Analysis"),
        info_pane, content, sizing_mode="stretch_width",
    )

    state: dict = {"last_step": -1}

    # ── time-series widget callbacks ─────────────────────────────────

    def _update_norm_time(event):
        _rebuild_time_series(
            ipc.read_gradient_stats(), event.new or [],
            norm_time_src, "grad_norm",
        )

    def _update_cos_time(event):
        _rebuild_time_series(
            [e for e in ipc.read_gradient_stats() if "cosine_sim" in e],
            event.new or [], cos_time_src, "cosine_sim",
        )

    norm_time_select.param.watch(_update_norm_time, "value")
    cos_time_select.param.watch(_update_cos_time, "value")

    # ── periodic update ──────────────────────────────────────────────

    def update():
        stats = ipc.read_gradient_stats()
        if not stats:
            info_pane.visible = True
            content.visible = False
            return

        info_pane.visible = False
        content.visible = True

        current_step = latest_step(stats)
        if current_step == state["last_step"]:
            return
        state["last_step"] = current_step
        step_label.object = f"**Latest step: {current_step}**"

        # update time-series selector options only (visuals generated on demand)
        all_layers = sorted({e["layer"] for e in stats})
        norm_time_select.options = all_layers
        cos_layers = sorted({e["layer"] for e in stats if "cosine_sim" in e})
        cos_time_select.options = cos_layers

    return layout, update


def _rebuild_time_series(all_stats, selected_layers, source, field):
    """Rebuild multi_line data for the selected layers."""
    xs, ys, colors, labels = [], [], [], []
    for i, layer in enumerate(selected_layers):
        records = sorted(
            [e for e in all_stats if e["layer"] == layer],
            key=lambda e: e["step"],
        )
        if records:
            xs.append([e["step"] for e in records])
            ys.append([e.get(field, 0) for e in records])
            colors.append(PALETTE[i % len(PALETTE)])
            labels.append(short_layer_name(layer))
    source.data = {"xs": xs, "ys": ys, "colors": colors, "labels": labels}
