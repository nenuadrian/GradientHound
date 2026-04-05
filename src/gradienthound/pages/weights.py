"""Weights page -- histograms, SVD spectrum, norms, heatmaps, sparsity."""
from __future__ import annotations

import panel as pn
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker, HoverTool
from bokeh.palettes import RdBu11

from ._common import (
    make_figure, make_hbar_figure, short_layer_name, short_layer_names,
    PALETTE, BLUE, latest_entries, latest_step,
)


def create(ipc):
    """Build the weights page.  Returns ``(layout, update_fn)``."""
    import pandas as pd

    info_pane = pn.pane.Alert(
        "Waiting for weight snapshots (captured every N steps via `gh.watch`)...",
        alert_type="info",
    )
    step_label = pn.pane.Markdown("")

    # ── Histogram section ────────────────────────────────────────────
    hist_select = pn.widgets.Select(name="Layer", options=[], width=400)
    hist_src = ColumnDataSource(data={"center": [], "count": [], "width": []})
    hist_fig = make_figure(title="Weight Distribution", x_label="Weight Value",
                           y_label="Count", height=280)
    hist_fig.vbar(x="center", top="count", width="width", source=hist_src,
                  color=BLUE, alpha=0.8)
    hist_metrics = pn.Row(sizing_mode="stretch_width")

    hist_card = pn.Card(
        hist_select,
        pn.pane.Bokeh(hist_fig, sizing_mode="stretch_width"),
        hist_metrics,
        title="Weight Distribution Histograms", collapsed=True,
        sizing_mode="stretch_width",
    )

    # ── SVD section ──────────────────────────────────────────────────
    svd_select = pn.widgets.Select(name="Layer", options=[], width=400)

    sv_src = ColumnDataSource(data={"index": [], "sv": []})
    sv_fig = make_figure(title="Singular Values", x_label="Index",
                         y_label="Value", height=260)
    sv_fig.vbar(x="index", top="sv", width=0.7, source=sv_src, color="#b08ddb")

    ce_src = ColumnDataSource(data={"component": [], "energy": []})
    ce_fig = make_figure(title="Cumulative Energy", x_label="Component",
                         y_label="Energy", height=260)
    ce_fig.line("component", "energy", source=ce_src, color="#8bb8d4",
                line_width=2)
    ce_label = pn.pane.Markdown("")
    svd_metrics = pn.Row(sizing_mode="stretch_width")

    svd_card = pn.Card(
        svd_select,
        pn.Row(
            pn.pane.Bokeh(sv_fig, sizing_mode="stretch_width"),
            pn.pane.Bokeh(ce_fig, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        ),
        ce_label, svd_metrics,
        title="Spectral Analysis (SVD)", collapsed=True,
        sizing_mode="stretch_width",
    )

    # ── Norms section ────────────────────────────────────────────────
    norm_src = ColumnDataSource(data={"layer": [], "norm": []})
    norm_fig = make_hbar_figure(title="Parameter Norms (L2)", y_range=[],
                                height=300)
    norm_fig.hbar(y="layer", right="norm", source=norm_src, height=0.7,
                  color=BLUE)
    norm_button = pn.widgets.Button(name="Compute", button_type="primary")

    def _render_norms(latest: list[dict]) -> None:
        layer_names = short_layer_names([e["layer"] for e in latest])
        norms = [e["norm_l2"] for e in latest]
        norm_src.data = {"layer": layer_names, "norm": norms}
        norm_fig.y_range.factors = list(reversed(layer_names))

    def _on_norm_click(event):
        stats = ipc.read_weight_stats()
        if not stats:
            return
        _render_norms(latest_entries(stats))

    norm_button.on_click(_on_norm_click)

    norm_card = pn.Card(
        norm_button,
        pn.pane.Bokeh(norm_fig, sizing_mode="stretch_width"),
        title="Parameter Norms (L2)", collapsed=True,
        sizing_mode="stretch_width",
    )

    # ── Effective rank table ─────────────────────────────────────────
    rank_pane = pn.pane.DataFrame(sizing_mode="stretch_width")
    rank_button = pn.widgets.Button(name="Compute", button_type="primary")

    def _render_rank_table(latest: list[dict]) -> None:
        rank_entries = [e for e in latest if "effective_rank" in e]
        if rank_entries:
            rank_pane.object = pd.DataFrame([{
                "Layer": short_layer_name(e["layer"]),
                "Effective Rank": round(e["effective_rank"], 1),
                "Max Rank": e.get("max_rank", 0),
                "Utilization": (f"{e['effective_rank'] / e['max_rank'] * 100:.0f}%"
                                if e.get("max_rank", 0) > 0 else "N/A"),
            } for e in rank_entries])
        else:
            rank_pane.object = pd.DataFrame([
                {"Status": "No 2D weight matrices with SVD data in the latest snapshot."},
            ])

    def _on_rank_click(event):
        stats = ipc.read_weight_stats()
        if not stats:
            return
        _render_rank_table(latest_entries(stats))

    rank_button.on_click(_on_rank_click)

    rank_card = pn.Card(
        rank_button, rank_pane,
        title="Effective Rank (2D layers)", collapsed=True,
        sizing_mode="stretch_width",
    )

    # ── Distribution over time ───────────────────────────────────────
    time_select = pn.widgets.Select(name="Layer", options=[], width=400)
    time_src = ColumnDataSource(data={"step": [], "mean": [], "std": [],
                                      "min": [], "max": []})
    time_fig = make_figure(title="Weight Stats Over Time", x_label="step",
                           height=280)
    for col, color, label in [
        ("mean", "#8bb8d4", "mean"), ("std", "#6dbf8b", "std"),
        ("min", "#e85d6f", "min"), ("max", "#e8a87c", "max"),
    ]:
        time_fig.line("step", col, source=time_src, color=color,
                      line_width=2, legend_label=label)
    from ._common import style_legend
    style_legend(time_fig)

    time_card = pn.Card(
        time_select,
        pn.pane.Bokeh(time_fig, sizing_mode="stretch_width"),
        title="Weight Distribution Over Time", collapsed=True,
        sizing_mode="stretch_width",
    )

    # ── Heatmap (on-demand) ──────────────────────────────────────────
    hm_select = pn.widgets.Select(name="Layer", options=[], width=400)
    hm_button = pn.widgets.Button(name="Compute Heatmap", button_type="primary")
    hm_pane = pn.pane.Bokeh(sizing_mode="stretch_width")
    hm_caption = pn.pane.Markdown("")

    def _on_hm_click(event):
        layer = hm_select.value
        if not layer:
            return
        ipc.clear_response("weight_heatmap")
        ipc.write_request({
            "id": "weight_heatmap", "type": "weight_heatmap", "layer": layer,
        })
        hm_caption.object = "*Computing...*"

    hm_button.on_click(_on_hm_click)

    hm_card = pn.Card(
        hm_select, hm_button, hm_caption, hm_pane,
        title="Weight Heatmap (on-demand)", collapsed=True,
        sizing_mode="stretch_width",
    )

    # ── Sparsity ─────────────────────────────────────────────────────
    sparsity_src = ColumnDataSource(data={"layer": [], "pct": []})
    sparsity_fig = make_hbar_figure(title="Near-Zero Weights (%)",
                                     y_range=[], height=300)
    sparsity_fig.hbar(y="layer", right="pct", source=sparsity_src,
                      height=0.7, color="#e8a87c")
    sparsity_button = pn.widgets.Button(name="Compute", button_type="primary")

    def _on_sparsity_click(event):
        stats = ipc.read_weight_stats()
        if not stats:
            return
        latest = latest_entries(stats)
        sp_layers = short_layer_names([e["layer"] for e in latest])
        sp_pcts = [e.get("near_zero_pct", 0) for e in latest]
        sparsity_src.data = {"layer": sp_layers, "pct": sp_pcts}
        sparsity_fig.y_range.factors = list(reversed(sp_layers))

    sparsity_button.on_click(_on_sparsity_click)

    sparsity_card = pn.Card(
        sparsity_button,
        pn.pane.Bokeh(sparsity_fig, sizing_mode="stretch_width"),
        title="Near-Zero Weights (Sparsity)", collapsed=True,
        sizing_mode="stretch_width",
    )

    # ── layout ───────────────────────────────────────────────────────
    content = pn.Column(
        step_label,
        hist_card, svd_card, norm_card, rank_card,
        time_card, hm_card, sparsity_card,
        visible=False, sizing_mode="stretch_width",
    )

    layout = pn.Column(
        pn.pane.Markdown("## Weight Analysis"),
        info_pane, content, sizing_mode="stretch_width",
    )

    state: dict = {"last_step": -1}

    # ── widget callbacks ─────────────────────────────────────────────

    def _populate_hist(event):
        stats = ipc.read_weight_stats()
        if not stats:
            return
        latest = {e["layer"]: e for e in latest_entries(stats)}
        e = latest.get(event.new)
        if not e or "hist_counts" not in e:
            return
        centers = e["hist_centers"]
        counts = e["hist_counts"]
        w = centers[1] - centers[0] if len(centers) > 1 else 0.01
        hist_src.data = {"center": centers, "count": counts,
                         "width": [w] * len(centers)}
        hist_metrics.clear()
        hist_metrics.extend([
            pn.indicators.Number(name="Mean", value=round(e["mean"], 4),
                                 font_size="16pt", title_size="9pt"),
            pn.indicators.Number(name="Std", value=round(e["std"], 4),
                                 font_size="16pt", title_size="9pt"),
            pn.indicators.Number(name="Near-Zero %",
                                 value=round(e.get("near_zero_pct", 0), 1),
                                 font_size="16pt", title_size="9pt"),
            pn.indicators.Number(name="Kurtosis",
                                 value=round(e.get("kurtosis", 0), 2),
                                 font_size="16pt", title_size="9pt"),
        ])

    hist_select.param.watch(_populate_hist, "value")

    def _populate_svd(event):
        stats = ipc.read_weight_stats()
        if not stats:
            return
        latest = {e["layer"]: e for e in latest_entries(stats)}
        e = latest.get(event.new)
        if not e or "singular_values" not in e:
            return
        svs = e["singular_values"]
        sv_src.data = {"index": list(range(len(svs))), "sv": svs}

        if "cumulative_energy" in e:
            ce = e["cumulative_energy"]
            ce_src.data = {"component": list(range(len(ce))), "energy": ce}
            t90 = next((i for i, v in enumerate(ce) if v >= 0.9), len(ce))
            ce_label.object = f"90% energy at component {t90} of {len(ce)}"

        svd_metrics.clear()
        svd_metrics.extend([
            pn.indicators.Number(name="Stable Rank",
                                 value=round(e.get("stable_rank", 0), 1),
                                 font_size="16pt", title_size="9pt"),
            pn.indicators.Number(name="Effective Rank",
                                 value=round(e.get("effective_rank", 0), 1),
                                 font_size="16pt", title_size="9pt"),
            pn.indicators.Number(name="Condition #",
                                 value=round(e.get("condition_number", 0)),
                                 font_size="16pt", title_size="9pt"),
        ])

    svd_select.param.watch(_populate_svd, "value")

    def _populate_time(event):
        stats = ipc.read_weight_stats()
        if not stats:
            return
        layer_data = sorted(
            [e for e in stats if e["layer"] == event.new],
            key=lambda e: e["step"],
        )
        if not layer_data:
            return
        time_src.data = {
            "step": [e["step"] for e in layer_data],
            "mean": [e["mean"] for e in layer_data],
            "std": [e["std"] for e in layer_data],
            "min": [e["min"] for e in layer_data],
            "max": [e["max"] for e in layer_data],
        }

    time_select.param.watch(_populate_time, "value")

    # ── periodic update ──────────────────────────────────────────────

    def update():
        stats = ipc.read_weight_stats()
        if not stats:
            info_pane.visible = True
            content.visible = False
            return

        info_pane.visible = False
        content.visible = True

        current_step = latest_step(stats)
        if current_step == state["last_step"]:
            _check_heatmap_response()
            return
        state["last_step"] = current_step
        step_label.object = f"**Latest snapshot: step {current_step}**"

        latest = latest_entries(stats)
        _render_norms(latest)
        _render_rank_table(latest)

        # update selector options only (visuals generated on demand)
        hist_entries = [e for e in latest if "hist_counts" in e]
        if hist_entries:
            hist_select.options = [e["layer"] for e in hist_entries]

        svd_entries = [e for e in latest if "singular_values" in e]
        if svd_entries:
            svd_select.options = [e["layer"] for e in svd_entries]

        time_select.options = sorted({e["layer"] for e in stats})

        weight_2d = [e for e in latest
                     if len(e.get("shape", [])) == 2 and "bias" not in e["layer"]]
        if weight_2d:
            hm_select.options = [e["layer"] for e in weight_2d]

        _check_heatmap_response()

    def _check_heatmap_response():
        resp = ipc.read_response("weight_heatmap")
        if not resp:
            return
        if "error" in resp:
            hm_caption.object = f"**Error:** {resp['error']}"
            return
        import numpy as np
        matrix = np.array(resp["matrix"])
        vmax = resp["vmax"]
        hm_caption.object = (
            f"Shape: {resp['shape']} | Display: {resp['display_shape']} | "
            f"Sparsity: {resp['sparsity']:.1f}%"
        )
        palette = list(reversed(RdBu11))
        mapper = LinearColorMapper(palette=palette, low=-vmax, high=vmax)
        fig = make_figure(
            title=short_layer_name(resp.get("layer", "")),
            x_label="Column", y_label="Row", height=400,
        )
        fig.image(image=[matrix], x=0, y=0,
                  dw=matrix.shape[1], dh=matrix.shape[0],
                  color_mapper=mapper)
        cb = ColorBar(color_mapper=mapper, location=(0, 0))
        fig.add_layout(cb, "right")
        hm_pane.object = fig

    return layout, update
