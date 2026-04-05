"""Training page -- predictions, CKA similarity, attention patterns, activations."""
from __future__ import annotations

import panel as pn
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar

from ._common import (
    make_figure, make_hbar_figure, short_layer_name, short_layer_names,
    BLUE, latest_entries,
)


def create(ipc):
    """Build the training dynamics page.  Returns ``(layout, update_fn)``."""

    info_pane = pn.pane.Alert(
        "No training dynamics data yet.  Use `gh.watch(model, log_activations=True)`, "
        "`gh.log_predictions()`, or `gh.log_attention()` to populate this page.",
        alert_type="info",
    )

    # ── Predictions vs Actual ────────────────────────────────────────
    pred_select = pn.widgets.Select(name="Source", options=[], width=300)
    pred_src = ColumnDataSource(data={"actual": [], "predicted": []})
    pred_fig = make_figure(title="Predictions vs Actual (Value Calibration)",
                           x_label="Actual", y_label="Predicted", height=350)
    pred_fig.scatter("actual", "predicted", source=pred_src, size=5,
                     color=BLUE, alpha=0.6)
    # diagonal reference line
    pred_fig.line([0, 1], [0, 1], line_dash="dashed", line_color="#5e4a4e",
                  line_width=1)
    pred_caption = pn.pane.Markdown("")

    pred_card = pn.Card(
        pred_select,
        pn.pane.Bokeh(pred_fig, sizing_mode="stretch_width"),
        pred_caption,
        title="Predictions vs Actual", collapsed=True,
        sizing_mode="stretch_width", visible=False,
    )

    # ── CKA (on-demand) ──────────────────────────────────────────────
    cka_button = pn.widgets.Button(name="Compute CKA Matrix",
                                    button_type="primary")
    cka_pane = pn.pane.Bokeh(sizing_mode="stretch_width")
    cka_caption = pn.pane.Markdown(
        "*Click the button to compute CKA similarity between all 2D weight layers.*")
    cka_summary = pn.Row(sizing_mode="stretch_width")

    def _on_cka_click(event):
        ipc.clear_response("cka")
        ipc.write_request({"id": "cka", "type": "cka"})
        cka_caption.object = "*Computing...*"

    cka_button.on_click(_on_cka_click)

    cka_card = pn.Card(
        cka_button, cka_caption, cka_pane, cka_summary,
        title="CKA Layer Similarity (on-demand)", collapsed=True,
        sizing_mode="stretch_width",
    )

    # ── Attention Patterns ───────────────────────────────────────────
    attn_caption = pn.pane.Markdown("")
    attn_slider = pn.widgets.IntSlider(name="Head", start=0, end=0, value=0,
                                        width=300)
    attn_pane = pn.pane.Bokeh(sizing_mode="stretch_width")

    attn_card = pn.Card(
        attn_caption, attn_slider, attn_pane,
        title="Attention Patterns", collapsed=True,
        sizing_mode="stretch_width", visible=False,
    )

    # ── Activation Stats ─────────────────────────────────────────────
    act_table = pn.pane.DataFrame(sizing_mode="stretch_width")
    act_src = ColumnDataSource(data={"layer": [], "zero_frac": []})
    act_fig = make_hbar_figure(title="Zero Fraction by Layer", y_range=[],
                                height=300)
    act_fig.hbar(y="layer", right="zero_frac", source=act_src, height=0.7,
                 color="#e8a87c")
    act_button = pn.widgets.Button(name="Compute", button_type="primary")

    def _on_act_click(event):
        import pandas as pd
        act_stats = ipc.read_activation_stats()
        if not act_stats:
            return
        latest_act = latest_entries(act_stats)
        act_table.object = pd.DataFrame([{
            "Layer": short_layer_name(e["layer"]),
            "Mean": round(e["mean"], 4),
            "Std": round(e["std"], 4),
            "Min": round(e["min"], 4),
            "Max": round(e["max"], 4),
            "Zero %": f"{e.get('zero_fraction', 0) * 100:.1f}%",
        } for e in latest_act])
        al = short_layer_names([e["layer"] for e in latest_act])
        zf = [e.get("zero_fraction", 0) for e in latest_act]
        act_src.data = {"layer": al, "zero_frac": zf}
        act_fig.y_range.factors = list(reversed(al))

    act_button.on_click(_on_act_click)

    act_card = pn.Card(
        act_button, act_table,
        pn.pane.Bokeh(act_fig, sizing_mode="stretch_width"),
        title="Activation Statistics", collapsed=True,
        sizing_mode="stretch_width", visible=False,
    )

    # ── layout ───────────────────────────────────────────────────────
    content = pn.Column(
        pred_card, cka_card, attn_card, act_card,
        sizing_mode="stretch_width",
    )

    layout = pn.Column(
        pn.pane.Markdown("## Training Dynamics"),
        info_pane, content, sizing_mode="stretch_width",
    )

    state: dict = {"last_attn": None}

    # ── attention head selector ──────────────────────────────────────

    def _render_attn_head(event):
        attention = ipc.read_attention()
        if not attention:
            return
        latest = attention[-1]
        weights = latest["weights"]
        head = event.new
        if head >= len(weights):
            return
        import numpy as np
        from bokeh.palettes import Viridis256
        matrix = np.array(weights[head])
        mapper = LinearColorMapper(palette=Viridis256, low=float(matrix.min()),
                                    high=float(matrix.max()))
        fig = make_figure(title=f"Head {head}", x_label="Key",
                          y_label="Query", height=350)
        fig.image(image=[matrix], x=0, y=0,
                  dw=matrix.shape[1], dh=matrix.shape[0],
                  color_mapper=mapper)
        cb = ColorBar(color_mapper=mapper, location=(0, 0))
        fig.add_layout(cb, "right")
        attn_pane.object = fig

    attn_slider.param.watch(_render_attn_head, "value")

    # ── prediction source selector ───────────────────────────────────

    def _render_predictions(event):
        predictions = ipc.read_predictions()
        entries = [e for e in predictions if e["name"] == event.new]
        if not entries:
            return
        latest = entries[-1]
        pred = latest["predicted"]
        actual = latest["actual"]
        n = min(len(pred), len(actual))
        pred_src.data = {"actual": actual[:n], "predicted": pred[:n]}
        # adjust diagonal
        all_vals = actual[:n] + pred[:n]
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            pred_fig.renderers[-1].data_source.data = {
                "x": [lo, hi], "y": [lo, hi],
            }
        pred_caption.object = (
            f"Step {latest.get('step', '?')} | {n} points | "
            "Points on diagonal = perfect calibration"
        )

    pred_select.param.watch(_render_predictions, "value")

    # ── periodic update ──────────────────────────────────────────────

    def update():
        predictions = ipc.read_predictions()
        attention = ipc.read_attention()
        act_stats = ipc.read_activation_stats()

        has_data = bool(predictions or attention or act_stats)
        info_pane.visible = not has_data

        # predictions - update selector options only (visual on demand via select)
        if predictions:
            pred_card.visible = True
            names = sorted({e["name"] for e in predictions})
            if pred_select.options != names:
                pred_select.options = names

        # CKA response check
        resp = ipc.read_response("cka")
        if resp and "error" not in resp:
            import numpy as np
            from bokeh.palettes import Viridis256
            matrix = np.array(resp["matrix"])
            short_names = resp["short_names"]
            mapper = LinearColorMapper(palette=Viridis256, low=0, high=1)
            fig = make_figure(
                title="Linear CKA Similarity", height=400,
                x_range=short_names, y_range=list(reversed(short_names)),
            )
            fig.image(image=[np.flipud(matrix)], x=0, y=0,
                      dw=len(short_names), dh=len(short_names),
                      color_mapper=mapper)
            cb = ColorBar(color_mapper=mapper, location=(0, 0))
            fig.add_layout(cb, "right")
            fig.xaxis.major_label_orientation = 0.8
            cka_pane.object = fig
            cka_caption.object = (
                f"Step {resp.get('step', '?')} | {resp['n']} weight layers")

            n = len(matrix)
            upper = [matrix[i][j] for i in range(n) for j in range(i+1, n)]
            if upper:
                cka_summary.clear()
                cka_summary.extend([
                    pn.indicators.Number(name="Mean CKA (off-diag)",
                                         value=round(sum(upper)/len(upper), 3),
                                         font_size="16pt", title_size="9pt"),
                    pn.indicators.Number(name="Max CKA (off-diag)",
                                         value=round(max(upper), 3),
                                         font_size="16pt", title_size="9pt"),
                ])
        elif resp and "error" in resp:
            cka_caption.object = f"**Error:** {resp['error']}"

        # attention - update visibility and slider config only (visual on demand via slider)
        if attention:
            attn_card.visible = True
            latest = attention[-1]
            n_heads = len(latest["weights"])
            attn_caption.object = (
                f"Step {latest.get('step', '?')} | Layer: {latest['name']} | "
                f"Shape: {latest.get('shape', '?')}")
            if attn_slider.end != max(0, n_heads - 1):
                attn_slider.end = max(0, n_heads - 1)

        # activations - toggle visibility only (visual on demand via button)
        if act_stats:
            act_card.visible = True

    return layout, update
