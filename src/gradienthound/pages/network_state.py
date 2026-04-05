"""Network State page -- layer-by-layer tables of all weight values."""
from __future__ import annotations

import uuid

import panel as pn

from ._common import short_layer_name


_MAX_PARAMS = 1_000_000


def create(ipc):
    """Build the network state page.  Returns ``(layout, update_fn)``."""

    model_select = pn.widgets.Select(name="Model", options=[], width=300)
    info_pane = pn.pane.Alert(
        "Waiting for models to be registered...", alert_type="info")

    param_row = pn.Row(sizing_mode="stretch_width")
    too_large_pane = pn.pane.Alert("", alert_type="warning", visible=False)

    capture_btn = pn.widgets.Button(
        name="Capture Network State", button_type="primary", width=300)
    capture_status = pn.pane.Markdown("")

    state_container = pn.Column(sizing_mode="stretch_width")
    precision_slider = pn.widgets.IntSlider(
        name="Display precision (decimal places)", start=2, end=8, value=4,
        width=300,
    )

    layout = pn.Column(
        pn.pane.Markdown("## Network State"),
        model_select, info_pane,
        param_row, too_large_pane,
        capture_btn, capture_status,
        precision_slider, state_container,
        sizing_mode="stretch_width",
    )

    cache: dict = {}
    # Pending request state -- written by button click, polled by update()
    pending: dict = {"req_id": None, "model": None}

    # ── button click: fire-and-forget, no blocking ───────────────────

    def _on_capture(event):
        selected = model_select.value
        if not selected:
            return
        req_id = f"network_state_{uuid.uuid4().hex[:8]}"
        ipc.write_request({
            "type": "network_state", "id": req_id, "model": selected,
        })
        pending["req_id"] = req_id
        pending["model"] = selected
        capture_status.object = (
            "*Requesting network state from training process -- "
            "waiting for next `gh.step()` call...*")
        capture_btn.disabled = True

    capture_btn.on_click(_on_capture)

    # ── render helper ────────────────────────────────────────────────

    def _render_state(state_data):
        import numpy as np
        import pandas as pd

        layers = state_data["layers"]
        total_params = state_data["total_params"]
        step = state_data.get("step", "?")
        model_name = state_data.get("model", "?")
        precision = precision_slider.value

        state_container.clear()
        state_container.append(pn.pane.Markdown(
            f"**Snapshot at step {step}** | Model: **{model_name}** | "
            f"Total parameters: **{total_params:,}**"))

        # summary table
        summary_rows = []
        for layer in layers:
            vals = np.array(layer["values"]).flatten()
            summary_rows.append({
                "Layer": short_layer_name(layer["name"]),
                "Shape": str(layer["shape"]),
                "Params": f"{layer['numel']:,}",
                "Mean": f"{vals.mean():.{precision}f}",
                "Std": f"{vals.std():.{precision}f}",
                "Min": f"{vals.min():.{precision}f}",
                "Max": f"{vals.max():.{precision}f}",
                "Grad": "Yes" if layer["requires_grad"] else "No",
            })

        state_container.append(pn.pane.Markdown("### Layer Summary"))
        state_container.append(pn.pane.DataFrame(
            pd.DataFrame(summary_rows), sizing_mode="stretch_width"))

        # layer-by-layer tables
        state_container.append(pn.pane.Markdown("### Layer-by-Layer Weights"))

        for layer in layers:
            shape = layer["shape"]
            numel = layer["numel"]
            values = layer["values"]
            label = (f"{layer['name']}  |  shape={shape}  |  "
                     f"{numel:,} params")

            if len(shape) <= 1:
                flat = values[0] if values else []
                df = pd.DataFrame(
                    [flat],
                    columns=[str(i) for i in range(len(flat))],
                    index=["value"],
                )
            elif len(shape) == 2:
                rows, cols = shape
                df = pd.DataFrame(
                    values,
                    columns=[str(c) for c in range(cols)],
                    index=[str(r) for r in range(rows)],
                )
            else:
                flat_cols = len(values[0]) if values else 0
                df = pd.DataFrame(
                    values,
                    columns=[str(c) for c in range(flat_cols)],
                )

            styled = df.style.format(f"{{:.{precision}f}}")
            if df.size > 0 and len(shape) >= 2:
                vmax = abs(df.values).max()
                if vmax > 0:
                    styled = styled.background_gradient(
                        cmap="RdBu_r", axis=None,
                        vmin=-vmax, vmax=vmax)

            card = pn.Card(
                pn.pane.DataFrame(
                    styled, sizing_mode="stretch_width",
                    max_height=600),
                title=label, collapsed=True,
                sizing_mode="stretch_width",
            )
            state_container.append(card)

    precision_slider.param.watch(
        lambda e: _render_state(cache[model_select.value])
        if model_select.value and model_select.value in cache else None,
        "value",
    )

    # ── periodic update -- also polls for pending response ───────────

    def update():
        models = ipc.read_models()
        if not models:
            info_pane.visible = True
            return
        info_pane.visible = False

        names = list(models.keys())
        if model_select.options != names:
            model_select.options = names
            if model_select.value not in names:
                model_select.value = names[0]

        selected = model_select.value
        if not selected:
            return

        graph_data = models[selected]
        total_params = graph_data.get("total_params", 0)

        param_row.clear()
        param_row.extend([
            pn.indicators.Number(name="Total Parameters",
                                 value=total_params, format="{value:,.0f}",
                                 font_size="20pt", title_size="9pt"),
            pn.indicators.Number(
                name="Status", value=0,
                format="Eligible" if total_params <= _MAX_PARAMS else "Too Large",
                font_size="20pt", title_size="9pt"),
        ])

        if total_params > _MAX_PARAMS:
            too_large_pane.object = (
                f"Network state view is limited to models with "
                f"<= {_MAX_PARAMS:,} parameters.  This model has "
                f"{total_params:,}.")
            too_large_pane.visible = True
            capture_btn.disabled = True
        else:
            too_large_pane.visible = False
            if not pending["req_id"]:
                capture_btn.disabled = False

        # render cached state if available
        if selected in cache and not state_container.objects:
            _render_state(cache[selected])

        # poll for pending network-state response (non-blocking)
        if pending["req_id"]:
            resp = ipc.read_response(pending["req_id"])
            if resp is not None:
                ipc.clear_response(pending["req_id"])
                req_model = pending["model"]
                pending["req_id"] = None
                pending["model"] = None
                capture_btn.disabled = False

                if "error" in resp:
                    capture_status.object = f"**Error:** {resp['error']}"
                else:
                    cache[req_model] = resp
                    capture_status.object = ""
                    _render_state(resp)

    return layout, update
