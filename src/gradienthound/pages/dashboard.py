"""Dashboard page -- run overview without live auto-refresh churn."""
from __future__ import annotations

import panel as pn

from ._common import latest_step


def create(ipc):
    """Build the dashboard page. Returns ``(layout, update_fn)``."""

    run_state = pn.pane.Markdown(
        "Run details will appear here once models or metadata are registered.",
        sizing_mode="stretch_width",
    )
    metadata_card = pn.Card(
        pn.pane.Str("No metadata"),
        title="Run Metadata",
        collapsed=False,
        visible=False,
        sizing_mode="stretch_width",
    )
    summary_row = pn.Row(sizing_mode="stretch_width")
    data_row = pn.Row(sizing_mode="stretch_width")
    notes_pane = pn.pane.Alert(
        "This page stays stable while you browse. Open Metrics for live charts.",
        alert_type="info",
        sizing_mode="stretch_width",
    )

    layout = pn.Column(
        pn.pane.Markdown("## Run Overview"),
        run_state,
        summary_row,
        data_row,
        metadata_card,
        notes_pane,
        sizing_mode="stretch_width",
    )

    def update():
        models = ipc.read_models()
        metadata = ipc.read_metadata()
        metrics = ipc.read_metrics()
        gradients = ipc.read_gradient_stats()
        weights = ipc.read_weight_stats()
        activations = ipc.read_activation_stats()
        optimizers = ipc.read_optimizers()

        total_params = sum(m.get("total_params", 0) for m in models.values())
        model_names = ", ".join(models.keys()) if models else "None yet"
        run_state.object = (
            f"**Models:** {model_names}\n\n"
            f"**Latest metric step:** {latest_step(metrics, key='_step'):,}\n\n"
            f"**Latest gradient step:** {latest_step(gradients):,}"
        )

        summary_row.clear()
        summary_row.extend([
            pn.indicators.Number(
                name="Models",
                value=len(models),
                font_size="24pt",
                title_size="10pt",
            ),
            pn.indicators.Number(
                name="Parameters",
                value=total_params,
                format="{value:,.0f}",
                font_size="24pt",
                title_size="10pt",
            ),
            pn.indicators.Number(
                name="Optimizers",
                value=len(optimizers),
                font_size="24pt",
                title_size="10pt",
            ),
        ])

        data_row.clear()
        data_row.extend([
            pn.indicators.Number(
                name="Metric Records",
                value=len(metrics),
                font_size="20pt",
                title_size="10pt",
            ),
            pn.indicators.Number(
                name="Gradient Snapshots",
                value=len(gradients),
                font_size="20pt",
                title_size="10pt",
            ),
            pn.indicators.Number(
                name="Weight Snapshots",
                value=len(weights),
                font_size="20pt",
                title_size="10pt",
            ),
            pn.indicators.Number(
                name="Activation Snapshots",
                value=len(activations),
                font_size="20pt",
                title_size="10pt",
            ),
        ])

        if metadata:
            metadata_card.visible = True
            metadata_card[0] = pn.pane.Str(
                "\n".join(f"{k}: {v}" for k, v in metadata.items()),
            )
        else:
            metadata_card.visible = False

    return layout, update
