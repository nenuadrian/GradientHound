"""Optimizer page -- detailed optimizer state inspection."""
from __future__ import annotations

import math

import panel as pn
from bokeh.models import ColumnDataSource

from ._common import (
    make_figure, make_hbar_figure, short_layer_name, style_legend,
    fmt_bytes, fmt_num, PALETTE,
)


def create(ipc):
    """Build the optimizer page.  Returns ``(layout, update_fn)``."""

    info_pane = pn.pane.Alert("Waiting for optimizers to be registered...",
                               alert_type="info")
    content = pn.Column(sizing_mode="stretch_width", visible=False)

    layout = pn.Column(
        pn.pane.Markdown("## Optimizer Inspector"),
        info_pane, content, sizing_mode="stretch_width",
    )

    state: dict = {"last_step": -1, "built": False}

    def update():
        optimizers = ipc.read_optimizers()
        if not optimizers:
            info_pane.visible = True
            content.visible = False
            return
        info_pane.visible = False
        content.visible = True

        opt_state_history = ipc.read_optimizer_state()
        grad_stats = ipc.read_gradient_stats()

        # detect change
        cur_step = 0
        for s in opt_state_history:
            for gs in s.get("groups", []):
                cur_step = max(cur_step, gs.get("optimizer_step", 0))
        if cur_step == state["last_step"] and state["built"]:
            return
        state["last_step"] = cur_step

        # rebuild content
        sections: list = []

        for opt_name, opt_info in optimizers.items():
            opt_type = opt_info["type"]
            total_numel = opt_info.get("total_numel", 0)
            total_mem = opt_info.get("total_memory_bytes", 0)
            n_groups = len(opt_info.get("param_groups", []))
            groups = opt_info.get("param_groups", [])

            opt_states = [s for s in opt_state_history
                          if s.get("optimizer") == opt_name]
            latest_state = opt_states[-1] if opt_states else None
            opt_step = 0
            if latest_state:
                for gs in latest_state.get("groups", []):
                    opt_step = max(opt_step, gs.get("optimizer_step", 0))

            # summary metrics
            summary = pn.Row(
                pn.indicators.Number(name="Parameters",
                                     value=total_numel,
                                     format="{value:,.0f}",
                                     font_size="20pt", title_size="9pt"),
                pn.indicators.Number(name="Param Groups",
                                     value=n_groups,
                                     font_size="20pt", title_size="9pt"),
                pn.indicators.Number(name="State Memory",
                                     value=0,
                                     format=fmt_bytes(total_mem),
                                     font_size="20pt", title_size="9pt"),
                pn.indicators.Number(name="Optimizer Step",
                                     value=opt_step,
                                     format="{value:,}",
                                     font_size="20pt", title_size="9pt"),
                sizing_mode="stretch_width",
            )

            # configuration
            import pandas as pd
            config_df = pd.DataFrame([
                {"Parameter": k, "Value": v}
                for k, v in opt_info.get("defaults", {}).items()
            ])
            config_card = pn.Card(
                pn.pane.DataFrame(config_df, sizing_mode="stretch_width"),
                title="Configuration", collapsed=False,
                sizing_mode="stretch_width",
            ) if not config_df.empty else pn.Spacer(height=0)

            # param groups
            pg_rows = []
            for pg in groups:
                row = {
                    "Group": pg.get("index", "?"),
                    "Params": pg.get("num_params", 0),
                    "Elements": fmt_num(pg.get("total_numel", 0)),
                    "% Total": f"{pg.get('pct_of_total', 0):.1f}%",
                    "Memory": fmt_bytes(pg.get("memory_bytes", 0)),
                    "LR": f"{pg.get('lr', 0):.2e}",
                }
                if "weight_decay" in pg:
                    row["Weight Decay"] = pg["weight_decay"]
                if "momentum" in pg:
                    row["Momentum"] = pg["momentum"]
                if "betas" in pg:
                    row["Betas"] = str(pg["betas"])
                pg_rows.append(row)

            pg_card_children = [
                pn.pane.DataFrame(pd.DataFrame(pg_rows),
                                  sizing_mode="stretch_width")
            ] if pg_rows else []
            lrs = [pg.get("lr", 0) for pg in groups]
            if len(set(lrs)) > 1:
                pg_card_children.append(pn.pane.Alert(
                    f"LRs differ: {', '.join(f'{lr:.2e}' for lr in lrs)}",
                    alert_type="info",
                ))
            pg_card = pn.Card(
                *pg_card_children,
                title="Parameter Groups", collapsed=False,
                sizing_mode="stretch_width",
            ) if pg_card_children else pn.Spacer(height=0)

            opt_section_parts = [
                pn.pane.Markdown(f"### {opt_name} ({opt_type})"),
                summary, config_card, pg_card,
            ]

            # effective LR
            if latest_state and any(
                "effective_lr" in gs for gs in latest_state.get("groups", [])
            ):
                eff_row = pn.Row(sizing_mode="stretch_width")
                for gs in latest_state.get("groups", []):
                    if "effective_lr" in gs:
                        eff_row.append(pn.indicators.Number(
                            name=f"Group {gs.get('group_index', 0)}",
                            value=gs["effective_lr"],
                            format="{value:.2e}",
                            font_size="18pt", title_size="9pt",
                        ))

                eff_parts = [
                    pn.pane.Markdown(
                        "*Actual update magnitude: lr / (sqrt(mean(v)) + eps)*"),
                    eff_row,
                ]
                if len(opt_states) > 1:
                    eff_src, eff_fig = _build_time_chart(
                        opt_states, "effective_lr", "Effective LR Over Time")
                    eff_parts.append(
                        pn.pane.Bokeh(eff_fig, sizing_mode="stretch_width"))

                opt_section_parts.append(pn.Card(
                    *eff_parts, title="Effective Learning Rate",
                    collapsed=False, sizing_mode="stretch_width",
                ))

            # optimizer state health
            if latest_state:
                has_mom = any(
                    "exp_avg_norm_mean" in gs or "momentum_norm_mean" in gs
                    for gs in latest_state.get("groups", [])
                )
                if has_mom:
                    health_parts = []
                    for gs in latest_state.get("groups", []):
                        gi = gs.get("group_index", 0)
                        row = pn.Row(sizing_mode="stretch_width")
                        if "exp_avg_norm_mean" in gs:
                            row.extend([
                                pn.indicators.Number(
                                    name="Avg Momentum Norm",
                                    value=round(gs["exp_avg_norm_mean"], 4),
                                    font_size="16pt", title_size="9pt"),
                                pn.indicators.Number(
                                    name="Max Momentum Norm",
                                    value=round(gs.get("exp_avg_norm_max", 0), 4),
                                    font_size="16pt", title_size="9pt"),
                            ])
                        elif "momentum_norm_mean" in gs:
                            row.extend([
                                pn.indicators.Number(
                                    name="Avg Momentum Norm",
                                    value=round(gs["momentum_norm_mean"], 4),
                                    font_size="16pt", title_size="9pt"),
                                pn.indicators.Number(
                                    name="Max Momentum Norm",
                                    value=round(gs.get("momentum_norm_max", 0), 4),
                                    font_size="16pt", title_size="9pt"),
                            ])
                        if "exp_avg_sq_mean" in gs:
                            row.append(pn.indicators.Number(
                                name="Mean 2nd Moment",
                                value=gs["exp_avg_sq_mean"],
                                format="{value:.2e}",
                                font_size="16pt", title_size="9pt"))
                        if "warmup_pct" in gs:
                            row.append(pn.indicators.Number(
                                name="Bias Correction Warmup",
                                value=round(gs["warmup_pct"]),
                                format="{value}%",
                                font_size="16pt", title_size="9pt"))
                        health_parts.append(pn.pane.Markdown(f"**Group {gi}**"))
                        health_parts.append(row)

                    if len(opt_states) > 1:
                        field = "exp_avg_norm_mean"
                        if not any(field in gs for gs in latest_state.get("groups", [])):
                            field = "momentum_norm_mean"
                        msrc, mfig = _build_time_chart(
                            opt_states, field, "Momentum Buffer Norms Over Time")
                        health_parts.append(
                            pn.pane.Bokeh(mfig, sizing_mode="stretch_width"))

                    opt_section_parts.append(pn.Card(
                        *health_parts, title="Optimizer State Health",
                        collapsed=False, sizing_mode="stretch_width",
                    ))

            # update magnitude
            if grad_stats and groups:
                latest_gs = max(e["step"] for e in grad_stats)
                latest_grads = [e for e in grad_stats if e["step"] == latest_gs]
                mag_parts = [pn.pane.Markdown(
                    "*lr * grad_norm / weight_norm. Healthy range: [1e-4, 1e-2]*")]
                for gi, pg in enumerate(groups):
                    lr = pg.get("lr", 1e-3)
                    # Build layers and ratios, deduplicating short names
                    # so the FactorRange never receives duplicate factors.
                    seen: dict[str, int] = {}
                    layers: list[str] = []
                    ratios: list[float] = []
                    for e in latest_grads:
                        name = short_layer_name(e["layer"])
                        count = seen.get(name, 0)
                        seen[name] = count + 1
                        if count > 0:
                            name = f"{name} [{count}]"
                        ratio = (lr * e.get("grad_norm", 0)) / max(e.get("weight_norm", 1e-20), 1e-20)
                        if not math.isfinite(ratio):
                            ratio = 0.0
                        layers.append(name)
                        ratios.append(ratio)
                    if not layers:
                        continue
                    src = ColumnDataSource(data={"layer": layers, "ratio": ratios})
                    fig = make_hbar_figure(
                        title=f"Group {gi} (lr={lr:.2e})",
                        y_range=list(reversed(layers)), height=250,
                    )
                    fig.hbar(y="layer", right="ratio", source=src, height=0.7,
                             color=PALETTE[gi % len(PALETTE)])
                    mag_parts.append(
                        pn.pane.Bokeh(fig, sizing_mode="stretch_width"))

                opt_section_parts.append(pn.Card(
                    *mag_parts, title="Update Magnitude by Param Group",
                    collapsed=True, sizing_mode="stretch_width",
                ))

            # tips
            tips = _generate_tips(opt_type, opt_info, latest_state)
            opt_section_parts.append(pn.Card(
                pn.pane.Markdown("\n".join(f"- {t}" for t in tips)),
                title="Optimizer Insights", collapsed=True,
                sizing_mode="stretch_width",
            ))

            sections.extend(opt_section_parts)

        content.clear()
        content.extend(sections)
        state["built"] = True

    return layout, update


def _build_time_chart(opt_states, field, title):
    """Build a multi-group time-series Bokeh chart."""
    time_data: dict[int, dict[str, float]] = {}
    for snap in opt_states:
        step = snap["step"]
        for gs in snap.get("groups", []):
            if field in gs:
                gi = gs.get("group_index", 0)
                if step not in time_data:
                    time_data[step] = {}
                time_data[step][f"Group {gi}"] = gs[field]

    if not time_data:
        src = ColumnDataSource(data={"xs": [], "ys": [], "colors": []})
        fig = make_figure(title=title, x_label="step", height=250)
        fig.multi_line("xs", "ys", source=src, line_color="colors",
                       line_width=2)
        return src, fig

    steps = sorted(time_data.keys())
    group_names = sorted({k for v in time_data.values() for k in v})
    xs, ys, colors = [], [], []
    for i, gn in enumerate(group_names):
        gx, gy = [], []
        for s in steps:
            if gn in time_data[s]:
                val = time_data[s][gn]
                if val is not None and math.isfinite(val):
                    gx.append(s)
                    gy.append(val)
        if gx:
            xs.append(gx)
            ys.append(gy)
            colors.append(PALETTE[i % len(PALETTE)])

    src = ColumnDataSource(data={"xs": xs, "ys": ys, "colors": colors})
    fig = make_figure(title=title, x_label="step", height=250)
    fig.multi_line("xs", "ys", source=src, line_color="colors", line_width=2)
    return src, fig


def _generate_tips(opt_type, opt_info, latest_state):
    tips = []
    defaults = opt_info.get("defaults", {})
    total_mem = opt_info.get("total_memory_bytes", 0)
    buffers = opt_info.get("buffers_per_param", 0)

    if opt_type in ("Adam", "AdamW", "NAdam", "RAdam"):
        tips.append(
            f"**{opt_type}** maintains 2 state buffers per parameter "
            f"(momentum + variance), using ~{fmt_bytes(total_mem)} of memory.")
        if opt_type == "Adam" and defaults.get("weight_decay", 0) > 0:
            tips.append(
                "Using `Adam` with weight_decay applies L2 to the gradient. "
                "Consider `AdamW` for decoupled weight decay.")
        eps = defaults.get("eps", 1e-8)
        if eps == 1e-8:
            tips.append(
                "Epsilon is at default 1e-8. For mixed-precision, "
                "consider 1e-5 or 1e-4.")
        betas = defaults.get("betas")
        if isinstance(betas, (list, tuple)) and len(betas) == 2 and betas[1] == 0.999:
            tips.append(
                "Beta2=0.999 is default. For noisy problems, "
                "try 0.98 or 0.95.")

    elif opt_type == "SGD":
        mom = defaults.get("momentum", 0)
        if mom == 0:
            tips.append(
                "**SGD without momentum.** Consider momentum=0.9 "
                "to accelerate convergence.")
        elif buffers == 1:
            tips.append(
                f"**SGD + momentum ({mom})** uses ~{fmt_bytes(total_mem)} "
                "of state memory.")
        tips.append(
            "SGD applies the same LR to all parameters. "
            "Use param groups for fine-tuning.")

    wd = defaults.get("weight_decay", 0)
    if wd == 0:
        tips.append(
            "Weight decay is **0**. Consider 1e-4 to 1e-2 for regularization.")

    if latest_state:
        for gs in latest_state.get("groups", []):
            wp = gs.get("warmup_pct", 100)
            if wp < 50:
                tips.append(
                    f"Group {gs.get('group_index', '?')} bias correction "
                    f"is only {wp:.0f}% converged -- consider longer warmup.")
                break

    return tips or ["No specific recommendations for this configuration."]
