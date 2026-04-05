"""Optimizer page -- detailed optimizer state inspection."""
from __future__ import annotations

import streamlit as st

from gradienthound.pages._common import inject_css, get_ipc, short_layer_name

inject_css()

st.header("Optimizer Inspector")

ipc = get_ipc()
if ipc is None:
    st.error("IPC channel not initialised.")
    st.stop()


def _fmt_bytes(b: int | float) -> str:
    if b < 1024:
        return f"{b:.0f} B"
    if b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    if b < 1024 ** 3:
        return f"{b / 1024 ** 2:.1f} MB"
    return f"{b / 1024 ** 3:.2f} GB"


def _fmt_num(n: int | float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(int(n))


@st.fragment(run_every=2)
def _optimizer_inspector() -> None:
    import pandas as pd

    optimizers = ipc.read_optimizers()
    if not optimizers:
        st.info("Waiting for optimizers to be registered...")
        return

    opt_state_history = ipc.read_optimizer_state()
    grad_stats = ipc.read_gradient_stats()

    for opt_name, opt_info in optimizers.items():
        opt_type = opt_info["type"]
        st.subheader(f"{opt_name} ({opt_type})")

        # ── Summary Metrics ──────────────────────────────────────────
        total_numel = opt_info.get("total_numel", 0)
        total_mem = opt_info.get("total_memory_bytes", 0)
        n_groups = len(opt_info.get("param_groups", []))
        buffers = opt_info.get("buffers_per_param", 0)

        # Get optimizer step from latest state
        opt_step = 0
        latest_state = None
        opt_states = [s for s in opt_state_history if s.get("optimizer") == opt_name]
        if opt_states:
            latest_state = opt_states[-1]
            for gs in latest_state.get("groups", []):
                opt_step = max(opt_step, gs.get("optimizer_step", 0))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Parameters", _fmt_num(total_numel))
        c2.metric("Param Groups", str(n_groups))
        c3.metric("State Memory", _fmt_bytes(total_mem) if total_mem else "0 B")
        c4.metric("Optimizer Step", f"{opt_step:,}" if opt_step else "N/A")

        # ── Configuration ────────────────────────────────────────────
        if opt_info["defaults"]:
            with st.expander("Configuration", expanded=True):
                defaults_df = pd.DataFrame(
                    [{"Parameter": k, "Value": v} for k, v in opt_info["defaults"].items()]
                )
                st.dataframe(defaults_df, use_container_width=True, hide_index=True)

        # ── Parameter Groups ─────────────────────────────────────────
        groups = opt_info.get("param_groups", [])
        if groups:
            with st.expander("Parameter Groups", expanded=True):
                display_rows = []
                for pg in groups:
                    row = {
                        "Group": pg.get("index", "?"),
                        "Params": pg.get("num_params", 0),
                        "Elements": _fmt_num(pg.get("total_numel", 0)),
                        "% of Total": f"{pg.get('pct_of_total', 0):.1f}%",
                        "Memory": _fmt_bytes(pg.get("memory_bytes", 0)),
                        "LR": f"{pg.get('lr', 0):.2e}",
                    }
                    if "weight_decay" in pg:
                        row["Weight Decay"] = pg["weight_decay"]
                    if "momentum" in pg:
                        row["Momentum"] = pg["momentum"]
                    if "betas" in pg:
                        row["Betas"] = str(pg["betas"])
                    if "eps" in pg:
                        row["Eps"] = f"{pg['eps']:.1e}"
                    display_rows.append(row)
                st.dataframe(
                    pd.DataFrame(display_rows),
                    use_container_width=True, hide_index=True,
                )

                # Highlight LR differences across groups
                lrs = [pg.get("lr", 0) for pg in groups]
                if len(set(lrs)) > 1:
                    st.info(
                        f"Learning rates differ across groups: "
                        f"{', '.join(f'{lr:.2e}' for lr in lrs)}"
                    )

        # ── Effective Learning Rate (Adam family) ────────────────────
        if latest_state and any("effective_lr" in gs for gs in latest_state.get("groups", [])):
            with st.expander("Effective Learning Rate", expanded=True):
                st.caption(
                    "Actual update magnitude: lr / (sqrt(mean(v)) + eps). "
                    "Shows how adaptive optimization scales the base LR."
                )
                eff_cols = st.columns(max(len(latest_state["groups"]), 1))
                for idx, gs in enumerate(latest_state.get("groups", [])):
                    if "effective_lr" in gs:
                        base = gs.get("lr", 0)
                        eff = gs["effective_lr"]
                        col = eff_cols[idx % len(eff_cols)]
                        col.metric(
                            f"Group {gs.get('group_index', idx)}",
                            f"{eff:.2e}",
                            delta=f"base: {base:.2e}",
                            delta_color="off",
                        )

                # Effective LR over time
                if len(opt_states) > 1:
                    st.markdown("**Effective LR Over Time**")
                    time_data: dict[int, dict[str, float]] = {}
                    for snap in opt_states:
                        step = snap["step"]
                        for gs in snap.get("groups", []):
                            if "effective_lr" in gs:
                                gi = gs.get("group_index", 0)
                                if step not in time_data:
                                    time_data[step] = {}
                                time_data[step][f"Group {gi}"] = gs["effective_lr"]
                    if time_data:
                        df = pd.DataFrame.from_dict(time_data, orient="index").sort_index()
                        df.index.name = "step"
                        st.line_chart(df, use_container_width=True)

        # ── Optimizer State Health ───────────────────────────────────
        if latest_state:
            has_momentum = any(
                "exp_avg_norm_mean" in gs or "momentum_norm_mean" in gs
                for gs in latest_state.get("groups", [])
            )
            if has_momentum:
                with st.expander("Optimizer State Health", expanded=True):
                    for gs in latest_state.get("groups", []):
                        gi = gs.get("group_index", 0)
                        st.markdown(f"**Group {gi}**")
                        mcols = st.columns(4)

                        if "exp_avg_norm_mean" in gs:
                            mcols[0].metric("Avg Momentum Norm", f"{gs['exp_avg_norm_mean']:.4f}")
                            mcols[1].metric("Max Momentum Norm", f"{gs.get('exp_avg_norm_max', 0):.4f}")
                        if "momentum_norm_mean" in gs:
                            mcols[0].metric("Avg Momentum Norm", f"{gs['momentum_norm_mean']:.4f}")
                            mcols[1].metric("Max Momentum Norm", f"{gs.get('momentum_norm_max', 0):.4f}")
                        if "exp_avg_sq_mean" in gs:
                            mcols[2].metric("Mean Second Moment", f"{gs['exp_avg_sq_mean']:.2e}")
                        if "warmup_pct" in gs:
                            mcols[3].metric("Bias Correction Warmup", f"{gs['warmup_pct']:.0f}%")

                    # Momentum norms over time
                    if len(opt_states) > 1:
                        st.markdown("**Momentum Buffer Norms Over Time**")
                        time_data = {}
                        for snap in opt_states:
                            step = snap["step"]
                            for gs in snap.get("groups", []):
                                gi = gs.get("group_index", 0)
                                val = gs.get("exp_avg_norm_mean") or gs.get("momentum_norm_mean")
                                if val is not None:
                                    if step not in time_data:
                                        time_data[step] = {}
                                    time_data[step][f"Group {gi}"] = val
                        if time_data:
                            df = pd.DataFrame.from_dict(time_data, orient="index").sort_index()
                            df.index.name = "step"
                            st.line_chart(df, use_container_width=True)

        # ── Update Magnitude Analysis ────────────────────────────────
        if grad_stats and groups:
            with st.expander("Update Magnitude by Param Group", expanded=False):
                st.caption(
                    "lr * grad_norm / weight_norm per layer, "
                    "grouped by parameter group. Healthy range: [1e-4, 1e-2]."
                )
                latest_step = max(e["step"] for e in grad_stats)
                latest_grads = [e for e in grad_stats if e["step"] == latest_step]

                # Build a map of layer -> param group index
                # Use the number of params per group to approximate assignment
                group_param_counts = [pg.get("num_params", 0) for pg in groups]
                cumulative = []
                total = 0
                for c in group_param_counts:
                    cumulative.append(total)
                    total += c

                for gi, pg in enumerate(groups):
                    lr = pg.get("lr", 1e-3)
                    ratio_data = []
                    for e in latest_grads:
                        gn = e.get("grad_norm", 0)
                        wn = e.get("weight_norm", 1e-20)
                        ratio = (lr * gn) / max(wn, 1e-20)
                        ratio_data.append({
                            "Layer": short_layer_name(e["layer"]),
                            "Update/Weight": ratio,
                        })
                    if ratio_data:
                        df = pd.DataFrame(ratio_data)
                        if not df.empty:
                            st.markdown(f"**Group {gi}** (lr={lr:.2e})")
                            st.bar_chart(
                                df.set_index("Layer"), horizontal=True,
                            )

        # ── Optimizer Tips ───────────────────────────────────────────
        with st.expander("Optimizer Insights", expanded=False):
            tips = _generate_tips(opt_type, opt_info, latest_state)
            for tip in tips:
                st.markdown(f"- {tip}")


def _generate_tips(
    opt_type: str,
    opt_info: dict,
    latest_state: dict | None,
) -> list[str]:
    tips = []
    defaults = opt_info.get("defaults", {})
    total_mem = opt_info.get("total_memory_bytes", 0)
    buffers = opt_info.get("buffers_per_param", 0)

    if opt_type in ("Adam", "AdamW", "NAdam", "RAdam"):
        tips.append(
            f"**{opt_type}** maintains 2 state buffers per parameter "
            f"(momentum + variance), using ~{_fmt_bytes(total_mem)} of memory."
        )
        if opt_type == "Adam" and defaults.get("weight_decay", 0) > 0:
            tips.append(
                "Using `Adam` with weight_decay applies L2 penalty to the gradient, "
                "not the weights. Consider `AdamW` for decoupled weight decay."
            )
        eps = defaults.get("eps", 1e-8)
        if eps == 1e-8:
            tips.append(
                "Epsilon is at the default 1e-8. For mixed-precision training, "
                "consider increasing to 1e-5 or 1e-4."
            )
        betas = defaults.get("betas")
        if isinstance(betas, (list, tuple)):
            if len(betas) == 2 and betas[1] == 0.999:
                tips.append(
                    "Beta2=0.999 is the default. For noisy problems or large batch sizes, "
                    "0.98 or 0.95 can help stabilize training."
                )

    elif opt_type == "SGD":
        mom = defaults.get("momentum", 0)
        if mom == 0:
            tips.append(
                "**SGD without momentum.** Training may be slow. "
                "Momentum (0.9) accelerates convergence in most cases."
            )
        elif buffers == 1:
            tips.append(
                f"**SGD + momentum ({mom})** uses 1 buffer per parameter, "
                f"~{_fmt_bytes(total_mem)} of state memory."
            )
        tips.append(
            "SGD applies the same learning rate to all parameters. "
            "Use param groups with different LRs for fine-tuning."
        )

    elif opt_type == "RMSprop":
        tips.append(
            f"**RMSprop** uses {buffers} buffer(s) per parameter, "
            f"~{_fmt_bytes(total_mem)} of state memory."
        )

    # Universal tips
    wd = defaults.get("weight_decay", 0)
    if wd == 0:
        tips.append(
            "Weight decay is **0**. Consider adding weight decay (e.g., 1e-4 to 1e-2) "
            "for regularization, especially on large models."
        )

    if latest_state:
        for gs in latest_state.get("groups", []):
            wp = gs.get("warmup_pct", 100)
            if wp < 50:
                tips.append(
                    f"Group {gs.get('group_index', '?')} bias correction is only "
                    f"{wp:.0f}% converged. Variance estimates are still noisy -- "
                    "consider a longer warmup or learning rate schedule."
                )
                break

    if not tips:
        tips.append("No specific recommendations for this optimizer configuration.")

    return tips


_optimizer_inspector()
