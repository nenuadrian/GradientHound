"""Optimizer page -- detailed optimizer state inspection."""
from __future__ import annotations

import streamlit as st

from gradienthound.pages._common import inject_css, get_ipc

inject_css()

st.header("Optimizer Inspector")

ipc = get_ipc()
if ipc is None:
    st.error("IPC channel not initialised.")
    st.stop()


@st.fragment(run_every=2)
def _optimizer_inspector() -> None:
    import pandas as pd

    optimizers = ipc.read_optimizers()
    if not optimizers:
        st.info("Waiting for optimizers to be registered...")
        return

    for opt_name, opt_info in optimizers.items():
        st.subheader(f"{opt_name} ({opt_info['type']})")

        # ── Defaults ──────────────────────────────────────────────────
        if opt_info["defaults"]:
            with st.expander("Defaults", expanded=True):
                defaults_df = pd.DataFrame(
                    [{"Parameter": k, "Value": v} for k, v in opt_info["defaults"].items()]
                )
                st.dataframe(defaults_df, use_container_width=True, hide_index=True)

        # ── Param Groups ──────────────────────────────────────────────
        if opt_info["param_groups"]:
            with st.expander("Parameter Groups", expanded=True):
                groups = opt_info["param_groups"]
                st.dataframe(groups, use_container_width=True)

                # Visual comparison of key hyperparams across groups
                if len(groups) > 1:
                    st.markdown("**Per-Group Hyperparameter Comparison**")
                    hyperparams = ["lr", "weight_decay", "momentum", "eps", "betas"]
                    for hp in hyperparams:
                        values = [g.get(hp) for g in groups if hp in g]
                        if values and any(v is not None for v in values):
                            chart_data = pd.DataFrame([
                                {"Group": f"Group {g.get('index', i)}", hp: g.get(hp, 0)}
                                for i, g in enumerate(groups)
                                if hp in g
                            ])
                            if not chart_data.empty:
                                st.bar_chart(chart_data.set_index("Group"))

        # ── Learning Rate Schedule ────────────────────────────────────
        # Show LR from gradient stats over time if available
        grad_stats = ipc.read_gradient_stats()
        if grad_stats and opt_info["defaults"].get("lr"):
            with st.expander("Learning Rate", expanded=False):
                base_lr = opt_info["defaults"]["lr"]
                st.metric("Base Learning Rate", f"{base_lr:.2e}")
                st.caption(
                    "LR schedule tracking requires `gh.log_scalars()`. "
                    "Currently showing the base LR from optimizer config."
                )


_optimizer_inspector()
