"""Weights page -- histograms, SVD spectrum, norms, effective rank, on-demand heatmaps."""
from __future__ import annotations

import streamlit as st

from gradienthound.pages._common import (
    inject_css, get_ipc, short_layer_name, rank_health_color,
)

inject_css()

st.header("Weight Analysis")

ipc = get_ipc()
if ipc is None:
    st.error("IPC channel not initialised.")
    st.stop()


@st.fragment(run_every=2)
def _weight_analysis() -> None:
    import pandas as pd

    stats = ipc.read_weight_stats()
    if not stats:
        st.info("Waiting for weight snapshots (captured every N steps via `gh.watch`)...")
        return

    latest_step = max(e["step"] for e in stats)
    latest = [e for e in stats if e["step"] == latest_step]
    st.caption(f"Latest snapshot: step {latest_step}")

    # ── Weight Distribution Histograms ────────────────────────────────
    hist_entries = [e for e in latest if "hist_counts" in e]
    if hist_entries:
        with st.expander("Weight Distribution Histograms", expanded=True):
            layers = [e["layer"] for e in hist_entries]
            layer_map = {e["layer"]: e for e in hist_entries}
            selected = st.selectbox(
                "Layer", layers, format_func=short_layer_name, key="hist_layer",
            )
            if selected and selected in layer_map:
                e = layer_map[selected]
                centers = e["hist_centers"]
                counts = e["hist_counts"]
                df = pd.DataFrame({"Weight Value": centers, "Count": counts})
                st.bar_chart(df.set_index("Weight Value"), use_container_width=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean", f"{e['mean']:.4f}")
                c2.metric("Std", f"{e['std']:.4f}")
                c3.metric("Near-Zero %", f"{e.get('near_zero_pct', 0):.1f}%")
                c4.metric("Kurtosis", f"{e.get('kurtosis', 0):.2f}")

    # ── Spectral Analysis (SVD) ───────────────────────────────────────
    svd_entries = [e for e in latest if "singular_values" in e]
    if svd_entries:
        with st.expander("Spectral Analysis (SVD)", expanded=True):
            svd_layers = [e["layer"] for e in svd_entries]
            svd_map = {e["layer"]: e for e in svd_entries}
            selected_svd = st.selectbox(
                "Layer", svd_layers, format_func=short_layer_name, key="svd_layer",
            )
            if selected_svd and selected_svd in svd_map:
                e = svd_map[selected_svd]
                svs = e["singular_values"]

                tab_sv, tab_energy = st.tabs(["Singular Values", "Cumulative Energy"])

                with tab_sv:
                    sv_df = pd.DataFrame({
                        "Index": list(range(len(svs))),
                        "Singular Value": svs,
                    })
                    st.bar_chart(sv_df.set_index("Index"), use_container_width=True)

                with tab_energy:
                    if "cumulative_energy" in e:
                        ce = e["cumulative_energy"]
                        ce_df = pd.DataFrame({
                            "Component": list(range(len(ce))),
                            "Cumulative Energy": ce,
                        })
                        st.line_chart(ce_df.set_index("Component"), use_container_width=True)
                        # 90% energy threshold
                        threshold_90 = next(
                            (i for i, v in enumerate(ce) if v >= 0.9), len(ce)
                        )
                        st.caption(f"90% energy at component {threshold_90} of {len(ce)}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Stable Rank", f"{e.get('stable_rank', 0):.1f}")
                c2.metric("Effective Rank", f"{e.get('effective_rank', 0):.1f}")
                c3.metric("Condition #", f"{e.get('condition_number', 0):.0f}")

    # ── Parameter Norms ───────────────────────────────────────────────
    with st.expander("Parameter Norms (L2)", expanded=False):
        df = pd.DataFrame([
            {"Layer": short_layer_name(e["layer"]), "L2 Norm": e["norm_l2"]}
            for e in latest
        ])
        if not df.empty:
            st.bar_chart(df.set_index("Layer"), horizontal=True)

    # ── Effective Rank ────────────────────────────────────────────────
    rank_entries = [e for e in latest if "effective_rank" in e]
    if rank_entries:
        with st.expander("Effective Rank (2D layers)", expanded=False):
            rank_data = []
            for e in rank_entries:
                eff = e["effective_rank"]
                mx = e.get("max_rank", 1)
                rank_data.append({
                    "Layer": short_layer_name(e["layer"]),
                    "Effective Rank": round(eff, 1),
                    "Max Rank": mx,
                    "Utilization": f"{eff / mx * 100:.0f}%" if mx > 0 else "N/A",
                })
            st.dataframe(pd.DataFrame(rank_data), use_container_width=True, hide_index=True)

    # ── Weight Distribution Over Time ─────────────────────────────────
    with st.expander("Weight Distribution Over Time", expanded=False):
        layers = sorted({e["layer"] for e in stats})
        layer_display = {l: short_layer_name(l) for l in layers}
        selected_layer = st.selectbox(
            "Layer", layers, format_func=lambda x: layer_display[x],
            key="weight_dist_layer",
        )
        if selected_layer:
            layer_data = [e for e in stats if e["layer"] == selected_layer]
            if layer_data:
                df = pd.DataFrame([
                    {
                        "step": e["step"],
                        "mean": e["mean"],
                        "std": e["std"],
                        "min": e["min"],
                        "max": e["max"],
                    }
                    for e in layer_data
                ]).set_index("step")
                st.line_chart(df, use_container_width=True)

    # ── On-Demand: Weight Heatmap ─────────────────────────────────────
    weight_2d = [e for e in latest if len(e.get("shape", [])) == 2 and "bias" not in e["layer"]]
    if weight_2d:
        with st.expander("Weight Heatmap (on-demand)", expanded=False):
            hm_layers = [e["layer"] for e in weight_2d]
            selected_hm = st.selectbox(
                "Layer", hm_layers, format_func=short_layer_name, key="hm_layer",
            )
            if selected_hm:
                btn_key = f"hm_btn_{selected_hm}"
                if st.button("Compute Heatmap", key=btn_key):
                    ipc.clear_response("weight_heatmap")
                    ipc.write_request({
                        "id": "weight_heatmap",
                        "type": "weight_heatmap",
                        "layer": selected_hm,
                    })

                resp = ipc.read_response("weight_heatmap")
                if resp and resp.get("layer") == selected_hm and "error" not in resp:
                    import numpy as np
                    matrix = np.array(resp["matrix"])
                    vmax = resp["vmax"]
                    st.caption(
                        f"Shape: {resp['shape']} | Display: {resp['display_shape']} | "
                        f"Sparsity: {resp['sparsity']:.1f}%"
                    )
                    try:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
                        ax.set_xlabel("Column")
                        ax.set_ylabel("Row")
                        ax.set_title(short_layer_name(selected_hm))
                        plt.colorbar(im, ax=ax)
                        st.pyplot(fig)
                        plt.close(fig)
                    except ImportError:
                        st.dataframe(pd.DataFrame(matrix), use_container_width=True)
                elif resp and "error" in resp:
                    st.warning(resp["error"])

    # ── Near-Zero Weight % ────────────────────────────────────────────
    with st.expander("Near-Zero Weights (Sparsity)", expanded=False):
        sparsity_df = pd.DataFrame([
            {"Layer": short_layer_name(e["layer"]), "Near-Zero %": e.get("near_zero_pct", 0)}
            for e in latest
        ])
        if not sparsity_df.empty:
            st.bar_chart(sparsity_df.set_index("Layer"), horizontal=True)


_weight_analysis()
