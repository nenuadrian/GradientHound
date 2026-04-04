"""Training page -- predictions vs actual, attention, activations, CKA similarity."""
from __future__ import annotations

import streamlit as st

from gradienthound.pages._common import inject_css, get_ipc, short_layer_name

inject_css()

st.header("Training Dynamics")

ipc = get_ipc()
if ipc is None:
    st.error("IPC channel not initialised.")
    st.stop()


@st.fragment(run_every=2)
def _training_dynamics() -> None:
    import pandas as pd

    # ── Predictions vs Actual (Value Calibration) ─────────────────────
    predictions = ipc.read_predictions()
    if predictions:
        with st.expander("Predictions vs Actual (Value Calibration)", expanded=True):
            names = sorted({e["name"] for e in predictions})
            selected_name = st.selectbox("Source", names, key="pred_name")
            entries = [e for e in predictions if e["name"] == selected_name]

            if entries:
                latest = entries[-1]
                pred = latest["predicted"]
                actual = latest["actual"]
                min_len = min(len(pred), len(actual))

                df = pd.DataFrame({
                    "Predicted": pred[:min_len],
                    "Actual": actual[:min_len],
                })
                st.scatter_chart(df, x="Actual", y="Predicted", use_container_width=True)
                st.caption(
                    f"Step {latest.get('step', '?')} | "
                    f"{min_len} points | "
                    f"Points on diagonal = perfect calibration"
                )

    # ── CKA Similarity (on-demand) ────────────────────────────────────
    with st.expander("CKA Layer Similarity (on-demand)", expanded=True):
        if st.button("Compute CKA Matrix", key="cka_btn"):
            ipc.clear_response("cka")
            ipc.write_request({"id": "cka", "type": "cka"})

        resp = ipc.read_response("cka")
        if resp and "error" not in resp:
            import numpy as np
            matrix = np.array(resp["matrix"])
            short_names = resp["short_names"]

            st.caption(f"Step {resp.get('step', '?')} | {resp['n']} weight layers")

            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 7))
                im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
                ax.set_xticks(range(len(short_names)))
                ax.set_yticks(range(len(short_names)))
                ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
                ax.set_yticklabels(short_names, fontsize=7)
                ax.set_title("Linear CKA Similarity")
                plt.colorbar(im, ax=ax)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except ImportError:
                st.dataframe(
                    pd.DataFrame(matrix, columns=short_names, index=short_names),
                    use_container_width=True,
                )

            # Summary stats
            n = len(matrix)
            upper_vals = [matrix[i][j] for i in range(n) for j in range(i + 1, n)]
            if upper_vals:
                c1, c2 = st.columns(2)
                c1.metric("Mean CKA (off-diag)", f"{sum(upper_vals) / len(upper_vals):.3f}")
                c2.metric("Max CKA (off-diag)", f"{max(upper_vals):.3f}")
        elif resp and "error" in resp:
            st.warning(resp["error"])
        else:
            st.caption("Click the button to compute CKA similarity between all 2D weight layers.")

    # ── Attention Patterns ────────────────────────────────────────────
    attention = ipc.read_attention()
    if attention:
        with st.expander("Attention Patterns", expanded=True):
            import numpy as np

            entries = sorted(attention, key=lambda e: e.get("step", 0))
            latest = entries[-1]

            st.caption(f"Step {latest.get('step', '?')} | Layer: {latest['name']} | Shape: {latest.get('shape', '?')}")

            weights = latest["weights"]
            n_heads = len(weights)
            selected_head = st.slider("Head", 0, max(0, n_heads - 1), 0, key="attn_head")

            if selected_head < n_heads:
                matrix = weights[selected_head]
                arr = np.array(matrix)
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(arr, cmap="viridis", aspect="auto")
                    ax.set_xlabel("Key")
                    ax.set_ylabel("Query")
                    ax.set_title(f"Head {selected_head}")
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)
                except ImportError:
                    st.dataframe(pd.DataFrame(arr), use_container_width=True)

    # ── Activation Stats ──────────────────────────────────────────────
    act_stats = ipc.read_activation_stats()
    if act_stats:
        with st.expander("Activation Statistics", expanded=False):
            latest_step = max(e["step"] for e in act_stats)
            latest_act = [e for e in act_stats if e["step"] == latest_step]
            st.caption(f"Latest step: {latest_step}")

            act_df = pd.DataFrame([
                {
                    "Layer": short_layer_name(e["layer"]),
                    "Mean": round(e["mean"], 4),
                    "Std": round(e["std"], 4),
                    "Min": round(e["min"], 4),
                    "Max": round(e["max"], 4),
                    "Zero %": f"{e.get('zero_fraction', 0) * 100:.1f}%",
                }
                for e in latest_act
            ])
            if not act_df.empty:
                st.dataframe(act_df, use_container_width=True, hide_index=True)

            zf_df = pd.DataFrame([
                {"Layer": short_layer_name(e["layer"]), "Zero Fraction": e.get("zero_fraction", 0)}
                for e in latest_act
            ])
            if not zf_df.empty:
                st.bar_chart(zf_df.set_index("Layer"), horizontal=True)

    # ── Empty state ───────────────────────────────────────────────────
    if not predictions and not attention and not act_stats:
        st.info(
            "No training dynamics data yet. Use `gh.watch(model, log_activations=True)`, "
            "`gh.log_predictions()`, or `gh.log_attention()` to populate this page."
        )


_training_dynamics()
