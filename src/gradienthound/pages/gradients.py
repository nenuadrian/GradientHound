"""Gradients page -- gradient flow, update ratios, dead neurons, noise, cosine similarity."""
from __future__ import annotations

import streamlit as st

from gradienthound.pages._common import (
    inject_css, get_ipc, short_layer_name, grad_health_color,
    GREEN, YELLOW, RED,
)

inject_css()

st.header("Gradient Analysis")

ipc = get_ipc()
if ipc is None:
    st.error("IPC channel not initialised.")
    st.stop()


@st.fragment(run_every=2)
def _gradient_analysis() -> None:
    import pandas as pd

    stats = ipc.read_gradient_stats()
    if not stats:
        st.info("Waiting for gradient data (captured via `gh.watch` + `gh.step`)...")
        return

    latest_step = max(e["step"] for e in stats)
    latest = [e for e in stats if e["step"] == latest_step]
    st.caption(f"Latest step: {latest_step}")

    # ── Gradient Flow ─────────────────────────────────────────────────
    with st.expander("Gradient Flow (per-layer L2 norms)", expanded=True):
        flow_data = []
        for e in latest:
            gn = e.get("grad_norm", 0)
            color = grad_health_color(gn)
            flow_data.append({
                "Layer": short_layer_name(e["layer"]),
                "Gradient Norm": gn,
                "Health": "Healthy" if color == GREEN else ("Weak" if color == YELLOW else "Critical"),
            })
        df = pd.DataFrame(flow_data)
        if not df.empty:
            st.bar_chart(df.set_index("Layer")[["Gradient Norm"]], horizontal=True)

            healthy = sum(1 for d in flow_data if d["Health"] == "Healthy")
            weak = sum(1 for d in flow_data if d["Health"] == "Weak")
            critical = sum(1 for d in flow_data if d["Health"] == "Critical")
            cols = st.columns(3)
            cols[0].metric("Healthy", str(healthy))
            cols[1].metric("Weak", str(weak))
            cols[2].metric("Critical", str(critical))

    # ── Gradient Cosine Similarity ────────────────────────────────────
    cosine_entries = [e for e in latest if "cosine_sim" in e]
    if cosine_entries:
        with st.expander("Gradient Cosine Similarity (vs previous step)", expanded=True):
            cos_data = []
            for e in cosine_entries:
                cs = e["cosine_sim"]
                if cs > 0.5:
                    health = "Stable"
                elif cs >= 0:
                    health = "Noisy"
                else:
                    health = "Oscillating"
                cos_data.append({
                    "Layer": short_layer_name(e["layer"]),
                    "Cosine Similarity": cs,
                    "Status": health,
                })
            df = pd.DataFrame(cos_data)
            st.bar_chart(df.set_index("Layer")[["Cosine Similarity"]], horizontal=True)

            stable = sum(1 for d in cos_data if d["Status"] == "Stable")
            noisy = sum(1 for d in cos_data if d["Status"] == "Noisy")
            osc = sum(1 for d in cos_data if d["Status"] == "Oscillating")
            cols = st.columns(3)
            cols[0].metric("Stable (>0.5)", str(stable))
            cols[1].metric("Noisy (0-0.5)", str(noisy))
            cols[2].metric("Oscillating (<0)", str(osc))

    # ── Update-to-Weight Ratio ────────────────────────────────────────
    with st.expander("Update-to-Weight Ratio", expanded=True):
        optimizers = ipc.read_optimizers()
        lr = 1e-3
        for opt_info in optimizers.values():
            for pg in opt_info.get("param_groups", []):
                if "lr" in pg:
                    lr = pg["lr"]
                    break
            if lr != 1e-3:
                break

        ratio_data = []
        for e in latest:
            gn = e.get("grad_norm", 0)
            wn = e.get("weight_norm", 1e-20)
            ratio = (lr * gn) / max(wn, 1e-20)
            ratio_data.append({
                "Layer": short_layer_name(e["layer"]),
                "Update/Weight Ratio": ratio,
            })
        df = pd.DataFrame(ratio_data)
        if not df.empty:
            st.bar_chart(df.set_index("Layer"), horizontal=True)
            st.caption(f"Using lr={lr:.1e}. Healthy range: [1e-4, 1e-2]")

    # ── Dead Neurons ──────────────────────────────────────────────────
    with st.expander("Dead Neurons (zero-gradient %)", expanded=False):
        dead_data = pd.DataFrame([
            {
                "Layer": short_layer_name(e["layer"]),
                "Dead Grad %": e.get("dead_grad_pct", 0),
                "Near-Zero Weight %": e.get("near_zero_weight_pct", 0),
            }
            for e in latest
        ])
        if not dead_data.empty:
            st.bar_chart(dead_data.set_index("Layer"), horizontal=True)

    # ── Gradient Noise Scale ──────────────────────────────────────────
    with st.expander("Gradient Noise Scale", expanded=False):
        noise_data = pd.DataFrame([
            {
                "Layer": short_layer_name(e["layer"]),
                "Noise Scale": e.get("grad_noise_scale", 0),
            }
            for e in latest
        ])
        if not noise_data.empty:
            st.bar_chart(noise_data.set_index("Layer"), horizontal=True)
            st.caption("Var(g) / E[g]^2 -- high values indicate noisy gradients")

    # ── Gradient Norms Over Time ──────────────────────────────────────
    with st.expander("Gradient Norms Over Time", expanded=False):
        layers = sorted({e["layer"] for e in stats})
        selected = st.multiselect(
            "Layers", layers,
            default=layers[:5],
            format_func=short_layer_name,
            key="grad_norm_layers",
        )
        if selected:
            records = [e for e in stats if e["layer"] in selected]
            if records:
                pivot: dict[int, dict[str, float]] = {}
                for e in records:
                    step = e["step"]
                    if step not in pivot:
                        pivot[step] = {}
                    pivot[step][short_layer_name(e["layer"])] = e.get("grad_norm", 0)
                df = pd.DataFrame.from_dict(pivot, orient="index").sort_index()
                df.index.name = "step"
                st.line_chart(df, use_container_width=True)

    # ── Cosine Similarity Over Time ───────────────────────────────────
    cosine_all = [e for e in stats if "cosine_sim" in e]
    if cosine_all:
        with st.expander("Cosine Similarity Over Time", expanded=False):
            cos_layers = sorted({e["layer"] for e in cosine_all})
            sel_cos = st.multiselect(
                "Layers", cos_layers,
                default=cos_layers[:5],
                format_func=short_layer_name,
                key="cos_layers",
            )
            if sel_cos:
                records = [e for e in cosine_all if e["layer"] in sel_cos]
                pivot: dict[int, dict[str, float]] = {}
                for e in records:
                    step = e["step"]
                    if step not in pivot:
                        pivot[step] = {}
                    pivot[step][short_layer_name(e["layer"])] = e["cosine_sim"]
                df = pd.DataFrame.from_dict(pivot, orient="index").sort_index()
                df.index.name = "step"
                st.line_chart(df, use_container_width=True)


_gradient_analysis()
