"""Dashboard page -- metrics overview with live-updating charts."""
from __future__ import annotations

from collections import defaultdict

import streamlit as st

from gradienthound.pages._common import inject_css, get_ipc, BRAND_RED

inject_css()

st.markdown(f'<h1 style="color:{BRAND_RED}; margin:0;">GradientHound</h1>', unsafe_allow_html=True)
st.caption("Real-time model architecture inspector")

ipc = get_ipc()
if ipc is None:
    st.error("IPC channel not initialised.")
    st.stop()

# ── Run metadata ──────────────────────────────────────────────────────
metadata = ipc.read_metadata()
if metadata:
    with st.expander("Run metadata", expanded=False):
        for key, value in metadata.items():
            st.text(f"{key}: {value}")


def _categorise_metrics(keys: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        prefix, sep, _ = key.partition("/")
        category = prefix.strip() if sep and prefix.strip() else "Other"
        buckets[category].append(key)
    return dict(buckets)


# ── Live metrics fragment ─────────────────────────────────────────────


@st.fragment(run_every=2)
def _live_metrics() -> None:
    models = ipc.read_models()
    metrics = ipc.read_metrics()
    grad_stats = ipc.read_gradient_stats()

    # Summary cards
    total_params = sum(m.get("total_params", 0) for m in models.values())
    num_models = len(models)
    num_steps = max((e.get("_step", 0) for e in metrics), default=0) if metrics else 0
    num_grad_steps = max((e.get("step", 0) for e in grad_stats), default=0) if grad_stats else 0
    effective_steps = max(num_steps, num_grad_steps)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Step", f"{effective_steps:,}")
    c2.metric("Models", str(num_models))
    c3.metric("Parameters", f"{total_params:,}")
    c4.metric("Metrics", str(len({k for e in metrics for k in e if not k.startswith("_")})) if metrics else "0")

    # Health badges from gradient stats
    if grad_stats:
        latest_step = max(e["step"] for e in grad_stats)
        latest = [e for e in grad_stats if e["step"] == latest_step]
        vanishing = sum(1 for e in latest if e.get("grad_norm", 1) < 1e-7)
        exploding = sum(1 for e in latest if e.get("grad_norm", 0) > 1e3)
        if vanishing > 0:
            st.warning(f"{vanishing} layers with vanishing gradients")
        if exploding > 0:
            st.error(f"{exploding} layers with exploding gradients")

    # Metric charts
    if not metrics:
        st.info("Waiting for metrics...")
        return

    import pandas as pd

    all_keys = sorted({k for entry in metrics for k in entry if not k.startswith("_")})
    if not all_keys:
        return

    categories = _categorise_metrics(all_keys)

    for cat_name, cat_keys in categories.items():
        with st.expander(cat_name, expanded=False):
            for i in range(0, len(cat_keys), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(cat_keys):
                        break
                    metric_key = cat_keys[idx]
                    steps = [
                        e.get("_step", k)
                        for k, e in enumerate(metrics)
                        if metric_key in e
                    ]
                    values = [e[metric_key] for e in metrics if metric_key in e]
                    if values:
                        with col:
                            with st.container(border=True):
                                st.caption(metric_key)
                                df = pd.DataFrame(
                                    {"step": steps, metric_key: values},
                                ).set_index("step")
                                st.line_chart(df, use_container_width=True)


_live_metrics()
