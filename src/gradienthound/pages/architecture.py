"""Architecture page -- model graph visualization and module details."""
from __future__ import annotations

import streamlit as st

from gradienthound.graph import render_graphviz
from gradienthound.pages._common import inject_css, get_ipc

inject_css()

st.header("Model Architecture")

ipc = get_ipc()
if ipc is None:
    st.error("IPC channel not initialised.")
    st.stop()


@st.fragment(run_every=2)
def _architecture() -> None:
    models = ipc.read_models()

    if not models:
        st.info("Waiting for models to be registered...")
        return

    model_names = list(models.keys())
    selected = st.selectbox("Model", model_names)
    if selected is None:
        return

    graph_data = models[selected]

    # ── Summary metrics ───────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Class", graph_data["class"])
    c2.metric("Parameters", f"{graph_data['total_params']:,}")
    c3.metric("Modules", str(len(graph_data["modules"])))

    # ── Architecture diagram ──────────────────────────────────────────
    with st.expander("Architecture Diagram", expanded=True):
        dot = render_graphviz(graph_data)
        st.graphviz_chart(dot.source, use_container_width=True)

    # ── Raw PyTorch module dump ──────────────────────────────────────
    with st.expander("PyTorch Module Dump (print(model))", expanded=False):
        model_dump = graph_data.get("pytorch_repr")
        if model_dump:
            st.code(model_dump)
        else:
            st.info("Raw module dump not available for this capture.")

    # ── Module details table ──────────────────────────────────────────
    with st.expander("Module Details", expanded=False):
        rows = []
        for mod in graph_data["modules"]:
            if not mod["is_leaf"]:
                continue
            attr_str = ", ".join(f"{k}={v}" for k, v in mod["attributes"].items())
            rows.append({
                "Path": mod["path"],
                "Type": mod["type"],
                "Parameters": mod["params"],
                "Attributes": attr_str,
            })
        if rows:
            st.dataframe(rows, use_container_width=True)

_architecture()
