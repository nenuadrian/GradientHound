"""Network State page -- layer-by-layer tables of all weight values."""
from __future__ import annotations

import time
import uuid

import streamlit as st

from gradienthound.pages._common import inject_css, get_ipc, short_layer_name

inject_css()

st.header("Network State")

ipc = get_ipc()
if ipc is None:
    st.error("IPC channel not initialised.")
    st.stop()

_MAX_PARAMS = 1_000_000


def _request_network_state(model_name: str) -> str:
    """Send a network_state request and return the request ID."""
    req_id = f"network_state_{uuid.uuid4().hex[:8]}"
    ipc.write_request({
        "type": "network_state",
        "id": req_id,
        "model": model_name,
    })
    return req_id


def _poll_response(req_id: str, timeout: float = 30.0) -> dict | None:
    """Poll for a response with timeout."""
    start = time.time()
    while time.time() - start < timeout:
        resp = ipc.read_response(req_id)
        if resp is not None:
            ipc.clear_response(req_id)
            return resp
        time.sleep(0.3)
    return None


def _render_state(state: dict) -> None:
    """Render the full network state as layer-by-layer tables."""
    import numpy as np
    import pandas as pd

    layers = state["layers"]
    total_params = state["total_params"]
    step = state.get("step", "?")
    model_name = state.get("model", "?")

    st.caption(f"Snapshot at step **{step}** | Model: **{model_name}** | "
               f"Total parameters: **{total_params:,}**")

    # Summary table
    summary_rows = []
    for layer in layers:
        vals = np.array(layer["values"]).flatten()
        summary_rows.append({
            "Layer": short_layer_name(layer["name"]),
            "Shape": str(layer["shape"]),
            "Params": f"{layer['numel']:,}",
            "Mean": f"{vals.mean():.6f}",
            "Std": f"{vals.std():.6f}",
            "Min": f"{vals.min():.6f}",
            "Max": f"{vals.max():.6f}",
            "Grad": "Yes" if layer["requires_grad"] else "No",
        })
    st.subheader("Layer Summary")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Precision selector
    precision = st.slider("Display precision (decimal places)", 2, 8, 4, key="precision")

    # Layer-by-layer expandable tables
    st.subheader("Layer-by-Layer Weights")

    for layer in layers:
        shape = layer["shape"]
        numel = layer["numel"]
        label = f"{layer['name']}  |  shape={shape}  |  {numel:,} params"

        with st.expander(label, expanded=False):
            values = layer["values"]

            if len(shape) <= 1:
                # Scalar or 1D: single row
                flat = values[0] if values else []
                df = pd.DataFrame(
                    [flat],
                    columns=[str(i) for i in range(len(flat))],
                    index=["value"],
                )
                st.dataframe(
                    df.style.format(f"{{:.{precision}f}}"),
                    use_container_width=True,
                )

            elif len(shape) == 2:
                rows, cols = shape
                df = pd.DataFrame(
                    values,
                    columns=[str(c) for c in range(cols)],
                    index=[str(r) for r in range(rows)],
                )
                st.caption(f"{rows} rows x {cols} cols")
                st.dataframe(
                    df.style.format(f"{{:.{precision}f}}").background_gradient(
                        cmap="RdBu_r", axis=None, vmin=-abs(df.values).max(),
                        vmax=abs(df.values).max(),
                    ),
                    use_container_width=True,
                    height=min(35 * (rows + 1), 600),
                )

            else:
                # Higher dimensional: reshaped to 2D
                flat_rows = len(values)
                flat_cols = len(values[0]) if values else 0
                st.caption(
                    f"Original shape {shape} -> displayed as "
                    f"{flat_rows} x {flat_cols}"
                )
                df = pd.DataFrame(
                    values,
                    columns=[str(c) for c in range(flat_cols)],
                )
                st.dataframe(
                    df.style.format(f"{{:.{precision}f}}").background_gradient(
                        cmap="RdBu_r", axis=None, vmin=-abs(df.values).max(),
                        vmax=abs(df.values).max(),
                    ),
                    use_container_width=True,
                    height=min(35 * (flat_rows + 1), 600),
                )


def _network_state_page() -> None:
    models = ipc.read_models()

    if not models:
        st.info("Waiting for models to be registered...")
        return

    model_names = list(models.keys())
    selected = st.selectbox("Model", model_names, key="ns_model")
    if selected is None:
        return

    graph_data = models[selected]
    total_params = graph_data.get("total_params", 0)

    c1, c2 = st.columns(2)
    c1.metric("Total Parameters", f"{total_params:,}")
    c2.metric("Status",
              "Eligible" if total_params <= _MAX_PARAMS else "Too Large",
              delta=f"limit: {_MAX_PARAMS:,}",
              delta_color="normal" if total_params <= _MAX_PARAMS else "inverse")

    if total_params > _MAX_PARAMS:
        st.warning(
            f"Network state view is limited to models with <= {_MAX_PARAMS:,} parameters. "
            f"This model has {total_params:,}. The full weight tables would be too large "
            f"to display interactively."
        )
        return

    # Check for cached state in session
    cache_key = f"network_state_{selected}"
    if cache_key in st.session_state and st.session_state[cache_key] is not None:
        _render_state(st.session_state[cache_key])
        st.divider()

    if st.button("Capture Network State", type="primary", use_container_width=True):
        with st.spinner("Requesting network state from training process..."):
            req_id = _request_network_state(selected)
            resp = _poll_response(req_id)

        if resp is None:
            st.error(
                "Timed out waiting for response. Make sure the training process "
                "is running and calling `gh.step()` or `self._gh_step()`."
            )
        elif "error" in resp:
            st.error(resp["error"])
        else:
            st.session_state[cache_key] = resp
            st.rerun()


_network_state_page()
