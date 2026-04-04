"""GradientHound multi-page Streamlit dashboard.  Launched as a subprocess by core.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st

# ── Parse IPC directory from script args ──────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--ipc-dir", required=True)
_args, _ = _parser.parse_known_args(sys.argv[1:])

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from gradienthound.ipc import IPCChannel  # noqa: E402

# Store IPC channel in session state so pages can access it
if "ipc" not in st.session_state:
    st.session_state.ipc = IPCChannel(directory=_args.ipc_dir)

st.set_page_config(
    page_title="GradientHound",
    page_icon="\U0001f43e",
    layout="wide",
)

# ── Multi-page navigation ────────────────────────────────────────────
_pages_dir = Path(__file__).parent / "pages"

dashboard = st.Page(str(_pages_dir / "dashboard.py"), title="Dashboard", icon=":material/dashboard:", default=True)
architecture = st.Page(str(_pages_dir / "architecture.py"), title="Architecture", icon=":material/account_tree:")
weights = st.Page(str(_pages_dir / "weights.py"), title="Weights", icon=":material/analytics:")
gradients = st.Page(str(_pages_dir / "gradients.py"), title="Gradients", icon=":material/trending_up:")
training = st.Page(str(_pages_dir / "training.py"), title="Training", icon=":material/timeline:")
optimizer_page = st.Page(str(_pages_dir / "optimizer.py"), title="Optimizers", icon=":material/tune:")

nav = st.navigation([dashboard, architecture, weights, gradients, training, optimizer_page])
nav.run()
