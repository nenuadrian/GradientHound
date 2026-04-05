"""GradientHound Panel + Bokeh dashboard.  Launched as a subprocess by core.py."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import panel as pn

pn.extension(sizing_mode="stretch_width")

# Ensure the package is importable when launched via ``panel serve``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gradienthound.ipc import IPCChannel  # noqa: E402
from gradienthound.pages import (  # noqa: E402
    dashboard, architecture, weights, gradients,
    training, optimizer, network_state,
)

# ── IPC channel ──────────────────────────────────────────────────────

ipc_dir = os.environ.get("GRADIENTHOUND_IPC_DIR")
if not ipc_dir:
    raise RuntimeError("GRADIENTHOUND_IPC_DIR environment variable not set")

ipc = IPCChannel(directory=ipc_dir)

# ── Build pages ──────────────────────────────────────────────────────

_page_defs = {
    "Dashboard":     dashboard.create,
    "Architecture":  architecture.create,
    "Weights":       weights.create,
    "Gradients":     gradients.create,
    "Training":      training.create,
    "Optimizers":    optimizer.create,
    "Network State": network_state.create,
}

pages: dict[str, tuple] = {}
for name, factory in _page_defs.items():
    pages[name] = factory(ipc)

# ── Navigation ───────────────────────────────────────────────────────

nav = pn.widgets.RadioBoxGroup(
    options=list(pages.keys()),
    value="Dashboard",
    inline=False,
)

content_area = pn.Column(
    pages["Dashboard"][0],
    sizing_mode="stretch_both",
)


def _on_page_change(event):
    layout, _update = pages[event.new]
    content_area.clear()
    content_area.append(layout)
    _update()  # immediate refresh on page switch


nav.param.watch(_on_page_change, "value")


# ── Periodic live-update ─────────────────────────────────────────────

def _periodic_update():
    _, update_fn = pages[nav.value]
    try:
        update_fn()
    except Exception:
        pass  # keep callback alive even if a single update fails


pn.state.add_periodic_callback(_periodic_update, period=2000)

# ── Template ─────────────────────────────────────────────────────────

template = pn.template.FastListTemplate(
    title="GradientHound",
    accent_base_color="#c62828",
    header_background="#c62828",
    sidebar=[
        pn.pane.Markdown("### Navigation", margin=(10, 10)),
        nav,
    ],
    main=[content_area],
    theme="dark",
)

template.servable()
