"""GradientHound Panel + Bokeh dashboard.  Launched as a subprocess by core.py."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import panel as pn

_RAW_CSS = """
:root {
  --gh-bg: #0f0a0a;
  --gh-surface: #1a1012;
  --gh-surface-elevated: rgba(38, 20, 24, 0.92);
  --gh-surface-soft: rgba(62, 34, 40, 0.58);
  --gh-border: rgba(178, 132, 140, 0.2);
  --gh-border-strong: rgba(178, 132, 140, 0.34);
  --gh-text: #f0e6e8;
  --gh-text-muted: #b09a9e;
  --gh-accent: #d4707a;
  --gh-accent-strong: #c2525e;
  --gh-glow: rgba(212, 112, 122, 0.18);
  --gh-danger: #e85d6f;
}

html, body, .bk-root {
  background:
    radial-gradient(circle at top left, rgba(180, 60, 70, 0.10), transparent 28%),
    radial-gradient(circle at top right, rgba(212, 112, 122, 0.08), transparent 24%),
    linear-gradient(180deg, #0f0a0a 0%, #0a0606 100%);
  color: var(--gh-text);
}

.bk-panel-models-fastlisttemplate,
.bk-panel-models-fastlisttemplate .main-area,
.bk-panel-models-fastlisttemplate .main-content,
.bk-panel-models-fastlisttemplate .content,
.bk-panel-models-fastlisttemplate .sidebar {
  background: transparent !important;
}

.bk-panel-models-fastlisttemplate .header {
  background: linear-gradient(135deg, #1a0e10 0%, #2a1418 100%) !important;
  border-bottom: 1px solid var(--gh-border) !important;
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
}

.bk-panel-models-fastlisttemplate .title {
  color: #fff5f6 !important;
  font-weight: 700;
  letter-spacing: 0.04em;
}

.bk-panel-models-fastlisttemplate .sidebar {
  border-right: 1px solid var(--gh-border) !important;
  padding-top: 14px;
  min-width: 60px;
  max-width: 500px;
  transition: width 0.25s ease, min-width 0.25s ease, padding 0.25s ease;
  position: relative;
}

.bk-panel-models-fastlisttemplate .sidebar .bk-panel-models-markup-Markdown {
  color: var(--gh-text-muted);
}

.bk-panel-models-fastlisttemplate .main-area {
  padding-top: 22px !important;
}

.bk-panel-models-fastlisttemplate .bk-panel-models-layout-Card,
.bk-panel-models-fastlisttemplate .bk-panel-models-indicators-Number,
.bk-panel-models-fastlisttemplate .bk-panel-models-widgets-button-Button,
.bk-panel-models-fastlisttemplate .bk-panel-models-widgets-select-Select,
.bk-panel-models-fastlisttemplate .bk-panel-models-widgets-slider-IntSlider,
.bk-panel-models-fastlisttemplate .bk-panel-models-markup-Alert,
.bk-panel-models-fastlisttemplate .bk-panel-models-widgetbox-WidgetBox {
  border-radius: 18px !important;
}

.bk-panel-models-fastlisttemplate .bk-panel-models-layout-Card,
.bk-panel-models-fastlisttemplate .bk-panel-models-indicators-Number,
.bk-panel-models-fastlisttemplate .bk-panel-models-markup-Alert,
.bk-panel-models-fastlisttemplate .bk-Column,
.bk-panel-models-fastlisttemplate .bk-Row {
  backdrop-filter: blur(14px);
}

.bk-panel-models-fastlisttemplate .bk-panel-models-layout-Card {
  background: linear-gradient(180deg, rgba(26, 16, 18, 0.94), rgba(18, 10, 12, 0.92)) !important;
  border: 1px solid var(--gh-border) !important;
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
  transition: box-shadow 0.2s ease, border-color 0.2s ease;
}

.bk-panel-models-fastlisttemplate .bk-panel-models-layout-Card:hover {
  border-color: var(--gh-border-strong) !important;
  box-shadow: 0 22px 48px rgba(0, 0, 0, 0.30);
}

.bk-panel-models-fastlisttemplate .bk-panel-models-layout-Card .card-header {
  background: transparent !important;
  border-bottom: 1px solid rgba(178, 132, 140, 0.12) !important;
  color: var(--gh-text) !important;
  font-weight: 600;
  letter-spacing: 0.02em;
}

.bk-panel-models-fastlisttemplate .bk-panel-models-indicators-Number {
  background: linear-gradient(160deg, rgba(20, 12, 14, 0.96), rgba(34, 20, 24, 0.92)) !important;
  border: 1px solid var(--gh-border) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03), 0 10px 24px rgba(0, 0, 0, 0.22);
  padding: 10px 14px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.bk-panel-models-fastlisttemplate .bk-panel-models-indicators-Number:hover {
  transform: translateY(-2px);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03), 0 14px 32px rgba(0, 0, 0, 0.28);
}

.bk-panel-models-fastlisttemplate .bk-panel-models-indicators-Number .title,
.bk-panel-models-fastlisttemplate .bk-panel-models-indicators-Number .label {
  color: var(--gh-text-muted) !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.bk-panel-models-fastlisttemplate .bk-panel-models-indicators-Number .value {
  color: #fff5f6 !important;
}

.bk-panel-models-fastlisttemplate .bk-input,
.bk-panel-models-fastlisttemplate .choices__inner,
.bk-panel-models-fastlisttemplate .noUi-target {
  background: rgba(18, 10, 14, 0.94) !important;
  border: 1px solid var(--gh-border) !important;
  color: var(--gh-text) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
  border-radius: 10px !important;
  transition: border-color 0.2s ease;
}

.bk-panel-models-fastlisttemplate .bk-input:focus,
.bk-panel-models-fastlisttemplate .choices.is-focused .choices__inner {
  border-color: var(--gh-accent) !important;
  box-shadow: 0 0 0 3px var(--gh-glow) !important;
}

.bk-panel-models-fastlisttemplate .bk-btn-primary {
  background: linear-gradient(135deg, var(--gh-accent) 0%, var(--gh-accent-strong) 100%) !important;
  border: none !important;
  color: #0f0a0a !important;
  font-weight: 700;
  box-shadow: 0 10px 24px rgba(194, 82, 94, 0.24);
}

.bk-panel-models-fastlisttemplate .bk-btn-primary:hover {
  filter: brightness(1.06);
  transform: translateY(-1px);
}

.bk-panel-models-fastlisttemplate .bk-btn-primary:active {
  transform: translateY(0);
  filter: brightness(0.95);
}

.bk-panel-models-fastlisttemplate .bk-btn-default {
  background: rgba(34, 18, 22, 0.92) !important;
  border: 1px solid var(--gh-border) !important;
  color: var(--gh-text) !important;
}

.bk-panel-models-fastlisttemplate .bk-btn-group .bk-btn,
.bk-panel-models-fastlisttemplate .bk-btn {
  border-radius: 12px !important;
  transition: all 160ms ease;
}

.bk-panel-models-fastlisttemplate .bk-markup,
.bk-panel-models-fastlisttemplate .markdown-text {
  color: var(--gh-text);
}

.bk-panel-models-fastlisttemplate .bk-markup h1,
.bk-panel-models-fastlisttemplate .bk-markup h2,
.bk-panel-models-fastlisttemplate .bk-markup h3,
.bk-panel-models-fastlisttemplate .bk-markup strong {
  color: #fff5f6;
}

.bk-panel-models-fastlisttemplate .bk-panel-models-markup-Alert {
  border: 1px solid var(--gh-border) !important;
  background: linear-gradient(180deg, rgba(30, 16, 20, 0.96), rgba(18, 10, 12, 0.92)) !important;
  color: var(--gh-text) !important;
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.16);
}

.bk-panel-models-fastlisttemplate .bk-panel-models-markup-Alert.alert-danger {
  border-color: rgba(232, 93, 111, 0.4) !important;
}

.bk-panel-models-fastlisttemplate .bk-panel-models-markup-Alert.alert-warning {
  border-color: rgba(255, 193, 7, 0.28) !important;
}

.bk-panel-models-fastlisttemplate .bk-panel-models-markup-Alert.alert-info {
  border-color: rgba(212, 112, 122, 0.26) !important;
}

.bk-panel-models-fastlisttemplate .bk-tabulator,
.bk-panel-models-fastlisttemplate .dataframe {
  background: rgba(18, 10, 14, 0.92) !important;
  color: var(--gh-text) !important;
  border: 1px solid var(--gh-border) !important;
  border-radius: 16px !important;
  overflow: hidden;
}

.bk-panel-models-fastlisttemplate .tabulator-header,
.bk-panel-models-fastlisttemplate .dataframe thead th {
  background: rgba(38, 20, 24, 0.94) !important;
  color: #fff5f6 !important;
}

.bk-panel-models-fastlisttemplate .tabulator-row,
.bk-panel-models-fastlisttemplate .dataframe tbody tr {
  background: transparent !important;
  color: var(--gh-text) !important;
}

.bk-panel-models-fastlisttemplate .bk-panel-models-widgets-box-RadioBoxGroup label {
  display: block;
  margin: 6px 0;
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(18, 10, 14, 0.72);
  border: 1px solid transparent;
  color: var(--gh-text-muted);
  transition: all 160ms ease;
}

.bk-panel-models-fastlisttemplate .bk-panel-models-widgets-box-RadioBoxGroup input:checked + span,
.bk-panel-models-fastlisttemplate .bk-panel-models-widgets-box-RadioBoxGroup label:hover {
  color: var(--gh-text);
}

.bk-panel-models-fastlisttemplate .bk-panel-models-widgets-box-RadioBoxGroup label:has(input:checked) {
  background: linear-gradient(135deg, rgba(38, 18, 22, 0.96), rgba(52, 24, 30, 0.9));
  border-color: rgba(212, 112, 122, 0.34);
  box-shadow: 0 8px 20px rgba(194, 82, 94, 0.12);
}

/* ── Sidebar resize handle ─────────────────────────────── */
.bk-panel-models-fastlisttemplate .sidebar::after {
  content: '';
  position: absolute;
  top: 0; right: -3px; bottom: 0;
  width: 6px;
  cursor: col-resize;
  background: transparent;
  transition: background 0.2s ease;
  z-index: 100;
}
.bk-panel-models-fastlisttemplate .sidebar::after:hover,
.bk-panel-models-fastlisttemplate .sidebar.gh-resizing::after {
  background: var(--gh-accent);
}

/* ── Sidebar toggle button ─────────────────────────────── */
#gh-sidebar-toggle {
  position: fixed;
  top: 10px;
  left: 10px;
  z-index: 10000;
  width: 34px; height: 34px;
  border-radius: 10px;
  border: 1px solid var(--gh-border);
  background: rgba(26, 16, 18, 0.94);
  color: var(--gh-text-muted);
  font-size: 18px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 160ms ease;
  backdrop-filter: blur(10px);
  line-height: 1;
}
#gh-sidebar-toggle:hover {
  background: rgba(38, 20, 24, 0.96);
  color: var(--gh-accent);
  border-color: var(--gh-accent);
}

.bk-panel-models-fastlisttemplate .sidebar.gh-sidebar-collapsed {
  width: 0 !important;
  min-width: 0 !important;
  padding: 0 !important;
  overflow: hidden;
  border-right: none !important;
}

/* ── Bokeh tooltip ─────────────────────────────────────── */
.bk-tooltip {
  background: rgba(38, 20, 24, 0.95) !important;
  border: 1px solid rgba(178, 132, 140, 0.34) !important;
  border-radius: 10px !important;
  color: #f0e6e8 !important;
  padding: 8px 12px !important;
  font-size: 12px !important;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3) !important;
}

/* ── Custom scrollbar ──────────────────────────────────── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--gh-bg); }
::-webkit-scrollbar-thumb {
  background: var(--gh-surface-soft);
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: var(--gh-border-strong); }

/* ── Page fade-in animation ────────────────────────────── */
@keyframes gh-fade-in {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.bk-panel-models-fastlisttemplate .main-area .bk-Column {
  animation: gh-fade-in 0.3s ease;
}
"""

pn.extension(sizing_mode="stretch_width", raw_css=[_RAW_CSS])

# Ensure the package is importable when launched via ``panel serve``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gradienthound.ipc import IPCChannel  # noqa: E402
from gradienthound.pages import (  # noqa: E402
    dashboard, metrics, architecture, weights, gradients,
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
    "Metrics":       metrics.create,
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

_LIVE_PAGES = {"Dashboard", "Metrics", "Weights", "Gradients", "Training"}


def _on_page_change(event):
    layout, _update = pages[event.new]
    content_area.clear()
    content_area.append(layout)
    _update()  # immediate refresh on page switch


nav.param.watch(_on_page_change, "value")


# ── Periodic live-update ─────────────────────────────────────────────

def _periodic_update():
    if nav.value not in _LIVE_PAGES:
        return
    _, update_fn = pages[nav.value]
    try:
        update_fn()
    except Exception:
        pass  # keep callback alive even if a single update fails


pn.state.add_periodic_callback(_periodic_update, period=2000)

# ── Template ─────────────────────────────────────────────────────────

_SIDEBAR_JS = """
<div id="gh-sidebar-toggle">\u25c0</div>
<script>
(function() {
  function findSidebar() {
    var selectors = [
      '.bk-panel-models-fastlisttemplate .sidebar',
      '.sidebar',
      '.bk-panel-models-fastlisttemplate [class*="sidebar"]',
      '[class*="fastlisttemplate"] [class*="sidebar"]'
    ];
    for (var i = 0; i < selectors.length; i++) {
      var el = document.querySelector(selectors[i]);
      if (el) return el;
    }
    return null;
  }

  function applyCollapsed(sb, collapsed) {
    sb.classList.toggle('gh-sidebar-collapsed', collapsed);
    if (collapsed) {
      sb.style.setProperty('width', '0px', 'important');
      sb.style.setProperty('min-width', '0px', 'important');
      sb.style.setProperty('padding', '0px', 'important');
      sb.style.setProperty('overflow', 'hidden', 'important');
      sb.style.setProperty('border-right', 'none', 'important');
    } else {
      sb.style.removeProperty('width');
      sb.style.removeProperty('min-width');
      sb.style.removeProperty('padding');
      sb.style.removeProperty('overflow');
      sb.style.removeProperty('border-right');
    }
  }

  function initToggle() {
    var toggle = document.getElementById('gh-sidebar-toggle');
    if (!toggle) { setTimeout(initToggle, 200); return; }
    if (toggle.dataset.bound === '1') return;

    toggle.dataset.bound = '1';
    toggle.addEventListener('click', function() {
      var sb = findSidebar();
      if (!sb) return;
      var collapsed = !sb.classList.contains('gh-sidebar-collapsed');
      applyCollapsed(sb, collapsed);
      toggle.textContent = collapsed ? '\u25b6' : '\u25c0';
    });
  }

  function initResize() {
    var sb = findSidebar();
    if (!sb) { setTimeout(initResize, 200); return; }
    var dragging = false, startX = 0, startW = 0;
    sb.addEventListener('mousedown', function(e) {
      if (sb.classList.contains('gh-sidebar-collapsed')) return;
      var rect = sb.getBoundingClientRect();
      if (e.clientX < rect.right - 8) return;
      dragging = true; startX = e.clientX; startW = rect.width;
      sb.classList.add('gh-resizing');
      e.preventDefault();
    });
    document.addEventListener('mousemove', function(e) {
      if (!dragging) return;
      var newW = Math.min(500, Math.max(60, startW + (e.clientX - startX)));
      sb.style.width = newW + 'px';
    });
    document.addEventListener('mouseup', function() {
      if (dragging) { dragging = false; sb.classList.remove('gh-resizing'); }
    });
  }
  function initAll() {
    initToggle();
    initResize();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initAll);
  else initAll();
})();
</script>
"""

template = pn.template.FastListTemplate(
    title="GradientHound",
    accent_base_color="#d4707a",
    header_background="#1a0e10",
    sidebar=[
        pn.pane.Markdown(
            "### Navigation\n"
            "<span style='color:#b09a9e;'>Inspect architecture, training dynamics, and optimizer health in one place.</span>",
            margin=(10, 10),
        ),
        nav,
    ],
    main=[content_area],
    theme="dark",
    header=[pn.pane.HTML(_SIDEBAR_JS, sizing_mode="fixed", width=0, height=0)],
)

template.servable()
