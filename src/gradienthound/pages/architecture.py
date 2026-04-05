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
        svg_bytes = dot.pipe(format="svg")
        svg_str = svg_bytes.decode("utf-8")
        # Strip XML declaration if present so it embeds cleanly in HTML.
        if svg_str.startswith("<?xml"):
            svg_str = svg_str[svg_str.index("?>") + 2:].strip()
        # Make the SVG fill its container for pan-zoom.
        svg_str = svg_str.replace(
            "<svg ", '<svg id="arch-svg" style="width:100%;height:100%;" ', 1
        )
        html = f"""
        <div id="graph-container" style="
            width:100%;height:600px;border:1px solid #444;
            border-radius:8px;overflow:hidden;position:relative;
            background:#1e1e1e;">
          {svg_str}
          <div style="position:absolute;bottom:8px;right:8px;display:flex;gap:4px;">
            <button onclick="zoomIn()" style="
              width:32px;height:32px;border:none;border-radius:6px;
              background:#333;color:#ccc;font-size:18px;cursor:pointer;">+</button>
            <button onclick="zoomOut()" style="
              width:32px;height:32px;border:none;border-radius:6px;
              background:#333;color:#ccc;font-size:18px;cursor:pointer;">&minus;</button>
            <button onclick="resetZoom()" style="
              height:32px;border:none;border-radius:6px;padding:0 10px;
              background:#333;color:#ccc;font-size:12px;cursor:pointer;">Reset</button>
          </div>
        </div>
        <script>
        (function() {{
          const container = document.getElementById('graph-container');
          const svg = document.getElementById('arch-svg');
          let scale = 1, panX = 0, panY = 0, isPanning = false, startX, startY;

          function applyTransform() {{
            svg.style.transform = 'translate(' + panX + 'px,' + panY + 'px) scale(' + scale + ')';
            svg.style.transformOrigin = '0 0';
          }}

          container.addEventListener('wheel', function(e) {{
            e.preventDefault();
            const rect = container.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            const prevScale = scale;
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            scale = Math.max(0.1, Math.min(10, scale * delta));
            panX = mouseX - (mouseX - panX) * (scale / prevScale);
            panY = mouseY - (mouseY - panY) * (scale / prevScale);
            applyTransform();
          }}, {{passive: false}});

          container.addEventListener('mousedown', function(e) {{
            isPanning = true;
            startX = e.clientX - panX;
            startY = e.clientY - panY;
            container.style.cursor = 'grabbing';
          }});
          window.addEventListener('mousemove', function(e) {{
            if (!isPanning) return;
            panX = e.clientX - startX;
            panY = e.clientY - startY;
            applyTransform();
          }});
          window.addEventListener('mouseup', function() {{
            isPanning = false;
            container.style.cursor = 'grab';
          }});
          container.style.cursor = 'grab';

          window.zoomIn = function() {{
            const rect = container.getBoundingClientRect();
            const cx = rect.width / 2, cy = rect.height / 2;
            const prevScale = scale;
            scale = Math.min(10, scale * 1.3);
            panX = cx - (cx - panX) * (scale / prevScale);
            panY = cy - (cy - panY) * (scale / prevScale);
            applyTransform();
          }};
          window.zoomOut = function() {{
            const rect = container.getBoundingClientRect();
            const cx = rect.width / 2, cy = rect.height / 2;
            const prevScale = scale;
            scale = Math.max(0.1, scale / 1.3);
            panX = cx - (cx - panX) * (scale / prevScale);
            panY = cy - (cy - panY) * (scale / prevScale);
            applyTransform();
          }};
          window.resetZoom = function() {{
            scale = 1; panX = 0; panY = 0;
            applyTransform();
          }};
        }})();
        </script>
        """
        st.components.v1.html(html, height=620)

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
