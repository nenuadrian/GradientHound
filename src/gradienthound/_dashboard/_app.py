"""Dash app factory and callbacks."""
from __future__ import annotations

import json
import threading
from fnmatch import fnmatch
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, callback, Output, Input, State, no_update, ctx, ALL
import dash_cytoscape as cyto

cyto.load_extra_layouts()

from ._constants import PAGES, PAGE_CATEGORIES, SERIES_COLORS, HEALTH_COLORS
from ._helpers import (
    plotly_layout, short_layer, placeholder_page, node_detail_panel,
    compute_checkpoint_change_tables, render_checkpoint_change_table,
    compute_effective_rank_table, compute_distribution_stats_table,
    compute_scalar_metric_tables,
    compute_spectral_gap_table, compute_spectral_gap_ratios,
)
from ._health import weight_health
from ._wandb import parse_wandb_project_run_id, fetch_wandb_run_metrics, metrics_page_wandb
from ._pages import (
    overview_page, landing_page_empty, architecture_page,
    weight_health_page, distributions_page, spectral_page, dynamics_page,
    gradient_flow_page,
    checkpoints_page, checkpoints_page_empty,
    on_demand_page, raw_data_page, tools_page,
)
from ._tool_registry import ToolRegistry, ToolInfo, Requirement, register_builtin_tools


def _merge_model_exports(gh_files: list[Path]) -> dict:
    """Merge multiple ``.gh.json`` model exports into a single descriptor.

    Each file's module tree, parameters, and FX graph are prefixed with the
    model name so they coexist without collisions.  This lets GradientHound
    display multi-network architectures (e.g. actor + critic) as one view.
    """
    merged: dict = {
        "format_version": "1.0",
        "model_name": "",
        "model_class": "",
        "total_params": 0,
        "trainable_params": 0,
        "inputs": [],
        "outputs": [],
        "module_tree": {"name": "root", "modules": []},
        "parameters": {},
    }

    names: list[str] = []
    sub_root_paths: list[str] = []
    for gh_file in gh_files:
        with open(gh_file) as f:
            data = json.load(f)

        name = data.get("model_name", gh_file.stem)
        names.append(name)
        prefix = name

        merged["total_params"] += data.get("total_params", 0)
        merged["trainable_params"] += data.get("trainable_params", 0)

        # Merge inputs/outputs with prefix
        for inp in data.get("inputs", []):
            inp_copy = dict(inp)
            inp_copy["name"] = f"{prefix}/{inp_copy.get('name', '')}"
            merged["inputs"].append(inp_copy)
        for out in data.get("outputs", []):
            out_copy = dict(out)
            out_copy["name"] = f"{prefix}/{out_copy.get('name', '')}"
            merged["outputs"].append(out_copy)

        # Merge parameters with prefix
        for param_name, param_info in data.get("parameters", {}).items():
            merged["parameters"][f"{prefix}.{param_name}"] = param_info

        # Merge module tree: prefix all paths
        mt = data.get("module_tree", {})
        for mod in mt.get("modules", []):
            mod_copy = dict(mod)
            mod_copy["path"] = f"{prefix}.{mod_copy['path']}" if mod_copy.get("path") else prefix
            if "children" in mod_copy:
                mod_copy["children"] = [f"{prefix}.{c}" for c in mod_copy["children"]]
            merged["module_tree"]["modules"].append(mod_copy)

        # Determine the prefixed root path for this sub-model's tree.
        mt_root_name = mt.get("name", "")
        root_modules = [m for m in mt.get("modules", []) if m.get("path") == mt_root_name]
        if root_modules:
            # Root module was prefixed to "{prefix}.{mt_root_name}"
            sub_root_paths.append(f"{prefix}.{mt_root_name}" if mt_root_name else prefix)
        else:
            # No root in the tree — create a synthetic parent
            top_children = [
                f"{prefix}.{m['path']}" for m in mt.get("modules", [])
                if m.get("path") and "." not in m["path"]
            ]
            merged["module_tree"]["modules"].append({
                "path": prefix,
                "type": data.get("model_name", name),
                "type_full": data.get("model_class", ""),
                "is_leaf": False,
                "children": top_children,
                "params": data.get("total_params", 0),
            })
            sub_root_paths.append(prefix)

        # Merge FX graph with prefixed node names
        fx = data.get("fx_graph")
        if fx:
            if "fx_graph" not in merged:
                merged["fx_graph"] = {"nodes": [], "edges": []}
            for node in fx.get("nodes", []):
                node_copy = dict(node)
                node_copy["name"] = f"{prefix}__{node_copy['name']}"
                if node_copy.get("nn_module"):
                    node_copy["nn_module"] = f"{prefix}.{node_copy['nn_module']}"
                node_copy["args"] = [f"{prefix}__{a}" for a in node_copy.get("args", [])]
                merged["fx_graph"]["nodes"].append(node_copy)
            for edge in fx.get("edges", []):
                merged["fx_graph"]["edges"].append({
                    "from": f"{prefix}__{edge['from']}",
                    "to": f"{prefix}__{edge['to']}",
                })

        # Merge graph signature
        sig = data.get("graph_signature")
        if sig:
            if "graph_signature" not in merged:
                merged["graph_signature"] = {
                    "parameters": [], "buffers": [],
                    "user_inputs": [], "user_outputs": [],
                }
            ms = merged["graph_signature"]
            ms["parameters"].extend(f"{prefix}.{p}" for p in sig.get("parameters", []))
            ms["buffers"].extend(f"{prefix}.{b}" for b in sig.get("buffers", []))
            ms["user_inputs"].extend(f"{prefix}__{i}" for i in sig.get("user_inputs", []))
            ms["user_outputs"].extend(f"{prefix}__{o}" for o in sig.get("user_outputs", []))

        # Merge live analysis (FLOPs, activations, pruning groups)
        sub_live = data.get("live_analysis")
        if sub_live:
            if "live_analysis" not in merged:
                merged["live_analysis"] = {
                    "flops": {"total_flops": 0, "by_module": {}, "by_operator": {}, "unsupported_ops": {}},
                    "activations": {},
                    "pruning_groups": [],
                }
            ml = merged["live_analysis"]

            sub_flops = sub_live.get("flops")
            if sub_flops:
                ml["flops"]["total_flops"] += sub_flops.get("total_flops", 0)
                for mod, count in sub_flops.get("by_module", {}).items():
                    ml["flops"]["by_module"][f"{prefix}.{mod}"] = count
                for op, count in sub_flops.get("by_operator", {}).items():
                    ml["flops"]["by_operator"][op] = ml["flops"]["by_operator"].get(op, 0) + count
                for op, count in sub_flops.get("unsupported_ops", {}).items():
                    ml["flops"]["unsupported_ops"][op] = ml["flops"]["unsupported_ops"].get(op, 0) + count

            sub_acts = sub_live.get("activations")
            if sub_acts:
                for mod, info in sub_acts.items():
                    ml["activations"][f"{prefix}.{mod}"] = info

            sub_pruning = sub_live.get("pruning_groups")
            if sub_pruning:
                for group in sub_pruning:
                    prefixed = dict(group)
                    prefixed["coupled_layers"] = [
                        f"{prefix}/{l}" for l in group.get("coupled_layers", [])
                    ]
                    prefixed["sub_model"] = prefix
                    ml["pruning_groups"].append(prefixed)

    merged["model_name"] = " + ".join(names)
    merged["model_class"] = ", ".join(names)
    merged["sub_models"] = names

    # Add root node with children pointing to each sub-model's root
    merged["module_tree"]["modules"].insert(0, {
        "path": "root",
        "type": merged["model_name"],
        "type_full": "",
        "is_leaf": False,
        "children": sub_root_paths,
        "params": merged["total_params"],
    })

    return merged


def create_app(
    data_dir: str | None = None,
    model_paths: list[str] | None = None,
    checkpoint_paths: list[str] | None = None,
    loader_path: str | None = None,
    wandb_entity: str | None = None,
    wandb_project_run_id: str | None = None,
    model_path: str | None = None,  # Deprecated, for backward compatibility
) -> Dash:
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        title="GradientHound",
        external_stylesheets=[dbc.themes.FLATLY],
    )

    # Backward compatibility: if model_path is provided (old parameter), convert to model_paths
    if model_path is not None and model_paths is None:
        model_paths = [model_path]

    # Load model export(s)
    model_data: dict | None = None
    available_models: dict[str, str] = {}  # {name: path} mapping
    selected_model: str | None = None
    
    if model_paths:
        # Build a mapping of model names (from filenames) to paths
        for model_path in model_paths:
            p = Path(model_path)
            name = p.stem.replace(".gh", "")  # Remove .gh.json extension
            if not name:
                name = p.name
            available_models[name] = model_path
        
        # Load the first model by default
        if available_models:
            selected_model = next(iter(available_models))
            model_path = available_models[selected_model]
            with open(model_path) as f:
                model_data = json.load(f)

    # IPC channel
    ipc = None
    if data_dir:
        from gradienthound.ipc import IPCChannel
        ipc = IPCChannel(directory=data_dir)

    # Checkpoint state
    ckpt_state: dict = {
        "paths": checkpoint_paths or [],
        "loader_path": loader_path,
        "processed": False,
        "snapshots": [],
        "selected_indices": None,  # Track which checkpoints are selected for processing
    }
    has_checkpoints = bool(ckpt_state["paths"])

    wandb_state: dict = {
        "entity": (wandb_entity or "").strip(),
        "project_run_id": (wandb_project_run_id or "").strip(),
        "data": None,
    }

    # Model state
    model_state: dict = {
        "available": available_models,
        "selected": selected_model,
        "current_data": model_data,
    }

    # ── Sidebar navigation ────────────────────────────────────────────
    sidebar_children = [
        html.Div("GradientHound", className="gh-sidebar-brand"),
    ]

    # Model selector (only when multiple models)
    if len(available_models) > 1:
        sidebar_children.append(html.Div([
            html.Small("Model", className="text-uppercase text-muted d-block mb-1"),
            dcc.Dropdown(
                id="model-selector",
                options=[{"label": name, "value": name} for name in sorted(available_models.keys())],
                value=selected_model,
                clearable=False,
            ),
        ], className="gh-sidebar-model"))

    # Build categorized nav links
    for category, paths in PAGE_CATEGORIES.items():
        sidebar_children.append(
            html.Div(category, className="gh-sidebar-category"),
        )
        nav_items = []
        for path in paths:
            if path in PAGES:
                title, _ = PAGES[path]
                nav_items.append(
                    dbc.NavLink(title, href=path, id=f"nav-{path}", active="exact"),
                )
        sidebar_children.append(dbc.Nav(nav_items, vertical=True, pills=True))

    sidebar = html.Div(sidebar_children, className="gh-sidebar-inner")

    # Background task state -- shared across callbacks, mutated by threads
    bg_tasks: dict = {
        "ckpt_running": False,
        "ckpt_done": False,
        "ckpt_error": None,
        "ckpt_message": None,
        "wandb_running": False,
        "wandb_done": False,
        "wandb_error": None,
        "wandb_data": None,
        "wandb_message": None,
    }

    # ── Tool registry ──────────────────────────────────────────────
    tool_registry = ToolRegistry()
    register_builtin_tools(
        tool_registry,
        ipc=ipc,
        has_checkpoints=has_checkpoints,
        ckpt_state=ckpt_state,
        wandb_state=wandb_state,
        model_data=model_data,
    )
    # Expose registry on the app so 3rd parties can register tools:
    #   app = create_app(...)
    #   app.tool_registry.register(ToolInfo(...))
    app.tool_registry = tool_registry  # type: ignore[attr-defined]

    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="ckpt-store", data=None),
        dcc.Store(id="wandb-store", data=None),
        dcc.Store(id="model-store", data={"selected": selected_model}),
        # Polling intervals for background tasks (disabled until a task starts)
        dcc.Interval(id="ckpt-poll", interval=1000, n_intervals=0, disabled=True),
        dcc.Interval(id="wandb-poll", interval=1000, n_intervals=0, disabled=True),
        # Global background-task status bar (visible from any page)
        html.Div(id="bg-task-bar"),
        dbc.Row([
            dbc.Col(sidebar, width=2, id="gh-sidebar"),
            dbc.Col(
                html.Div(id="gh-content", className="px-4 py-2"),
                width=10,
                id="gh-main",
            ),
        ], className="g-0"),
    ])

    def _empty_gradflow_figure(message: str):
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.update_layout(
            **plotly_layout(title="Gradient Flow"),
            height=500,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14, "color": "#666"},
            }],
        )
        return fig

    # ── Model selector ───────────────────────────────────────────────

    if len(available_models) > 1:
        @callback(
            Output("model-store", "data"),
            Input("model-selector", "value"),
            prevent_initial_call=True,
        )
        def _update_model_selection(selected_model_name):
            if selected_model_name and selected_model_name in model_state["available"]:
                model_path = model_state["available"][selected_model_name]
                with open(model_path) as f:
                    model_data_new = json.load(f)
                model_state["selected"] = selected_model_name
                model_state["current_data"] = model_data_new
                return {"selected": selected_model_name, "data": model_data_new}
            return {"selected": selected_model_name}

    # ── Global background-task status bar ─────────────────────────

    @callback(
        Output("bg-task-bar", "children"),
        Input("ckpt-poll", "n_intervals"),
        Input("wandb-poll", "n_intervals"),
    )
    def _update_bg_task_bar(_ckpt_n, _wandb_n):
        items = []
        if bg_tasks["ckpt_running"]:
            items.append(dbc.Alert([
                dbc.Spinner(size="sm", spinner_class_name="me-2"),
                "Processing checkpoints\u2026",
            ], color="info", className="mb-1 py-2", dismissable=False))
        if bg_tasks["wandb_running"]:
            items.append(dbc.Alert([
                dbc.Spinner(size="sm", spinner_class_name="me-2"),
                "Fetching W&B metrics\u2026",
            ], color="info", className="mb-1 py-2", dismissable=False))
        return items if items else ""

    # ── Process checkpoints (background) ───────────────────────────

    if has_checkpoints:
        # Store for tracking which checkpoint indices are currently visible in the table
        ckpt_state["visible_indices"] = []
        
        def _ckpt_worker():
            from gradienthound.checkpoint import process_checkpoints
            try:
                # Get selected checkpoint paths
                selected_paths = ckpt_state["paths"]  # Default to all
                if ckpt_state["selected_indices"]:
                    selected_indices_set = set(ckpt_state["selected_indices"])
                    selected_paths = [
                        path for i, path in enumerate(ckpt_state["paths"])
                        if i in selected_indices_set
                    ]
                
                if not selected_paths:
                    bg_tasks["ckpt_error"] = "No checkpoints selected."
                    bg_tasks["ckpt_done"] = True
                    return
                
                snapshots = process_checkpoints(
                    selected_paths, loader_path=ckpt_state["loader_path"],
                )
                ckpt_state["snapshots"] = snapshots
                ckpt_state["processed"] = True
                n_params = len({s["layer"] for snap in snapshots for s in snap["weight_stats"]})
                bg_tasks["ckpt_message"] = f"Done \u2014 {len(snapshots)} checkpoints, {n_params} params"
                bg_tasks["ckpt_done"] = True
            except Exception as exc:
                bg_tasks["ckpt_error"] = str(exc)
                bg_tasks["ckpt_done"] = True
            finally:
                bg_tasks["ckpt_running"] = False

        @callback(
            Output("ckpt-status", "children", allow_duplicate=True),
            Output("ckpt-process-btn", "disabled", allow_duplicate=True),
            Output("ckpt-poll", "disabled"),
            Input("ckpt-process-btn", "n_clicks"),
            State("ckpt-selector", "data"),
            prevent_initial_call=True,
        )
        def _start_ckpt_processing(n_clicks, selected_indices):
            if not n_clicks or bg_tasks["ckpt_running"]:
                return no_update, no_update, no_update
            
            # Store selected checkpoint indices
            if not selected_indices:
                return (
                    dbc.Alert("Please select at least one checkpoint to process.", color="warning", className="mb-0"),
                    False,
                    True,
                )
            
            ckpt_state["selected_indices"] = [int(idx) for idx in selected_indices]
            bg_tasks["ckpt_running"] = True
            bg_tasks["ckpt_done"] = False
            bg_tasks["ckpt_error"] = None
            bg_tasks["ckpt_message"] = None
            threading.Thread(target=_ckpt_worker, daemon=True).start()
            return (
                [dbc.Spinner(size="sm", spinner_class_name="me-2"),
                 "Processing checkpoints in background\u2026 you can navigate other pages."],
                True,
                False,  # enable polling
            )

        @callback(
            Output("ckpt-store", "data"),
            Output("ckpt-status", "children"),
            Output("ckpt-process-btn", "disabled"),
            Output("ckpt-poll", "disabled", allow_duplicate=True),
            Input("ckpt-poll", "n_intervals"),
            prevent_initial_call=True,
        )
        def _poll_ckpt_processing(_n):
            if not bg_tasks["ckpt_done"]:
                return no_update, no_update, no_update, no_update
            # Task finished
            if bg_tasks["ckpt_error"]:
                return (
                    no_update,
                    dbc.Alert(f"Error: {bg_tasks['ckpt_error']}", color="danger", className="mb-0"),
                    False,
                    True,  # disable polling
                )
            return (
                {"ready": True},
                dbc.Alert(bg_tasks["ckpt_message"], color="success", className="mb-0"),
                True,
                True,  # disable polling
            )

        # ── Build checkpoint table with embedded checkboxes ────────────

        @callback(
            Output("ckpt-table-body", "children"),
            Input("ckpt-selector", "data"),
            Input("ckpt-filter-input", "value"),
            prevent_initial_call=False,
        )
        def _build_ckpt_table(selected_value, filter_pattern):
            """Build table rows with checkboxes for checkpoint selection."""
            from gradienthound._dashboard._page_checkpoints import _fmt_bytes
            
            # Decode selection (dcc.Store stores as JSON)
            selected_indices = set()
            if selected_value:
                selected_indices = set(int(v) for v in selected_value) if isinstance(selected_value, list) else {int(selected_value)}
            
            # Build visible rows based on filter and track visible indices
            rows = []
            visible_indices = []
            for i, path in enumerate(ckpt_state["paths"]):
                p = Path(path)
                name = p.name
                
                # Apply filter if present
                if filter_pattern:
                    if not fnmatch(name, filter_pattern):
                        continue
                
                visible_indices.append(i)
                
                # Build row with checkbox in first column
                size = ""
                if p.exists():
                    size = _fmt_bytes(p.stat().st_size)
                
                status = ""
                if ckpt_state.get("snapshots"):
                    snap = next(
                        (s for s in ckpt_state["snapshots"] if s["path"] == path), 
                        None
                    )
                    if snap:
                        status = f"{len(snap['weight_stats'])} params"
                
                rows.append(html.Tr([
                    html.Td(
                        dbc.Checkbox(
                            id={"type": "ckpt-row-checkbox", "index": i},
                            value=i in selected_indices,
                            className="form-check-input",
                            style={"cursor": "pointer", "marginTop": "0.25rem"},
                        ),
                        style={"textAlign": "center", "verticalAlign": "middle"},
                    ),
                    html.Td(name),
                    html.Td(size),
                    html.Td(dbc.Badge(status, color="success") if status else "\u2014"),
                    html.Td(html.Code(str(path), style={"fontSize": "0.8em"})),
                ]))
            
            # Store visible indices for the checkbox callback
            ckpt_state["visible_indices"] = visible_indices
            
            return rows if rows else [html.Tr([html.Td("No checkpoints match filter", colSpan=5, className="text-center text-muted")])]

        # ── Handle checkbox changes ────────────────────────────────────

        @callback(
            Output("ckpt-selector", "data", allow_duplicate=True),
            Input({"type": "ckpt-row-checkbox", "index": ALL}, "value"),
            prevent_initial_call=True,
        )
        def _update_selection_from_checkbox(checkbox_values):
            """Update selection when individual checkboxes are toggled."""
            if not ctx.triggered or not checkbox_values:
                return no_update
            
            # Use visible indices to map checkbox values to checkpoint indices
            visible_indices = ckpt_state.get("visible_indices", [])
            if not visible_indices:
                return no_update
            
            # Build new selection based on which visible checkboxes are checked
            new_selection = []
            for visible_idx, is_checked in zip(visible_indices, checkbox_values):
                if is_checked:
                    new_selection.append(str(visible_idx))
            
            return new_selection

        # ── Filter checkpoints by pattern ────────────────────────────

        @callback(
            Output("ckpt-selector", "data", allow_duplicate=True),
            Input("ckpt-filter-input", "value"),
            State("ckpt-selector", "data"),
            prevent_initial_call=True,
        )
        def _handle_filter_change(filter_pattern, current_selection):
            """Maintain selection when filter changes."""
            # Just return current selection - the table builder will handle filtering
            return current_selection or [str(i) for i in range(len(ckpt_state["paths"]))]

        # ── Update selection count display ────────────────────────────

        @callback(
            Output("ckpt-selection-count", "children"),
            Input("ckpt-selector", "data"),
        )
        def _update_selection_count(selected_values):
            """Update the display showing how many checkpoints are selected."""
            count = len(selected_values) if selected_values else 0
            total = len(ckpt_state["paths"])
            if count == 0:
                return html.Span(f"0 of {total} selected", className="text-danger fw-bold")
            elif count == total:
                return html.Span(f"All {total} selected", className="text-success fw-bold")
            else:
                return html.Span(f"{count} of {total} selected", className="text-info fw-bold")

        # ── Select All / Clear checkpoint buttons ────────────────────

        @callback(
            Output("ckpt-selector", "data", allow_duplicate=True),
            Input("ckpt-select-all-btn", "n_clicks"),
            Input("ckpt-clear-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def _toggle_ckpt_selection(select_all_clicks, clear_clicks):
            if not ctx.triggered:
                return no_update

            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if button_id == "ckpt-select-all-btn":
                # Select all checkpoints
                return [str(i) for i in range(len(ckpt_state["paths"]))]
            elif button_id == "ckpt-clear-btn":
                # Clear all selections
                return []
            
            return no_update

    # ── Fetch W&B run metrics (background) ────────────────────────

    def _wandb_worker(entity: str, project_run_id_str: str):
        try:
            project, run_id = parse_wandb_project_run_id(project_run_id_str)
            entries, run_label = fetch_wandb_run_metrics(entity, project, run_id)
            metric_keys = sorted({k for entry in entries for k in entry if k != "_step"})

            data = {
                "entity": entity,
                "project": project,
                "run_id": run_id,
                "run_label": run_label,
                "entries": entries,
                "metric_keys": metric_keys,
            }
            wandb_state["entity"] = entity
            wandb_state["project_run_id"] = project_run_id_str
            wandb_state["data"] = data
            bg_tasks["wandb_data"] = data
            bg_tasks["wandb_message"] = f"Fetched {len(metric_keys)} metrics across {len(entries)} points."
            bg_tasks["wandb_done"] = True
        except Exception as exc:
            bg_tasks["wandb_error"] = str(exc)
            bg_tasks["wandb_done"] = True
        finally:
            bg_tasks["wandb_running"] = False

    @callback(
        Output("wandb-fetch-status", "children", allow_duplicate=True),
        Output("wandb-fetch-btn", "disabled"),
        Output("wandb-poll", "disabled"),
        Input("wandb-fetch-btn", "n_clicks"),
        State("wandb-entity-input", "value"),
        State("wandb-project-run-id-input", "value"),
        prevent_initial_call=True,
    )
    def _start_wandb_fetch(n_clicks, entity_value, project_run_id_value):
        if not n_clicks or bg_tasks["wandb_running"]:
            return no_update, no_update, no_update

        entity = (entity_value or "").strip()
        project_run_id_str = (project_run_id_value or "").strip()

        if not entity or not project_run_id_str:
            return (
                dbc.Alert("Both Entity and Project/Run ID are required.", color="warning", className="mb-0"),
                no_update,
                no_update,
            )

        bg_tasks["wandb_running"] = True
        bg_tasks["wandb_done"] = False
        bg_tasks["wandb_error"] = None
        bg_tasks["wandb_data"] = None
        bg_tasks["wandb_message"] = None
        threading.Thread(target=_wandb_worker, args=(entity, project_run_id_str), daemon=True).start()
        return (
            [dbc.Spinner(size="sm", spinner_class_name="me-2"),
             "Fetching W&B metrics in background\u2026 you can navigate other pages."],
            True,
            False,  # enable polling
        )

    @callback(
        Output("wandb-store", "data"),
        Output("wandb-fetch-status", "children"),
        Output("wandb-fetch-btn", "disabled", allow_duplicate=True),
        Output("wandb-poll", "disabled", allow_duplicate=True),
        Input("wandb-poll", "n_intervals"),
        prevent_initial_call=True,
    )
    def _poll_wandb_fetch(_n):
        if not bg_tasks["wandb_done"]:
            return no_update, no_update, no_update, no_update
        if bg_tasks["wandb_error"]:
            return (
                no_update,
                dbc.Alert(f"Failed to fetch W&B metrics: {bg_tasks['wandb_error']}", color="danger", className="mb-0"),
                False,
                True,  # disable polling
            )
        return (
            bg_tasks["wandb_data"],
            dbc.Alert(bg_tasks["wandb_message"], color="success", className="mb-0"),
            False,
            True,  # disable polling
        )

    # ── Routing ──────────────────────────────────────────────────────

    @callback(
        Output("gh-content", "children"),
        Input("url", "pathname"),
        Input("ckpt-store", "data"),
        Input("wandb-store", "data"),
        Input("model-store", "data"),
    )
    def _route(pathname, ckpt_data, wandb_data, model_store_data):
        snapshots = ckpt_state["snapshots"] if ckpt_state["processed"] else None

        # Use model data from store if available (for model switching), otherwise use initial model_data
        current_model_data = model_data
        if model_store_data and "data" in model_store_data:
            current_model_data = model_store_data["data"]

        # ── MODEL ────────────────────────────────────────────────────
        if pathname is None or pathname == "/":
            if current_model_data:
                return overview_page(current_model_data, snapshots=snapshots)
            return landing_page_empty()

        if pathname == "/architecture":
            if current_model_data:
                return architecture_page(current_model_data, snapshots=snapshots)
            return placeholder_page("Architecture", "Load a model first.")

        # ── ANALYSIS ─────────────────────────────────────────────────
        if pathname == "/weight-health":
            return weight_health_page(current_model_data, snapshots=snapshots)

        if pathname == "/distributions":
            return distributions_page(current_model_data, snapshots=snapshots)

        if pathname == "/spectral" or pathname == "/weightwatcher":
            return spectral_page(current_model_data, snapshots=snapshots)

        if pathname == "/dynamics":
            return dynamics_page(current_model_data, snapshots=snapshots)

        # ── LIVE ─────────────────────────────────────────────────────
        if pathname == "/gradient-flow":
            model_names = []
            if ipc is not None:
                model_names = sorted(ipc.read_models().keys())
            return gradient_flow_page(model_names=model_names)

        if pathname == "/metrics":
            page_data = wandb_data if wandb_data is not None else wandb_state.get("data")
            return metrics_page_wandb(
                page_data,
                default_entity=wandb_state.get("entity"),
                default_project_run_id=wandb_state.get("project_run_id"),
            )

        # ── DATA ─────────────────────────────────────────────────────
        if pathname == "/checkpoints":
            if has_checkpoints:
                return checkpoints_page(ckpt_state["paths"], snapshots)
            return checkpoints_page_empty()

        if pathname == "/on-demand":
            model_names = sorted(ipc.read_models().keys()) if ipc else []
            return on_demand_page(model_names=model_names, has_ipc=ipc is not None)

        if pathname == "/raw-data":
            return raw_data_page(
                has_ipc=ipc is not None,
                data_dir=str(ipc.directory) if ipc else None,
            )

        # ── SYSTEM ───────────────────────────────────────────────────
        if pathname == "/tools":
            return tools_page(tool_registry.all_status())

        if pathname in PAGES:
            title, desc = PAGES[pathname]
            return placeholder_page(title, desc)

        return placeholder_page("Not Found", f"No page at {pathname}")

    # ── Gradient flow page callbacks ───────────────────────────────

    @callback(
        Output("gradflow-chart", "figure"),
        Output("gradflow-model-select", "options"),
        Output("gradflow-model-select", "value"),
        Output("gradflow-summary", "children"),
        Input("gradflow-refresh", "n_intervals"),
        Input("gradflow-model-select", "value"),
        Input("gradflow-window", "value"),
        Input("gradflow-hide-bias", "value"),
        prevent_initial_call=True,
    )
    def _update_gradient_flow(_n_intervals, selected_model, window_steps, hide_bias_opts):
        import plotly.graph_objects as go

        if ipc is None:
            return (
                _empty_gradflow_figure("No IPC data directory configured."),
                [],
                None,
                "Pass --data-dir to load live gradient captures.",
            )

        # Quick count check avoids loading all rows when empty.
        if ipc._count_events("gradient_stats") == 0:
            model_options = [
                {"label": name, "value": name}
                for name in sorted(ipc.read_models().keys())
            ]
            model_value = selected_model or (model_options[0]["value"] if model_options else None)
            return (
                _empty_gradflow_figure("Waiting for gradient stats... call gradienthound.step() during training."),
                model_options,
                model_value,
                "No gradient records yet.",
            )

        # Discover model names from the kv store (fast) rather than
        # scanning all events.
        model_names = sorted(ipc.read_models().keys())
        if not model_names:
            # Fallback: read a small sample to extract model names.
            sample = ipc.read_gradient_stats(last_n=200)
            model_names = sorted({e.get("model", "") for e in sample if e.get("model")})
        model_options = [{"label": name, "value": name} for name in model_names]

        if selected_model not in model_names:
            selected_model = model_names[0] if model_names else None

        # Use SQL-level filtering: find latest step for this model,
        # then fetch only the step window we need.
        latest_step = ipc._max_step("gradient_stats", model=selected_model)
        if latest_step is None:
            return (
                _empty_gradflow_figure("No records for the selected model."),
                model_options,
                selected_model,
                "No gradient records for this model.",
            )

        window_steps = int(window_steps or 10)
        min_step = max(0, latest_step - window_steps + 1)
        window_entries = ipc.read_gradient_stats(
            model=selected_model, step_min=min_step, step_max=latest_step,
        )

        hide_bias = "hide_bias" in (hide_bias_opts or [])
        layer_stats: dict[str, dict[str, float]] = {}
        layer_order: list[str] = []

        for rec in window_entries:
            layer = str(rec.get("layer", ""))
            if not layer:
                continue
            if hide_bias and layer.endswith(".bias"):
                continue

            avg_grad = rec.get("grad_abs_mean")
            if avg_grad is None:
                avg_grad = abs(float(rec.get("grad_mean", 0.0)))

            max_grad = rec.get("grad_abs_max")
            if max_grad is None:
                max_grad = abs(float(rec.get("grad_mean", 0.0))) + float(rec.get("grad_std", 0.0))

            if layer not in layer_stats:
                layer_stats[layer] = {
                    "avg_sum": 0.0,
                    "avg_count": 0.0,
                    "max_val": 0.0,
                }
                layer_order.append(layer)

            layer_stats[layer]["avg_sum"] += float(avg_grad)
            layer_stats[layer]["avg_count"] += 1.0
            layer_stats[layer]["max_val"] = max(layer_stats[layer]["max_val"], float(max_grad))

        if not layer_order:
            return (
                _empty_gradflow_figure("No layers remain after applying filters."),
                model_options,
                selected_model,
                "No plottable layers after filters.",
            )

        x_layers = [short_layer(layer) for layer in layer_order]
        avg_vals = [layer_stats[layer]["avg_sum"] / max(layer_stats[layer]["avg_count"], 1.0) for layer in layer_order]
        max_vals = [layer_stats[layer]["max_val"] for layer in layer_order]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_layers,
            y=max_vals,
            name="max|grad|",
            marker_color="#f39c12",
            opacity=0.65,
            hovertemplate="layer=%{x}<br>max|grad|=%{y:.6g}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=x_layers,
            y=avg_vals,
            name="avg|grad|",
            marker_color="#375a7f",
            opacity=0.95,
            hovertemplate="layer=%{x}<br>avg|grad|=%{y:.6g}<extra></extra>",
        ))
        fig.update_layout(
            **plotly_layout(title="Gradient Flow Across Layers"),
            barmode="overlay",
            height=520,
            xaxis_title="Layer",
            yaxis_title="Gradient magnitude",
            xaxis_tickangle=-45,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        )

        summary = (
            f"Model: {selected_model or 'all'} | Steps: {min_step}-{latest_step} | "
            f"Layers: {len(layer_order)}"
        )
        return fig, model_options, selected_model, summary

    # ── Node click detail panels ─────────────────────────────────────

    @callback(
        Output("gh-node-detail", "children"),
        Input("gh-cyto", "tapNodeData"),
        prevent_initial_call=True,
    )
    def _on_fx_node_click(data):
        if data is None:
            return no_update
        return node_detail_panel(data)

    @callback(
        Output("gh-arch-detail", "children"),
        Input("gh-cyto-arch", "tapNodeData"),
        prevent_initial_call=True,
    )
    def _on_arch_node_click(data):
        if data is None:
            return no_update
        return node_detail_panel(data)

    # ── Histogram callback ───────────────────────────────────────────

    @callback(
        Output("ckpt-histogram", "figure"),
        Input("ckpt-layer-select", "value"),
        prevent_initial_call=True,
    )
    def _update_histogram(selected_layer):
        import plotly.graph_objects as go
        snapshots = ckpt_state.get("snapshots", [])
        fig = go.Figure()
        for i, snap in enumerate(snapshots):
            stat = next((s for s in snap["weight_stats"] if s["layer"] == selected_layer), None)
            if stat and "hist_counts" in stat:
                fig.add_trace(go.Bar(
                    x=stat["hist_centers"], y=stat["hist_counts"], name=snap["name"],
                    marker_color=SERIES_COLORS[i % len(SERIES_COLORS)], opacity=0.65,
                ))
        fig.update_layout(**plotly_layout(title=f"Weight Distribution \u2014 {short_layer(selected_layer)}"),
                          barmode="overlay", height=360)
        return fig

    # ── SVD spectrum callback ────────────────────────────────────────

    @callback(
        Output("ckpt-svd-spectrum", "figure"),
        Input("ckpt-svd-layer-select", "value"),
        prevent_initial_call=True,
    )
    def _update_svd_spectrum(selected_layer):
        import plotly.graph_objects as go
        snapshots = ckpt_state.get("snapshots", [])
        fig = go.Figure()
        for i, snap in enumerate(snapshots):
            stat = next((s for s in snap["weight_stats"] if s["layer"] == selected_layer), None)
            if stat and "singular_values" in stat:
                svs = stat["singular_values"]
                fig.add_trace(go.Scatter(
                    x=list(range(len(svs))), y=svs, mode="lines", name=snap["name"],
                    line={"color": SERIES_COLORS[i % len(SERIES_COLORS)]},
                ))
        fig.update_layout(**plotly_layout(title=f"Singular Values \u2014 {short_layer(selected_layer)}"),
                          xaxis_title="Index", yaxis_title="Singular Value", yaxis_type="log", height=360)
        return fig

    # ── ESD (spectral analysis) callback ───────────────────────────────

    @callback(
        Output("esd-chart", "figure"),
        Input("esd-layer-select", "value"),
        prevent_initial_call=True,
    )
    def _update_esd(selected_layer):
        import plotly.graph_objects as go
        snapshots = ckpt_state.get("snapshots", [])
        fig = go.Figure()
        for i, snap in enumerate(snapshots):
            stat = next((s for s in snap["weight_stats"] if s["layer"] == selected_layer), None)
            if stat and "esd" in stat:
                evals = stat["esd"]
                # Log-log histogram of eigenvalue density
                pos_evals = sorted([e for e in evals if e > 0], reverse=True)
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(pos_evals) + 1)),
                    y=pos_evals,
                    mode="lines",
                    name=snap["name"],
                    line={"color": SERIES_COLORS[i % len(SERIES_COLORS)]},
                    hovertemplate="rank=%{x}<br>eigenvalue=%{y:.4g}<extra></extra>",
                ))
                # Mark MP edge if available
                lambda_plus = stat.get("lambda_plus")
                if lambda_plus is not None and i == len(snapshots) - 1:
                    fig.add_hline(
                        y=lambda_plus,
                        line_dash="dash",
                        line_color="#dc3545",
                        annotation_text=f"MP edge ({lambda_plus:.3g})",
                        annotation_position="top right",
                    )
        alpha_str = ""
        if snapshots:
            last_stat = next(
                (s for s in snapshots[-1]["weight_stats"] if s["layer"] == selected_layer), None,
            )
            if last_stat and last_stat.get("alpha") is not None:
                alpha_str = f"  |  alpha={last_stat['alpha']:.2f}"
        fig.update_layout(
            **plotly_layout(title=f"ESD \u2014 {short_layer(selected_layer)}{alpha_str}"),
            xaxis_title="Rank",
            yaxis_title="Eigenvalue",
            xaxis_type="log",
            yaxis_type="log",
            height=380,
        )
        return fig

    @callback(
        Output("ckpt-svd-rank-table-wrap", "children"),
        Output("ckpt-svd-rank-slider-wrap", "style"),
        Output("ckpt-svd-rank-slider-label", "children"),
        Input("ckpt-svd-rank-mode", "value"),
        Input("ckpt-svd-rank-slider", "value"),
        prevent_initial_call=True,
    )
    def _update_svd_rank_table(mode, slider_idx):
        snapshots = ckpt_state.get("snapshots", [])
        if not snapshots:
            return no_update, no_update, no_update

        checkpoint_names, svd_layers, svd_rank_table = compute_effective_rank_table(snapshots)
        if not checkpoint_names or not svd_layers:
            return no_update, no_update, no_update

        idx = int(slider_idx or 0)
        idx = max(0, min(idx, len(checkpoint_names) - 1))

        table = render_checkpoint_change_table(
            checkpoint_names=checkpoint_names,
            all_layers=svd_layers,
            values_table=svd_rank_table,
            mode=mode or "full",
            selected_idx=idx,
            formatter=lambda v: f"{v:.4g}",
        )
        slider_style = {"display": "block" if mode == "single" else "none"}
        return table, slider_style, checkpoint_names[idx]

    # ── Gradients table mode callbacks ─────────────────────────────

    @callback(
        Output("grad-diff-table-wrap", "children"),
        Output("grad-diff-slider-wrap", "style"),
        Output("grad-diff-slider-label", "children"),
        Input("grad-diff-mode", "value"),
        Input("grad-diff-slider", "value"),
        prevent_initial_call=True,
    )
    def _update_grad_diff_table(mode, slider_idx):
        snapshots = ckpt_state.get("snapshots", [])
        if len(snapshots) < 2:
            return no_update, no_update, no_update

        checkpoint_names, all_layers, diff_table, _ = compute_checkpoint_change_tables(snapshots)
        idx = int(slider_idx or 0)
        idx = max(0, min(idx, len(checkpoint_names) - 1))

        table = render_checkpoint_change_table(
            checkpoint_names=checkpoint_names,
            all_layers=all_layers,
            values_table=diff_table,
            mode=mode or "full",
            selected_idx=idx,
            formatter=lambda v: f"{v:.6g}",
        )
        slider_style = {"display": "block" if mode == "single" else "none"}
        return table, slider_style, checkpoint_names[idx]

    @callback(
        Output("grad-rel-table-wrap", "children"),
        Output("grad-rel-slider-wrap", "style"),
        Output("grad-rel-slider-label", "children"),
        Input("grad-rel-mode", "value"),
        Input("grad-rel-slider", "value"),
        prevent_initial_call=True,
    )
    def _update_grad_rel_table(mode, slider_idx):
        snapshots = ckpt_state.get("snapshots", [])
        if len(snapshots) < 2:
            return no_update, no_update, no_update

        checkpoint_names, all_layers, _, rel_table = compute_checkpoint_change_tables(snapshots)
        idx = int(slider_idx or 0)
        idx = max(0, min(idx, len(checkpoint_names) - 1))

        table = render_checkpoint_change_table(
            checkpoint_names=checkpoint_names,
            all_layers=all_layers,
            values_table=rel_table,
            mode=mode or "full",
            selected_idx=idx,
            formatter=lambda v: f"{v:.2f}%",
        )
        slider_style = {"display": "block" if mode == "single" else "none"}
        return table, slider_style, checkpoint_names[idx]

    @callback(
        Output("dist-stats-table-wrap", "children"),
        Output("dist-stats-slider-wrap", "style"),
        Output("dist-stats-slider-label", "children"),
        Input("dist-stats-mode", "value"),
        Input("dist-stats-slider", "value"),
        prevent_initial_call=True,
    )
    def _update_distribution_stats_table(mode, slider_idx):
        snapshots = ckpt_state.get("snapshots", [])
        if not snapshots:
            return no_update, no_update, no_update

        checkpoint_names, stat_layers, stat_table = compute_distribution_stats_table(snapshots, max_layers=50)
        if not checkpoint_names or not stat_layers:
            return no_update, no_update, no_update

        idx = int(slider_idx or 0)
        idx = max(0, min(idx, len(checkpoint_names) - 1))

        table = render_checkpoint_change_table(
            checkpoint_names=checkpoint_names,
            all_layers=stat_layers,
            values_table=stat_table,
            mode=mode or "full",
            selected_idx=idx,
            formatter=lambda v: html.Code(v),
        )
        slider_style = {"display": "block" if mode == "single" else "none"}
        return table, slider_style, checkpoint_names[idx]

    @callback(
        Output("ww-metric-table-wrap", "children"),
        Output("ww-metric-slider-wrap", "style"),
        Output("ww-metric-slider-label", "children"),
        Input("ww-metric-select", "value"),
        Input("ww-metric-mode", "value"),
        Input("ww-metric-slider", "value"),
        prevent_initial_call=True,
    )
    def _update_weightwatcher_metrics_table(metric_key, mode, slider_idx):
        snapshots = ckpt_state.get("snapshots", [])
        if not snapshots:
            return no_update, no_update, no_update

        metric_meta = {
            "alpha": lambda v: f"{v:.3f}",
            "alpha_weighted": lambda v: f"{v:.3f}",
            "mp_softrank": lambda v: f"{v:.3f}",
            "num_spikes": lambda v: f"{v:.0f}",
            "log_spectral_norm": lambda v: f"{v:.3f}",
            "lambda_plus": lambda v: f"{v:.3g}",
        }
        checkpoint_names, layers, tables = compute_scalar_metric_tables(
            snapshots,
            list(metric_meta.keys()),
            max_layers=50,
        )
        if not checkpoint_names or not layers or not tables:
            return no_update, no_update, no_update

        key = metric_key if metric_key in tables else next(iter(tables))
        idx = int(slider_idx or 0)
        idx = max(0, min(idx, len(checkpoint_names) - 1))

        table = render_checkpoint_change_table(
            checkpoint_names=checkpoint_names,
            all_layers=layers,
            values_table=tables[key],
            mode=mode or "full",
            selected_idx=idx,
            formatter=metric_meta.get(key, lambda v: f"{v:.4g}"),
        )
        slider_style = {"display": "block" if mode == "single" else "none"}
        return table, slider_style, checkpoint_names[idx]

    @callback(
        Output("drift-metric-table-wrap", "children"),
        Output("drift-metric-slider-wrap", "style"),
        Output("drift-metric-slider-label", "children"),
        Input("drift-metric-select", "value"),
        Input("drift-metric-mode", "value"),
        Input("drift-metric-slider", "value"),
        prevent_initial_call=True,
    )
    def _update_drift_metrics_table(metric_key, mode, slider_idx):
        snapshots = ckpt_state.get("snapshots", [])
        if not snapshots:
            return no_update, no_update, no_update

        metric_meta = {
            "drift_cosine_prev": lambda v: f"{v:.4f}",
            "drift_subspace_overlap_prev": lambda v: f"{v:.4f}",
            "drift_cka_prev": lambda v: f"{v:.4f}",
        }
        checkpoint_names, layers, tables = compute_scalar_metric_tables(
            snapshots,
            list(metric_meta.keys()),
            max_layers=50,
        )
        if not checkpoint_names or not layers or not tables:
            return no_update, no_update, no_update

        key = metric_key if metric_key in tables else next(iter(tables))
        idx = int(slider_idx or 0)
        idx = max(0, min(idx, len(checkpoint_names) - 1))

        table = render_checkpoint_change_table(
            checkpoint_names=checkpoint_names,
            all_layers=layers,
            values_table=tables[key],
            mode=mode or "full",
            selected_idx=idx,
            formatter=metric_meta.get(key, lambda v: f"{v:.4g}"),
        )
        slider_style = {"display": "block" if mode == "single" else "none"}
        return table, slider_style, checkpoint_names[idx]

    # ── Spectral gap table mode callback ─────────────────────────

    @callback(
        Output("spectral-gap-table-wrap", "children"),
        Output("spectral-gap-slider-wrap", "style"),
        Output("spectral-gap-slider-label", "children"),
        Input("spectral-gap-mode", "value"),
        Input("spectral-gap-slider", "value"),
        prevent_initial_call=True,
    )
    def _update_spectral_gap_table(mode, slider_idx):
        snapshots = ckpt_state.get("snapshots", [])
        if not snapshots:
            return no_update, no_update, no_update

        checkpoint_names, layers, gap_table = compute_spectral_gap_table(snapshots, max_layers=50)
        if not checkpoint_names or not layers:
            return no_update, no_update, no_update

        idx = int(slider_idx or 0)
        idx = max(0, min(idx, len(checkpoint_names) - 1))

        table = render_checkpoint_change_table(
            checkpoint_names=checkpoint_names,
            all_layers=layers,
            values_table=gap_table,
            mode=mode or "full",
            selected_idx=idx,
            formatter=lambda v: f"{v:.2f}",
        )
        slider_style = {"display": "block" if mode == "single" else "none"}
        return table, slider_style, checkpoint_names[idx]

    # ── Spectral gap detail chart callback ─────────────────────────

    @callback(
        Output("spectral-gap-detail-chart", "figure"),
        Input("spectral-gap-layer-select", "value"),
        prevent_initial_call=True,
    )
    def _update_spectral_gap_detail(selected_layer):
        import plotly.graph_objects as go
        snapshots = ckpt_state.get("snapshots", [])
        if not snapshots or not selected_layer:
            return go.Figure()

        gap_labels = ["s1/s2", "s2/s3", "s3/s4", "s4/s5", "s5/s6"]
        fig = go.Figure()
        for i, snap in enumerate(snapshots):
            stat = next((s for s in snap["weight_stats"] if s["layer"] == selected_layer), None)
            if stat and "singular_values" in stat:
                ratios = compute_spectral_gap_ratios(stat["singular_values"], top_k=5)
                fig.add_trace(go.Bar(
                    x=gap_labels[:len(ratios)],
                    y=ratios,
                    name=snap["name"],
                    marker_color=SERIES_COLORS[i % len(SERIES_COLORS)],
                    opacity=0.75,
                ))
        fig.update_layout(
            **plotly_layout(title=f"Spectral Gaps \u2014 {short_layer(selected_layer)}"),
            barmode="group", height=340,
            xaxis_title="Gap", yaxis_title="Ratio", yaxis_type="log",
        )
        return fig

    # ── Network state: health node click ────────────────────────────

    @callback(
        Output("ns-layer-detail", "children"),
        Input("gh-cyto-health", "tapNodeData"),
        prevent_initial_call=True,
    )
    def _on_health_node_click(data):
        import plotly.graph_objects as go

        if data is None or not ckpt_state.get("snapshots"):
            return no_update

        mod_path = data.get("id", "")
        snapshots_list = ckpt_state["snapshots"]

        # Find all params belonging to this module
        matching: list[tuple[str, dict]] = []
        for snap in snapshots_list:
            for stat in snap["weight_stats"]:
                param = stat["layer"]
                parent = param.rsplit(".", 1)[0] if "." in param else ""
                if parent == mod_path:
                    matching.append((snap["name"], stat))

        if not matching:
            return dbc.Alert(
                f"No parameter data for module \"{mod_path}\".",
                color="secondary", className="mt-3",
            )

        cards: list = []
        seen_params: set[str] = set()
        for snap_name, stat in matching:
            param = stat["layer"]
            if param in seen_params:
                continue
            seen_params.add(param)

            state, reason = weight_health(stat)
            items = [
                f"**{short_layer(param)}** ({snap_name}) \u2014 "
                f"Health: **{state.title()}** \u2014 {reason}",
                f"L2={stat.get('norm_l2', 0):.4g}  "
                f"\u03bc={stat.get('mean', 0):.4g}  "
                f"\u03c3={stat.get('std', 0):.4g}  "
                f"shape={'x'.join(str(s) for s in stat.get('shape', []))}",
            ]

            children: list = [dbc.CardBody([
                html.P(items[0], style={"marginBottom": "4px"}),
                html.Code(items[1]),
            ])]

            # Mini histogram
            if "hist_counts" in stat and "hist_centers" in stat:
                fig = go.Figure(go.Bar(
                    x=stat["hist_centers"], y=stat["hist_counts"],
                    marker_color=HEALTH_COLORS[state], opacity=0.8,
                ))
                fig.update_layout(
                    **plotly_layout(), height=200,
                    margin={"l": 40, "r": 10, "t": 10, "b": 30},
                    xaxis_title="Value", yaxis_title="Count",
                )
                children.append(dcc.Graph(figure=fig, style={"height": "200px"}))

            # Mini SVD
            if "singular_values" in stat:
                svs = stat["singular_values"]
                fig = go.Figure(go.Scatter(
                    x=list(range(len(svs))), y=svs, mode="lines",
                    line={"color": HEALTH_COLORS[state]},
                ))
                fig.update_layout(
                    **plotly_layout(), height=200,
                    margin={"l": 40, "r": 10, "t": 10, "b": 30},
                    xaxis_title="Index", yaxis_title="SV", yaxis_type="log",
                )
                children.append(dcc.Graph(figure=fig, style={"height": "200px"}))

            cards.append(dbc.Card(children, className="mb-2",
                                  style={"borderLeft": f"4px solid {HEALTH_COLORS[state]}"}))

        return html.Div([
            html.H5(f"Details: {mod_path}", className="mt-3 mb-2"),
            *cards,
        ])

    @callback(
        Output("ww-metric-heatmap", "figure"),
        Output("ww-metric-trend", "figure"),
        Input("ww-global-metric-select", "value", allow_optional=True),
        Input("ww-checkpoint-select", "value", allow_optional=True),
    )
    def _update_ww_metric_views(metric_key, selected_checkpoint):
        import plotly.graph_objects as go

        snapshots = ckpt_state.get("snapshots", [])
        empty = go.Figure()
        if not snapshots:
            return empty, empty

        metric_labels = {
            "alpha": "Alpha",
            "alpha_weighted": "Alpha Weighted",
            "mp_softrank": "MP Softrank",
            "num_spikes": "Spikes",
            "log_spectral_norm": "log10(lambda_max)",
            "lambda_plus": "MP edge (lambda+)",
        }
        metric_keys = list(metric_labels.keys())
        checkpoint_names, layers, tables = compute_scalar_metric_tables(
            snapshots,
            metric_keys,
            max_layers=200,
        )
        if not checkpoint_names or not layers or not tables:
            return empty, empty

        key = metric_key if metric_key in tables else next(iter(tables))
        table = tables[key]
        label = metric_labels.get(key, key)
        selected_name = selected_checkpoint if selected_checkpoint in checkpoint_names else checkpoint_names[-1]
        selected_idx = checkpoint_names.index(selected_name)

        heatmap = go.Figure(go.Heatmap(
            z=table,
            x=checkpoint_names,
            y=[short_layer(layer) for layer in layers],
            colorscale="Viridis",
            colorbar={"title": label},
            hovertemplate="layer=%{y}<br>checkpoint=%{x}<br>value=%{z:.4g}<extra></extra>",
        ))
        heatmap.update_layout(
            **plotly_layout(title=f"{label} Heatmap"),
            height=520,
            yaxis={"autorange": "reversed"},
        )
        heatmap.add_vline(
            x=selected_name,
            line_width=2,
            line_dash="dash",
            line_color="#dc3545",
        )

        means: list[float | None] = []
        mins: list[float | None] = []
        maxs: list[float | None] = []
        for ci in range(len(checkpoint_names)):
            col_vals = [row[ci] for row in table if row[ci] is not None]
            if col_vals:
                means.append(sum(col_vals) / len(col_vals))
                mins.append(min(col_vals))
                maxs.append(max(col_vals))
            else:
                means.append(None)
                mins.append(None)
                maxs.append(None)

        trend = go.Figure()
        trend.add_trace(go.Scatter(
            x=checkpoint_names,
            y=means,
            mode="lines+markers",
            name="mean",
            line={"color": "#375a7f", "width": 3},
        ))
        trend.add_trace(go.Scatter(
            x=checkpoint_names,
            y=maxs,
            mode="lines",
            name="max",
            line={"color": "#e67e22", "dash": "dot"},
        ))
        trend.add_trace(go.Scatter(
            x=checkpoint_names,
            y=mins,
            mode="lines",
            name="min",
            line={"color": "#00bc8c", "dash": "dot"},
        ))
        selected_y = means[selected_idx] if selected_idx < len(means) else None
        if selected_y is not None:
            trend.add_trace(go.Scatter(
                x=[selected_name],
                y=[selected_y],
                mode="markers",
                name="selected",
                marker={"size": 12, "color": "#dc3545", "symbol": "diamond"},
            ))
        trend.update_layout(
            **plotly_layout(title=f"{label} Summary Across Checkpoints"),
            height=340,
            xaxis_title="Checkpoint",
            yaxis_title=label,
        )

        return heatmap, trend

    @callback(
        Output("ww-layer-multi-metric", "figure"),
        Output("ww-layer-esd", "figure"),
        Input("ww-layer-select", "value", allow_optional=True),
        Input("ww-checkpoint-select", "value", allow_optional=True),
    )
    def _update_ww_layer_views(selected_layer, selected_checkpoint):
        import plotly.graph_objects as go

        snapshots = ckpt_state.get("snapshots", [])
        multi = go.Figure()
        esd_fig = go.Figure()
        if not snapshots or not selected_layer:
            return multi, esd_fig

        checkpoint_names = [snap.get("name", "?") for snap in snapshots]
        selected_name = selected_checkpoint if selected_checkpoint in checkpoint_names else checkpoint_names[-1]
        metric_meta = {
            "alpha": {"name": "Alpha", "color": "#375a7f"},
            "alpha_weighted": {"name": "Alpha Weighted", "color": "#00bc8c"},
            "mp_softrank": {"name": "MP Softrank", "color": "#e67e22"},
            "num_spikes": {"name": "Spikes", "color": "#e74c3c"},
            "log_spectral_norm": {"name": "log10(lambda_max)", "color": "#9b59b6"},
            "lambda_plus": {"name": "MP edge", "color": "#1abc9c"},
        }

        for key, meta in metric_meta.items():
            vals: list[float | None] = []
            for snap in snapshots:
                stat = next((s for s in snap.get("weight_stats", []) if s.get("layer") == selected_layer), None)
                val = stat.get(key) if stat else None
                vals.append(float(val) if isinstance(val, (int, float)) else None)
            if any(v is not None for v in vals):
                multi.add_trace(go.Scatter(
                    x=checkpoint_names,
                    y=vals,
                    mode="lines+markers",
                    name=meta["name"],
                    line={"color": meta["color"]},
                ))

        multi.add_vline(
            x=selected_name,
            line_width=2,
            line_dash="dash",
            line_color="#dc3545",
        )

        multi.update_layout(
            **plotly_layout(title=f"Layer Metrics - {short_layer(selected_layer)}"),
            height=360,
            xaxis_title="Checkpoint",
            yaxis_title="Metric value",
        )

        for i, snap in enumerate(snapshots):
            stat = next((s for s in snap.get("weight_stats", []) if s.get("layer") == selected_layer), None)
            if not stat or not stat.get("esd"):
                continue
            evals = [float(v) for v in stat.get("esd", []) if isinstance(v, (int, float)) and v > 0]
            if not evals:
                continue
            evals = sorted(evals, reverse=True)
            esd_fig.add_trace(go.Scatter(
                x=list(range(1, len(evals) + 1)),
                y=evals,
                mode="lines",
                name=snap.get("name", f"ckpt_{i}"),
                line={
                    "color": SERIES_COLORS[i % len(SERIES_COLORS)],
                    "width": 3 if snap.get("name") == selected_name else 1.5,
                },
                opacity=1.0 if snap.get("name") == selected_name else 0.45,
                hovertemplate="rank=%{x}<br>eigenvalue=%{y:.4g}<extra></extra>",
            ))

            lambda_plus = stat.get("lambda_plus")
            if isinstance(lambda_plus, (int, float)) and i == len(snapshots) - 1:
                esd_fig.add_hline(
                    y=float(lambda_plus),
                    line_dash="dash",
                    line_color="#dc3545",
                    annotation_text=f"MP edge ({lambda_plus:.3g})",
                    annotation_position="top right",
                )

        esd_fig.update_layout(
            **plotly_layout(title=f"ESD Across Checkpoints - {short_layer(selected_layer)}"),
            height=420,
            xaxis_title="Rank",
            yaxis_title="Eigenvalue",
            xaxis_type="log",
            yaxis_type="log",
        )

        return multi, esd_fig

    @callback(
        Output("ww-leaderboard-wrap", "children"),
        Input("ww-checkpoint-select", "value", allow_optional=True),
        Input("ww-global-metric-select", "value", allow_optional=True),
    )
    def _update_ww_leaderboard(selected_checkpoint, metric_key):
        snapshots = ckpt_state.get("snapshots", [])
        if not snapshots:
            return dbc.Alert("No snapshots available.", color="secondary", className="mb-0")

        metric_labels = {
            "alpha": "Alpha",
            "alpha_weighted": "Alpha Weighted",
            "mp_softrank": "MP Softrank",
            "num_spikes": "Spikes",
            "log_spectral_norm": "log10(lambda_max)",
            "lambda_plus": "MP edge (lambda+)",
        }

        snap = next((s for s in snapshots if s.get("name") == selected_checkpoint), snapshots[-1])
        key = metric_key if metric_key in metric_labels else "alpha_weighted"

        rows: list[tuple[str, float]] = []
        for stat in snap.get("weight_stats", []):
            layer = stat.get("layer")
            val = stat.get(key)
            if isinstance(layer, str) and isinstance(val, (int, float)):
                rows.append((layer, float(val)))

        if not rows:
            return dbc.Alert(
                f"No values for {metric_labels.get(key, key)} in checkpoint {snap.get('name', '?')}.",
                color="secondary",
                className="mb-0",
            )

        reverse = key not in {"mp_softrank"}
        rows.sort(key=lambda x: x[1], reverse=reverse)

        body = [
            html.Tr([
                html.Td(str(i + 1)),
                html.Td(html.Code(short_layer(layer))),
                html.Td(f"{value:.6g}"),
            ])
            for i, (layer, value) in enumerate(rows[:80])
        ]

        return html.Div(
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Rank"),
                    html.Th("Layer"),
                    html.Th(metric_labels.get(key, key)),
                ])),
                html.Tbody(body),
            ], bordered=True, hover=True, responsive=True, size="sm"),
            style={"maxHeight": "520px", "overflowY": "auto"},
        )

    # ══════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════
    # ── On-Demand Analysis page callbacks ────────────────────────────
    # ══════════════════════════════════════════════════════════════════

    @callback(
        Output("od-heatmap-layer", "options"),
        Input("od-model-select", "value"),
        prevent_initial_call=True,
    )
    def _update_od_layer_options(model_name):
        if ipc is None:
            return []
        models = ipc.read_models()
        model_info = models.get(model_name, {})
        params = model_info.get("parameters", {})
        options = []
        for name, meta in params.items():
            shape = meta.get("shape", [])
            if len(shape) == 2:
                shape_str = "x".join(str(s) for s in shape)
                options.append({"label": f"{short_layer(name)} ({shape_str})", "value": name})
        return options

    # On-demand background task state
    od_tasks: dict = {
        "heatmap_req_id": None,
        "heatmap_running": False,
        "heatmap_result": None,
        "cka_req_id": None,
        "cka_running": False,
        "cka_result": None,
        "state_req_id": None,
        "state_running": False,
        "state_result": None,
    }

    def _od_poll_worker(req_id: str, task_key: str, ipc_channel, timeout: int = 20):
        """Background thread that polls IPC for a response.

        Receives ``ipc_channel`` as an explicit argument rather than
        capturing it from the enclosing scope, so stale-reference bugs
        are avoided if the surrounding state changes after the thread starts.
        """
        import time as _time
        if ipc_channel is None:
            od_tasks[f"{task_key}_result"] = {"error": "No IPC channel configured."}
            od_tasks[f"{task_key}_running"] = False
            return
        for _ in range(timeout * 2):
            try:
                resp = ipc_channel.read_response(req_id)
            except Exception as exc:
                od_tasks[f"{task_key}_result"] = {"error": f"IPC read failed: {exc}"}
                od_tasks[f"{task_key}_running"] = False
                return
            if resp is not None:
                try:
                    ipc_channel.clear_response(req_id)
                except Exception:
                    pass
                od_tasks[f"{task_key}_result"] = resp
                od_tasks[f"{task_key}_running"] = False
                return
            _time.sleep(0.5)
        od_tasks[f"{task_key}_result"] = {"error": "Timeout: training process did not respond. Ensure gradienthound.step() is being called."}
        od_tasks[f"{task_key}_running"] = False

    # ── Heatmap: submit ──────────────────────────────────────────
    @callback(
        Output("od-heatmap-status", "children", allow_duplicate=True),
        Output("od-heatmap-btn", "disabled"),
        Output("od-poll", "disabled", allow_duplicate=True),
        Input("od-heatmap-btn", "n_clicks"),
        State("od-model-select", "value"),
        State("od-heatmap-layer", "value"),
        prevent_initial_call=True,
    )
    def _submit_heatmap(n_clicks, model_name, layer):
        import uuid
        if not n_clicks or ipc is None:
            return no_update, no_update, no_update
        if not layer:
            return dbc.Alert("Select a layer first.", color="warning"), False, no_update

        req_id = f"heatmap_{uuid.uuid4().hex[:8]}"
        ipc.write_request({"type": "weight_heatmap", "id": req_id, "model": model_name, "layer": layer})
        od_tasks["heatmap_req_id"] = req_id
        od_tasks["heatmap_running"] = True
        od_tasks["heatmap_result"] = None
        threading.Thread(target=_od_poll_worker, args=(req_id, "heatmap", ipc, 20), daemon=True).start()
        return (
            [dbc.Spinner(size="sm", spinner_class_name="me-2"), "Computing\u2026"],
            True,
            False,  # enable polling
        )

    # ── CKA: submit ──────────────────────────────────────────────
    @callback(
        Output("od-cka-status", "children", allow_duplicate=True),
        Output("od-cka-btn", "disabled"),
        Output("od-poll", "disabled", allow_duplicate=True),
        Input("od-cka-btn", "n_clicks"),
        State("od-model-select", "value"),
        prevent_initial_call=True,
    )
    def _submit_cka(n_clicks, model_name):
        import uuid
        if not n_clicks or ipc is None:
            return no_update, no_update, no_update

        req_id = f"cka_{uuid.uuid4().hex[:8]}"
        ipc.write_request({"type": "cka", "id": req_id, "model": model_name})
        od_tasks["cka_req_id"] = req_id
        od_tasks["cka_running"] = True
        od_tasks["cka_result"] = None
        threading.Thread(target=_od_poll_worker, args=(req_id, "cka", ipc, 30), daemon=True).start()
        return (
            [dbc.Spinner(size="sm", spinner_class_name="me-2"), "Computing\u2026"],
            True,
            False,
        )

    # ── Network state: submit ────────────────────────────────────
    @callback(
        Output("od-state-status", "children", allow_duplicate=True),
        Output("od-state-btn", "disabled"),
        Output("od-poll", "disabled", allow_duplicate=True),
        Input("od-state-btn", "n_clicks"),
        State("od-model-select", "value"),
        prevent_initial_call=True,
    )
    def _submit_network_state(n_clicks, model_name):
        import uuid
        if not n_clicks or ipc is None:
            return no_update, no_update, no_update

        req_id = f"state_{uuid.uuid4().hex[:8]}"
        ipc.write_request({"type": "network_state", "id": req_id, "model": model_name})
        od_tasks["state_req_id"] = req_id
        od_tasks["state_running"] = True
        od_tasks["state_result"] = None
        threading.Thread(target=_od_poll_worker, args=(req_id, "state", ipc, 20), daemon=True).start()
        return (
            [dbc.Spinner(size="sm", spinner_class_name="me-2"), "Computing\u2026"],
            True,
            False,
        )

    # ── On-demand polling callback ───────────────────────────────
    @callback(
        Output("od-heatmap-chart", "figure"),
        Output("od-heatmap-chart", "style"),
        Output("od-heatmap-status", "children"),
        Output("od-heatmap-btn", "disabled", allow_duplicate=True),
        Output("od-cka-chart", "figure"),
        Output("od-cka-chart", "style"),
        Output("od-cka-status", "children"),
        Output("od-cka-btn", "disabled", allow_duplicate=True),
        Output("od-state-result", "children"),
        Output("od-state-status", "children"),
        Output("od-state-btn", "disabled", allow_duplicate=True),
        Output("od-poll", "disabled"),
        Input("od-poll", "n_intervals"),
        prevent_initial_call=True,
    )
    def _poll_on_demand(_n):
        import plotly.graph_objects as go

        hm_fig, hm_style, hm_status, hm_btn = no_update, no_update, no_update, no_update
        cka_fig, cka_style, cka_status, cka_btn = no_update, no_update, no_update, no_update
        st_result, st_status, st_btn = no_update, no_update, no_update
        any_running = od_tasks["heatmap_running"] or od_tasks["cka_running"] or od_tasks["state_running"]

        # ── Heatmap result ──
        if not od_tasks["heatmap_running"] and od_tasks["heatmap_result"] is not None:
            resp = od_tasks["heatmap_result"]
            od_tasks["heatmap_result"] = None
            hm_btn = False
            if "error" in resp:
                hm_fig = go.Figure()
                hm_style = {"display": "none"}
                hm_status = dbc.Alert(resp["error"], color="danger")
            else:
                matrix = resp.get("matrix", [])
                layer = resp.get("layer", "?")
                fig = go.Figure(go.Heatmap(
                    z=matrix, colorscale="RdBu_r", zmid=0,
                    hovertemplate="row=%{y}<br>col=%{x}<br>value=%{z:.4g}<extra></extra>",
                ))
                shape = resp.get("shape", [])
                disp = resp.get("display_shape", shape)
                sparsity = resp.get("sparsity", 0)
                fig.update_layout(
                    **plotly_layout(
                        title=f"{short_layer(layer)} "
                              f"({'x'.join(str(s) for s in shape)}, "
                              f"displayed {'x'.join(str(s) for s in disp)}) "
                              f"| Sparsity: {sparsity:.1f}%"
                    ),
                    height=500, yaxis={"autorange": "reversed"},
                )
                hm_fig = fig
                hm_style = {"display": "block"}
                hm_status = dbc.Badge("Done", color="success")

        # ── CKA result ──
        if not od_tasks["cka_running"] and od_tasks["cka_result"] is not None:
            resp = od_tasks["cka_result"]
            od_tasks["cka_result"] = None
            cka_btn = False
            if "error" in resp:
                cka_fig = go.Figure()
                cka_style = {"display": "none"}
                cka_status = dbc.Alert(resp["error"], color="danger")
            else:
                matrix = resp.get("matrix", [])
                names = resp.get("short_names", resp.get("layers", []))
                fig = go.Figure(go.Heatmap(
                    z=matrix, x=names, y=names, colorscale="Viridis",
                    zmin=0, zmax=1,
                    hovertemplate="%{y} vs %{x}<br>CKA=%{z:.3f}<extra></extra>",
                ))
                fig.update_layout(
                    **plotly_layout(title=f"CKA Similarity ({resp.get('n', 0)} layers)"),
                    height=550, yaxis={"autorange": "reversed"},
                )
                cka_fig = fig
                cka_style = {"display": "block"}
                cka_status = dbc.Badge("Done", color="success")

        # ── Network state result ──
        if not od_tasks["state_running"] and od_tasks["state_result"] is not None:
            resp = od_tasks["state_result"]
            od_tasks["state_result"] = None
            st_btn = False
            if "error" in resp:
                st_result = dbc.Alert(resp["error"], color="danger")
                st_status = ""
            else:
                layers = resp.get("layers", [])
                total = resp.get("total_params", 0)
                header = [html.Th("Parameter"), html.Th("Shape"), html.Th("Elements"),
                          html.Th("Dtype"), html.Th("Sample Values")]
                rows = []
                for layer in layers:
                    vals = layer.get("values", [])
                    sample = ""
                    if vals and isinstance(vals[0], list):
                        flat = [v for row in vals[:3] for v in row[:5]]
                        sample = ", ".join(f"{v:.4g}" for v in flat[:10])
                        if len(flat) > 10:
                            sample += ", ..."
                    elif vals:
                        sample = ", ".join(f"{v:.4g}" for v in vals[0][:10])
                    rows.append(html.Tr([
                        html.Td(html.Code(short_layer(layer["name"]))),
                        html.Td("x".join(str(s) for s in layer.get("shape", []))),
                        html.Td(f"{layer.get('numel', 0):,}"),
                        html.Td(layer.get("dtype", "")),
                        html.Td(html.Code(sample), style={"fontSize": "0.8em"}),
                    ]))
                st_result = html.Div([
                    html.P(f"Total parameters: {total:,}", className="fw-semibold"),
                    dbc.Table([html.Thead(html.Tr(header)), html.Tbody(rows)],
                              bordered=True, hover=True, responsive=True, size="sm"),
                ], style={"maxHeight": "600px", "overflowY": "auto"})
                st_status = dbc.Badge("Done", color="success")

        # Disable polling if nothing is running and all results consumed
        still_running = od_tasks["heatmap_running"] or od_tasks["cka_running"] or od_tasks["state_running"]
        has_pending = od_tasks["heatmap_result"] is not None or od_tasks["cka_result"] is not None or od_tasks["state_result"] is not None
        disable_poll = not still_running and not has_pending

        return (
            hm_fig, hm_style, hm_status, hm_btn,
            cka_fig, cka_style, cka_status, cka_btn,
            st_result, st_status, st_btn,
            disable_poll,
        )

    # ══════════════════════════════════════════════════════════════════
    # ── Raw Data page callbacks ──────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════

    @callback(
        Output("raw-data-table", "children"),
        Output("raw-data-count", "children"),
        Input("raw-data-refresh-btn", "n_clicks"),
        Input("raw-data-type", "value"),
        Input("raw-data-limit", "value"),
        prevent_initial_call=True,
    )
    def _update_raw_data(_n, data_type, limit):
        if ipc is None:
            return "", "No IPC channel."

        reader_map = {
            "gradient_stats": ipc.read_gradient_stats,
            "weight_stats": ipc.read_weight_stats,
            "activation_stats": ipc.read_activation_stats,
            "optimizer_state": ipc.read_optimizer_state,
            "metrics": ipc.read_metrics,
            "predictions": ipc.read_predictions,
            "attention": ipc.read_attention,
        }
        reader = reader_map.get(data_type)
        if reader is None:
            return "", f"Unknown data type: {data_type}"

        limit = int(limit or 50)
        total = ipc._count_events(data_type)
        entries = reader(last_n=limit) if limit > 0 else reader()
        if not entries:
            return dbc.Alert("No records found.", color="secondary"), f"0 records"

        # Auto-detect columns from first few entries
        all_keys: list[str] = []
        seen_keys: set[str] = set()
        for e in entries[:20]:
            for k in e:
                if k not in seen_keys:
                    all_keys.append(k)
                    seen_keys.add(k)

        # Skip large nested values in display
        skip_keys = {"hist_counts", "hist_centers", "singular_values", "esd",
                     "weights", "values", "predicted", "actual"}
        display_keys = [k for k in all_keys if k not in skip_keys][:15]

        header = [html.Th(k) for k in display_keys]
        rows = []
        for e in entries:
            cells = []
            for k in display_keys:
                v = e.get(k)
                if v is None:
                    cells.append(html.Td("-", className="text-muted"))
                elif isinstance(v, float):
                    cells.append(html.Td(f"{v:.6g}"))
                elif isinstance(v, list):
                    cells.append(html.Td(f"[{len(v)} items]", className="text-muted"))
                else:
                    cells.append(html.Td(str(v)))
            rows.append(html.Tr(cells))

        count_text = f"{total} total records"
        if limit > 0 and total > limit:
            count_text += f" (showing last {limit})"

        table = html.Div(
            dbc.Table([
                html.Thead(html.Tr(header)),
                html.Tbody(rows),
            ], bordered=True, hover=True, responsive=True, size="sm", striped=True),
            style={"maxHeight": "700px", "overflowY": "auto"},
        )
        return table, count_text

    # ── Tools page refresh callback ──────────────────────────────

    @callback(
        Output("tools-card-container", "children"),
        Input("tools-refresh-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _refresh_tools(_n_clicks):
        from ._page_tools import _tool_card, _CATEGORY_BADGE
        tools = tool_registry.all_status()

        category_order = ["capture", "on-demand", "analysis", "integration"]
        categories: dict[str, list] = {}
        for t in tools:
            categories.setdefault(t["category"], []).append(t)

        ordered_cats = [c for c in category_order if c in categories]
        ordered_cats += [c for c in categories if c not in ordered_cats]

        sections = []
        for cat in ordered_cats:
            cat_label = _CATEGORY_BADGE.get(cat, (cat.title(), "secondary"))[0]
            sections.append(html.H4(cat_label, className="mt-4 mb-3"))
            sections.append(dbc.Row([_tool_card(t) for t in categories[cat]]))
        return sections

    return app
