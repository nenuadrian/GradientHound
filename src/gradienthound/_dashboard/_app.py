"""Dash app factory and callbacks."""
from __future__ import annotations

import json
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, callback, Output, Input, State, no_update
import dash_cytoscape as cyto

cyto.load_extra_layouts()

from ._constants import PAGES, SERIES_COLORS, HEALTH_COLORS
from ._helpers import (
    plotly_layout, short_layer, placeholder_page, node_detail_panel,
    compute_checkpoint_change_tables, render_checkpoint_change_table,
    compute_effective_rank_table, compute_distribution_stats_table,
)
from ._health import weight_health
from ._wandb import parse_wandb_project_run_id, fetch_wandb_run_metrics, metrics_page_wandb
from ._pages import (
    dashboard_page, landing_page_empty,
    checkpoints_page, checkpoints_page_empty,
)


def create_app(
    data_dir: str | None = None,
    model_path: str | None = None,
    checkpoint_paths: list[str] | None = None,
    loader_path: str | None = None,
    wandb_entity: str | None = None,
    wandb_project_run_id: str | None = None,
) -> Dash:
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        title="GradientHound",
        external_stylesheets=[dbc.themes.FLATLY],
    )

    # Load model export
    model_data: dict | None = None
    if model_path:
        p = Path(model_path)
        if p.is_dir():
            gh_files = sorted(p.glob("*.gh.json"))
            if gh_files:
                with open(gh_files[0]) as f:
                    model_data = json.load(f)
        elif p.exists():
            with open(p) as f:
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
    }
    has_checkpoints = bool(ckpt_state["paths"])

    wandb_state: dict = {
        "entity": (wandb_entity or "").strip(),
        "project_run_id": (wandb_project_run_id or "").strip(),
        "data": None,
    }

    # Navbar
    nav_links = [
        dbc.NavLink(title, href=path, id=f"nav-{path}", active="exact")
        for path, (title, _) in PAGES.items()
    ]
    navbar = dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("GradientHound", href="/", className="fw-bold"),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                dbc.Nav(nav_links, className="ms-auto", navbar=True),
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ], fluid=True),
        color="light",
        dark=False,
        sticky="top",
        className="mb-3 shadow-sm border-bottom",
    )

    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="ckpt-store", data=None),
        dcc.Store(id="wandb-store", data=None),
        navbar,
        dbc.Container(html.Div(id="gh-content"), fluid=True, className="px-4"),
    ])

    # ── Navbar toggle (mobile) ──────────────────────────────────────

    @callback(
        Output("navbar-collapse", "is_open"),
        Input("navbar-toggler", "n_clicks"),
        State("navbar-collapse", "is_open"),
    )
    def _toggle_navbar(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open

    # ── Process checkpoints ──────────────────────────────────────────

    if has_checkpoints:
        @callback(
            Output("ckpt-store", "data"),
            Output("ckpt-status", "children"),
            Output("ckpt-process-btn", "disabled"),
            Input("ckpt-process-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def _process_ckpts(n_clicks):
            if not n_clicks:
                return no_update, no_update, no_update
            from gradienthound.checkpoint import process_checkpoints
            try:
                snapshots = process_checkpoints(
                    ckpt_state["paths"], loader_path=ckpt_state["loader_path"],
                )
                ckpt_state["snapshots"] = snapshots
                ckpt_state["processed"] = True
                n_params = len({s["layer"] for snap in snapshots for s in snap["weight_stats"]})
                return {"ready": True}, f"Done \u2014 {len(snapshots)} checkpoints, {n_params} params", True
            except Exception as exc:
                return no_update, f"Error: {exc}", False

    # ── Fetch W&B run metrics ───────────────────────────────────────

    @callback(
        Output("wandb-store", "data"),
        Output("wandb-fetch-status", "children"),
        Input("wandb-fetch-btn", "n_clicks"),
        State("wandb-entity-input", "value"),
        State("wandb-project-run-id-input", "value"),
        prevent_initial_call=True,
    )
    def _fetch_wandb_metrics(n_clicks, entity_value, project_run_id_value):
        if not n_clicks:
            return no_update, no_update

        entity = (entity_value or "").strip()
        project_run_id_str = (project_run_id_value or "").strip()

        if not entity or not project_run_id_str:
            return no_update, dbc.Alert(
                "Both Entity and Project/Run ID are required.",
                color="warning",
                className="mb-0",
            )

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

            return data, dbc.Alert(
                f"Fetched {len(metric_keys)} metrics across {len(entries)} points.",
                color="success",
                className="mb-0",
            )
        except Exception as exc:
            return no_update, dbc.Alert(
                f"Failed to fetch W&B metrics: {exc}",
                color="danger",
                className="mb-0",
            )

    # ── Routing ──────────────────────────────────────────────────────

    @callback(
        Output("gh-content", "children"),
        Input("url", "pathname"),
        Input("ckpt-store", "data"),
        Input("wandb-store", "data"),
    )
    def _route(pathname, ckpt_data, wandb_data):
        snapshots = ckpt_state["snapshots"] if ckpt_state["processed"] else None

        if pathname is None or pathname == "/":
            if model_data:
                return dashboard_page(model_data, snapshots=snapshots)
            return landing_page_empty()

        if pathname == "/metrics":
            page_data = wandb_data if wandb_data is not None else wandb_state.get("data")
            return metrics_page_wandb(
                page_data,
                default_entity=wandb_state.get("entity"),
                default_project_run_id=wandb_state.get("project_run_id"),
            )

        if pathname == "/checkpoints":
            if has_checkpoints:
                return checkpoints_page(ckpt_state["paths"], snapshots)
            return checkpoints_page_empty()

        if pathname in PAGES:
            title, desc = PAGES[pathname]
            return placeholder_page(title, desc)

        return placeholder_page("Not Found", f"No page at {pathname}")

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

    return app
