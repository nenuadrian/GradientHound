"""Weight Health page: L2 norms, anomalies, parameter health table, optimizer."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html, dcc

from ._constants import HEALTH_COLORS, HEALTH_SORT
from ._helpers import (
    plotly_layout, short_layer, column_summary, summary_chart,
)
from ._health import weight_health
from ._page_checkpoints import _optimizer_state_cards


def weight_health_page(model_data: dict | None, snapshots: list | None = None):
    import plotly.graph_objects as go

    children = [
        html.H2("Weight Health", className="mt-3 mb-1"),
        html.P("L2 norms, anomalies, and per-parameter health assessment.", className="text-muted mb-4"),
    ]

    if not snapshots:
        children.append(dbc.Alert(
            "Load checkpoints to see weight health analysis.",
            color="info",
        ))
        return dbc.Container(children, fluid=True)

    snap = snapshots[-1]
    stats = snap["weight_stats"]
    snap_name = snap["name"]

    # ── L2 Norm per Layer (health-colored) ───────────────────────────
    weight_layers = [s for s in stats if not s["layer"].endswith(".bias")]
    norm_fig = go.Figure()
    colors = [HEALTH_COLORS[weight_health(s)[0]] for s in weight_layers]
    norm_fig.add_trace(go.Bar(
        x=[short_layer(s["layer"]) for s in weight_layers],
        y=[s.get("norm_l2", 0) for s in weight_layers],
        marker_color=colors,
    ))
    norm_fig.update_layout(
        **plotly_layout(title=f"L2 Norm per Layer \u2014 {snap_name}"),
        xaxis_tickangle=-45, height=380,
    )
    children.append(dbc.Card(dbc.CardBody([
        html.H5("L2 Norm per Layer", className="card-title"),
        html.P("Weight norms colored by health state.", className="text-muted"),
        dcc.Graph(figure=norm_fig),
    ]), className="mb-3"))

    # ── Norm Evolution Across Checkpoints ────────────────────────────
    if len(snapshots) >= 2:
        ckpt_names_norm = [s["name"] for s in snapshots]
        weight_layer_names = [s["layer"] for s in weight_layers]
        norm_table: list[list[float | None]] = []
        for layer in weight_layer_names:
            row: list[float | None] = []
            for sn in snapshots:
                stat = next((s for s in sn["weight_stats"] if s["layer"] == layer), None)
                row.append(stat.get("norm_l2") if stat else None)
            norm_table.append(row)

        n_means, n_mins, n_maxs = column_summary(norm_table, len(ckpt_names_norm))
        norm_evo_fig = summary_chart(
            ckpt_names_norm, n_means, n_mins, n_maxs,
            "L2 Norm Evolution", "L2 Norm",
        )
        children.append(dbc.Card(dbc.CardBody([
            html.H5("Norm Evolution Across Checkpoints", className="card-title"),
            html.P("Mean, min, and max L2 norm across all weight layers per checkpoint.",
                    className="text-muted"),
            dcc.Graph(figure=norm_evo_fig),
        ]), className="mb-3"))

    # ── Top Anomalies ────────────────────────────────────────────────
    ranked_events: list[dict] = []
    for snap_item in snapshots[1:]:
        curr_name = snap_item.get("name", "?")
        compared_to = (snap_item.get("anomaly_summary") or {}).get("compared_to", snapshots[0].get("name", "?"))
        for event in snap_item.get("anomalies", []):
            row = dict(event)
            row["checkpoint"] = curr_name
            row["compared_to"] = compared_to
            ranked_events.append(row)

    ranked_events.sort(key=lambda e: e.get("score", 0.0), reverse=True)
    top_events = ranked_events[:20]

    if top_events:
        def _event_badge(event_type: str):
            if event_type == "rank_collapse":
                return dbc.Badge("Rank Collapse", color="danger")
            if event_type == "kurtosis_spike":
                return dbc.Badge("Kurtosis Spike", color="warning", text_color="dark")
            if event_type == "norm_jump_outlier":
                return dbc.Badge("Norm Jump Outlier", color="info")
            return dbc.Badge(event_type, color="secondary")

        rows = []
        for event in top_events:
            rows.append(html.Tr([
                html.Td(_event_badge(event.get("type", "unknown"))),
                html.Td(html.Code(short_layer(event.get("layer", "?")))),
                html.Td(event.get("checkpoint", "?")),
                html.Td(event.get("compared_to", "?"), className="text-muted"),
                html.Td(f"{event.get('score', 0.0):.2f}", className="fw-semibold"),
                html.Td(event.get("message", ""), className="text-muted", style={"fontSize": "0.85em"}),
            ]))

        children.append(dbc.Card(dbc.CardBody([
            html.H5("Top Anomalies", className="card-title"),
            html.P(
                "Ranked suspicious checkpoint transitions: effective-rank collapse, "
                "kurtosis spikes, and norm-jump outliers.",
                className="text-muted",
            ),
            html.Div(
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Type"), html.Th("Layer"), html.Th("Checkpoint"),
                        html.Th("Compared To"), html.Th("Score"), html.Th("Details"),
                    ])),
                    html.Tbody(rows),
                ], bordered=True, hover=True, responsive=True, size="sm"),
                style={"maxHeight": "480px", "overflowY": "auto"},
            ),
        ]), className="mb-3"))

    # ── Optimizer metrics ────────────────────────────────────────────
    children.extend(_optimizer_state_cards(snapshots))

    # ── Parameter Health table ───────────────────────────────────────
    health_rows: list[dict] = []
    for stat in stats:
        state, reason = weight_health(stat)
        shape_str = "x".join(str(s) for s in stat.get("shape", []))
        kind = "bias" if stat["layer"].endswith(".bias") else "weight"
        eff_rank = stat.get("effective_rank")
        max_rank = stat.get("max_rank")
        rank_str = (
            f"{100 * eff_rank / max(max_rank, 1):.0f}%"
            if eff_rank is not None and max_rank else "-"
        )
        health_rows.append({
            "layer": stat["layer"],
            "kind": kind,
            "shape": shape_str,
            "state": state,
            "reason": reason,
            "norm_l2": stat.get("norm_l2", 0),
            "near_zero_pct": stat.get("near_zero_pct"),
            "rank_util": rank_str,
            "condition": stat.get("condition_number"),
            "kurtosis": stat.get("kurtosis"),
            "alpha": stat.get("alpha"),
            "mp_softrank": stat.get("mp_softrank"),
            "num_spikes": stat.get("num_spikes"),
            "mean": stat.get("mean", 0),
            "std": stat.get("std", 0),
        })

    health_rows.sort(key=lambda r: (HEALTH_SORT.get(r["state"], 3), r["layer"]))

    def _badge(state):
        return dbc.Badge(
            state.title(),
            color=(
                "danger" if state == "critical"
                else "warning" if state == "warning"
                else "success" if state == "healthy"
                else "secondary"
            ),
        )

    def _metric(val, fmt=".4g"):
        if val is None:
            return "-"
        return f"{val:{fmt}}"

    has_spectral = any(r["alpha"] is not None for r in health_rows)

    table_rows = []
    for r in health_rows:
        cells = [
            html.Td(html.Code(short_layer(r["layer"]))),
            html.Td(r["kind"]),
            html.Td(html.Code(r["shape"])),
            html.Td(_badge(r["state"])),
            html.Td(r["reason"], className="text-muted", style={"fontSize": "0.85em"}),
            html.Td(_metric(r["norm_l2"])),
            html.Td(_metric(r["near_zero_pct"], ".1f")),
            html.Td(r["rank_util"]),
            html.Td(_metric(r["condition"], ".1e")),
            html.Td(_metric(r["kurtosis"], ".1f")),
        ]
        if has_spectral:
            cells.append(html.Td(_metric(r["alpha"], ".2f")))
            cells.append(html.Td(_metric(r["mp_softrank"], ".2f")))
            cells.append(html.Td(str(r["num_spikes"]) if r["num_spikes"] is not None else "-"))
        table_rows.append(html.Tr(cells, id={"type": "ns-row", "index": r["layer"]}))

    header_cells = [
        html.Th("Layer"), html.Th("Kind"), html.Th("Shape"),
        html.Th("Health"), html.Th("Reason"),
        html.Th("L2 Norm"), html.Th("Zero%"),
        html.Th("Rank"), html.Th("Cond#"), html.Th("Kurt"),
    ]
    if has_spectral:
        header_cells.extend([
            html.Th("Alpha", title="Power-law exponent of eigenvalue spectral density"),
            html.Th("MP Soft", title="Marchenko-Pastur softrank (1.0 = random)"),
            html.Th("Spikes", title="Eigenvalues above MP bulk edge"),
        ])

    health_table = dbc.Table([
        html.Thead(html.Tr(header_cells)),
        html.Tbody(table_rows),
    ], bordered=True, hover=True, responsive=True, size="sm")

    children.append(dbc.Card(dbc.CardBody([
        html.H5(f"Parameter Health ({len(health_rows)} entries)", className="card-title"),
        html.Div(health_table, style={"maxHeight": "600px", "overflowY": "auto"}),
    ]), className="mb-3"))

    return dbc.Container(children, fluid=True)
