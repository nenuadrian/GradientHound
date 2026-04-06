"""Checkpoints page layout and optimizer state cards."""
from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import html

from ._helpers import (
    compute_optimizer_summary_table, compute_optimizer_evolution_table,
)


def _fmt_bytes(n: int) -> str:
    if n >= 1_073_741_824:
        return f"{n / 1_073_741_824:.1f} GB"
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    if n >= 1_024:
        return f"{n / 1_024:.1f} KB"
    return f"{n} B"


def checkpoints_page(ckpt_paths: list[str], snapshots: list[dict] | None):
    children = [
        html.H2("Checkpoints", className="mt-3 mb-1"),
        html.P(f"{len(ckpt_paths)} checkpoint files loaded", className="text-muted mb-4"),
    ]

    file_rows = []
    for i, path in enumerate(ckpt_paths):
        p = Path(path)
        size = ""
        if p.exists():
            size = _fmt_bytes(p.stat().st_size)
        status = ""
        if snapshots:
            snap = next((s for s in snapshots if s["path"] == path), None)
            if snap:
                status = f"{len(snap['weight_stats'])} params"
        file_rows.append(html.Tr([
            html.Td(str(i + 1)),
            html.Td(p.name),
            html.Td(size),
            html.Td(dbc.Badge(status, color="success") if status else "\u2014"),
            html.Td(html.Code(str(path), style={"fontSize": "0.8em"})),
        ]))

    children.append(dbc.Card(dbc.CardBody([
        html.H5("Checkpoint Files", className="card-title"),
        dbc.Table([
            html.Thead(html.Tr([html.Th("#"), html.Th("File"), html.Th("Size"), html.Th("Status"), html.Th("Path")])),
            html.Tbody(file_rows),
        ], bordered=True, hover=True, responsive=True, size="sm"),
    ]), className="mb-3"))

    if snapshots:
        n_params = len({s["layer"] for snap in snapshots for s in snap["weight_stats"]})
        children.append(dbc.Alert([
            html.Strong(f"Processed \u2014 {len(snapshots)} checkpoints, {n_params} parameters. "),
            "Head back to the Dashboard to explore weight analysis, health charts, and optimizer metrics.",
        ], color="success", className="mb-3"))
    else:
        children.append(dbc.Card(dbc.CardBody([
            html.H5("Process", className="card-title"),
            html.P(
                "Load all checkpoints and compute per-parameter weight statistics "
                "(norms, distributions, SVD, kurtosis). This may take a moment for large models.",
                className="text-muted",
            ),
            dbc.Button("Process Checkpoints", id="ckpt-process-btn", color="primary", n_clicks=0),
            html.Div(id="ckpt-status", className="mt-2 text-muted"),
        ]), className="mb-3"))

    return dbc.Container(children, fluid=True)


def checkpoints_page_empty():
    return dbc.Container([
        html.H2("Checkpoints", className="mt-3 mb-1"),
        html.P("No checkpoints loaded.", className="text-muted mb-4"),
        dbc.Card(dbc.CardBody([
            html.H5("How to use", className="card-title"),
            html.P("Pass checkpoint files via the CLI to compare them:", className="text-muted"),
            html.Code("python -m gradienthound --checkpoints epoch1.pt epoch5.pt epoch10.pt",
                       className="d-block p-2 bg-dark rounded"),
        ])),
    ], fluid=True)


def _optimizer_state_cards(snapshots: list[dict]) -> list:
    """Build optimizer analysis cards for the checkpoints page."""
    has_any = any(snap.get("optimizer_states") for snap in snapshots)
    if not has_any:
        return []

    cards: list = [
        html.H3("Optimizer States", className="mt-4 mb-2"),
        html.P(
            "Optimizer state dicts detected in checkpoint files. "
            "Statistics are computed from the saved first/second moment buffers.",
            className="text-muted mb-3",
        ),
    ]

    # ── Summary table ────────────────────────────────────────────────
    ckpt_names, opt_rows = compute_optimizer_summary_table(snapshots)

    if opt_rows:
        header = [html.Th("Optimizer"), html.Th("Type")] + [
            html.Th(name) for name in ckpt_names
        ]
        body_rows = []
        for orow in opt_rows:
            cells = [
                html.Td(html.Strong(orow["name"])),
                html.Td(dbc.Badge(orow["type"], color="info")),
            ]
            for cell in orow["cells"]:
                if cell is None:
                    cells.append(html.Td("\u2014", className="text-muted"))
                else:
                    n_groups = len(cell.get("groups", []))
                    mem = _fmt_bytes(cell.get("total_state_bytes", 0))
                    n_tensors = cell.get("n_state_tensors", 0)
                    cells.append(html.Td([
                        html.Div(f"{n_groups} group{'s' if n_groups != 1 else ''}, {n_tensors} tensors"),
                        html.Small(f"State memory: {mem}", className="text-muted"),
                    ]))
            body_rows.append(html.Tr(cells))

        cards.append(dbc.Card(dbc.CardBody([
            html.H5("Overview", className="card-title"),
            html.Div(
                dbc.Table([
                    html.Thead(html.Tr(header)),
                    html.Tbody(body_rows),
                ], bordered=True, hover=True, responsive=True, size="sm"),
                style={"overflowX": "auto"},
            ),
        ]), className="mb-3"))

    # ── Hyperparameter comparison table ──────────────────────────────
    ckpt_names, evo_rows = compute_optimizer_evolution_table(snapshots)

    if evo_rows:
        # Hyperparameters
        hp_header = [html.Th("Optimizer"), html.Th("Group")] + [
            html.Th(name) for name in ckpt_names
        ]
        hp_body: list = []
        for erow in evo_rows:
            cells = [
                html.Td(erow["optimizer"]),
                html.Td(str(erow["group_index"])),
            ]
            for cell in erow["cells"]:
                if cell is None:
                    cells.append(html.Td("\u2014", className="text-muted"))
                else:
                    parts = []
                    lr = cell.get("lr")
                    if lr is not None:
                        parts.append(f"lr={lr:.2e}")
                    betas = cell.get("betas")
                    if betas:
                        parts.append(f"betas={betas}")
                    wd = cell.get("weight_decay")
                    if wd is not None and wd > 0:
                        parts.append(f"wd={wd:.2e}")
                    eps = cell.get("eps")
                    if eps is not None:
                        parts.append(f"eps={eps:.0e}")
                    mom = cell.get("momentum")
                    if mom is not None and mom > 0:
                        parts.append(f"mom={mom}")
                    cells.append(html.Td(html.Code(", ".join(parts) if parts else "\u2014")))
            hp_body.append(html.Tr(cells))

        cards.append(dbc.Card(dbc.CardBody([
            html.H5("Hyperparameters Across Checkpoints", className="card-title"),
            html.Div(
                dbc.Table([
                    html.Thead(html.Tr(hp_header)),
                    html.Tbody(hp_body),
                ], bordered=True, hover=True, responsive=True, size="sm"),
                style={"overflowX": "auto"},
            ),
        ]), className="mb-3"))

        # ── State statistics table ───────────────────────────────────
        st_header = [html.Th("Optimizer"), html.Th("Group")] + [
            html.Th(name) for name in ckpt_names
        ]
        st_body: list = []
        for erow in evo_rows:
            cells = [
                html.Td(erow["optimizer"]),
                html.Td(str(erow["group_index"])),
            ]
            for cell in erow["cells"]:
                if cell is None:
                    cells.append(html.Td("\u2014", className="text-muted"))
                else:
                    lines: list = []
                    step = cell.get("step")
                    if step is not None and step > 0:
                        lines.append(html.Div(f"Step: {step:,}"))
                    ean = cell.get("exp_avg_norm_mean")
                    if ean is not None:
                        lines.append(html.Div(
                            f"1st moment norm: {ean:.4g} "
                            f"(max {cell.get('exp_avg_norm_max', 0):.4g})"
                        ))
                    esm = cell.get("exp_avg_sq_mean")
                    if esm is not None:
                        lines.append(html.Div(f"2nd moment mean: {esm:.4g}"))
                    elr = cell.get("effective_lr")
                    if elr is not None:
                        lines.append(html.Div([
                            "Effective LR: ",
                            html.Strong(f"{elr:.4g}"),
                        ]))
                    bc = cell.get("bias_correction2")
                    if bc is not None:
                        wp = cell.get("warmup_pct", 0)
                        lines.append(html.Div(
                            f"Bias corr: {bc:.6f} ({wp:.1f}% warmed up)"
                        ))
                    mnm = cell.get("momentum_norm_mean")
                    if mnm is not None:
                        lines.append(html.Div(
                            f"Momentum norm: {mnm:.4g} "
                            f"(max {cell.get('momentum_norm_max', 0):.4g})"
                        ))
                    if not lines:
                        lines.append(html.Div("No state yet", className="text-muted"))
                    cells.append(html.Td(lines))
            st_body.append(html.Tr(cells))

        cards.append(dbc.Card(dbc.CardBody([
            html.H5("Optimizer State Statistics", className="card-title"),
            html.P(
                "Per-group statistics from the optimizer's internal buffers. "
                "Effective LR is estimated as lr / (sqrt(mean_v) + eps).",
                className="text-muted mb-2",
            ),
            html.Div(
                dbc.Table([
                    html.Thead(html.Tr(st_header)),
                    html.Tbody(st_body),
                ], bordered=True, hover=True, responsive=True, size="sm"),
                style={"maxHeight": "600px", "overflowY": "auto", "overflowX": "auto"},
            ),
        ]), className="mb-3"))

    return cards
