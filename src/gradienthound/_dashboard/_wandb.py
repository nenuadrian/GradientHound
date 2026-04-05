"""Weights & Biases integration: caching, fetching, and metrics page."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import html, dcc

from ._helpers import plotly_layout, lttb_downsample, _MAX_CHART_POINTS


def parse_wandb_project_run_id(project_run_id: str) -> tuple[str, str]:
    cleaned = project_run_id.strip().strip("/")
    parts = [p for p in cleaned.split("/") if p]
    if len(parts) != 2:
        raise ValueError("Expected format: project/run_id")
    return parts[0], parts[1]


_WANDB_CACHE_DIR = Path.home() / ".cache" / "gradienthound" / "wandb"
_WANDB_CACHE_TTL = 3600  # seconds


def _wandb_cache_path(entity: str, project: str, run_id: str) -> Path:
    key = f"{entity}/{project}/{run_id}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return _WANDB_CACHE_DIR / f"{entity}_{project}_{run_id}_{h}.json"


def _wandb_cache_read(entity: str, project: str, run_id: str) -> tuple[list[dict], str] | None:
    path = _wandb_cache_path(entity, project, run_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if time.time() - data.get("ts", 0) > _WANDB_CACHE_TTL:
            return None
        return data["entries"], data["run_label"]
    except (json.JSONDecodeError, KeyError):
        return None


def _wandb_cache_write(entity: str, project: str, run_id: str, entries: list[dict], run_label: str):
    path = _wandb_cache_path(entity, project, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ts": time.time(), "run_label": run_label, "entries": entries}))


def fetch_wandb_run_metrics(entity: str, project: str, run_id: str) -> tuple[list[dict], str]:
    cached = _wandb_cache_read(entity, project, run_id)
    if cached is not None:
        return cached

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed. Install it with: pip install wandb") from exc

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    entries: list[dict] = []
    for idx, row in enumerate(run.scan_history(page_size=1000)):
        entry: dict = {}
        step_val = row.get("_step")
        if isinstance(step_val, (int, float)) and math.isfinite(float(step_val)):
            entry["_step"] = int(step_val)
        else:
            entry["_step"] = idx

        for key, value in row.items():
            if key.startswith("_"):
                continue
            if isinstance(value, bool):
                entry[key] = int(value)
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                entry[key] = float(value)

        if len(entry) > 1:
            entries.append(entry)

    run_label = run.name or run.id
    _wandb_cache_write(entity, project, run_id, entries, run_label)
    return entries, run_label


def _wandb_metric_category(metric_key: str) -> str:
    if "/" in metric_key:
        return metric_key.split("/", 1)[0]
    return "Other"


def _wandb_metric_figure(metric_key: str, entries: list[dict]):
    import plotly.graph_objects as go

    xs: list[int | float] = []
    ys: list[float] = []
    for idx, entry in enumerate(entries):
        if metric_key not in entry:
            continue
        xs.append(entry.get("_step", idx))
        ys.append(float(entry[metric_key]))

    xs, ys = lttb_downsample(xs, ys, _MAX_CHART_POINTS)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line={"width": 2, "color": "#375a7f"},
        name=metric_key,
        hovertemplate="step=%{x}<br>value=%{y:.6g}<extra></extra>",
    ))
    fig.update_layout(
        **plotly_layout(title=metric_key, margin={"l": 50, "r": 20, "t": 40, "b": 45}),
        height=280,
        showlegend=False,
    )
    fig.update_xaxes(title="step")
    return fig


def metrics_page_wandb(
    wandb_data: dict | None,
    default_entity: str | None,
    default_project_run_id: str | None,
):
    controls = dbc.Card(dbc.CardBody([
        html.H5("Load Weights & Biases Run", className="card-title"),
        html.P(
            "Fetch scalar history from entity + project/run_id."
            " Metrics are grouped by prefix before '/'.",
            className="text-muted",
        ),
        dbc.Row([
            dbc.Col([
                dbc.Label("Entity", html_for="wandb-entity-input"),
                dbc.Input(
                    id="wandb-entity-input",
                    type="text",
                    value=default_entity or "",
                    placeholder="team-or-username",
                ),
            ], md=4),
            dbc.Col([
                dbc.Label("Project/Run ID", html_for="wandb-project-run-id-input"),
                dbc.Input(
                    id="wandb-project-run-id-input",
                    type="text",
                    value=default_project_run_id or "",
                    placeholder="project/run_id",
                ),
            ], md=5),
            dbc.Col([
                dbc.Label(" "),
                dbc.Button("Fetch Metrics", id="wandb-fetch-btn", color="primary", className="w-100"),
            ], md=3),
        ], className="g-3"),
        html.Div(id="wandb-fetch-status", className="mt-3"),
    ]), className="mb-3")

    children: list = [
        html.H2("Metrics", className="mt-3 mb-1"),
        html.P("Run metrics fetched directly from Weights & Biases.", className="text-muted mb-4"),
        controls,
    ]

    if not wandb_data:
        children.append(dbc.Alert(
            "No run loaded yet. Enter an entity and project/run_id, then click Fetch Metrics.",
            color="secondary",
        ))
        return dbc.Container(children, fluid=True)

    entries = wandb_data.get("entries", [])
    metric_keys = wandb_data.get("metric_keys", [])
    run_label = wandb_data.get("run_label", "")

    children.append(dbc.Card(dbc.CardBody([
        html.H6("Loaded Run", className="mb-2"),
        html.Code(
            f"{wandb_data.get('entity', '')}/{wandb_data.get('project', '')}/{wandb_data.get('run_id', '')}"
        ),
        html.Div(
            f"{run_label} | points: {len(entries)} | metrics: {len(metric_keys)}",
            className="text-muted mt-2",
        ),
    ]), className="mb-3"))

    if not metric_keys:
        children.append(dbc.Alert("No scalar metrics were found in this run.", color="warning"))
        return dbc.Container(children, fluid=True)

    grouped: dict[str, list[str]] = {}
    for key in sorted(metric_keys):
        grouped.setdefault(_wandb_metric_category(key), []).append(key)

    for category in sorted(grouped):
        children.append(html.H4(category, className="mt-4 mb-3"))
        keys = grouped[category]
        for i in range(0, len(keys), 2):
            cols = []
            for key in keys[i:i + 2]:
                cols.append(dbc.Col(dbc.Card(dbc.CardBody([
                    dcc.Graph(
                        figure=_wandb_metric_figure(key, entries),
                        config={"displayModeBar": False},
                    ),
                ])), md=6, xs=12))
            children.append(dbc.Row(cols, className="g-3 mb-3"))

    return dbc.Container(children, fluid=True)
