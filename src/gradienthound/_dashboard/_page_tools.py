"""Tools panel page: displays all registered tools and their status."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html, dcc


_STATE_BADGE = {
    "has_data":    ("Has Data",    "success"),
    "ready":       ("Ready",       "info"),
    "unavailable": ("Unavailable", "secondary"),
}

_CATEGORY_BADGE = {
    "capture":     ("Capture",     "primary"),
    "analysis":    ("Analysis",    "warning"),
    "integration": ("Integration", "info"),
    "on-demand":   ("On-Demand",   "dark"),
}


def _requirement_item(req: dict) -> dbc.ListGroupItem:
    met = req["met"]
    icon = "\u2713" if met else "\u2717"
    color = "success" if met else "danger"
    return dbc.ListGroupItem(
        [
            html.Span(icon, className=f"me-2 text-{color} fw-bold"),
            html.Span(req["label"]),
        ],
        className="py-1 px-2 border-0 bg-transparent",
    )


def _tool_card(tool: dict) -> dbc.Col:
    state = tool["state"]
    state_label, state_color = _STATE_BADGE.get(state, ("Unknown", "secondary"))
    cat_label, cat_color = _CATEGORY_BADGE.get(
        tool["category"], (tool["category"].title(), "secondary"),
    )

    header_badges = html.Div([
        dbc.Badge(cat_label, color=cat_color, className="me-2"),
        dbc.Badge(state_label, color=state_color),
    ], className="mb-2")

    req_list = dbc.ListGroup(
        [_requirement_item(r) for r in tool["requirements"]],
        flush=True,
        className="mb-2",
    ) if tool["requirements"] else html.P(
        "No special requirements", className="text-muted small mb-2",
    )

    footer = []
    if tool.get("page"):
        footer.append(
            dcc.Link(
                dbc.Button("Go to page", size="sm", color="primary", outline=True),
                href=tool["page"],
            )
        )

    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.H5(tool["name"], className="card-title mb-1"),
                header_badges,
                html.P(tool["description"], className="card-text small text-muted"),
                html.Hr(className="my-2"),
                html.Small("Requirements", className="fw-bold text-uppercase d-block mb-1"),
                req_list,
                html.Div(footer) if footer else None,
            ]),
        ], className="h-100 shadow-sm"),
        md=6, lg=4, className="mb-3",
    )


def _summary_badges(tools: list[dict]) -> html.Div:
    total = len(tools)
    with_data = sum(1 for t in tools if t["state"] == "has_data")
    ready = sum(1 for t in tools if t["state"] == "ready")
    unavail = sum(1 for t in tools if t["state"] == "unavailable")

    return html.Div([
        dbc.Badge(f"{total} total", color="dark", className="me-2 fs-6"),
        dbc.Badge(f"{with_data} active", color="success", className="me-2 fs-6"),
        dbc.Badge(f"{ready} ready", color="info", className="me-2 fs-6"),
        dbc.Badge(f"{unavail} unavailable", color="secondary", className="fs-6"),
    ], className="mb-3")


def tools_page(tools: list[dict] | None = None) -> dbc.Container:
    if tools is None:
        tools = []

    # Group by category (preserve order: capture, analysis, on-demand, integration, then any others)
    category_order = ["capture", "on-demand", "analysis", "integration"]
    categories: dict[str, list[dict]] = {}
    for t in tools:
        categories.setdefault(t["category"], []).append(t)

    ordered_cats = [c for c in category_order if c in categories]
    ordered_cats += [c for c in categories if c not in ordered_cats]

    sections = []
    for cat in ordered_cats:
        cat_label = _CATEGORY_BADGE.get(cat, (cat.title(), "secondary"))[0]
        cat_tools = categories[cat]
        sections.append(html.H4(cat_label, className="mt-4 mb-3"))
        sections.append(dbc.Row([_tool_card(t) for t in cat_tools]))

    return dbc.Container([
        html.H2("Tools", className="mt-3 mb-1"),
        html.P(
            "All registered tools and their current status. "
            "Tools are discovered automatically based on your configuration.",
            className="text-muted mb-3",
        ),

        # Auto-refresh toggle
        dbc.Row([
            dbc.Col(_summary_badges(tools), md=8),
            dbc.Col(
                dbc.Button(
                    "Refresh", id="tools-refresh-btn", color="primary",
                    size="sm", className="float-end",
                ),
                md=4, className="text-end",
            ),
        ], className="mb-2"),

        # Tool cards container (refreshed by callback)
        html.Div(sections, id="tools-card-container"),
    ], fluid=True)
