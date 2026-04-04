from __future__ import annotations

import re

from ..models.graph import ModelGraphModel, GraphNodeModel


def search_nodes(
    graph: ModelGraphModel,
    query: str,
    use_regex: bool = False,
) -> list[GraphNodeModel]:
    """Search nodes by name, id, or module_type."""
    results = []
    if use_regex:
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            return []
        for node in graph.nodes:
            if (pattern.search(node.id)
                or pattern.search(node.name)
                or (node.module_type and pattern.search(node.module_type))):
                results.append(node)
    else:
        q = query.lower()
        for node in graph.nodes:
            if (q in node.id.lower()
                or q in node.name.lower()
                or (node.module_type and q in node.module_type.lower())):
                results.append(node)
    return results


def get_node(graph: ModelGraphModel, node_id: str) -> GraphNodeModel | None:
    for node in graph.nodes:
        if node.id == node_id:
            return node
    return None
