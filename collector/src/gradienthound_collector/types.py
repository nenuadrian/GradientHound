from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class GraphNode:
    id: str
    name: str
    op: str  # module_group, call_module, call_function, call_method, placeholder, get_attr, output
    is_leaf: bool
    module_type: str | None = None
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    input_shapes: list[list[Any]] = field(default_factory=list)
    output_shapes: list[list[Any]] = field(default_factory=list)
    param_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove None values for cleaner JSON
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class GraphEdge:
    id: str
    source: str
    target: str
    tensor_shape: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class ModelGraph:
    model_name: str
    model_class: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    root_id: str = "root"
    input_nodes: list[str] = field(default_factory=list)
    output_nodes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_class": self.model_class,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "root_id": self.root_id,
            "input_nodes": self.input_nodes,
            "output_nodes": self.output_nodes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
