from __future__ import annotations

from pydantic import BaseModel
from typing import Any


class GraphNodeModel(BaseModel):
    id: str
    name: str
    op: str
    is_leaf: bool
    module_type: str | None = None
    parent_id: str | None = None
    children: list[str] = []
    attributes: dict[str, Any] = {}
    input_shapes: list[list[Any]] = []
    output_shapes: list[list[Any]] = []
    param_count: int = 0


class GraphEdgeModel(BaseModel):
    id: str
    source: str
    target: str
    tensor_shape: dict[str, Any] | None = None


class ModelGraphModel(BaseModel):
    model_name: str
    model_class: str
    nodes: list[GraphNodeModel] = []
    edges: list[GraphEdgeModel] = []
    root_id: str = "root"
    input_nodes: list[str] = []
    output_nodes: list[str] = []


class CheckpointManifest(BaseModel):
    format: str
    version: str
    created_at: str
    pytorch_version: str | None = None
    collector_version: str | None = None


class CheckpointMetadata(BaseModel):
    step: int | None = None
    epoch: int | None = None
    loss: float | None = None


class CheckpointInfo(BaseModel):
    filename: str
    manifest: CheckpointManifest
    metadata: CheckpointMetadata
    model_name: str
    node_count: int
    edge_count: int
    param_count: int


class ParameterStats(BaseModel):
    name: str
    shape: list[int]
    dtype: str
    numel: int
    mean: float
    std: float
    min: float
    max: float
    histogram: list[int]
    histogram_edges: list[float]
