from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..core.config import settings
from ..core.checkpoint_loader import CheckpointLoader
from ..models.graph import (
    CheckpointInfo,
    ModelGraphModel,
    CheckpointMetadata,
    GraphNodeModel,
    ParameterStats,
)
from ..services.graph_service import search_nodes, get_node

router = APIRouter(prefix="/api")


def _get_checkpoint_dir(dir_override: str | None = None) -> Path:
    return Path(dir_override) if dir_override else Path(settings.checkpoint_dir)


def _get_loader(filename: str, dir_override: str | None = None) -> CheckpointLoader:
    checkpoint_dir = _get_checkpoint_dir(dir_override)
    path = checkpoint_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {filename}")
    return CheckpointLoader(path)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/checkpoints", response_model=list[CheckpointInfo])
def list_checkpoints(dir: str | None = Query(None)):
    checkpoint_dir = _get_checkpoint_dir(dir)
    if not checkpoint_dir.exists():
        return []

    results = []
    for ghound_file in sorted(checkpoint_dir.glob("*.ghound")):
        try:
            with CheckpointLoader(ghound_file) as loader:
                manifest = loader.manifest()
                metadata = loader.metadata()
                graph = loader.graph()
                results.append(CheckpointInfo(
                    filename=ghound_file.name,
                    manifest=manifest,
                    metadata=metadata,
                    model_name=graph.model_name,
                    node_count=len(graph.nodes),
                    edge_count=len(graph.edges),
                    param_count=loader.total_param_count(),
                ))
        except Exception:
            continue
    return results


@router.get("/checkpoints/{filename}/graph", response_model=ModelGraphModel)
def get_graph(filename: str, dir: str | None = Query(None)):
    with _get_loader(filename, dir) as loader:
        return loader.graph()


@router.get("/checkpoints/{filename}/metadata", response_model=CheckpointMetadata)
def get_metadata(filename: str, dir: str | None = Query(None)):
    with _get_loader(filename, dir) as loader:
        return loader.metadata()


@router.get("/checkpoints/{filename}/node/{node_id}", response_model=GraphNodeModel)
def get_node_detail(filename: str, node_id: str, dir: str | None = Query(None)):
    with _get_loader(filename, dir) as loader:
        graph = loader.graph()
        node = get_node(graph, node_id)
        if node is None:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
        return node


@router.get("/checkpoints/{filename}/node/{node_id}/params", response_model=ParameterStats)
def get_node_params(filename: str, node_id: str, dir: str | None = Query(None)):
    with _get_loader(filename, dir) as loader:
        try:
            return loader.parameter_stats(node_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"No parameters for node: {node_id}")


@router.get("/checkpoints/{filename}/search", response_model=list[GraphNodeModel])
def search(
    filename: str,
    q: str = Query(...),
    regex: bool = Query(False),
    dir: str | None = Query(None),
):
    with _get_loader(filename, dir) as loader:
        graph = loader.graph()
        return search_nodes(graph, q, use_regex=regex)
