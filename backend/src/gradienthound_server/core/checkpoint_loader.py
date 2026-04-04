from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from safetensors import safe_open

from ..models.graph import (
    ModelGraphModel,
    CheckpointManifest,
    CheckpointMetadata,
    ParameterStats,
)


class CheckpointLoader:
    """Reads .ghound ZIP archives without requiring PyTorch."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.path}")
        self._zf = zipfile.ZipFile(self.path, "r")

    def close(self):
        self._zf.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def manifest(self) -> CheckpointManifest:
        data = json.loads(self._zf.read("manifest.json"))
        return CheckpointManifest(**data)

    def graph(self) -> ModelGraphModel:
        data = json.loads(self._zf.read("graph.json"))
        return ModelGraphModel(**data)

    def metadata(self) -> CheckpointMetadata:
        data = json.loads(self._zf.read("metadata.json"))
        return CheckpointMetadata(**data)

    def parameter_names(self) -> list[str]:
        """List all parameter tensor names."""
        # Extract safetensors to a temp location and read keys
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp.write(self._zf.read("parameters/params.safetensors"))
            tmp_path = tmp.name
        try:
            with safe_open(tmp_path, framework="numpy") as f:
                return list(f.keys())
        finally:
            os.unlink(tmp_path)

    def parameter_stats(self, name: str) -> ParameterStats:
        """Load a single parameter tensor and compute statistics."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp.write(self._zf.read("parameters/params.safetensors"))
            tmp_path = tmp.name
        try:
            with safe_open(tmp_path, framework="numpy") as f:
                tensor = f.get_tensor(name)
                flat = tensor.flatten().astype(np.float64)
                hist_counts, hist_edges = np.histogram(flat, bins=50)
                return ParameterStats(
                    name=name,
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    numel=int(tensor.size),
                    mean=float(np.mean(flat)),
                    std=float(np.std(flat)),
                    min=float(np.min(flat)),
                    max=float(np.max(flat)),
                    histogram=hist_counts.tolist(),
                    histogram_edges=hist_edges.tolist(),
                )
        finally:
            os.unlink(tmp_path)

    def total_param_count(self) -> int:
        graph = self.graph()
        # Sum param_count from the root node (which has total)
        for node in graph.nodes:
            if node.id == graph.root_id:
                return node.param_count
        return 0
