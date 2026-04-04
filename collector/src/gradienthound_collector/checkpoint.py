from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save

from .types import ModelGraph


def write_checkpoint(
    path: str | Path,
    graph: ModelGraph,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
    step: int | None = None,
    epoch: int | None = None,
    loss: float | None = None,
) -> Path:
    """Write a .ghound checkpoint file (ZIP archive).

    Args:
        path: Output file path (should end with .ghound)
        graph: Extracted model graph
        model: The PyTorch model (for state_dict)
        optimizer: Optional optimizer (for optimizer state)
        metadata: Optional custom metadata dict
        step: Optional training step number
        epoch: Optional epoch number
        loss: Optional loss value
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "ghound",
        "version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pytorch_version": torch.__version__,
        "collector_version": "0.1.0",
    }

    meta = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        **(metadata or {}),
    }

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Manifest
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        # Graph
        zf.writestr("graph.json", graph.to_json())

        # Metadata
        zf.writestr("metadata.json", json.dumps(meta, indent=2))

        # Parameters via safetensors
        state_dict = model.state_dict()
        # safetensors requires all tensors to be contiguous
        clean_state = {k: v.contiguous() for k, v in state_dict.items()}
        param_bytes = save(clean_state)
        zf.writestr("parameters/params.safetensors", param_bytes)

        # Optional optimizer state
        if optimizer is not None:
            opt_state = optimizer.state_dict()
            opt_tensors = {}
            for group_idx, group in enumerate(opt_state.get("param_groups", [])):
                for key, val in group.items():
                    if isinstance(val, torch.Tensor):
                        opt_tensors[f"group{group_idx}.{key}"] = val.contiguous()
            for param_id, state in opt_state.get("state", {}).items():
                for key, val in state.items():
                    if isinstance(val, torch.Tensor):
                        opt_tensors[f"state.{param_id}.{key}"] = val.contiguous()
            if opt_tensors:
                opt_bytes = save(opt_tensors)
                zf.writestr("optimizer/optimizer.safetensors", opt_bytes)

    return path
