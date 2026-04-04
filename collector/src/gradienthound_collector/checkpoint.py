from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
from safetensors.torch import save

from .types import ModelGraph


OptimizerLike = (
    torch.optim.Optimizer
    | Iterable[torch.optim.Optimizer]
    | Mapping[str, torch.optim.Optimizer]
    | None
)


def _state_dict_to_safetensors_tensors(
    state_dict: dict[str, Any],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}

    for group_idx, group in enumerate(state_dict.get("param_groups", [])):
        for key, val in group.items():
            if isinstance(val, torch.Tensor):
                tensors[f"group{group_idx}.{key}"] = val.contiguous()

    for param_id, state in state_dict.get("state", {}).items():
        for key, val in state.items():
            if isinstance(val, torch.Tensor):
                tensors[f"state.{param_id}.{key}"] = val.contiguous()

    return tensors


def _safe_optimizer_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", name).strip("._")
    return clean or "optimizer"


def _normalize_optimizers(optimizer: OptimizerLike) -> list[tuple[str, torch.optim.Optimizer]]:
    if optimizer is None:
        return []

    if isinstance(optimizer, torch.optim.Optimizer):
        return [("optimizer", optimizer)]

    if isinstance(optimizer, Mapping):
        out: list[tuple[str, torch.optim.Optimizer]] = []
        for name, opt in optimizer.items():
            if not isinstance(opt, torch.optim.Optimizer):
                raise TypeError("All values in optimizer mapping must be torch.optim.Optimizer instances")
            out.append((_safe_optimizer_name(name), opt))
        return out

    out = []
    for idx, opt in enumerate(optimizer):
        if not isinstance(opt, torch.optim.Optimizer):
            raise TypeError("All items in optimizer iterable must be torch.optim.Optimizer instances")
        out.append((f"optimizer_{idx}", opt))
    return out


def write_checkpoint(
    path: str | Path,
    graph: ModelGraph,
    model: nn.Module,
    optimizer: OptimizerLike = None,
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
        optimizer: Optional optimizer(s) to include state. Accepts a single optimizer,
            an iterable of optimizers, or a dict of named optimizers.
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

        # Optional optimizer state(s)
        normalized_optimizers = _normalize_optimizers(optimizer)
        used_names: set[str] = set()
        for base_name, opt in normalized_optimizers:
            file_stem = _safe_optimizer_name(base_name)
            if file_stem in used_names:
                suffix = 1
                while f"{file_stem}_{suffix}" in used_names:
                    suffix += 1
                file_stem = f"{file_stem}_{suffix}"
            used_names.add(file_stem)

            opt_tensors = _state_dict_to_safetensors_tensors(opt.state_dict())
            if not opt_tensors:
                continue

            opt_bytes = save(opt_tensors)
            if len(normalized_optimizers) == 1 and file_stem in {"optimizer", "optimizer_0"}:
                # Keep legacy path for single-optimizer checkpoints.
                zf.writestr("optimizer/optimizer.safetensors", opt_bytes)
            else:
                zf.writestr(f"optimizer/{file_stem}.safetensors", opt_bytes)

    return path
