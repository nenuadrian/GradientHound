"""Checkpoint comparison utilities.

Load multiple PyTorch checkpoints, compute per-parameter weight statistics,
and return structured data suitable for the Dash comparison dashboard.

Two loading modes are supported:

1. **With loader script** — user provides a ``.py`` file that defines
   ``load_checkpoint(path: str) -> torch.nn.Module``.
2. **Without loader** — GradientHound calls ``torch.load`` and auto-detects
   the ``state_dict`` from common checkpoint formats.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

_HIST_BINS = 80


# ── Standalone stat computation ──────────────────────────────────────


def compute_tensor_stats(
    named_tensors: Iterable[tuple[str, Any]],
    *,
    hist_bins: int = _HIST_BINS,
) -> list[dict[str, Any]]:
    """Compute per-tensor weight statistics.

    Works with ``model.named_parameters()``, ``state_dict.items()``, or any
    iterable of ``(name, tensor)`` pairs.

    Returns a list of stat dicts, one per tensor.
    """
    import torch

    entries: list[dict[str, Any]] = []

    for param_name, tensor in named_tensors:
        if not isinstance(tensor, torch.Tensor):
            continue

        data = tensor.data
        flat = data.flatten().float()

        entry: dict[str, Any] = {
            "layer": param_name,
            "norm_l2": data.norm(2).item(),
            "norm_frobenius": (
                data.norm("fro").item() if data.ndim >= 2 else data.norm(2).item()
            ),
            "mean": flat.mean().item(),
            "std": flat.std().item() if flat.numel() > 1 else 0.0,
            "min": flat.min().item(),
            "max": flat.max().item(),
            "near_zero_pct": (flat.abs() < 1e-6).float().mean().item() * 100,
            "numel": flat.numel(),
            "shape": list(data.shape),
        }

        # Kurtosis
        if flat.numel() > 4:
            mu = flat.mean()
            sigma = flat.std()
            if sigma > 1e-12:
                entry["kurtosis"] = ((flat - mu) / sigma).pow(4).mean().item() - 3.0
            else:
                entry["kurtosis"] = 0.0
        else:
            entry["kurtosis"] = 0.0

        # Histogram
        lo, hi = flat.min().item(), flat.max().item()
        if hi - lo < 1e-12:
            lo, hi = lo - 1.0, hi + 1.0
        counts = torch.histc(flat, bins=hist_bins, min=lo, max=hi)
        bin_width = (hi - lo) / hist_bins
        bin_centers = [lo + (i + 0.5) * bin_width for i in range(hist_bins)]
        entry["hist_counts"] = counts.tolist()
        entry["hist_centers"] = bin_centers

        # SVD for 2D weight matrices (skip bias)
        if data.ndim == 2 and "bias" not in param_name:
            try:
                svs = torch.linalg.svdvals(data.float().detach())
                entry["singular_values"] = svs.tolist()

                sv_sq = svs.pow(2)
                total_energy = sv_sq.sum()
                if total_energy > 1e-12:
                    cum_energy = sv_sq.cumsum(0) / total_energy
                    entry["cumulative_energy"] = cum_energy.tolist()

                sigma_max_sq = svs[0].item() ** 2
                entry["stable_rank"] = total_energy.item() / max(sigma_max_sq, 1e-12)
                entry["condition_number"] = svs[0].item() / max(svs[-1].item(), 1e-12)

                sv_sum = svs.sum()
                if sv_sum > 1e-12:
                    p = svs / sv_sum
                    p = p[p > 1e-12]
                    entropy = -(p * p.log()).sum().item()
                    entry["effective_rank"] = math.exp(entropy)
                    entry["max_rank"] = min(data.shape[0], data.shape[1])
                else:
                    entry["effective_rank"] = 0.0
                    entry["max_rank"] = min(data.shape[0], data.shape[1])
            except Exception:
                pass

        entries.append(entry)

    return entries


# ── State dict detection ─────────────────────────────────────────────


def extract_state_dict(raw: Any) -> dict[str, Any]:
    """Extract a ``state_dict`` from a loaded checkpoint object.

    Handles common formats:

    - Raw ``state_dict`` (dict whose values are all tensors)
    - ``{"state_dict": ...}``
    - ``{"model_state_dict": ...}``
    - ``{"model": ...}``
    - Nested sub-dicts (e.g. ``{"actor_mean": {tensor_dict}, "critic": {tensor_dict}, ...}``)
      are flattened with dotted prefixes like ``"actor_mean.0.weight"``.

    Non-tensor top-level values (optimizers, scalars, etc.) are silently skipped.

    Raises :class:`ValueError` if no tensors can be found.
    """
    import torch

    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected a dict from torch.load, got {type(raw).__name__}"
        )

    def _is_tensor_dict(d: Any) -> bool:
        return isinstance(d, dict) and d and all(
            isinstance(v, torch.Tensor) for v in d.values()
        )

    # Case 1: already a flat state_dict
    if _is_tensor_dict(raw):
        return raw

    # Case 2: well-known wrapper keys
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = raw.get(key)
        if candidate is not None and _is_tensor_dict(candidate):
            return candidate

    # Case 3: nested sub-dicts — flatten with dotted prefix
    # e.g. {"actor_mean": {"0.weight": tensor, ...}, "critic": {...}, ...}
    merged: dict[str, torch.Tensor] = {}
    for key, val in raw.items():
        if isinstance(val, torch.Tensor):
            merged[key] = val
        elif _is_tensor_dict(val):
            for sub_key, tensor in val.items():
                merged[f"{key}.{sub_key}"] = tensor

    if merged:
        return merged

    raise ValueError(
        "Could not detect a state_dict in the checkpoint. "
        f"Top-level keys: {list(raw.keys())[:20]}"
    )


# ── Loader script import ─────────────────────────────────────────────


def import_loader(loader_path: str) -> Callable[[str], nn.Module]:
    """Import a user-provided loader script and return its ``load_checkpoint``.

    The script must define::

        def load_checkpoint(path: str) -> torch.nn.Module:
            ...
    """
    path = Path(loader_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Loader script not found: {path}")

    spec = importlib.util.spec_from_file_location("_gh_user_loader", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import loader from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_gh_user_loader"] = module
    spec.loader.exec_module(module)

    fn = getattr(module, "load_checkpoint", None)
    if fn is None or not callable(fn):
        raise ImportError(
            f"Loader script {path} must define a callable 'load_checkpoint(path)'"
        )
    return fn


# ── Checkpoint name derivation ───────────────────────────────────────


def derive_checkpoint_name(path: str) -> str:
    """Derive a human-readable name from a checkpoint file path.

    ``"checkpoints/epoch_10.pt"`` → ``"epoch_10"``
    """
    return Path(path).stem


# ── Orchestration ────────────────────────────────────────────────────


def process_checkpoints(
    checkpoint_paths: list[str],
    *,
    loader_path: str | None = None,
) -> list[dict[str, Any]]:
    """Load checkpoints and compute per-parameter weight statistics.

    Args:
        checkpoint_paths: Paths to ``.pt`` / ``.pth`` / ``.ckpt`` files.
        loader_path: Optional path to a Python script with
            ``load_checkpoint(path) -> nn.Module``.

    Returns:
        List of snapshot dicts, each containing ``name``, ``path``, and
        ``weight_stats`` (list of per-parameter stat dicts).
    """
    import torch

    loader_fn: Callable | None = None
    if loader_path:
        loader_fn = import_loader(loader_path)

    snapshots: list[dict[str, Any]] = []

    for ckpt_path in checkpoint_paths:
        name = derive_checkpoint_name(ckpt_path)

        if loader_fn is not None:
            model = loader_fn(ckpt_path)
            stats = compute_tensor_stats(model.named_parameters())
        else:
            try:
                raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            except Exception:
                raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = extract_state_dict(raw)
            stats = compute_tensor_stats(sd.items())

        snapshots.append({
            "name": name,
            "path": ckpt_path,
            "weight_stats": stats,
        })

    return snapshots
