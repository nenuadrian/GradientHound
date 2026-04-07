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
_DEFAULT_SUBSPACE_TOP_K = 8
_RANK_COLLAPSE_RATIO_THRESHOLD = 0.6
_KURTOSIS_SPIKE_THRESHOLD = 2.0
_NORM_OUTLIER_Z_THRESHOLD = 2.5

# Known optimizer state buffer keys (used for detection & type inference)
_ADAM_BUFFERS = {"exp_avg", "exp_avg_sq"}
_SGD_BUFFERS = {"momentum_buffer"}
_RMSPROP_BUFFERS = {"square_avg"}
_ADADELTA_BUFFERS = {"square_avg", "acc_delta"}
_ALL_OPT_BUFFERS = (
    _ADAM_BUFFERS | _SGD_BUFFERS | _RMSPROP_BUFFERS | _ADADELTA_BUFFERS | {"step", "max_exp_avg_sq", "grad_avg"}
)


# ── Optimizer state detection ──────────────────────────────────────────


def is_optimizer_state_dict(obj: Any) -> bool:
    """Return ``True`` if *obj* looks like a PyTorch optimizer ``state_dict()``.

    Detection is structural — we look for the ``state`` / ``param_groups``
    layout produced by :meth:`torch.optim.Optimizer.state_dict`, not at key
    names in the parent checkpoint.
    """
    if not isinstance(obj, dict):
        return False

    state = obj.get("state")
    param_groups = obj.get("param_groups")

    # Must have both top-level keys
    if not isinstance(state, dict) or not isinstance(param_groups, list):
        return False

    # param_groups entries must look like optimizer groups
    if not param_groups:
        return False
    for pg in param_groups:
        if not isinstance(pg, dict):
            return False
        if "params" not in pg:
            return False

    # state keys are integer param indices (may be int or str-of-int)
    if state:
        sample_key = next(iter(state))
        if not isinstance(sample_key, int):
            try:
                int(sample_key)
            except (ValueError, TypeError):
                return False
        # At least one entry should contain known buffer keys
        for s in state.values():
            if isinstance(s, dict) and (set(s.keys()) & _ALL_OPT_BUFFERS):
                return True
        # Even without buffers (step 0, empty state), the structure is enough
        return True

    # Empty state (before first optimizer step) — structure is still valid
    return True


def extract_optimizer_states(raw: dict) -> dict[str, dict]:
    """Scan a raw checkpoint dict and return all optimizer state dicts found.

    Returns ``{key_name: optimizer_state_dict}`` for each top-level value that
    passes :func:`is_optimizer_state_dict`.
    """
    found: dict[str, dict] = {}
    for key, val in raw.items():
        if is_optimizer_state_dict(val):
            found[key] = val
    return found


def _infer_optimizer_type(opt_sd: dict) -> str:
    """Heuristically infer the optimizer type from its state dict."""
    state = opt_sd.get("state", {})
    pg0 = (opt_sd.get("param_groups") or [{}])[0]

    # Collect all buffer keys across state entries
    all_keys: set[str] = set()
    for s in state.values():
        if isinstance(s, dict):
            all_keys.update(s.keys())

    if "acc_delta" in all_keys:
        return "Adadelta"
    if "exp_avg" in all_keys and "exp_avg_sq" in all_keys:
        # Distinguish Adam variants by param_group keys
        if pg0.get("amsgrad"):
            return "Adam (amsgrad)"
        if "decoupled_weight_decay" in pg0 or "weight_decay" in pg0:
            # AdamW typically appears with weight_decay > 0 but we can't be sure
            pass
        return "Adam-family"
    if "square_avg" in all_keys:
        return "RMSprop"
    if "momentum_buffer" in all_keys:
        return "SGD (momentum)"
    if "betas" in pg0:
        return "Adam-family"
    if pg0.get("momentum", 0) > 0:
        return "SGD (momentum)"

    return "Unknown"


def compute_optimizer_stats(name: str, opt_sd: dict) -> dict[str, Any]:
    """Compute aggregate statistics from a single optimizer state dict.

    Returns a dict with type, hyperparameters, per-group state stats, and
    memory estimates.
    """
    import torch

    opt_type = _infer_optimizer_type(opt_sd)
    state = opt_sd.get("state", {})
    param_groups = opt_sd.get("param_groups", [])

    # ── Per-group hyperparameters ────────────────────────────────────
    groups_info: list[dict[str, Any]] = []
    for i, pg in enumerate(param_groups):
        info: dict[str, Any] = {"group_index": i}
        for k, v in pg.items():
            if k == "params":
                info["num_params"] = len(v)
                info["param_indices"] = v
            elif isinstance(v, (int, float, bool, str)):
                info[k] = v
            elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
                info[k] = list(v)
        groups_info.append(info)

    # ── Map param indices → group index ──────────────────────────────
    param_to_group: dict[int, int] = {}
    for gi, pg in enumerate(param_groups):
        for pid in pg.get("params", []):
            p_int = int(pid) if not isinstance(pid, int) else pid
            param_to_group[p_int] = gi

    # ── Per-group state buffer statistics ────────────────────────────
    n_groups = len(param_groups)
    group_exp_avg_norms: list[list[float]] = [[] for _ in range(n_groups)]
    group_exp_avg_sq_means: list[list[float]] = [[] for _ in range(n_groups)]
    group_momentum_norms: list[list[float]] = [[] for _ in range(n_groups)]
    group_steps: list[int] = [0] * n_groups
    total_state_numel = 0
    total_state_bytes = 0
    n_state_tensors = 0

    for pid, s in state.items():
        if not isinstance(s, dict):
            continue
        p_int = int(pid) if not isinstance(pid, int) else pid
        gi = param_to_group.get(p_int, 0)

        for buf_key, buf_val in s.items():
            if buf_key == "step":
                # step may be a scalar tensor or a plain int/float
                step_val = int(buf_val.item()) if hasattr(buf_val, "item") else int(buf_val)
                group_steps[gi] = max(group_steps[gi], step_val)
            elif isinstance(buf_val, torch.Tensor):
                numel = buf_val.numel()
                total_state_numel += numel
                total_state_bytes += numel * buf_val.element_size()
                n_state_tensors += 1

                if buf_key == "exp_avg":
                    group_exp_avg_norms[gi].append(buf_val.norm(2).item())
                elif buf_key == "exp_avg_sq":
                    group_exp_avg_sq_means[gi].append(buf_val.mean().item())
                elif buf_key == "momentum_buffer":
                    group_momentum_norms[gi].append(buf_val.norm(2).item())

    # ── Assemble per-group stats ─────────────────────────────────────
    for gi, info in enumerate(groups_info):
        info["step"] = group_steps[gi]

        ea = group_exp_avg_norms[gi]
        if ea:
            info["exp_avg_norm_mean"] = sum(ea) / len(ea)
            info["exp_avg_norm_max"] = max(ea)

        esq = group_exp_avg_sq_means[gi]
        if esq:
            avg_v = sum(esq) / len(esq)
            info["exp_avg_sq_mean"] = avg_v
            eps = info.get("eps", 1e-8)
            lr = info.get("lr", 0)
            info["effective_lr"] = lr / (avg_v ** 0.5 + eps) if avg_v > 0 else lr

            # Bias correction
            betas = info.get("betas")
            step = group_steps[gi]
            if betas and step > 0 and isinstance(betas, (list, tuple)) and len(betas) == 2:
                bc2 = 1 - betas[1] ** step
                info["bias_correction2"] = bc2
                import math
                steps_to_converge = math.log(0.01) / math.log(betas[1]) if betas[1] < 1 else 1000
                info["warmup_pct"] = min(100.0, round(step / steps_to_converge * 100, 1))

        mn = group_momentum_norms[gi]
        if mn:
            info["momentum_norm_mean"] = sum(mn) / len(mn)
            info["momentum_norm_max"] = max(mn)

    return {
        "name": name,
        "type": opt_type,
        "n_state_tensors": n_state_tensors,
        "total_state_numel": total_state_numel,
        "total_state_bytes": total_state_bytes,
        "groups": groups_info,
    }


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

                # WeightWatcher-style spectral metrics
                if "singular_values" in entry:
                    from gradienthound.spectral import compute_spectral_metrics
                    spectral = compute_spectral_metrics(
                        entry["singular_values"],
                        (data.shape[0], data.shape[1]),
                    )
                    entry.update(spectral)
            except Exception:
                pass

        entries.append(entry)

    return entries


def _linear_cka(a: Any, b: Any) -> float | None:
    """Compute a linear CKA-style similarity score between two 2D tensors."""
    import torch

    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return None
    if a.ndim != 2 or b.ndim != 2:
        return None

    min_rows = min(a.shape[0], b.shape[0])
    if min_rows <= 0:
        return None

    a_t = a[:min_rows].float()
    b_t = b[:min_rows].float()

    cross = (a_t.T @ b_t).pow(2).sum().item()
    self_a = (a_t.T @ a_t).pow(2).sum().item()
    self_b = (b_t.T @ b_t).pow(2).sum().item()
    denom = max(math.sqrt(self_a * self_b), 1e-12)
    return cross / denom


def _subspace_overlap_topk(a: Any, b: Any, top_k: int) -> float | None:
    """Compute overlap of right-singular top-k subspaces in [0, 1]."""
    import torch

    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return None
    if a.ndim != 2 or b.ndim != 2:
        return None
    if a.shape[1] != b.shape[1]:
        return None

    max_k = min(a.shape[0], a.shape[1], b.shape[0], b.shape[1], int(top_k))
    if max_k <= 0:
        return None

    try:
        _, _, va_t = torch.linalg.svd(a.float(), full_matrices=False)
        _, _, vb_t = torch.linalg.svd(b.float(), full_matrices=False)
    except Exception:
        return None

    va = va_t[:max_k].T
    vb = vb_t[:max_k].T
    overlap = (va.T @ vb).pow(2).sum().item() / max_k
    return max(0.0, min(1.0, overlap))


def annotate_directional_drift(
    snapshots: list[dict[str, Any]],
    *,
    subspace_top_k: int = _DEFAULT_SUBSPACE_TOP_K,
    compute_cka: bool = False,
) -> None:
    """Annotate snapshot stats with checkpoint-to-checkpoint directional drift.

    Adds the following keys on each layer entry for snapshots i >= 1:
    - ``drift_cosine_prev``: cosine similarity of flattened weights.
    - ``drift_subspace_overlap_prev``: top-k right-singular subspace overlap.
    - ``drift_cka_prev`` (optional): linear CKA similarity for 2D weights.
    """
    import torch

    if len(snapshots) < 2:
        return

    top_k = max(1, int(subspace_top_k))

    for i in range(1, len(snapshots)):
        prev_lookup = snapshots[i - 1].get("_tensor_lookup", {})
        curr_lookup = snapshots[i].get("_tensor_lookup", {})
        if not isinstance(prev_lookup, dict) or not isinstance(curr_lookup, dict):
            continue

        curr_entries = snapshots[i].get("weight_stats", [])
        cosine_vals: list[float] = []
        overlap_vals: list[float] = []
        cka_vals: list[float] = []

        for entry in curr_entries:
            layer = entry.get("layer")
            if not isinstance(layer, str):
                continue
            prev_t = prev_lookup.get(layer)
            curr_t = curr_lookup.get(layer)
            if not isinstance(prev_t, torch.Tensor) or not isinstance(curr_t, torch.Tensor):
                continue

            if prev_t.numel() == curr_t.numel() and prev_t.numel() > 0:
                prev_flat = prev_t.detach().float().flatten()
                curr_flat = curr_t.detach().float().flatten()
                denom = max(prev_flat.norm().item() * curr_flat.norm().item(), 1e-12)
                cosine = float(torch.dot(prev_flat, curr_flat).item() / denom)
                entry["drift_cosine_prev"] = max(-1.0, min(1.0, cosine))
                cosine_vals.append(entry["drift_cosine_prev"])

            overlap = _subspace_overlap_topk(prev_t, curr_t, top_k)
            if overlap is not None:
                entry["drift_subspace_overlap_prev"] = overlap
                overlap_vals.append(overlap)

            if compute_cka:
                cka_score = _linear_cka(prev_t, curr_t)
                if cka_score is not None:
                    entry["drift_cka_prev"] = cka_score
                    cka_vals.append(cka_score)

        summary: dict[str, Any] = {
            "compared_to": snapshots[i - 1].get("name"),
            "n_layers": len(curr_entries),
            "n_cosine": len(cosine_vals),
            "n_subspace": len(overlap_vals),
        }
        if cosine_vals:
            summary["cosine_mean"] = sum(cosine_vals) / len(cosine_vals)
        if overlap_vals:
            summary["subspace_overlap_mean"] = sum(overlap_vals) / len(overlap_vals)
        if compute_cka:
            summary["n_cka"] = len(cka_vals)
            if cka_vals:
                summary["cka_mean"] = sum(cka_vals) / len(cka_vals)
        snapshots[i]["drift_summary"] = summary


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def annotate_checkpoint_events(snapshots: list[dict[str, Any]]) -> None:
    """Detect and score suspicious checkpoint-to-checkpoint transitions.

    Events are attached under each snapshot as ``anomalies`` for i >= 1.
    """
    if len(snapshots) < 2:
        return

    for i in range(1, len(snapshots)):
        prev_snapshot = snapshots[i - 1]
        curr_snapshot = snapshots[i]

        prev_lookup = {s.get("layer"): s for s in prev_snapshot.get("weight_stats", [])}
        curr_lookup = {s.get("layer"): s for s in curr_snapshot.get("weight_stats", [])}

        # Collect relative norm changes for robust outlier detection.
        rel_changes: list[float] = []
        rel_change_by_layer: dict[str, float] = {}
        for layer, curr in curr_lookup.items():
            if not isinstance(layer, str):
                continue
            prev = prev_lookup.get(layer)
            if not isinstance(prev, dict):
                continue
            prev_norm = prev.get("norm_l2")
            curr_norm = curr.get("norm_l2")
            if not isinstance(prev_norm, (int, float)) or not isinstance(curr_norm, (int, float)):
                continue
            if prev_norm <= 1e-12:
                continue
            rel = abs(curr_norm - prev_norm) / prev_norm
            rel_changes.append(rel)
            rel_change_by_layer[layer] = rel

        med = _median(rel_changes)
        mad = _median([abs(v - med) for v in rel_changes])
        # Scale 1.4826 * MAD approximates standard deviation for normal data.
        robust_sigma = max(1.4826 * mad, 1e-6)

        anomalies: list[dict[str, Any]] = []

        for layer, curr in curr_lookup.items():
            if not isinstance(layer, str):
                continue
            prev = prev_lookup.get(layer)
            if not isinstance(prev, dict):
                continue

            # 1) Effective-rank collapse
            prev_rank = prev.get("effective_rank")
            curr_rank = curr.get("effective_rank")
            if isinstance(prev_rank, (int, float)) and isinstance(curr_rank, (int, float)) and prev_rank > 1e-6:
                rank_ratio = curr_rank / prev_rank
                if rank_ratio < _RANK_COLLAPSE_RATIO_THRESHOLD:
                    severity = (1.0 - rank_ratio) * 100.0
                    anomalies.append({
                        "type": "rank_collapse",
                        "layer": layer,
                        "score": round(severity, 3),
                        "current": curr_rank,
                        "previous": prev_rank,
                        "ratio": rank_ratio,
                        "message": (
                            f"Effective rank dropped from {prev_rank:.3g} to {curr_rank:.3g} "
                            f"({rank_ratio * 100:.1f}% of previous)."
                        ),
                    })

            # 2) Kurtosis spikes
            prev_kurt = prev.get("kurtosis")
            curr_kurt = curr.get("kurtosis")
            if isinstance(prev_kurt, (int, float)) and isinstance(curr_kurt, (int, float)):
                kurt_delta = abs(curr_kurt) - abs(prev_kurt)
                if kurt_delta >= _KURTOSIS_SPIKE_THRESHOLD:
                    severity = kurt_delta * 12.5
                    anomalies.append({
                        "type": "kurtosis_spike",
                        "layer": layer,
                        "score": round(severity, 3),
                        "current": curr_kurt,
                        "previous": prev_kurt,
                        "delta_abs": kurt_delta,
                        "message": (
                            f"|kurtosis| increased by {kurt_delta:.3g} "
                            f"(from {abs(prev_kurt):.3g} to {abs(curr_kurt):.3g})."
                        ),
                    })

            # 3) Large norm-jump outliers per layer
            rel = rel_change_by_layer.get(layer)
            if rel is not None:
                z = (rel - med) / robust_sigma
                if z >= _NORM_OUTLIER_Z_THRESHOLD:
                    severity = z * 10.0 + rel * 100.0
                    anomalies.append({
                        "type": "norm_jump_outlier",
                        "layer": layer,
                        "score": round(severity, 3),
                        "relative_change": rel,
                        "robust_z": z,
                        "median_relative_change": med,
                        "message": (
                            f"Relative L2 norm change {rel * 100:.2f}% "
                            f"(robust z={z:.2f} vs transition median {med * 100:.2f}%)."
                        ),
                    })

        anomalies.sort(key=lambda a: a.get("score", 0.0), reverse=True)
        curr_snapshot["anomalies"] = anomalies
        curr_snapshot["anomaly_summary"] = {
            "compared_to": prev_snapshot.get("name"),
            "count": len(anomalies),
            "top_score": anomalies[0]["score"] if anomalies else 0.0,
        }


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


# ── Checkpoint discovery ─────────────────────────────────────────────


_CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt"}


def discover_checkpoints(locations: list[str]) -> list[str]:
    """Discover checkpoint files from a mix of files and directories.

    Recursively searches directories for files with checkpoint extensions
    (``.pt``, ``.pth``, ``.ckpt``). Returns a sorted list of discovered paths.

    Args:
        locations: List of file paths or directory paths. Files are included
            directly; directories are recursively searched for checkpoints.

    Returns:
        Sorted list of discovered checkpoint file paths (absolute or relative,
        depending on input).
    """
    discovered: set[str] = set()

    for location in locations:
        p = Path(location)

        if p.is_file():
            # If it's a file, include it if it has a checkpoint extension
            if p.suffix.lower() in _CHECKPOINT_EXTENSIONS:
                discovered.add(str(p))
        elif p.is_dir():
            # Recursively search directory for checkpoint files
            for ext in _CHECKPOINT_EXTENSIONS:
                discovered.update(str(f) for f in p.glob(f"**/*{ext}"))
        # else: path doesn't exist yet; silently skip

    # Return sorted list for deterministic ordering
    return sorted(discovered)


def discover_model_exports(locations: list[str]) -> list[str]:
    """Discover model export files from a mix of files and directories.

    Recursively searches directories for files with ``.gh.json`` extension.
    Returns a sorted list of discovered paths.

    Args:
        locations: List of file paths or directory paths. Files are included
            directly; directories are recursively searched for model exports.

    Returns:
        Sorted list of discovered model export file paths (absolute or relative,
        depending on input).
    """
    discovered: set[str] = set()

    for location in locations:
        p = Path(location)

        if p.is_file():
            # If it's a file, include it if it has a .gh.json extension
            if p.suffix.lower() == ".json" and p.name.endswith(".gh.json"):
                discovered.add(str(p))
        elif p.is_dir():
            # Recursively search directory for .gh.json files
            discovered.update(str(f) for f in p.glob("**/*.gh.json"))
        # else: path doesn't exist yet; silently skip

    # Return sorted list for deterministic ordering
    return sorted(discovered)


# ── Orchestration ────────────────────────────────────────────────────


def process_checkpoints(
    checkpoint_paths: list[str],
    *,
    loader_path: str | None = None,
    compute_cka: bool = False,
    subspace_top_k: int = _DEFAULT_SUBSPACE_TOP_K,
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

        opt_stats: list[dict[str, Any]] = []

        if loader_fn is not None:
            model = loader_fn(ckpt_path)
            params = list(model.named_parameters())
            stats = compute_tensor_stats(params)
            tensor_lookup = {
                name: tensor.detach().cpu()
                for name, tensor in params
                if isinstance(tensor, torch.Tensor)
            }
        else:
            try:
                raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            except Exception:
                raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = extract_state_dict(raw)
            sd_items = list(sd.items())
            stats = compute_tensor_stats(sd_items)
            tensor_lookup = {
                name: tensor.detach().cpu()
                for name, tensor in sd_items
                if isinstance(tensor, torch.Tensor)
            }

            # Detect and analyse optimizer states
            if isinstance(raw, dict):
                for opt_key, opt_sd in extract_optimizer_states(raw).items():
                    try:
                        opt_stats.append(compute_optimizer_stats(opt_key, opt_sd))
                    except Exception:
                        pass  # skip malformed optimizer data

        snapshots.append({
            "name": name,
            "path": ckpt_path,
            "weight_stats": stats,
            "optimizer_states": opt_stats,
            "_tensor_lookup": tensor_lookup,
        })

    annotate_directional_drift(
        snapshots,
        subspace_top_k=subspace_top_k,
        compute_cka=compute_cka,
    )
    annotate_checkpoint_events(snapshots)

    for snapshot in snapshots:
        snapshot.pop("_tensor_lookup", None)

    return snapshots
