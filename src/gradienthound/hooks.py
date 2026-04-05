"""PyTorch hook system for automatic gradient, weight, and activation capture.

``WatchState`` registers backward/forward hooks on a model and accumulates
per-layer statistics in memory.  Call :meth:`flush_gradient_stats` (typically
via ``GradientHound.step()``) to write buffered data to the IPC channel.
"""
from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    from .ipc import IPCChannel

_HIST_BINS = 80
_MAX_HEATMAP_DIM = 128


class WatchState:
    """Per-model watch state.  Manages PyTorch hooks and buffered stats."""

    def __init__(
        self,
        model: nn.Module,
        name: str,
        *,
        log_gradients: bool = True,
        log_activations: bool = False,
        weight_every: int = 50,
    ) -> None:
        self.model = model
        self.name = name
        self.weight_every = weight_every

        self._grad_buffer: list[dict[str, Any]] = []
        self._activation_buffer: list[dict[str, Any]] = []
        self._hook_handles: list[Any] = []
        # Previous gradient directions for cosine similarity
        self._prev_grad_dirs: dict[str, Any] = {}

        if log_gradients:
            self._register_grad_hooks()
        if log_activations:
            self._register_activation_hooks()

    # ── Gradient hooks ────────────────────────────────────────────────

    def _register_grad_hooks(self) -> None:
        for param_name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            handle = param.register_post_accumulate_grad_hook(
                _make_grad_hook(self, param_name),
            )
            self._hook_handles.append(handle)

    def _record_grad(self, param_name: str, param: torch.Tensor) -> None:
        """Called by the grad hook -- accumulates stats in the buffer."""
        import torch

        grad = param.grad
        if grad is None:
            return

        grad_flat = grad.data.flatten().float()
        weight_flat = param.data.flatten().float()

        grad_norm = grad.data.norm(2).item()
        weight_norm = param.data.norm(2).item()

        grad_mean = grad_flat.mean().item()
        grad_std = grad_flat.std().item() if grad_flat.numel() > 1 else 0.0
        grad_var = grad_flat.var().item() if grad_flat.numel() > 1 else 0.0

        dead_grad_pct = (grad_flat.abs() < 1e-7).float().mean().item() * 100
        near_zero_weight_pct = (weight_flat.abs() < 1e-6).float().mean().item() * 100
        grad_noise = grad_var / max(grad_mean ** 2, 1e-20)

        # Cosine similarity with previous step's gradient
        cosine_sim: float | None = None
        cur_dir = grad_flat.detach().clone()
        cur_norm = cur_dir.norm()
        if cur_norm > 1e-12 and param_name in self._prev_grad_dirs:
            prev_dir = self._prev_grad_dirs[param_name]
            prev_norm = prev_dir.norm()
            if prev_norm > 1e-12:
                cosine_sim = torch.nn.functional.cosine_similarity(
                    cur_dir.unsqueeze(0), prev_dir.unsqueeze(0),
                ).item()
        self._prev_grad_dirs[param_name] = cur_dir

        rec: dict[str, Any] = {
            "layer": param_name,
            "grad_norm": grad_norm,
            "grad_mean": grad_mean,
            "grad_std": grad_std,
            "weight_norm": weight_norm,
            "dead_grad_pct": dead_grad_pct,
            "near_zero_weight_pct": near_zero_weight_pct,
            "grad_noise_scale": grad_noise,
        }
        if cosine_sim is not None:
            rec["cosine_sim"] = cosine_sim
        self._grad_buffer.append(rec)

    # ── Activation hooks ──────────────────────────────────────────────

    def _register_activation_hooks(self) -> None:
        for mod_name, module in self.model.named_modules():
            if list(module.children()):
                continue
            handle = module.register_forward_hook(
                _make_activation_hook(self, mod_name),
            )
            self._hook_handles.append(handle)

    def _record_activation(self, layer_name: str, output: torch.Tensor) -> None:
        import torch

        if not isinstance(output, torch.Tensor):
            return
        flat = output.data.flatten().float()
        self._activation_buffer.append({
            "layer": layer_name,
            "mean": flat.mean().item(),
            "std": flat.std().item() if flat.numel() > 1 else 0.0,
            "min": flat.min().item(),
            "max": flat.max().item(),
            "zero_fraction": (flat.abs() < 1e-6).float().mean().item(),
        })

    # ── Flush to IPC ──────────────────────────────────────────────────

    def flush_gradient_stats(self, step: int, ipc: IPCChannel) -> None:
        if not self._grad_buffer:
            return
        ts = time.time()
        entries = []
        for rec in self._grad_buffer:
            rec["step"] = step
            rec["model"] = self.name
            rec["_timestamp"] = ts
            entries.append(rec)
        ipc.append_gradient_stats(entries)
        self._grad_buffer.clear()

    def flush_activation_stats(self, step: int, ipc: IPCChannel) -> None:
        if not self._activation_buffer:
            return
        ts = time.time()
        entries = []
        for rec in self._activation_buffer:
            rec["step"] = step
            rec["model"] = self.name
            rec["_timestamp"] = ts
            entries.append(rec)
        ipc.append_activation_stats(entries)
        self._activation_buffer.clear()

    def compute_weight_stats(self, step: int, ipc: IPCChannel) -> None:
        """Compute per-layer weight statistics and write to IPC."""
        import torch

        ts = time.time()
        entries: list[dict[str, Any]] = []

        for param_name, param in self.model.named_parameters():
            data = param.data
            flat = data.flatten().float()

            entry: dict[str, Any] = {
                "step": step,
                "model": self.name,
                "layer": param_name,
                "norm_l2": data.norm(2).item(),
                "norm_frobenius": data.norm("fro").item() if data.ndim >= 2 else data.norm(2).item(),
                "mean": flat.mean().item(),
                "std": flat.std().item() if flat.numel() > 1 else 0.0,
                "min": flat.min().item(),
                "max": flat.max().item(),
                "near_zero_pct": (flat.abs() < 1e-6).float().mean().item() * 100,
                "numel": flat.numel(),
                "shape": list(data.shape),
                "_timestamp": ts,
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

            # Histogram bins (80 bins, matching Minerva)
            lo, hi = flat.min().item(), flat.max().item()
            if hi - lo < 1e-12:
                lo, hi = lo - 1.0, hi + 1.0
            counts = torch.histc(flat, bins=_HIST_BINS, min=lo, max=hi)
            bin_width = (hi - lo) / _HIST_BINS
            bin_centers = [lo + (i + 0.5) * bin_width for i in range(_HIST_BINS)]
            entry["hist_counts"] = counts.tolist()
            entry["hist_centers"] = bin_centers

            # SVD for 2D weight matrices
            if data.ndim == 2 and "bias" not in param_name:
                try:
                    svs = torch.linalg.svdvals(data.float().detach())
                    entry["singular_values"] = svs.tolist()

                    # Cumulative energy
                    sv_sq = svs.pow(2)
                    total_energy = sv_sq.sum()
                    if total_energy > 1e-12:
                        cum_energy = sv_sq.cumsum(0) / total_energy
                        entry["cumulative_energy"] = cum_energy.tolist()

                    # Stable rank: ||W||_F^2 / sigma_max^2
                    sigma_max_sq = svs[0].item() ** 2
                    entry["stable_rank"] = total_energy.item() / max(sigma_max_sq, 1e-12)

                    # Condition number
                    entry["condition_number"] = svs[0].item() / max(svs[-1].item(), 1e-12)

                    # Effective rank via entropy
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

        if entries:
            ipc.append_weight_stats(entries)

    # ── On-demand request processing ──────────────────────────────────

    def process_requests(self, step: int, ipc: IPCChannel) -> None:
        """Process pending on-demand computation requests from the UI."""
        requests = ipc.read_requests()
        if not requests:
            return
        ipc.clear_requests()

        for req in requests:
            req_type = req.get("type")
            req_id = req.get("id", req_type)
            model_name = req.get("model")
            if model_name and model_name != self.name:
                continue
            try:
                if req_type == "weight_heatmap":
                    self._compute_weight_heatmap(req, step, req_id, ipc)
                elif req_type == "cka":
                    self._compute_cka(step, req_id, ipc)
                elif req_type == "network_state":
                    self._compute_network_state(step, req_id, ipc)
            except Exception:
                ipc.write_response(req_id, {"error": "computation failed"})

    def _compute_weight_heatmap(
        self, req: dict, step: int, req_id: str, ipc: IPCChannel,
    ) -> None:
        import torch
        import torch.nn.functional as F

        layer = req.get("layer", "")
        param = dict(self.model.named_parameters()).get(layer)
        if param is None or param.ndim != 2:
            ipc.write_response(req_id, {"error": f"layer '{layer}' not found or not 2D"})
            return

        w = param.data.float().detach()
        rows, cols = w.shape

        # Downsample if too large
        if rows > _MAX_HEATMAP_DIM or cols > _MAX_HEATMAP_DIM:
            target_h = min(rows, _MAX_HEATMAP_DIM)
            target_w = min(cols, _MAX_HEATMAP_DIM)
            w = F.adaptive_avg_pool2d(
                w.unsqueeze(0).unsqueeze(0), (target_h, target_w),
            ).squeeze(0).squeeze(0)

        vmax = max(w.abs().max().item(), 1e-8)
        sparsity = (param.data.float().abs() < 1e-6).float().mean().item() * 100

        ipc.write_response(req_id, {
            "step": step,
            "layer": layer,
            "matrix": w.tolist(),
            "shape": [rows, cols],
            "display_shape": list(w.shape),
            "vmax": vmax,
            "sparsity": sparsity,
        })

    def _compute_network_state(self, step: int, req_id: str, ipc: IPCChannel) -> None:
        """Dump all parameter values for models under 1M params."""
        import numpy as np

        total_params = sum(p.numel() for p in self.model.parameters())
        if total_params > 1_000_000:
            ipc.write_response(req_id, {
                "error": f"Model has {total_params:,} parameters (limit: 1,000,000)",
            })
            return

        layers: list[dict[str, Any]] = []
        for param_name, param in self.model.named_parameters():
            data = param.data.detach().float().cpu()
            shape = list(data.shape)
            # Vectorised rounding -- avoids per-element Python round() calls
            arr = np.around(data.numpy(), decimals=6)

            if arr.ndim == 0:
                values = [[float(arr)]]
            elif arr.ndim == 1:
                values = [arr.tolist()]
            elif arr.ndim == 2:
                values = arr.tolist()
            else:
                # Reshape higher-dim to 2D: (product of leading dims, last dim)
                values = arr.reshape(-1, arr.shape[-1]).tolist()

            layers.append({
                "name": param_name,
                "shape": shape,
                "numel": param.numel(),
                "requires_grad": param.requires_grad,
                "dtype": str(param.dtype),
                "values": values,
            })

        ipc.write_response(req_id, {
            "step": step,
            "model": self.name,
            "total_params": total_params,
            "layers": layers,
            "_timestamp": time.time(),
        })

    def _compute_cka(self, step: int, req_id: str, ipc: IPCChannel) -> None:
        import torch

        # Collect all 2D weight matrices
        layers: list[str] = []
        weights: list[torch.Tensor] = []
        for name, param in self.model.named_parameters():
            if param.ndim == 2 and "bias" not in name:
                layers.append(name)
                weights.append(param.data.float().detach())

        n = len(layers)
        if n < 2:
            ipc.write_response(req_id, {"error": "need at least 2 weight matrices"})
            return

        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                wi, wj = weights[i], weights[j]
                # Align rows
                min_rows = min(wi.shape[0], wj.shape[0])
                wi_t = wi[:min_rows]
                wj_t = wj[:min_rows]
                cross = (wi_t.T @ wj_t).pow(2).sum().item()
                self_i = (wi_t.T @ wi_t).pow(2).sum().item()
                self_j = (wj_t.T @ wj_t).pow(2).sum().item()
                denom = max(math.sqrt(self_i * self_j), 1e-12)
                cka_val = cross / denom
                matrix[i][j] = cka_val
                matrix[j][i] = cka_val

        # Short layer names for display
        short_names = [_short_name(l) for l in layers]

        ipc.write_response(req_id, {
            "step": step,
            "layers": layers,
            "short_names": short_names,
            "matrix": matrix,
            "n": n,
        })

    # ── Cleanup ───────────────────────────────────────────────────────

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._grad_buffer.clear()
        self._activation_buffer.clear()
        self._prev_grad_dirs.clear()


# ── Utilities ─────────────────────────────────────────────────────────


def _short_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) > 3:
        parts = parts[-3:]
    return ".".join(parts)


# ── Hook factories ────────────────────────────────────────────────────


def _make_grad_hook(state: WatchState, param_name: str):
    def hook(param):
        state._record_grad(param_name, param)
    return hook


def _make_activation_hook(state: WatchState, layer_name: str):
    def hook(module, input, output):
        state._record_activation(layer_name, output)
    return hook
