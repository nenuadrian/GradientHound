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

_MAX_HEATMAP_DIM = 128
# Maximum number of unflushed records held in memory per buffer.
# Each record is a small dict (~10 floats), so 10 000 entries ≈ 1–2 MB.
# If step() is not called frequently enough the oldest entries are dropped
# to prevent unbounded memory growth.
_MAX_BUFFER_SIZE = 10_000


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
        grad_abs = grad_flat.abs()
        grad_abs_mean = grad_abs.mean().item()
        grad_abs_max = grad_abs.max().item()

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
            "grad_abs_mean": grad_abs_mean,
            "grad_abs_max": grad_abs_max,
            "weight_norm": weight_norm,
            "dead_grad_pct": dead_grad_pct,
            "near_zero_weight_pct": near_zero_weight_pct,
            "grad_noise_scale": grad_noise,
        }
        if cosine_sim is not None:
            rec["cosine_sim"] = cosine_sim
        if len(self._grad_buffer) >= _MAX_BUFFER_SIZE:
            self._grad_buffer.pop(0)
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
        if len(self._activation_buffer) >= _MAX_BUFFER_SIZE:
            self._activation_buffer.pop(0)
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
        from gradienthound.checkpoint import compute_tensor_stats

        ts = time.time()
        entries = compute_tensor_stats(self.model.named_parameters())

        for entry in entries:
            entry["step"] = step
            entry["model"] = self.name
            entry["_timestamp"] = ts

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
