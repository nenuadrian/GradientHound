"""PyTorch hook system for automatic gradient, weight, and activation capture.

``WatchState`` registers backward/forward hooks on a model and accumulates
per-layer statistics in memory.  Call :meth:`flush_gradient_stats` (typically
via ``GradientHound.step()``) to flush buffered data.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

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

    # ── Flush buffers ─────────────────────────────────────────────────

    def flush_gradient_stats(self, step: int) -> None:
        if not self._grad_buffer:
            return
        ts = time.time()
        for rec in self._grad_buffer:
            rec["step"] = step
            rec["model"] = self.name
            rec["_timestamp"] = ts
        self._grad_buffer.clear()

    def flush_activation_stats(self, step: int) -> None:
        if not self._activation_buffer:
            return
        ts = time.time()
        for rec in self._activation_buffer:
            rec["step"] = step
            rec["model"] = self.name
            rec["_timestamp"] = ts
        self._activation_buffer.clear()

    def compute_weight_stats(self, step: int) -> None:
        """Compute per-layer weight statistics."""
        from gradienthound.checkpoint import compute_tensor_stats

        ts = time.time()
        entries = compute_tensor_stats(self.model.named_parameters())

        for entry in entries:
            entry["step"] = step
            entry["model"] = self.name
            entry["_timestamp"] = ts

    # ── Cleanup ───────────────────────────────────────────────────────

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._grad_buffer.clear()
        self._activation_buffer.clear()
        self._prev_grad_dirs.clear()


# ── Hook factories ────────────────────────────────────────────────────


def _make_grad_hook(state: WatchState, param_name: str):
    def hook(param):
        state._record_grad(param_name, param)
    return hook


def _make_activation_hook(state: WatchState, layer_name: str):
    def hook(module, input, output):
        state._record_activation(layer_name, output)
    return hook
