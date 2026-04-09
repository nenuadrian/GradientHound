from __future__ import annotations

import atexit
import time
from typing import TYPE_CHECKING, Any

from .graph import extract_model_graph
from .hooks import WatchState

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim


class GradientHound:
    """Main entry point for GradientHound.
    """

    def __init__(
        self,
        metadata: dict | None = None,
    ) -> None:
        self._models: dict[str, dict] = {}
        self._optimizers: dict[str, dict] = {}
        self._optimizer_refs: dict[str, optim.Optimizer] = {}
        self._watches: dict[str, WatchState] = {}
        self._metadata: dict = metadata or {}
        self._wandb_original_log: Any = None
        self._step: int = 0
        self._last_flushed_step: int | None = None
        self._shutdown_called: bool = False

        atexit.register(self.shutdown)

    def register_model(self, name: str, model: nn.Module) -> None:
        """Register a PyTorch model. The UI will display its architecture."""
        graph = extract_model_graph(name, model)
        self._models[name] = graph

    def register_optimizer(self, name: str, optimizer: optim.Optimizer) -> None:
        """Register an optimizer. The UI will display its configuration."""
        info = _extract_optimizer_info(optimizer)
        self._optimizers[name] = info
        self._optimizer_refs[name] = optimizer

    # ── Hook-based capture ──────────────────────────────────────────

    def watch(
        self,
        model: nn.Module,
        name: str | None = None,
        *,
        log_gradients: bool = True,
        log_activations: bool = False,
        weight_every: int = 50,
    ) -> None:
        """Register PyTorch hooks for automatic gradient/weight/activation capture."""
        if name is None:
            name = type(model).__name__
        if name not in self._models:
            self.register_model(name, model)
        self._watches[name] = WatchState(
            model, name,
            log_gradients=log_gradients,
            log_activations=log_activations,
            weight_every=weight_every,
        )

    def _sync_step(self, step: int | None = None, *, advance: bool = False) -> int:
        """Synchronize the internal step counter with an external training step."""
        if step is None:
            if advance:
                self._step += 1
            return self._step

        try:
            resolved = int(step)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid step value: {step!r}") from exc

        if resolved < 0:
            raise ValueError(f"Step must be non-negative, got {resolved}")

        if resolved > self._step:
            self._step = resolved
        return self._step

    def step(self, step: int | None = None) -> None:
        """Advance the step counter and flush buffered stats."""
        current_step = self._sync_step(step, advance=step is None)
        has_pending_buffers = any(
            ws._grad_buffer or ws._activation_buffer  # noqa: SLF001
            for ws in self._watches.values()
        )
        should_emit_weights = self._last_flushed_step != current_step

        if self._last_flushed_step == current_step and not has_pending_buffers:
            return

        for ws in self._watches.values():
            ws.flush_gradient_stats(current_step)
            ws.flush_activation_stats(current_step)
            if should_emit_weights and current_step % ws.weight_every == 0:
                ws.compute_weight_stats(current_step)
        self._last_flushed_step = current_step

    def log_weights(self, name: str | None = None) -> None:
        """Force an immediate weight snapshot."""
        targets = [self._watches[name]] if name else list(self._watches.values())
        for ws in targets:
            ws.compute_weight_stats(self._step)

    def log_attention(
        self,
        name: str,
        weights: torch.Tensor,
    ) -> None:
        """Log attention weight matrix for visualization.

        *weights* shape: ``(batch, heads, seq_q, seq_kv)`` or ``(heads, seq_q, seq_kv)``.
        Large sequences are downsampled to 64x64.
        """
        import torch
        import torch.nn.functional as F

        w = weights.detach().float()
        if w.ndim == 4:
            w = w[0]  # take first batch element
        # w is now (heads, seq_q, seq_kv)
        heads, sq, skv = w.shape

        # Downsample large sequences
        max_seq = 64
        if sq > max_seq or skv > max_seq:
            w = F.adaptive_avg_pool2d(w.unsqueeze(0), (min(sq, max_seq), min(skv, max_seq))).squeeze(0)

    def log_predictions(
        self,
        predicted: Any,
        actual: Any,
        name: str = "default",
    ) -> None:
        """Log prediction vs actual for calibration scatter plot."""
        import torch

        def _to_list(x: Any) -> list[float]:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().flatten().tolist()
            if isinstance(x, (list, tuple)):
                return [float(v) for v in x]
            return [float(x)]

        pred = _to_list(predicted)
        act = _to_list(actual)
        # Subsample if too many points
        if len(pred) > 500:
            stride = len(pred) // 500
            pred = pred[::stride]
            act = act[::stride]

    # ── Wandb capture ─────────────────────────────────────────────────

    def capture_wandb(self) -> None:
        """Monkey-patch wandb.log to also capture metrics for the GradientHound UI.

        Call this after wandb.init(). All subsequent wandb.log() calls will
        also be recorded and displayed as live time-series charts.
        """
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")

        if self._wandb_original_log is not None:
            return  # already patched

        original_log = wandb.log
        self._wandb_original_log = original_log

        def _patched_log(data: dict, *args: Any, **kwargs: Any) -> Any:
            # Forward to real wandb.log first
            result = original_log(data, *args, **kwargs)

            logged_step = kwargs.get("step")
            if logged_step is not None:
                self.step(step=logged_step)
            else:
                self._step += 1

            return result

        wandb.log = _patched_log

    def _restore_wandb(self) -> None:
        """Restore the original wandb.log if we patched it."""
        if self._wandb_original_log is not None:
            try:
                import wandb
                wandb.log = self._wandb_original_log
            except ImportError:
                pass
            self._wandb_original_log = None

    def shutdown(self) -> None:
        """Clean up hooks and wandb patching.

        Idempotent -- safe to call more than once (e.g. from both a
        ``with`` block and the atexit handler).
        """
        if self._shutdown_called:
            return
        self._shutdown_called = True

        try:
            self._restore_wandb()
        except Exception:
            pass

        try:
            for ws in self._watches.values():
                ws.remove_hooks()
            self._watches.clear()
        except Exception:
            pass

    def __enter__(self) -> GradientHound:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


def _extract_optimizer_info(optimizer: optim.Optimizer) -> dict:
    defaults = {}
    for k, v in optimizer.defaults.items():
        if isinstance(v, (int, float, bool, str)):
            defaults[k] = v

    opt_type = type(optimizer).__name__
    # Determine state buffer multiplier for memory estimation
    adam_types = {"Adam", "AdamW", "NAdam", "RAdam", "Adagrad", "Adadelta"}
    if opt_type in adam_types:
        buffers_per_param = 2  # exp_avg + exp_avg_sq
    elif opt_type in {"SGD", "ASGD"} and defaults.get("momentum", 0) > 0:
        buffers_per_param = 1  # momentum_buffer
    elif opt_type == "RMSprop":
        buffers_per_param = 2 if defaults.get("momentum", 0) > 0 else 1
    else:
        buffers_per_param = 0

    total_numel = 0
    param_groups = []
    for i, group in enumerate(optimizer.param_groups):
        group_numel = sum(p.numel() for p in group["params"])
        # Estimate bytes: param element size * numel * buffers
        elem_size = 4  # default float32
        if group["params"]:
            elem_size = group["params"][0].element_size()
        group_memory_bytes = group_numel * elem_size * buffers_per_param

        pg: dict = {
            "index": i,
            "num_params": len(group["params"]),
            "total_numel": group_numel,
            "memory_bytes": group_memory_bytes,
        }
        for k, v in group.items():
            if k == "params":
                continue
            if isinstance(v, (int, float, bool, str)):
                pg[k] = v
        param_groups.append(pg)
        total_numel += group_numel

    # Back-fill percentage of total
    for pg in param_groups:
        pg["pct_of_total"] = round(pg["total_numel"] / max(total_numel, 1) * 100, 1)

    total_memory = sum(pg["memory_bytes"] for pg in param_groups)

    return {
        "type": opt_type,
        "defaults": defaults,
        "param_groups": param_groups,
        "total_numel": total_numel,
        "total_memory_bytes": total_memory,
        "buffers_per_param": buffers_per_param,
    }
