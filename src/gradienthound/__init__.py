from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .checkpoint import compute_tensor_stats, process_checkpoints
from .core import GradientHound
from .export import export_model

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim

__all__ = [
    "GradientHound",
    "init", "register_model", "register_optimizer",
    "capture_wandb", "shutdown",
    "watch", "step", "log_weights", "log_attention", "log_predictions",
    "export_model",
    "compute_tensor_stats", "process_checkpoints",
]
__version__ = "0.1.0"

_run: GradientHound | None = None


def _get_run() -> GradientHound | None:
    return _run


def init(
    ui: bool = True,
    port: int | None = None,
    metadata: dict | None = None,
) -> GradientHound:
    """Initialize GradientHound (wandb-style entry point)."""
    global _run
    if _run is not None:
        _run.shutdown()
    _run = GradientHound(ui=ui, port=port, metadata=metadata)
    return _run


def register_model(name: str, model: nn.Module) -> None:
    run = _get_run()
    if run is None:
        return
    run.register_model(name, model)


def register_optimizer(name: str, optimizer: optim.Optimizer) -> None:
    run = _get_run()
    if run is None:
        return
    run.register_optimizer(name, optimizer)


def capture_wandb() -> None:
    run = _get_run()
    if run is None:
        return
    run.capture_wandb()


def watch(
    model: nn.Module,
    name: str | None = None,
    *,
    log_gradients: bool = True,
    log_activations: bool = False,
    weight_every: int = 50,
) -> None:
    """Enable automatic gradient/activation capture via PyTorch hooks."""
    run = _get_run()
    if run is None:
        return
    run.watch(
        model, name,
        log_gradients=log_gradients,
        log_activations=log_activations,
        weight_every=weight_every,
    )


def step(step: int | None = None) -> None:
    """Flush buffered stats to IPC at the current or supplied training step."""
    run = _get_run()
    if run is None:
        return
    run.step(step=step)


def log_weights(name: str | None = None) -> None:
    """Force an immediate weight snapshot."""
    run = _get_run()
    if run is None:
        return
    run.log_weights(name)


def log_attention(name: str, weights: torch.Tensor) -> None:
    """Log attention weight matrix for visualization."""
    run = _get_run()
    if run is None:
        return
    run.log_attention(name, weights)


def log_predictions(
    predicted: Any,
    actual: Any,
    name: str = "default",
) -> None:
    """Log prediction vs actual for calibration scatter plot."""
    run = _get_run()
    if run is None:
        return
    run.log_predictions(predicted, actual, name)


def shutdown() -> None:
    global _run
    if _run is not None:
        _run.shutdown()
        _run = None
