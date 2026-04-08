"""Tool registry for tracking available analysis tools and their status."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Requirement:
    """A single requirement for a tool to function."""

    label: str
    check: Callable[[], bool]

    def satisfied(self) -> bool:
        try:
            return self.check()
        except Exception:
            return False


@dataclass
class ToolInfo:
    """Descriptor for a registered tool."""

    id: str
    name: str
    description: str
    category: str  # "capture", "analysis", "integration", "on-demand"
    requires: list[Requirement] = field(default_factory=list)
    check_has_data: Callable[[], bool] = lambda: False
    page: str | None = None  # link to relevant dashboard page

    def available(self) -> bool:
        """All requirements satisfied."""
        return all(r.satisfied() for r in self.requires)

    def status(self) -> dict:
        reqs = [
            {"label": r.label, "met": r.satisfied()}
            for r in self.requires
        ]
        has_data = False
        try:
            has_data = self.check_has_data()
        except Exception:
            pass

        avail = all(r["met"] for r in reqs)
        if has_data:
            state = "has_data"
        elif avail:
            state = "ready"
        else:
            state = "unavailable"

        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "requirements": reqs,
            "state": state,
            "has_data": has_data,
            "available": avail,
            "page": self.page,
        }


class ToolRegistry:
    """Central registry of all tools (built-in and 3rd-party)."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolInfo] = {}

    def register(self, tool: ToolInfo) -> None:
        self._tools[tool.id] = tool

    def unregister(self, tool_id: str) -> None:
        self._tools.pop(tool_id, None)

    def get(self, tool_id: str) -> ToolInfo | None:
        return self._tools.get(tool_id)

    def all_status(self) -> list[dict]:
        return [t.status() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())


def _check_package(name: str) -> bool:
    """Check if a Python package is importable."""
    import importlib
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def register_builtin_tools(
    registry: ToolRegistry,
    *,
    ipc=None,
    has_checkpoints: bool = False,
    ckpt_state: dict | None = None,
    wandb_state: dict | None = None,
    model_data=None,
) -> None:
    """Populate the registry with GradientHound's built-in tools."""

    # ── Capture tools ──────────────────────────────────────────────

    registry.register(ToolInfo(
        id="gradient_capture",
        name="Gradient Capture",
        description="Automatic per-layer gradient statistics via PyTorch hooks. "
                    "Records norm, mean, std, dead percentage, and cosine similarity.",
        category="capture",
        requires=[
            Requirement("PyTorch installed", lambda: _check_package("torch")),
            Requirement("IPC channel active", lambda: ipc is not None),
        ],
        check_has_data=lambda: (
            ipc is not None and bool(ipc.read_gradient_stats(last_n=1))
        ),
        page="/gradient-flow",
    ))

    registry.register(ToolInfo(
        id="weight_capture",
        name="Weight Tracking",
        description="Periodic weight statistics capture (norm, mean, std, "
                    "min, max, sparsity) logged every N training steps.",
        category="capture",
        requires=[
            Requirement("PyTorch installed", lambda: _check_package("torch")),
            Requirement("IPC channel active", lambda: ipc is not None),
        ],
        check_has_data=lambda: (
            ipc is not None and bool(ipc.read_weight_stats(last_n=1))
        ),
        page="/weight-health",
    ))

    registry.register(ToolInfo(
        id="activation_capture",
        name="Activation Capture",
        description="Forward-hook based activation statistics "
                    "(mean, std, min, max, zero fraction) per layer.",
        category="capture",
        requires=[
            Requirement("PyTorch installed", lambda: _check_package("torch")),
            Requirement("IPC channel active", lambda: ipc is not None),
        ],
        check_has_data=lambda: (
            ipc is not None and bool(ipc.read_activation_stats(last_n=1))
        ),
        page="/weight-health",
    ))

    registry.register(ToolInfo(
        id="optimizer_tracking",
        name="Optimizer State Tracking",
        description="Live optimizer buffer statistics — momentum norms, "
                    "adaptive learning-rate estimates, warmup progress.",
        category="capture",
        requires=[
            Requirement("PyTorch installed", lambda: _check_package("torch")),
            Requirement("IPC channel active", lambda: ipc is not None),
        ],
        check_has_data=lambda: (
            ipc is not None and bool(ipc.read_optimizer_state(last_n=1))
        ),
        page="/weight-health",
    ))

    registry.register(ToolInfo(
        id="attention_logging",
        name="Attention Logging",
        description="Manual attention-weight capture for transformer models. "
                    "Visualises head-level attention matrices.",
        category="capture",
        requires=[
            Requirement("PyTorch installed", lambda: _check_package("torch")),
            Requirement("IPC channel active", lambda: ipc is not None),
        ],
        check_has_data=lambda: (
            ipc is not None and bool(ipc.read_attention(last_n=1))
        ),
        page="/",
    ))

    registry.register(ToolInfo(
        id="prediction_logging",
        name="Prediction Logging",
        description="Log predicted vs actual values for calibration analysis. "
                    "Produces scatter plots and residual distributions.",
        category="capture",
        requires=[
            Requirement("IPC channel active", lambda: ipc is not None),
        ],
        check_has_data=lambda: (
            ipc is not None and bool(ipc.read_predictions(last_n=1))
        ),
        page="/",
    ))

    # ── On-demand tools ────────────────────────────────────────────

    registry.register(ToolInfo(
        id="weight_heatmap",
        name="Weight Heatmap",
        description="On-demand 2D heatmap of a layer's weight tensor. "
                    "Requested from the dashboard, computed in the training process.",
        category="on-demand",
        requires=[
            Requirement("IPC channel active", lambda: ipc is not None),
            Requirement("Model registered", lambda: ipc is not None and bool(ipc.read_models())),
        ],
        check_has_data=lambda: False,  # ephemeral request/response
        page="/on-demand",
    ))

    registry.register(ToolInfo(
        id="cka_similarity",
        name="CKA Similarity",
        description="Centered Kernel Alignment between layer activations. "
                    "Measures representational similarity across layers.",
        category="on-demand",
        requires=[
            Requirement("IPC channel active", lambda: ipc is not None),
            Requirement("Model registered", lambda: ipc is not None and bool(ipc.read_models())),
        ],
        check_has_data=lambda: False,
        page="/on-demand",
    ))

    registry.register(ToolInfo(
        id="network_state",
        name="Network State Dump",
        description="Full snapshot of current parameter and buffer values "
                    "for offline inspection.",
        category="on-demand",
        requires=[
            Requirement("IPC channel active", lambda: ipc is not None),
            Requirement("Model registered", lambda: ipc is not None and bool(ipc.read_models())),
        ],
        check_has_data=lambda: False,
        page="/on-demand",
    ))

    # ── Analysis tools ─────────────────────────────────────────────

    registry.register(ToolInfo(
        id="checkpoint_comparison",
        name="Checkpoint Comparison",
        description="Compare weight distributions, norms, and layer-level "
                    "changes across saved checkpoints.",
        category="analysis",
        requires=[
            Requirement("Checkpoint paths provided", lambda: has_checkpoints),
        ],
        check_has_data=lambda: (
            ckpt_state is not None and ckpt_state.get("processed", False)
        ),
        page="/checkpoints",
    ))

    registry.register(ToolInfo(
        id="weightwatcher",
        name="WeightWatcher / Spectral Analysis",
        description="Power-law fits and effective rank computation on weight "
                    "matrices. Detects over-parameterisation and collapse.",
        category="analysis",
        requires=[
            Requirement("Checkpoint paths provided", lambda: has_checkpoints),
            Requirement("powerlaw package installed", lambda: _check_package("powerlaw")),
        ],
        check_has_data=lambda: (
            ckpt_state is not None and ckpt_state.get("processed", False)
        ),
        page="/spectral",
    ))

    # ── Integration tools ──────────────────────────────────────────

    registry.register(ToolInfo(
        id="wandb_integration",
        name="Weights & Biases",
        description="Capture wandb.log() calls and display run metrics "
                    "as live time-series charts inside GradientHound.",
        category="integration",
        requires=[
            Requirement("wandb package installed", lambda: _check_package("wandb")),
        ],
        check_has_data=lambda: (
            (ipc is not None and bool(ipc.read_metrics(last_n=1)))
            or (wandb_state is not None and wandb_state.get("data") is not None)
        ),
        page="/metrics",
    ))
