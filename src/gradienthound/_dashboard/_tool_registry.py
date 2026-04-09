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
    category: str  # "analysis", "integration"
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
    has_checkpoints: bool = False,
    ckpt_state: dict | None = None,
    wandb_state: dict | None = None,
    model_data=None,
) -> None:
    """Populate the registry with GradientHound's built-in tools."""

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
            wandb_state is not None and wandb_state.get("data") is not None
        ),
        page="/metrics",
    ))
