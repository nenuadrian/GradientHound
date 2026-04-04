from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn

from .graph_extractor import extract_graph
from .checkpoint import write_checkpoint
from .types import ModelGraph


class GradientHoundCollector:
    """Collects model architecture and training data for GradientHound visualization.

    Usage:
        model = MyModel()
        collector = GradientHoundCollector(model, example_input=torch.randn(1, 3, 224, 224))
        # ... training loop ...
        collector.save("checkpoints/step_1000.ghound", step=1000, epoch=5, loss=0.123)
    """

    def __init__(
        self,
        model: nn.Module,
        example_input: torch.Tensor | tuple[torch.Tensor, ...],
        model_name: str | None = None,
    ):
        self.model = model
        self.graph: ModelGraph = extract_graph(model, example_input, model_name)

    def save(
        self,
        path: str | Path,
        optimizer: (
            torch.optim.Optimizer
            | Iterable[torch.optim.Optimizer]
            | Mapping[str, torch.optim.Optimizer]
            | None
        ) = None,
        step: int | None = None,
        epoch: int | None = None,
        loss: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save a .ghound checkpoint.

        Args:
            path: Output file path
            optimizer: Optional optimizer(s) to include state. Accepts a single
                optimizer, an iterable of optimizers, or a dict of named optimizers.
            step: Training step number
            epoch: Epoch number
            loss: Current loss value
            metadata: Any additional metadata
        """
        return write_checkpoint(
            path=path,
            graph=self.graph,
            model=self.model,
            optimizer=optimizer,
            metadata=metadata,
            step=step,
            epoch=epoch,
            loss=loss,
        )
