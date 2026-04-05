from __future__ import annotations

import torch.nn as nn

from gradienthound.graph import extract_model_graph, render_graphviz


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 16),
            nn.Dropout(0.1),
            nn.Linear(16, 4),
        )


def test_render_graphviz_uses_clusters_and_card_labels() -> None:
    graph = extract_model_graph("TinyNet", TinyNet())
    source = render_graphviz(graph).source

    assert "subgraph cluster_" in source
    assert "features | Sequential" in source
    assert "classifier | Sequential" in source
    assert "<<TABLE BORDER=" in source
    assert "tooltip=" in source
