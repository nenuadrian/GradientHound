"""Generate a .ghound checkpoint from a simple CNN model."""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add collector to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "collector" / "src"))

from gradienthound_collector import GradientHoundCollector


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    model = SimpleCNN(num_classes=10)
    example_input = torch.randn(1, 3, 32, 32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Simulate a training step
    output = model(example_input)
    loss = output.sum()
    loss.backward()
    optimizer.step()

    collector = GradientHoundCollector(
        model=model,
        example_input=example_input,
        model_name="SimpleCNN",
    )

    out_dir = Path(__file__).parent.parent / "checkpoints"
    out_path = out_dir / "simple_cnn.ghound"

    collector.save(
        path=out_path,
        optimizer=optimizer,
        step=1,
        epoch=1,
        loss=loss.item(),
    )
    print(f"Checkpoint saved to {out_path}")
    print(f"Graph has {len(collector.graph.nodes)} nodes and {len(collector.graph.edges)} edges")


if __name__ == "__main__":
    main()
