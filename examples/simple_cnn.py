"""Demo: visualize a simple CNN with GradientHound hooks."""
import time

import torch
import torch.nn as nn

import gradienthound


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
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    gradienthound.init(metadata={
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 1,
        "optimizer": "Adam",
        "dataset": "synthetic",
    })

    model = SimpleCNN(num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    gradienthound.register_model("SimpleCNN", model)
    gradienthound.register_optimizer("Adam", optimizer)

    # Enable automatic gradient + weight capture via PyTorch hooks
    gradienthound.watch(model, name="SimpleCNN")

    example_input = torch.randn(1, 3, 32, 32)
    target = torch.randint(0, 10, (1,))
    criterion = nn.CrossEntropyLoss()

    print("Training... (use the standalone dashboard to inspect exported models/checkpoints)")
    try:
        for epoch in range(100):
            output = model(example_input)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Flush gradient stats + periodic weight snapshots to the dashboard
            gradienthound.step()

            print(f"  Epoch {epoch + 1}/100  loss={loss.item():.4f}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        gradienthound.shutdown()


if __name__ == "__main__":
    main()
