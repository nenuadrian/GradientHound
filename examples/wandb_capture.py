"""Demo: capture wandb metrics and display them in the GradientHound UI."""
import time

import torch
import torch.nn as nn
import wandb

import gradienthound


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    wandb.init(project="gradienthound-demo", mode="offline")

    gradienthound.init(metadata={"project": "gradienthound-demo"})
    gradienthound.capture_wandb()

    model = SimpleCNN()
    gradienthound.register_model("SimpleCNN", model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    gradienthound.register_optimizer("SGD", optimizer)

    example_input = torch.randn(4, 3, 32, 32)
    target = torch.randint(0, 10, (4,))
    criterion = nn.CrossEntropyLoss()

    print("Training... (open the GradientHound UI to see live metrics)")
    try:
        for epoch in range(200):
            optimizer.zero_grad()
            output = model(example_input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            acc = (output.argmax(1) == target).float().mean().item()

            # This goes to wandb AND to GradientHound automatically
            wandb.log({"loss": loss.item(), "accuracy": acc})

            print(f"  Epoch {epoch + 1}  loss={loss.item():.4f}  acc={acc:.2f}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        gradienthound.shutdown()
        wandb.finish()


if __name__ == "__main__":
    main()
