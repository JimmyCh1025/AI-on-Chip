import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, in_channels=3, in_size=32, num_classes=10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        num_features = int(16 * (((in_size - 4) / 2 - 4) / 2) ** 2)
        self.fc3 = nn.Linear(num_features, 120)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        return x


if __name__ == "__main__":
    model = LeNet5(in_channels=1, in_size=28)
    inputs = torch.randn(1, 1, 28, 28)
    print(model)
    print(f"{inputs.shape} -> {model(inputs).shape}")
