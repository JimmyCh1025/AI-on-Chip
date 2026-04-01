import torch
from torch import nn
import torch.ao.quantization as tq


class MLP3(nn.Module):
    def __init__(self, in_channels=3, in_size=32, num_classes=10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels * in_size**2, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    def fuse_modules(self):
        tq.fuse_modules(self.eval(), [["fc1", "relu1"], ["fc2", "relu2"]], inplace=True)


if __name__ == "__main__":
    model = MLP3(in_channels=1, in_size=28)
    inputs = torch.randn(1, 1, 28, 28)
    print(model)
    print(f"{inputs.shape} -> {model(inputs).shape}")
    model.fuse_modules()
    print(model)
