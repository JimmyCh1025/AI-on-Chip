import torch
import torch.nn as nn
import torch.ao.quantization as tq


class VGG(nn.Module):
    def __init__(self, in_channels=3, in_size=32, num_classes=10) -> None:
        super().__init__()
        self.fmap_size = in_size
        self.conv1 = self.make_conv_layer(in_channels, 64, max_pool=True)
        self.conv2 = self.make_conv_layer(64, 192, max_pool=True)
        self.conv3 = self.make_conv_layer(192, 384, max_pool=False)
        self.conv4 = self.make_conv_layer(384, 256, max_pool=False)
        self.conv5 = self.make_conv_layer(256, 256, max_pool=True)

        self.fc6 = nn.Sequential(
            nn.Linear(256 * self.fmap_size**2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc8 = nn.Linear(128, num_classes)

        self._initialize_weights()

    def make_conv_layer(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        max_pool: bool = False,
    ) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if max_pool:
            self.fmap_size //= 2
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        print("Initializing weights")
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x

    def fuse_modules(self):
        tq.fuse_modules(
            self.eval(),
            [
                *[[f"conv{i}.0", f"conv{i}.1", f"conv{i}.2"] for i in range(1, 6)],
                *[[f"fc{i}.0", f"fc{i}.1"] for i in range(6, 8)],
            ],
            inplace=True,
        )


if __name__ == "__main__":
    model = VGG()
    inputs = torch.randn(1, 3, 32, 32)
    print(model)
    model.fuse_modules()
    print(model)
