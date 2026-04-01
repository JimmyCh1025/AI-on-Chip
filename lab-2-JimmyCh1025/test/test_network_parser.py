from dataclasses import asdict
from pathlib import Path
from types import ModuleType
import unittest
import sys

import onnx
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "src"))

from src.layer_info import Conv2DShapeParam, MaxPool2DShapeParam, LinearShapeParam
from test.utils import load_module_from_path


def load_student_module(base_dir: Path) -> ModuleType:
    return load_module_from_path(
        module_name="parser_student",
        module_path=base_dir / "src/network_parser/network_parser.py",
    )


def build_expected_layers() -> list[dict[str, int]]:
    layers = [
        Conv2DShapeParam(N=1, H=32, W=32, R=3, S=3, E=32, F=32, C=3, M=64, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        Conv2DShapeParam(N=1, H=16, W=16, R=3, S=3, E=16, F=16, C=64, M=192, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=192, M=384, U=1, P=1),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=384, M=256, U=1, P=1),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=256, M=256, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        LinearShapeParam(N=1, in_features=4096, out_features=256),
        LinearShapeParam(N=1, in_features=256, out_features=128),
        LinearShapeParam(N=1, in_features=128, out_features=10),
    ]
    return [asdict(i) for i in layers]


class TestNetworkParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (BASE_DIR / "src/lib/models").exists():
            raise unittest.SkipTest("Missing lib submodule models (src/lib/models)")

        from src.lib.models import VGG
        from src.network_parser import torch2onnx

        cls.module = load_student_module(BASE_DIR)
        cls.expected_layers = build_expected_layers()
        cls.torch_model = VGG(in_channels=3, in_size=32, num_classes=10)

        temp_dir = BASE_DIR / ".test_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = temp_dir / "vgg.onnx"
        dummy_input = torch.randn(1, 3, 32, 32)
        torch2onnx(cls.torch_model, onnx_path, dummy_input)
        cls.onnx_model = onnx.load(onnx_path)

    def test_parse_pytorch(self) -> None:
        layers = self.module.parse_pytorch(self.torch_model, input_shape=(1, 3, 32, 32))
        self.assertEqual([asdict(i) for i in layers], self.expected_layers)

    def test_parse_onnx(self) -> None:
        layers = self.module.parse_onnx(self.onnx_model)
        self.assertEqual([asdict(i) for i in layers], self.expected_layers)


if __name__ == "__main__":
    unittest.main()
