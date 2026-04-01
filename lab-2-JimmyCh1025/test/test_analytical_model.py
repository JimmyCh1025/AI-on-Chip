from types import ModuleType
from pathlib import Path
import unittest
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "src"))

from src.layer_info import Conv2DShapeParam, MaxPool2DShapeParam
from test.utils import load_module_from_path


def load_student_module(base_dir: Path) -> ModuleType:
    module_path = base_dir / "src/analytical_model/eyeriss.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find student file: {module_path}")
    return load_module_from_path(module_name="analyzer_student", module_path=module_path)


def create_analyzer(module: ModuleType):
    analyzer = module.EyerissAnalyzer()
    analyzer.hardware = module.EyerissHardwareParam(
        pe_array_h=6,
        pe_array_w=8,
        ifmap_spad_size=12,
        filter_spad_size=48,
        psum_spad_size=16,
        glb_size=64 << 10,
        bus_bw=4,
        noc_bw=4,
    )
    analyzer.mapping = module.EyerissMappingParam(m=16, n=1, e=8, p=4, q=4, r=1, t=2)
    analyzer._conv_shape = Conv2DShapeParam(
        N=1, H=32, W=32, R=3, S=3, E=32, F=32, C=3, M=64, U=1, P=1
    )
    analyzer._maxpool_shape = MaxPool2DShapeParam(N=1, kernel_size=2, stride=2)
    return analyzer


class TestAnalyticalModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module = load_student_module(BASE_DIR)
        cls.analyzer = create_analyzer(module)

    def test_glb_usage_per_pass(self):
        self.assertEqual(self.analyzer.glb_usage_per_pass["total"], 17984)

    def test_dram_access_per_layer(self):
        self.assertEqual(self.analyzer.dram_access_per_layer["total"], 47104)

    def test_glb_access_per_layer(self):
        self.assertEqual(self.analyzer.glb_access_per_layer["total"], 591872)


if __name__ == "__main__":
    unittest.main()
