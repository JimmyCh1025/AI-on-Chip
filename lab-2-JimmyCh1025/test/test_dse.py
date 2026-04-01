from dataclasses import asdict
from pathlib import Path
from types import ModuleType
import unittest
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "src"))

from src.analytical_model import EyerissAnalyzer, EyerissHardwareParam
from src.layer_info import Conv2DShapeParam, MaxPool2DShapeParam
from test.utils import load_module_from_path


def load_student_module(base_dir: Path) -> ModuleType:
    return load_module_from_path(
        module_name="mapper_student",
        module_path=base_dir / "src/analytical_model/mapper.py",
    )


def create_mapper(module):
    mapper = module.EyerissMapper(name="test_mapper")
    mapper.analyzer = EyerissAnalyzer(name="test_analyzer")
    mapper.analyzer.hardware = EyerissHardwareParam(
        pe_array_h=6,
        pe_array_w=8,
        ifmap_spad_size=12,
        filter_spad_size=48,
        psum_spad_size=16,
        glb_size=64 << 10,
        bus_bw=4,
        noc_bw=4,
    )
    mapper.analyzer._conv_shape = Conv2DShapeParam(
        N=1, H=32, W=32, R=3, S=3, E=32, F=32, C=3, M=64, U=1, P=1
    )
    mapper.analyzer._maxpool_shape = MaxPool2DShapeParam(N=1, kernel_size=2, stride=2)
    return mapper


class TestDSE(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module = load_student_module(BASE_DIR)
        cls.mapper = create_mapper(module)
        cls.mapper.mode = "none"
        cls.mappings = cls.mapper.generate_mappings()
        cls.has_mapping = len(cls.mappings) > 0
        cls.all_valid = cls.has_mapping and all(
            map(lambda x: cls.mapper.validate(asdict(x).values()), cls.mappings)
        )

    def test_generate_mappings_has_mapping(self):
        self.assertTrue(self.has_mapping)

    def test_generate_mappings_all_valid(self):
        self.assertTrue(self.all_valid)


if __name__ == "__main__":
    unittest.main()
