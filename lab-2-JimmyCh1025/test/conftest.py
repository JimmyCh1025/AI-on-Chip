import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

TEMP_DIR = ROOT_DIR / ".test_temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
