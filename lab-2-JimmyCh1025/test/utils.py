import importlib
import sys
from pathlib import Path

def load_module_from_path(
    module_path: str | Path, module_name: str | None = None
) -> object:
    module_path = Path(module_path)
    if not module_path.exists():
        raise FileNotFoundError(f"File {module_path} does not exist.")
    if module_name is None:
        module_name = module_path.stem

    src_dir = module_path.resolve().parents[1]
    root_dir = src_dir.parent
    sys.path.append(str(src_dir))
    sys.path.append(str(root_dir))

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
