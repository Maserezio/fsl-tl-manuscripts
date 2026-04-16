import sys
from pathlib import Path


def setup_project_imports():
    project_root = Path(__file__).resolve().parents[2]
    repo_root = project_root.parent
    vendored_detectron2 = repo_root / "50_modelling" / "detectron2"

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if vendored_detectron2.exists() and str(vendored_detectron2) not in sys.path:
        sys.path.append(str(vendored_detectron2))
    return project_root, repo_root


def collect_image_paths(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return sorted(path for path in input_path.iterdir() if path.suffix.lower() in extensions)