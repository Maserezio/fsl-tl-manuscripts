"""Put the center-line and analysis code folders on sys.path (both import each other's modules)."""
import sys
from pathlib import Path

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "50_modelling").is_dir() and (p / "99_evaluation").is_dir())
for _d in ("50_modelling/semantic_segmentation/center_line", "99_evaluation/analysis", "50_modelling/common"):
    if str(_REPO / _d) not in sys.path:
        sys.path.insert(1, str(_REPO / _d))
