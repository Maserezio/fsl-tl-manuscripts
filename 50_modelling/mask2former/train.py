"""Root-level entry point — delegates to scripts/train.py."""
import runpy
import sys
from pathlib import Path

sys.argv[0] = str(Path(__file__).parent / "scripts" / "train.py")
runpy.run_path(str(Path(__file__).parent / "scripts" / "train.py"), run_name="__main__")
