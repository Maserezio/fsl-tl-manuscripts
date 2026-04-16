"""Root-level entry point — delegates to scripts/evaluate.py."""
import runpy
import sys
from pathlib import Path

sys.argv[0] = str(Path(__file__).parent / "scripts" / "evaluate.py")
runpy.run_path(str(Path(__file__).parent / "scripts" / "evaluate.py"), run_name="__main__")
