"""Expand the per-job evaluation rows back into one row per (subset, method, k) cell.

Cells whose selection methods landed on the same pages share a trained model, so the
run script evaluates each unique job once and tags the row with the job id. This maps
those rows back onto all 90 cells.

    python 99_evaluation/scripts/expand_shot_selection.py
"""

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "99_evaluation" / "summaries" / "rq2"
MANIFEST = SUMMARY / "shot_selection_manifest.json"
CURVE = SUMMARY / "shot_selection_diva.csv"
OUT = SUMMARY / "shot_selection_diva_by_method.csv"
METRICS = ["Pixel_IU", "Line_IU", "DR", "RA", "FM"]


def main():
    manifest = json.loads(MANIFEST.read_text())
    if not CURVE.exists():
        raise SystemExit(f"missing {CURVE} -- nothing evaluated yet")
    scored = pd.read_csv(CURVE)
    # The run script writes the job id into the `method` column.
    by_job = {r["method"]: r for _, r in scored.iterrows()}

    rows, missing = [], 0
    for cell in manifest["cells"]:
        hit = by_job.get(cell["job"])
        if hit is None:
            missing += 1
            continue
        rows.append({"subset": cell["subset"], "method": cell["method"], "k": cell["k"],
                     "pages": cell["k"], "job": cell["job"],
                     **{m: round(float(hit[m]), 4) for m in METRICS if m in hit}})

    frame = pd.DataFrame(rows).sort_values(["subset", "method", "k"])
    frame.to_csv(OUT, index=False)
    print(f"{len(rows)} of {len(manifest['cells'])} cells filled"
          + (f", {missing} still unscored" if missing else ""))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
