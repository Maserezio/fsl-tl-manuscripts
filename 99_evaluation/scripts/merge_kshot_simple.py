"""Fold the one-stage k-shot rows into the shared curve CSV.

evaluate_lines.py scores the whole sweep in one call and writes rows keyed by run
folder name; the curve needs them keyed by k, next to the two-stage rows that
predict_and_eval_rtdetr_diva.py appends per run. This script does that join and
nothing else.

    K_VALUES="1 3 5 10 15" SUBSET=CB55 \
        CURVE_CSV=99_evaluation/summaries/rq2/kshot_cb55.csv \
        python 99_evaluation/scripts/merge_kshot_simple.py
"""

import os
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
METRICS = ["Pixel_IU", "Line_IU", "DR", "RA", "FM"]

SUBSET = os.environ.get("SUBSET", "CB55")
CURVE_CSV = REPO / os.environ.get(
    "CURVE_CSV", "99_evaluation/summaries/rq2/kshot_cb55.csv"
)
RAW = REPO / "99_evaluation" / "summaries" / "rq2" / "kshot_cb55_simple_raw.csv"
K_VALUES = [int(k) for k in os.environ.get("K_VALUES", "1 3 5 10 15").split()]


def main():
    if not RAW.exists():
        raise SystemExit(f"missing {RAW} -- run evaluate_lines.py first")
    raw = pd.read_csv(RAW)

    rows = []
    for _, r in raw.iterrows():
        # evaluate_lines.py puts the run folder name in `encoder` when --runs is used.
        match = re.search(r"_k(\d+)$", str(r["encoder"]))
        if not match:
            continue
        k = int(match.group(1))
        if k not in K_VALUES:
            continue
        rows.append({"approach": "simple_seg", "k": k, "pages": k, "subset": SUBSET,
                     "run": r["encoder"],
                     **{m: round(float(r[m]), 4) for m in METRICS if m in r}})
    if not rows:
        raise SystemExit(f"no rows in {RAW} matched a _k<N> suffix for k in {K_VALUES}")

    frame = pd.DataFrame(rows)
    if CURVE_CSV.exists():
        frame = pd.concat([pd.read_csv(CURVE_CSV), frame], ignore_index=True)
    frame = (frame.drop_duplicates(["approach", "k", "subset"], keep="last")
             .sort_values(["approach", "k"]))
    CURVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CURVE_CSV, index=False)
    print(f"{len(rows)} one-stage row(s) merged -> {CURVE_CSV}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
