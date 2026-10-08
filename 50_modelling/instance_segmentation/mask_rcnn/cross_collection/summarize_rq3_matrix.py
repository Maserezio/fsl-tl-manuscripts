#!/usr/bin/env python3
"""Collect RQ3 protocols A and B from the per-run summary.json files.

Writes, for each DIVA metric, the A matrix (rows = training collection) and the B matrix
(rows = held-out collection of the LOCO model), both with columns = test collection and
two row means: over all seven columns and over the transfer cells only (A: off-diagonal,
B: the held-out column alone is the transfer cell, so its mean is over the six seen ones).

    python 50_modelling/instance_segmentation/mask_rcnn/cross_collection/summarize_rq3_matrix.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
EVAL = REPO / "99_evaluation/instance_segmentation/mask_rcnn/cross_collection"
OUT = EVAL / "rq3_matrix"
DATASETS = ["Pinkas", "ONB", "RASAM", "RASM", "Phil_gr_130", "GRPOLY", "NorHand_v3"]
METRICS = ["PixelIU", "LinesIU", "LinesRecall", "LinesPrecision", "LinesFMeasure"]
STEM = "maskrcnn_convnext_tiny_catmus_1152"


def load(tag: str):
    path = EVAL / tag / "summary.json"
    return json.loads(path.read_text())["test"] if path.exists() else None


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = []
    for row in DATASETS:
        for col in DATASETS:
            a = f"{col}_{STEM}_k3_matrix" + ("" if row == col else f"_from_{row}")
            b = f"{col}_{STEM}_k6_loco" if row == col else f"{col}_{STEM}_k3_loco_from_no{row}"
            for protocol, tag in (("A", a), ("B", b)):
                test = load(tag)
                if test is not None:
                    rows.append({"protocol": protocol, "row": row, "target": col, **test})
    long = pd.DataFrame(rows)
    long.to_csv(OUT / "rq3_matrix_long.csv", index=False)

    # LOCO page-budget variants (run_rq3_loco_budget.sh): held-out test only, three seeds.
    for variant, k in (("loco3", 3), ("loco10", 10)):
        runs = [{"holdout": d, "seed": seed, **test} for d in DATASETS for seed in (42, 43, 44)
                if (test := load(f"{d}_{STEM}_k{k}_{variant}_s{seed}")) is not None]
        if not runs:
            continue
        runs = pd.DataFrame(runs)
        runs.to_csv(OUT / f"{variant}_long.csv", index=False)
        agg = runs.groupby("holdout")[METRICS].agg(["mean", "std", "count"]).reindex(DATASETS)
        agg.columns = [f"{m}_{s}" for m, s in agg.columns]
        agg.to_csv(OUT / f"{variant}_summary.csv", float_format="%.4f")
        print(f"\n{variant} LinesFMeasure (mean, sample std over seeds)")
        print(agg[["LinesFMeasure_mean", "LinesFMeasure_std", "LinesFMeasure_count"]].round(4).to_string())

    for metric in METRICS:
        for protocol in ("A", "B"):
            m = (long[long.protocol == protocol]
                 .pivot(index="row", columns="target", values=metric)
                 .reindex(index=DATASETS, columns=DATASETS))
            diag = np.eye(len(DATASETS), dtype=bool)
            m["mean_all"] = m[DATASETS].mean(axis=1)
            if protocol == "A":
                m["mean_transfer"] = m[DATASETS].where(~diag).mean(axis=1)
            else:
                m["mean_seen"] = m[DATASETS].where(~diag).mean(axis=1)
            m.index.name = "train" if protocol == "A" else "holdout"
            m.to_csv(OUT / f"{protocol}_{metric}.csv", float_format="%.4f")
            if metric == "LinesFMeasure":
                print(f"\n{protocol} {metric} ({m[DATASETS].notna().sum().sum()}/49 cells)")
                print(m.round(3).to_string())


if __name__ == "__main__":
    main()
