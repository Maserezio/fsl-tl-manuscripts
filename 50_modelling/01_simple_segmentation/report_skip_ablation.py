"""Assemble the skip-connection ablation table (run_skip_ablation.sh).

Runs both evaluations for the six checkpoints and joins them into one
backbone x n_skips table:

  Pixel IU / Line IU / FM   -- official DIVA Java evaluator, via evaluate_lines.py
                               (same PAGE-XML export + post-processing preset as
                               the backbone matrix, so the n_skips=4 row is
                               directly comparable to diva_backbone_matrix.csv)
  small_cc_recall           -- eval_small_components.py: fraction of GT connected
                               components under 200 px recovered by the raw mask

Usage:
  python report_skip_ablation.py            # run both evals, then print + save
  python report_skip_ablation.py --no-run   # just re-join the existing CSVs
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_DIR = REPO / "99_evaluation" / "01_simple_segmentation"

BACKBONES = ["resnet34", "dinov2"]
SKIPS = [4, 2, 0]
SUBSET = "CB55"

RUNS = [f"skipabl_{enc}_diva_{SUBSET}_s{n}" for enc in BACKBONES for n in SKIPS]
LINES_CSV = OUT_DIR / "skip_ablation_lines.csv"
SMALL_CSV = OUT_DIR / "skip_ablation_small_cc.csv"
FINAL_CSV = OUT_DIR / "skip_ablation_CB55.csv"

_RUN_RE = re.compile(r"^skipabl_(?P<enc>.+)_diva_(?P<subset>[^_]+)_s(?P<n>\d)$")


def parse_run(run: str) -> tuple[str, int]:
    m = _RUN_RE.match(run)
    if not m:
        raise ValueError(f"unexpected run name {run!r}")
    return m["enc"], int(m["n"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-run", action="store_true", help="reuse existing per-eval CSVs")
    args = parser.parse_args()

    if not args.no_run:
        subprocess.run([sys.executable, str(HERE / "evaluate_lines.py"),
                        "--family", "diva", "--subsets", SUBSET,
                        "--runs", *RUNS, "--out", str(LINES_CSV)], check=True)
        subprocess.run([sys.executable, str(HERE / "eval_small_components.py"),
                        "--runs", *RUNS, "--out", str(SMALL_CSV)], check=True)

    lines = pd.read_csv(LINES_CSV).rename(columns={"encoder": "run"})
    small = pd.read_csv(SMALL_CSV)

    df = lines.merge(small, on="run", how="outer", suffixes=("", "_cc"))
    df[["backbone", "n_skips"]] = df["run"].apply(lambda r: pd.Series(parse_run(r)))
    df["arm"] = df["backbone"].map({"resnet34": "CNN (native pyramid)",
                                    "dinov2": "SFP-ViT (flat tokens)"}).fillna(df["backbone"])

    cols = ["arm", "backbone", "n_skips", "Pixel_IU", "Line_IU", "FM", "DR", "RA",
            "small_cc_recall", "small_cc_pixel_recall", "small_components",
            "large_cc_recall"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(["backbone", "n_skips"], ascending=[False, False]).reset_index(drop=True)

    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FINAL_CSV, index=False)

    print("\n" + "=" * 110)
    print(f"SKIP-CONNECTION ABLATION -- DIVA-HisDB {SUBSET}, full-page semantic segmentation")
    print("=" * 110)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Relative degradation per arm: the actual quantity the hypothesis is about.
    print("\nDrop relative to the arm's own 4-skip baseline (percentage points):")
    for backbone, group in df.groupby("backbone", sort=False):
        base = group[group["n_skips"] == 4]
        if base.empty:
            continue
        base = base.iloc[0]
        for _, row in group[group["n_skips"] != 4].iterrows():
            deltas = "  ".join(
                f"{k}={100 * (row[k] - base[k]):+6.2f}"
                for k in ("Pixel_IU", "Line_IU", "FM", "small_cc_recall")
                if k in row and pd.notna(row[k]) and pd.notna(base[k])
            )
            print(f"  {backbone:10s} {int(base['n_skips'])}->{int(row['n_skips'])} skips:  {deltas}")

    print(f"\nsaved -> {FINAL_CSV}")


if __name__ == "__main__":
    main()
