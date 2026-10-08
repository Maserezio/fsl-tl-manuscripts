#!/usr/bin/env python3
"""Rerun RQ3 experiments A (7x7 transfer matrix) and B (LOCO-6) with the final "ab" setup.

"ab" = # FIXED: constants of the robustness ablation set to: keep the CATMuS-pretrained RPN
(RQ3_REPLACE_ANCHORS=0), train-only scale jitter (RQ3_SCALE_JITTER=1) with enlargement bound
JITTER_MAX, everything else as in the base runs (mask ROI 14, no freezing, last of 50 epochs,
source-validation grid for score/mask, box NMS 0.5, 300 detections, unchanged 1152 eval fit).
Post-processing is inference-only: every model is evaluated without it and with all rules on,
with thresholds re-selected on the source validation pages in each case.

JITTER_MAX is 2.0 if ab_j2 of the seed-42 ablation has a higher mean source-validation FM
than ab over its five sources and loses at most 3 FM on ONB->ONB; otherwise 1.6.
NorHand v3 is trained on draw d1 (00_data/RQ3/norhand_draws/d1); base is retrained on d1 once
so that the NorHand row of base and ab is comparable. LOCO keeps the original LOCO-6 pages.
Seed 42. Outputs: 99_evaluation/instance_segmentation/mask_rcnn/cross_collection/final_ab/{decision.json,A.csv,B.csv};
final checkpoints: 80_models/instance_segmentation/mask_rcnn/cross_collection/final_ab/<tag>.pt

    .venv/bin/python 50_modelling/instance_segmentation/mask_rcnn/cross_collection/run_rq3_final_ab.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PY = REPO / ".venv/bin/python"
EVAL = REPO / "99_evaluation/instance_segmentation/mask_rcnn/cross_collection"
MODELS = REPO / "80_models/instance_segmentation/mask_rcnn/cross_collection"
OUT = EVAL / "final_ab"
CKPT = MODELS / "final_ab"
STEM = "maskrcnn_convnext_tiny_catmus_1152"
DATASETS = ["Pinkas", "ONB", "RASAM", "RASM", "Phil_gr_130", "GRPOLY", "NorHand_v3"]
ABL_SOURCES = ["ONB", "RASAM", "Phil_gr_130", "RASM", "GRPOLY"]
SEED = 42
COMMON = ["--arm", "convnext_tiny_catmus", "--image-size", "1152", "--epochs", "50",
          "--fast-grid", "--clean-predictions", "--seed", str(SEED)]
# NorHand is trained on draw d1; the link keeps the root name NorHand_v3 for tags.
NORHAND_D1 = REPO / "00_data/RQ3/final_roots/NorHand_v3"


def choose_jitter() -> tuple[float, dict]:
    def val(cond, src):
        return json.loads((EVAL / f"{src}_{STEM}_k3_abl_{cond}_s42/summary.json").read_text())
    ab = [val("ab", s)["validation"]["LinesFMeasure"] for s in ABL_SOURCES]
    j2 = [val("ab_j2", s)["validation"]["LinesFMeasure"] for s in ABL_SOURCES]
    onb_ab = val("ab", "ONB")["test"]["LinesFMeasure"]
    onb_j2 = val("ab_j2", "ONB")["test"]["LinesFMeasure"]
    higher = float(np.mean(j2)) > float(np.mean(ab))
    no_drop = (onb_j2 - onb_ab) >= -0.03
    jitter = 2.0 if higher and no_drop else 1.6
    info = {"jitter_max": jitter, "rule": "2.0 if mean source-val FM(ab_j2) > mean(ab) and "
            "ONB->ONB FM(ab_j2) - FM(ab) >= -3 points, else 1.6",
            "source_val_fm_ab": dict(zip(ABL_SOURCES, ab)), "source_val_fm_ab_j2": dict(zip(ABL_SOURCES, j2)),
            "mean_ab": float(np.mean(ab)), "mean_ab_j2": float(np.mean(j2)),
            "onb_onb_ab": onb_ab, "onb_onb_ab_j2": onb_j2, "higher_val": higher, "no_onb_drop": no_drop}
    return jitter, info


def env(model: str, jitter: float, post: bool) -> dict:
    e = {k: v for k, v in os.environ.items() if not k.startswith("RQ3_")}
    e["DIVA_WORKERS"] = "10"
    if model == "ab":
        e.update(RQ3_REPLACE_ANCHORS="0", RQ3_SCALE_JITTER="1", RQ3_JITTER_UP_MAX=str(jitter))
    if post:
        e["RQ3_POST"] = "1"
    return e


def run(args: list[str], e: dict, tag: str) -> dict:
    if not (EVAL / tag / "summary.json").exists():
        print(f"  run {tag}", flush=True)
        with (EVAL / f"{tag}.log").open("w") as fh:
            subprocess.run([str(PY), "maskrcnn_rq3_loo.py", *args], cwd=HERE, env=e,
                           stdout=fh, stderr=subprocess.STDOUT, check=True)
    return json.loads((EVAL / tag / "summary.json").read_text())


def keep_checkpoint(tag: str) -> Path:
    """Move the final checkpoint of a training run to final_ab/ and drop the run folder."""
    dst = CKPT / f"{tag}.pt"
    src = MODELS / tag / "best.pt"
    if src.exists():
        shutil.move(src, dst)
        shutil.rmtree(MODELS / tag, ignore_errors=True)
    return dst


def row(exp, model, post, source, target, own, test):
    return {"experiment": exp, "model": model, "post": post, "source": source, "target": target,
            "FM": test["LinesFMeasure"], "recall": test["LinesRecall"],
            "precision": test["LinesPrecision"], "PixelIU": test["PixelIU"], "LinesIU": test["LinesIU"],
            "source_val_FM": (own.get("validation") or {}).get("LinesFMeasure"),
            "score_threshold": own["score_threshold"], "mask_threshold": own["mask_threshold"]}


def source_block(exp, model, jitter, source, root, targets, rows, reuse=None):
    """Train (or reuse) one model and evaluate it without and with post-processing."""
    label = f"final_{model}" + (f"_j{jitter:g}" if model == "ab" else "") + ("_d1" if exp == "A" and source == "NorHand_v3" else "")
    tag = f"{source}_{STEM}_k{'6' if exp == 'B' else '3'}_{label}"
    ckpt = CKPT / f"{tag}.pt"
    if reuse is not None and not ckpt.exists():
        os.link(reuse, ckpt)
        print(f"  reuse {reuse.name} for {tag}", flush=True)
    for post in (False, True):
        suffix = label + ("_post" if post else "")
        stag = f"{root.name}_{STEM}_k{'6' if exp == 'B' else '3'}_{suffix}"
        args = ["--data-root", str(root), "--result-suffix", suffix, *COMMON]
        if ckpt.exists():
            args += ["--eval-checkpoint", str(ckpt)]
        elif post:
            raise RuntimeError(f"missing checkpoint for {tag}")
        own = run(args, env(model, jitter, post), stag)
        if not ckpt.exists():
            ckpt = keep_checkpoint(stag)
        rows.append(row(exp, model, post, source, source if exp == "A" else root.name, own, own["test"]))
        for target in targets:
            ttag = f"{target}_{STEM}_k3_{suffix}_from_{source}"
            targs = ["--data-root", f"00_data/RQ3/matrix/{target}", "--result-suffix",
                     f"{suffix}_from_{source}", *COMMON, "--eval-checkpoint", str(ckpt),
                     "--fixed-score", str(own["score_threshold"]), "--fixed-mask", str(own["mask_threshold"])]
            test = run(targs, env(model, jitter, post), ttag)["test"]
            shutil.rmtree(MODELS / ttag, ignore_errors=True)
            rows.append(row(exp, model, post, source, target, own, test))
            print(f"  {exp} {model} post={post} {source}->{target} FM={100 * test['LinesFMeasure']:.2f}", flush=True)
        shutil.rmtree(MODELS / stag, ignore_errors=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    NORHAND_D1.parent.mkdir(parents=True, exist_ok=True)
    if not NORHAND_D1.exists():
        NORHAND_D1.symlink_to(REPO / "00_data/RQ3/norhand_draws/d1")
    jitter, info = choose_jitter()
    reuse_dir = CKPT / ("_reuse_ab_j2_s42" if jitter == 2.0 else "_reuse_ab_s42")
    if jitter != 2.0:  # the kept ab_j2 checkpoints are not needed (disk is nearly full)
        shutil.rmtree(CKPT / "_reuse_ab_j2_s42", ignore_errors=True)
    info["reused_checkpoints"] = sorted(p.stem for p in reuse_dir.glob("*.pt"))
    (OUT / "decision.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)

    rows: list[dict] = []
    for source in DATASETS:  # experiment A
        root = NORHAND_D1 if source == "NorHand_v3" else REPO / f"00_data/RQ3/matrix/{source}"
        reuse = reuse_dir / f"{source}.pt"
        print(f"[A] ab {source}", flush=True)
        source_block("A", "ab", jitter, source, root, [t for t in DATASETS if t != source], rows,
                     reuse if reuse.exists() else None)
        pd.DataFrame(rows).to_csv(OUT / "A.csv", index=False, float_format="%.4f")
    print("[A] base NorHand_v3 (draw d1)", flush=True)
    source_block("A", "base", jitter, "NorHand_v3", NORHAND_D1,
                 [t for t in DATASETS if t != "NorHand_v3"], rows)
    pd.DataFrame(rows).to_csv(OUT / "A.csv", index=False, float_format="%.4f")

    rows_b: list[dict] = []
    for holdout in DATASETS:  # experiment B: LOCO-6, held-out test only
        print(f"[B] ab LOCO {holdout}", flush=True)
        source_block("B", "ab", jitter, holdout, REPO / f"00_data/RQ3/loco/{holdout}", [], rows_b)
        pd.DataFrame(rows_b).to_csv(OUT / "B.csv", index=False, float_format="%.4f")
    print("FINAL_AB_DONE", flush=True)


if __name__ == "__main__":
    main()
