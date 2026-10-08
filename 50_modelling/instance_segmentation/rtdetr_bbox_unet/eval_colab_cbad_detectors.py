#!/usr/bin/env python3
"""Evaluate the RQ2 two-stage DINOv3 (cBAD) detectors trained on Colab, locally.

The Colab notebook trains the detectors (MyDrive/thesis_rq2_twostage_cbad/detectors/sel_cbad_<job>/
best_model) but its downloaded DIVA jar lacks the evaluator class, so evaluation runs here with the
local evaluator and the existing crop segmenters -- exactly stage 2 of run_shot_selection_init.sh.
Polls Drive until all page sets are evaluated; idempotent (markers in 80_models/.shotsel_markers_cbad_colab).

    nice -n 19 .venv/bin/python 50_modelling/instance_segmentation/rtdetr_bbox_unet/eval_colab_cbad_detectors.py
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
PY = REPO / ".venv/bin/python"
RCLONE = Path.home() / ".local/bin/rclone"
DRIVE = "gdrive:thesis_rq2_twostage_cbad/detectors"
MANIFEST = REPO / "99_evaluation/summaries/rq2/shot_selection_manifest.json"
SEG_ROOT = "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256"
DET_ROOT = REPO / "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/detection/rtdetr_hf"
BACKUP = REPO / "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/detection/rtdetr_hf_cbad_local_backup"
MARK = REPO / "80_models/.shotsel_markers_cbad_colab"
CURVE = "99_evaluation/summaries/rq2/shot_selection_diva_cbad.csv"
BY_METHOD = REPO / "99_evaluation/summaries/rq2/shot_selection_diva_by_method_cbad.csv"


def drive_detectors():
    r = subprocess.run([str(RCLONE), "lsf", "--dirs-only", DRIVE], capture_output=True, text=True)
    return {d.strip("/") for d in r.stdout.split() if d.startswith("sel_cbad_")}


def expand_cells(m):
    if not (REPO / CURVE).exists():
        return 0
    by_job = {r["method"]: r for _, r in pd.read_csv(REPO / CURVE).iterrows()}
    metrics = ["Pixel_IU", "Line_IU", "DR", "RA", "FM"]
    rows = [{"subset": c["subset"], "method": c["method"], "k": c["k"], "pages": c["k"], "job": c["job"],
             **{k: round(float(by_job[c["job"]][k]), 4) for k in metrics}}
            for c in m["cells"] if c["job"] in by_job]
    pd.DataFrame(rows).sort_values(["subset", "method", "k"]).to_csv(BY_METHOD, index=False)
    return len(rows)


def main():
    m = json.loads(MANIFEST.read_text())
    jobs = {j["job"]: j for j in m["jobs"]}
    MARK.mkdir(parents=True, exist_ok=True)
    while True:
        done = {p.name for p in MARK.iterdir()}
        todo = [j for j in jobs if j not in done]
        if not todo:
            break
        ready = drive_detectors()
        for job in todo:
            run = f"sel_cbad_{job}"
            if run not in ready:
                continue
            sub, k = jobs[job]["subset"], jobs[job]["k"]
            model_dir = DET_ROOT / ("" if sub == "CB55" else sub) / run
            if model_dir.exists() and not (model_dir / ".from_colab").exists():
                BACKUP.mkdir(parents=True, exist_ok=True)
                shutil.move(str(model_dir), str(BACKUP / run))       # keep the local run aside
            r = subprocess.run([str(RCLONE), "copy", f"{DRIVE}/{run}/best_model", str(model_dir / "best_model")],
                               capture_output=True, text=True)
            if r.returncode != 0 or not any((model_dir / "best_model").glob("*.safetensors")):
                print(f"{job}: download incomplete, retry later", flush=True)
                continue
            (model_dir / ".from_colab").touch()
            env = {**os.environ, "SUBSET": sub, "RUN_NAME": run, "K_SHOT": str(k),
                   "SEG_KSHOT_ROOT": f"{SEG_ROOT}/{sub}/{job}/k{k}/tversky", "CURVE_CSV": CURVE,
                   "CURVE_APPROACH": "two_stage_cbad", "CURVE_METHOD": job,
                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
            t0 = time.time()
            r = subprocess.run([str(PY), "instance_segmentation/rtdetr_bbox_unet/predict_and_eval_rtdetr_diva.py"], cwd=REPO / "50_modelling",
                               env=env, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{job}: evaluation FAILED\n{r.stderr[-800:]}", flush=True)
                continue
            (MARK / job).touch()
            n = expand_cells(m)
            fm = pd.read_csv(REPO / CURVE).set_index("method").loc[job, "FM"]
            print(time.strftime("%H:%M"), f"{job}: FM {100 * fm:.2f} ({(time.time() - t0) / 60:.1f} min); "
                  f"{len(done) + 1}/{len(jobs)} page sets, {n}/{len(m['cells'])} cells", flush=True)
            done.add(job)
        if len(done) < len(jobs):
            time.sleep(300)
    print("CBAD_COLAB_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
