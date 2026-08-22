"""Score the U-DIADS size-axis matrix (48 runs) with the Zottin metric, in parallel.

Same computation as evaluate_lines.py's udiads path -- ARU-Net baseline fusion,
seam-carve disconnection + small-object cleanup with the per-subset params from
postproc.PER_MS_PARAMS, then evaluate_metrics -- just arranged so it finishes in
minutes instead of hours. Verified to reproduce evaluate_lines.py's numbers.

Two things make it slow otherwise:

1. evaluate_metrics is ~23 s/page and everything else is ~0.4 s. find_best_matches
   compares every GT component against every predicted one as full-page boolean
   masks: ~30x30 passes over a multi-megapixel array. That is the entire runtime,
   and each page is independent, so it goes in a process pool.
2. The ARU-Net baseline depends only on the image, never on the run, but the
   original loop recomputes it for all 48 runs. Here it is computed once per page
   and cached to disk, so TensorFlow is touched 45 times instead of 720.

Pool workers deliberately never import TensorFlow: they read the cached baseline,
the run's probability map and the GT from disk, so nothing large is pickled.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "71_misc"))

from postproc import PER_MS_PARAMS, fuse_masks, run_pipeline  # noqa: E402
from evaluate_util import evaluate_metrics  # noqa: E402

SUBSETS = ["Latin14396", "Latin2", "Syr341"]
ARCHS = [
    "vit_tiny_patch16_224.augreg_in21k",
    "tu-convnext_femto",
    "tu-pvt_v2_b0",
    "tu-convnext_pico",
    "tu-pvt_v2_b1",
    "tu-convnext_tiny",
    "vit_small_patch16_224.augreg_in21k",
    "tu-pvt_v2_b2",
]
# (init label -> how the run folder names that arm). Most arms are a suffix on the
# architecture name; the DINO-SSL arm is a different timm/hier_encoder checkpoint
# entirely, so it is spelled out per architecture instead.
ARMS = {"imagenet": "_sz", "random": "_szrand"}
SSL_RUNS = {
    "tu-convnext_tiny": "tu-convnext_tiny.dinov3_lvd1689m",
    "vit_small_patch16_224.augreg_in21k": "vit_small_patch16_dinov3",
}
SSL_LABEL = "dinov3"


def run_token(arch, init):
    """The <token> in unet_<token>_udiads_<subset> for one matrix cell."""
    if init == SSL_LABEL:
        return SSL_RUNS.get(arch)
    return f"{arch}{ARMS[init]}"


def inits_for(arch):
    return list(ARMS) + ([SSL_LABEL] if arch in SSL_RUNS else [])


SPLIT = "test"
N_WORKERS = max(1, (os.cpu_count() or 4) - 2)

DATA = REPO / "00_data/U-DIADS-TL"
PROB_CACHE = REPO / "99_evaluation/01_simple_segmentation/u-diads-tl/prob_cache"
BASE_CACHE = REPO / "99_evaluation/01_simple_segmentation/u-diads-tl/arunet_baseline_cache"
OUT_CSV = REPO / "99_evaluation/01_simple_segmentation/u-diads-tl/size_axis_zottin.csv"
ARUNET_PB = REPO / "80_models/01_simple_segmentation/u-diads-tl/pretrained/arunet/model100_ema.pb"

_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_METRICS = ("Pixel_IU", "Line_IU", "DR", "RA", "FM")


def img_dir(ms):
    return DATA / ms / f"img-{ms}" / SPLIT


def gt_dir(ms):
    return DATA / ms / f"text-line-gt-{ms}" / SPLIT


def stems(ms):
    return sorted(p.stem for p in img_dir(ms).iterdir() if p.suffix.lower() in _EXTS)


def build_baseline_cache():
    """One ARU-Net forward per test page, cached to disk. Imports TF only if needed."""
    todo = [(ms, s) for ms in SUBSETS for s in stems(ms)
            if not (BASE_CACHE / ms / f"{s}.npy").exists()]
    if not todo:
        print(f"[baseline] all {sum(len(stems(m)) for m in SUBSETS)} pages already cached")
        return
    print(f"[baseline] computing {len(todo)} ARU-Net baselines (once per page, reused by every run)")
    from models.arunet import ARUNetWrapper
    from postproc import arunet_baseline_prob
    aru = ARUNetWrapper(str(ARUNET_PB), device="cpu", scale=0.33)
    for ms, s in todo:
        out = BASE_CACHE / ms / f"{s}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.cvtColor(cv2.imread(str(next(img_dir(ms).glob(s + ".*")))), cv2.COLOR_BGR2RGB)
        np.save(out, arunet_baseline_prob(img, aru).astype(np.float32))
        print(f"  {ms}/{s}")


def score_page(args):
    """Runs in a worker: fuse -> disconnect/cleanup -> Zottin metric for one page."""
    run, ms, stem = args
    prob = np.load(PROB_CACHE / run / f"{stem}.npy").astype(np.float32)
    base = np.load(BASE_CACHE / ms / f"{stem}.npy").astype(np.float32)
    gt = (cv2.imread(str(gt_dir(ms) / f"{stem}.png")).sum(axis=-1) > 0).astype(np.uint8)

    params = PER_MS_PARAMS.get(ms, PER_MS_PARAMS["Latin14396"])
    disc = {k: params[k] for k in ("line_sigma", "min_line_distance", "peak_frac", "valley_ratio")}
    refined = run_pipeline(fuse_masks(base, prob), disc)
    _, labels = cv2.connectedComponents(refined.astype(np.uint8))
    return (run, ms) + tuple(evaluate_metrics(gt, labels))


def main():
    build_baseline_cache()

    # Incremental by default: a cell already in the CSV is left alone, so adding an
    # arm costs only its own pages instead of re-scoring the whole matrix (~23 s per
    # page, 54 cells x 15 pages -- hours). Set RESCORE_ALL=1 to force a full pass,
    # which is what you want after changing the postprocessing or the metric.
    done_cells = set()
    previous = pd.DataFrame()
    if OUT_CSV.exists() and not os.environ.get("RESCORE_ALL"):
        previous = pd.read_csv(OUT_CSV)
        done_cells = set(zip(previous.encoder, previous.init, previous.subset))
        if done_cells:
            print(f"[incremental] {len(done_cells)} cell(s) already scored in {OUT_CSV.name}; "
                  f"set RESCORE_ALL=1 to redo them")

    tasks, missing = [], []
    for arch in ARCHS:
        for init in inits_for(arch):
            for ms in SUBSETS:
                if (arch, init, ms) in done_cells:
                    continue
                run = f"unet_{run_token(arch, init)}_udiads_{ms}"
                if not (PROB_CACHE / run).is_dir() or not any((PROB_CACHE / run).glob("*.npy")):
                    missing.append(run)
                    continue
                tasks += [(run, ms, s) for s in stems(ms)]
    if missing:
        print(f"\n[WARN] {len(missing)} run(s) have no cached probability maps and are skipped;\n"
              f"       run preprocessing/cache_predictions.py for them first. e.g. {missing[0]}")
    print(f"\n[score] {len(tasks)} page-jobs across {N_WORKERS} workers "
          f"(~23 s each single-threaded)\n")

    acc = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(score_page, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            run, ms, *vals = fut.result()
            acc.setdefault((run, ms), []).append(vals)
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} pages")

    rows = []
    for arch in ARCHS:
        for init in inits_for(arch):
            for ms in SUBSETS:
                run = f"unet_{run_token(arch, init)}_udiads_{ms}"
                vals = acc.get((run, ms))
                if not vals:
                    continue
                mean = np.mean(np.array(vals, dtype=float), axis=0)
                rows.append({"encoder": arch, "init": init, "subset": ms, "pages": len(vals),
                             **{k: round(float(v), 4) for k, v in zip(_METRICS, mean)}})

    df = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True) if len(previous) \
        else pd.DataFrame(rows)
    # Stable order so the CSV diffs cleanly between incremental runs.
    order = {a: i for i, a in enumerate(ARCHS)}
    df = (df.assign(_a=df.encoder.map(order), _s=df.subset.map({s: i for i, s in enumerate(SUBSETS)}))
            .sort_values(["_a", "_s", "init"]).drop(columns=["_a", "_s"]).reset_index(drop=True))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nsaved -> {OUT_CSV}  ({len(df)} cells, {len(rows)} newly scored)\n")
    if len(df):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
