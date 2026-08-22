"""Detection-only evaluation of the backbone matrix trained by run_train_all.sh.

Scores every `yolov8_<encoder>_<family>_<subset>` run on its subset's *test* split and
writes one CSV row per run: mAP50, mAP50-95, precision, recall, plus the checkpoint's
own best-epoch bookkeeping.

This is the detector in isolation -- not the 2-stage pipeline. For the line-level
Zottin/DIVA metrics after segmentation and stitching, see evaluate_2stage_detectors.py
(U-DIADS) and 01_simple_segmentation/evaluate_lines.py (both families).

Usage:
  python evaluate_detectors.py                                  # everything found
  python evaluate_detectors.py --family diva --subsets CB55
  python evaluate_detectors.py --encoders dinov2 am-radio
"""
import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

# Importing these registers the custom module classes (SFP neck, timm backbones) that
# the transformer checkpoints were pickled with -- without them torch.load cannot
# resolve the module paths and YOLO() raises on the affected runs.
import hier_encoder_yolo   # noqa: F401,E402
import timm_backbone       # noqa: E402
from train_detector import BACKBONES, resolve_data  # noqa: E402

from ultralytics import YOLO  # noqa: E402

# The native foundation backbones were built by torch.hub, so their layers are pickled
# under the hub repo's own top-level package (`dinov2.*`, `dinov3.*`, `radio.*`). Those
# packages only exist inside the hub cache, and torch.hub puts them on sys.path when it
# loads a model -- which we never do here, we just unpickle. Without this the affected
# checkpoints die with ModuleNotFoundError: No module named 'dinov2'.
_HUB = Path(torch.hub.get_dir())
for _repo in ("facebookresearch_dinov2_main", "facebookresearch_dinov3_main", "NVlabs_RADIO_main"):
    _p = _HUB / _repo
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))

FAMILIES = {
    "diva": dict(
        subsets=["CB55", "CS18", "CS863"],
        out="80_models/02_2stage/diva-hisdb/detection",
        data={"CB55": "configs/data/diva_cb55_detect.yaml",
              "CS18": "configs/data/diva_cs18_detect.yaml",
              "CS863": "configs/data/diva_cs863_detect.yaml"},
    ),
    "udiads": dict(
        subsets=["Latin14396", "Latin2", "Syr341"],
        out="80_models/02_2stage/u-diads-tl/detection",
        data={"Latin14396": "configs/data/udiads_latin14396_detect.yaml",
              "Latin2": "configs/data/udiads_latin2_detect.yaml",
              "Syr341": "configs/data/udiads_syr341_detect.yaml"},
    ),
}


def eval_one(ckpt: Path, data_yaml: str, imgsz: int, split: str, batch: int) -> dict | None:
    timm_backbone.register()
    model = YOLO(str(ckpt))
    try:
        r = model.val(data=resolve_data(HERE / data_yaml), split=split,
                      imgsz=imgsz, batch=batch, verbose=False, plots=False)
        b = r.box
        return dict(mAP50=round(float(b.map50), 4), mAP50_95=round(float(b.map), 4),
                    precision=round(float(b.mp), 4), recall=round(float(b.mr), 4))
    finally:
        # Every run builds a fresh model on the GPU; without an explicit teardown the
        # allocator holds the previous one and later (larger) backbones OOM even though
        # each fits on its own.
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(FAMILIES), default=None, help="default: both")
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--encoders", nargs="+", default=None, help="default: all in BACKBONES")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=1,
                    help="val batch; 1 keeps the big SFP backbones inside 8 GB at imgsz 1280")
    ap.add_argument("--out", default=str(REPO / "99_evaluation" / "02_2stage" / "detection_eval.csv"))
    args = ap.parse_args()

    encoders = args.encoders or list(BACKBONES)
    families = [args.family] if args.family else list(FAMILIES)

    rows = []
    for family in families:
        spec = FAMILIES[family]
        for ms in (args.subsets or spec["subsets"]):
            if ms not in spec["data"]:
                continue
            for enc in encoders:
                run = f"yolov8_{enc}_{family}_{ms}"
                ckpt = REPO / spec["out"] / run / "weights" / "best.pt"
                if not ckpt.exists():
                    continue
                try:
                    m = eval_one(ckpt, spec["data"][ms], args.imgsz, args.split, args.batch)
                except Exception as e:
                    print(f"[FAILED] {run}: {type(e).__name__}: {e}")
                    continue
                row = dict(family=family, subset=ms, encoder=enc, run=run, **m)
                rows.append(row)
                print(f"{family:7} {enc:38} {ms:12} " +
                      "  ".join(f"{k}={row[k]:.4f}" for k in
                                ("mAP50", "mAP50_95", "precision", "recall")))

    if not rows:
        raise SystemExit("no checkpoints found -- nothing evaluated")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n" + "=" * 100)
    print(f"DETECTION-ONLY EVALUATION -- {args.split} split, imgsz={args.imgsz}")
    print("=" * 100)
    print(df.drop(columns=["run"]).to_string(index=False))
    print("\nmAP50 by encoder x subset:")
    print(df.pivot_table(index="encoder", columns=["family", "subset"], values="mAP50").to_string())
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
