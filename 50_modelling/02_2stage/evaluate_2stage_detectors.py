"""Compare 2-stage detectors on U-DIADS-TL with the inference_eval pipeline.

For every trained detector (YOLO CNN / RT-DETR transformer), the SAME fixed
native single-line U-Net segments each detected box; predictions are connected
into one line and stitched into an instance map, then scored with the OFFICIAL
Zottin metric (Pixel_IU, Line_IU, DR, RA, FM). Only the detector varies.

Detector boxes are de-duplicated (greedy overlap merge, like the playground's
dedup_boxes) before stitching, so a detector that emits many overlapping boxes
per line (e.g. yolo26) isn't unfairly penalised on RA.

--conf-sweep tries several confidence thresholds per detector and reports each at
its best-FM conf (boxes are segmented once at the lowest conf and reused, so the
sweep is cheap). Without it, a single --conf is used.

Usage:
  python evaluate_2stage_detectors.py --subset latin14396                      # conf 0.1, dedup 0.1
  python evaluate_2stage_detectors.py --subset latin14396 --conf-sweep         # per-detector best conf
  python evaluate_2stage_detectors.py --subset latin14396 --models yolov8n rtdetr-l
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from ultralytics import YOLO, RTDETR

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.append(str(REPO / "71_misc"))
from evaluate_util import evaluate_metrics   # noqa: E402  (official Zottin metric)

DATA = REPO / "00_data" / "U-DIADS-TL"
PROJECT = REPO / "71_misc" / "runs" / "twostage_udiads_comparison"
SEG_DIR = REPO / "80_models" / "02_2stage" / "u-diads-tl" / "segmentation"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MS_NAME = {"latin14396": "Latin14396", "latin2": "Latin2", "syr341": "Syr341"}
DEFAULT_MODELS = ["yolov8n", "yolov8s", "yolo26n", "yolo26s", "rtdetr-l"]
PAD_INF, CLOSE_FRAC, BIN = 15, 0.04, 0.5
_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def load_seg(subset):
    ck = torch.load(SEG_DIR / f"unet_crops_{subset}_native.pth", map_location=DEVICE)
    state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    enc = ck.get("encoder", "resnet34") if isinstance(ck, dict) else "resnet34"
    m = smp.Unet(enc, encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
    m.load_state_dict(state); m.eval()
    return m


def load_detector(model_stem, subset):
    best = PROJECT / f"{model_stem}_{subset}" / "weights" / "best.pt"
    if not best.exists():
        return None
    Cls = RTDETR if "rtdetr" in model_stem.lower() else YOLO
    return Cls(str(best))


def _to32(v):
    return int(np.ceil(max(v, 1) / 32) * 32)


def connect_line(mask, close_frac=CLOSE_FRAC):
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return mask
    w = max(3, int(round(mask.shape[1] * close_frac))); w += 1 - (w & 1)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (w, 3)))
    n, lb, st, _ = cv2.connectedComponentsWithStats(closed, 8)
    if n <= 1:
        return closed
    return (lb == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


@torch.no_grad()
def segment_crop_native(crop_bgr, seg):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    ih, iw = x.shape[2], x.shape[3]
    xp = F.pad(x, (0, _to32(iw) - iw, 0, _to32(ih) - ih)).to(DEVICE)
    prob = torch.sigmoid(seg(xp))[0, 0, :ih, :iw].cpu().numpy()
    return connect_line((prob > BIN).astype(np.uint8))


def _box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1]); area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def dedup_indices(boxes, scores, thr):
    """Greedy: keep highest-conf box, drop any later box overlapping a kept one > thr."""
    kept = []
    for i in np.argsort(-scores):
        if all(_box_iou(boxes[i], boxes[k]) < thr for k in kept):
            kept.append(int(i))
    return kept


def segment_box(seg, page_bgr, box):
    """Segment one padded box crop once -> (padded_coords, single-line mask) or None."""
    H, W = page_bgr.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1 - PAD_INF), max(0, y1 - PAD_INF)
    x2, y2 = min(W, x2 + PAD_INF), min(H, y2 + PAD_INF)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2, segment_crop_native(page_bgr[y1:y2, x1:x2], seg).astype(bool))


def stitch(cells, H, W):
    inst = np.zeros((H, W), np.int32); nid = 1
    for cell in cells:
        if cell is None:
            continue
        x1, y1, x2, y2, pm = cell
        reg = inst[y1:y2, x1:x2]; reg[pm & (reg == 0)] = nid; nid += 1
    return inst


def read_bgr(img_dir, stem):
    for e in _EXTS:
        p = Path(img_dir) / f"{stem}{e}"
        if p.exists():
            return cv2.imread(str(p))
    raise FileNotFoundError(stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="latin14396", choices=list(MS_NAME))
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--conf-sweep", action="store_true", dest="conf_sweep",
                    help="try several confs per detector, report each at its best-FM conf")
    ap.add_argument("--conf-values", nargs="+", type=float, default=[0.05, 0.1, 0.25, 0.4, 0.5],
                    dest="conf_values")
    ap.add_argument("--dedup-iou", type=float, default=0.1, dest="dedup_iou",
                    help="drop a box overlapping a higher-conf kept box by more than this IoU")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    conf_grid = sorted(args.conf_values) if args.conf_sweep else [args.conf]
    conf_min = conf_grid[0]

    ms = MS_NAME[args.subset]
    img_dir = DATA / ms / f"img-{ms}" / "test"
    gt_dir = DATA / ms / f"text-line-gt-{ms}" / "test"
    stems = sorted({p.stem for p in img_dir.iterdir() if p.suffix.lower() in _EXTS})
    seg = load_seg(args.subset)
    print(f"device={DEVICE}  subset={ms}  test pages={len(stems)}  "
          f"conf={'sweep ' + str(conf_grid) if args.conf_sweep else conf_grid[0]}  dedup_iou={args.dedup_iou}")

    METRICS = ("Pixel_IU", "Line_IU", "DR", "RA", "FM")
    rows = []
    for stem_model in args.models:
        det = load_detector(stem_model, args.subset)
        if det is None:
            print(f"[skip] no trained detector for {stem_model}_{args.subset}")
            continue
        fam = "transformer" if "rtdetr" in stem_model.lower() else "cnn"
        # per-conf accumulators
        acc = {c: {k: [] for k in METRICS} for c in conf_grid}
        boxes_at = {c: [] for c in conf_grid}
        for stem in stems:
            page = read_bgr(img_dir, stem)
            gt = (cv2.imread(str(gt_dir / f"{stem}.png")).sum(axis=-1) > 0).astype(np.uint8)
            H, W = page.shape[:2]
            r = det(page, conf=conf_min, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                boxes = np.empty((0, 4), np.float32); scores = np.empty((0,), np.float32); cells = []
            else:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                cells = [segment_box(seg, page, b) for b in boxes]   # segment each box ONCE
            for c in conf_grid:
                sel = np.where(scores >= c)[0]
                keep = dedup_indices(boxes[sel], scores[sel], args.dedup_iou) if len(sel) else []
                chosen = [cells[sel[k]] for k in keep]
                inst = stitch(chosen, H, W)
                pix, line, DR, RA, FM = evaluate_metrics(gt, inst)
                for k, v in zip(METRICS, (pix, line, DR, RA, FM)):
                    acc[c][k].append(v)
                boxes_at[c].append(len([x for x in chosen if x is not None]))

        best_c = max(conf_grid, key=lambda c: float(np.mean(acc[c]["FM"])))
        row = {"detector": stem_model, "family": fam, "subset": ms, "conf": best_c,
               "mean_boxes": round(float(np.mean(boxes_at[best_c])), 1),
               **{k: round(float(np.mean(acc[best_c][k])), 4) for k in METRICS}}
        rows.append(row)
        tag = f"(best of {conf_grid})" if args.conf_sweep else ""
        print(f"{stem_model:12s} [{fam}]  conf={best_c:<4}  Pixel_IU={row['Pixel_IU']:.3f}  "
              f"Line_IU={row['Line_IU']:.3f}  DR={row['DR']:.3f}  RA={row['RA']:.3f}  FM={row['FM']:.3f} {tag}")

    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else REPO / "99_evaluation" / "02_2stage" / "u-diads-tl" / "twostage_comparison" / f"zottin_{args.subset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n" + "=" * 92)
    print(f"2-STAGE DETECTOR COMPARISON — {ms} — official Zottin metric (seg fixed, dedup + conf per detector)")
    print("=" * 92)
    print(df.to_string(index=False))
    print(f"\nsaved -> {out}\nDONE")


if __name__ == "__main__":
    main()
