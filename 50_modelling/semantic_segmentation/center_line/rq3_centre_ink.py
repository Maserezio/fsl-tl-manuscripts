#!/usr/bin/env python3
"""Centre lines + ink assignment: ceiling of the representation on the RQ3 dev pages.

Each text line is reduced to its centre line (per-column midpoint of the GT polygon). Every pixel
is given to the nearest centre line (Voronoi cell) if it lies within CUT x the page's line pitch;
the resulting cells go through the same PAGE export as Mask R-CNN and are scored with DIVA.
DIVA scores foreground (ink) pixels only, so the cells assign ink to lines, and no line height or
polygon convention is predicted at all. No model is run; dev pages only.

    DIVA_WORKERS=3 nice -n 19 .venv/bin/python 50_modelling/semantic_segmentation/center_line/rq3_centre_ink.py [CUT ...]
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json, os, sys
from pathlib import Path

os.environ.setdefault("DIVA_WORKERS", "3")
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq3_cheap_exps as ce  # noqa: E402
from rq3_linefield import line_geometry, page_polys  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT / "00_data/RQ3/dev"
OUT = ROOT / "99_evaluation/semantic_segmentation/center_line"
WORK_LONG = 2000   # working resolution (long side) for the distance transform


def centre_lines(polys, shape, s):
    lines = []
    for p in polys:
        g = line_geometry(p * s, shape)
        if g is not None:
            xs, cy, _, _ = g
            lines.append(np.stack([xs, cy], 1).astype(np.float32))
    return lines


def pitch(lines):
    """Median vertical gap between horizontally overlapping neighbouring centre lines."""
    rows = sorted((l[:, 1].mean(), l[:, 0].min(), l[:, 0].max()) for l in lines)
    gaps = []
    for i, (cy, a1, a2) in enumerate(rows):
        for cy2, b1, b2 in rows[i + 1:]:
            if min(a2, b2) - max(a1, b1) > 0.3 * min(a2 - a1, b2 - b1):
                gaps.append(cy2 - cy)
                break
    return float(np.median(gaps)) if gaps else 40.0


def cells(lines, shape, cut_px):
    """Instance map: label of the nearest centre line within cut_px, else 0."""
    lab = np.zeros(shape, np.int32)
    for i, l in enumerate(lines):
        cv2.polylines(lab, [np.rint(l).astype(np.int32)], False, i + 1, 1)
    src = np.where(lab > 0, 0, 255).astype(np.uint8)
    dist, near = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(near.max() + 1, np.int32)
    lut[near[lab > 0]] = lab[lab > 0]
    inst = lut[near]
    inst[dist > cut_px] = 0
    return inst.astype(np.uint16)


def main(cuts):
    res = {}
    for coll in sorted(p.name for p in DEV.iterdir() if p.is_dir()):
        root = DEV / coll
        coco = json.loads((root / "coco_instances/dev.json").read_text())
        by = page_polys(coco)
        works = {c: OUT / "pred" / f"cut{c}" / coll for c in cuts}
        for w in works.values():
            w.mkdir(parents=True, exist_ok=True)
        for im in coco["images"]:
            H, W = im["height"], im["width"]
            s = min(1.0, WORK_LONG / max(H, W))
            shape = (round(H * s), round(W * s))
            lines = centre_lines(by.get(im["id"], []), shape, s)
            p = pitch(lines)
            meta = {"path": im["file_name"], "orig_w": W, "orig_h": H}
            for c, w in works.items():
                inst = cv2.resize(cells(lines, shape, c * p), (W, H), interpolation=cv2.INTER_NEAREST)
                ce.m.page_xml(meta, inst, w / f"{Path(im['file_name']).stem}.xml")
        for c, w in works.items():
            _, metrics = ce.m.diva_eval(root, "dev", w)
            res.setdefault(f"cut{c}", {})[coll] = metrics
            print(coll, f"cut {c}", "FM %.1f" % (100 * metrics["LinesFMeasure"]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ceiling_dev.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main([float(c) for c in sys.argv[1:]] or [0.75, 1.5])
