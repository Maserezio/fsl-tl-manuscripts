#!/usr/bin/env python3
"""Center-line model on U-DIADS-TL, fine-tuned on the three training pages (exploratory).

  prepare   roots 00_data/RQ3/udiads_lf/<SUB>: COCO train/val/test whose line polygons are the column envelopes
            (4-px columns) of the GT connected components (the thesis conversion), images symlinked.
  train     CATMuS-pretrained center-line model fine-tuned 1500 steps per subset (no WiSE; in-domain, as the
            DIVA rows of tab:rq2-center-line) -> 80_models/semantic_segmentation/center_line/UDIADS<SUB>_cm.pt
  eval      test pages: the decoder of rq3_udiads_hybrid.py with the fine-tuned model instead of the zero-shot
            one -- RQ1 U-Net foreground (prob >= 0.5) cut by the center-line cells (thesis decoder row:1.0);
            'center' = components inside each cell, 'whole' = one label per cell. FEST metric, paired with the
            RQ1 pipeline. -> 99_evaluation/semantic_segmentation/center_line/udiads_ft/summary.json
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "50_modelling").is_dir())  # repo root
sys.path.insert(0, str(HERE))
import rq3_udiads_hybrid as HY  # noqa: E402

LFROOT = ROOT / "00_data/RQ3/udiads_lf"
OUT = ROOT / "99_evaluation/semantic_segmentation/center_line/udiads_ft"
SPLITS = {"train": ("training", "train"), "val": ("validation", "val"), "test": ("test",)}


def _dir(base, names):
    return next(base / n for n in names if (base / n).is_dir())


def envelope(comp, step=4):
    ys, xs = np.nonzero(comp)
    x0, x1 = xs.min(), xs.max() + 1
    top, bot = [], []
    for a in range(x0, x1, step):
        col = comp[:, a:min(a + step, x1)].any(axis=1)
        r = np.nonzero(col)[0]
        if len(r):
            top += [(a, r[0]), (min(a + step, x1), r[0])]
            bot += [(a, r[-1] + 1), (min(a + step, x1), r[-1] + 1)]
    return np.array(top + bot[::-1], np.int32)


def prepare():
    for sub in HY.SUBSETS:
        r = LFROOT / sub
        (r / "coco_instances").mkdir(parents=True, exist_ok=True)
        (r / "images").mkdir(parents=True, exist_ok=True)
        for split, names in SPLITS.items():
            idir = _dir(HY.DATA / sub / f"img-{sub}", names)
            gdir = _dir(HY.DATA / sub / f"text-line-gt-{sub}", names)
            link = r / "images" / split
            if not link.exists():
                link.symlink_to(idir)
            images, anns = [], []
            for i, gp in enumerate(sorted(gdir.glob("*.png"))):
                g = cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE) > 0
                H, W = g.shape
                images.append({"id": i, "file_name": next(idir.glob(gp.stem + ".*")).name, "width": W, "height": H})
                n, lab, st, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8))
                for c in range(1, n):
                    if st[c, 4] < 50:
                        continue
                    p = envelope(lab == c)
                    x, y, w, h = st[c, :4]
                    anns.append({"id": len(anns), "image_id": i, "category_id": 1, "bbox": [int(x), int(y), int(w), int(h)],
                                 "area": int(st[c, 4]), "iscrowd": 0, "segmentation": [p.ravel().tolist()]})
            (r / f"coco_instances/{split}.json").write_text(json.dumps(
                {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "TextLine", "supercategory": "text"}]}))
            print("prepared", sub, split, len(images), "pages", len(anns), "lines", flush=True)


def train():
    import rq3_linefield as LF
    for sub in HY.SUBSETS:
        LF.train(f"UDIADS{sub}", "cm", int(os.environ.get("UD_STEPS", "1500")), root=LFROOT / sub,
                 init=LF.MODELS / "catmus_pretrain.pt", out=LF.MODELS / f"UDIADS{sub}_cm.pt")


def cells(sub):
    """Center-line cells (thesis decoder row:1.0, two-pass inference) of the fine-tuned model on the test pages."""
    import torch
    from PIL import Image
    import rq3_linefield as LF
    import rq3_centre_lf as CL
    net = LF.build_model()
    net.load_state_dict(torch.load(LF.MODELS / f"UDIADS{sub}_cm.pt", map_location="cpu", weights_only=False)["model"])
    net = net.to(LF.DEV).eval()
    (OUT / "cells" / sub).mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for f in sorted((HY.DATA / sub / f"img-{sub}" / "test").glob("*.jpg")):
            dst = OUT / "cells" / sub / f"{f.stem}.png"
            if dst.exists():
                continue
            img = np.asarray(Image.open(f).convert("RGB"))
            H, W = img.shape[:2]
            s1 = 1152 / max(H, W)
            cen, _, up, dn = LF.run_net(net, cv2.resize(img, (int(W * s1), int(H * s1)), interpolation=cv2.INTER_AREA))
            sel = cen > 0.5
            thick = float(np.median(up[sel] + dn[sel])) if sel.sum() > 50 else LF.T0
            s2 = min(s1 * LF.T0 / max(thick, 1.0), 3200 / max(H, W))
            cen, end, up, dn = LF.run_net(net, cv2.resize(img, (int(W * s2), int(H * s2)),
                                                          interpolation=cv2.INTER_AREA if s2 < 1 else cv2.INTER_LINEAR))
            frags = [{"xs": fr["xs"] / s2, "cy": fr["cy"] / s2} for fr in LF._fragments(cen, end, 0.4)]
            page = {"coll": sub, "im": {"height": H, "width": W, "file_name": f.name}, "frags": frags, "T0": LF.T0 / s2}
            cv2.imwrite(str(dst), CL.page_instances(page, "row", 1.0).astype(np.uint16))
    print("cells", sub, flush=True)


def evaluate():
    HY.OUT = OUT                                          # HY.score reads OUT/cells/<sub>/<stem>.png
    for sub in HY.SUBSETS:
        cells(sub)
    res = {}
    for mode in ("center", "whole"):
        os.environ["HYBRID_MODE"] = mode
        jobs = [(s, f.stem) for s in HY.SUBSETS for f in sorted((HY.DATA / s / f"img-{s}" / "test").glob("*.jpg"))]
        with ProcessPoolExecutor(4) as ex:
            rows = list(ex.map(HY.score, jobs))
        per = {s: [r for r in rows if r[0] == s] for s in HY.SUBSETS}
        res[mode] = {s: round(100 * float(np.mean([r[3][4] for r in v])), 2) for s, v in per.items()}
        res[mode]["Mean"] = round(float(np.mean(list(res[mode].values()))), 2)
        if mode == "center":
            res["rq1"] = {s: round(100 * float(np.mean([r[2][4] for r in v])), 2) for s, v in per.items()}
            res["rq1"]["Mean"] = round(float(np.mean(list(res["rq1"].values()))), 2)
        print(mode, res[mode], flush=True)
    print("rq1", res["rq1"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    {"prepare": prepare, "train": train, "eval": evaluate}[sys.argv[1]]()
