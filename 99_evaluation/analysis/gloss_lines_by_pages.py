#!/usr/bin/env python3
"""Predicted lines of CATMuS Mask R-CNN (ConvNeXt-Tiny) on the DIVA-HisDB test pages, classified by the ink class
that dominates inside the polygon (pixel GT bits: 0x08 main text, 0x02 comment/gloss, 0x04 decoration), without
fine-tuning and after fine-tuning on 1, 3, and 20 pages (random selection, seed 42).
    .venv/bin/python 99_evaluation/analysis/gloss_lines_by_pages.py
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_qualitative_v2 as Q  # noqa: E402
import overlap_lines_diva as O  # noqa: E402

EV = Q.ROOT / "99_evaluation/instance_segmentation/mask_rcnn/diva-hisdb"
RUNS = {"zero-shot": "maskrcnn_convnext_tiny_catmus_704_zeroshot", "1 page": "maskrcnn_convnext_tiny_catmus_704_kshot_random_k1",
        "3 pages": "maskrcnn_convnext_tiny_catmus_704_kshot_random_k3", "20 pages": "maskrcnn_convnext_tiny_catmus_704"}


def page(args):
    sub, xml = args
    _, pix, _ = Q.files(sub, xml.stem)
    px = cv2.imread(str(pix))[:, :, 0]
    main, gl, de = (px & 0x08) > 0, (px & 0x02) > 0, (px & 0x04) > 0
    c = {"main": 0, "gloss": 0, "deco": 0, "none": 0}
    for p in O.polys(xml):
        m = np.zeros(px.shape, np.uint8); cv2.fillPoly(m, [p.astype(np.int32)], 1); m = m > 0
        v = {"main": int((m & main).sum()), "gloss": int((m & gl).sum()), "deco": int((m & de).sum())}
        c["none" if max(v.values()) == 0 else max(v, key=v.get)] += 1
    return c


RT = Q.ROOT / "99_evaluation/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/rtdetr_hf"
CL = Q.ROOT / "99_evaluation/semantic_segmentation/center_line/diva_cbad"
OTHER = {
    "RT-DETR 1 page": {"CB55": RT / "sel_catmus_CB55_k1_2c752f4a/pred_xml", "CS18": RT / "CS18/sel_catmus_CS18_k1_eec0092d/pred_xml",
                       "CS863": RT / "CS863/sel_catmus_CS863_k1_f4f0a4c0/pred_xml"},
    "RT-DETR 3 pages": {"CB55": RT / "sel_catmus_CB55_k3_9c4b2a61/pred_xml", "CS18": RT / "CS18/sel_catmus_CS18_k3_e5d541e7/pred_xml",
                        "CS863": RT / "CS863/sel_catmus_CS863_k3_002db31f/pred_xml"},
    "RT-DETR 20 pages": {"CB55": RT / "rtdetr_convnext_tiny_catmus/pred_xml", "CS18": RT / "CS18/rtdetr_convnext_tiny_catmus/pred_xml",
                         "CS863": RT / "CS863/rtdetr_convnext_tiny_catmus/pred_xml"},
    "center-line zero-shot": {s: CL / f"pred_cmw0_cellseam/test/{s}_k3/bandseamstroke_inside" for s in ("CB55", "CS18", "CS863")},
    "center-line 1 page": {s: CL / f"pred_cm_cellseam/test/{s}_k1/bandseamstroke_inside" for s in ("CB55", "CS18", "CS863")},
    "center-line 3 pages": {s: CL / f"pred_cm_cellseam/test/{s}_k3/bandseamstroke_inside" for s in ("CB55", "CS18", "CS863")},
    "center-line 20 pages": {s: CL / f"pred_cm_cellseam/test/{s}_full/bandseamstroke_inside" for s in ("CB55", "CS18", "CS863")},
}


def is_test(sub, stem):
    return (Q.files(sub, stem)[1]).exists() if stem else False


if __name__ == "__main__" and sys.argv[1:2] == ["other"]:
    for name, dirs in OTHER.items():
        jobs = []
        for sub, d in dirs.items():
            for x in sorted(Path(d).glob("*.xml")):
                try:
                    if is_test(sub, x.stem): jobs.append((sub, x))
                except Exception:
                    pass
        if not jobs:
            print(name, "missing"); continue
        with ProcessPoolExecutor(6) as ex:
            res = list(ex.map(page, jobs))
        tot = {k: sum(r[k] for r in res) for k in res[0]}
        print(f"{name:22s} pages {len(res)}  lines {sum(tot.values()):5d}  " + "  ".join(f"{k} {v}" for k, v in tot.items()), flush=True)
elif __name__ == "__main__":
    for name, run in RUNS.items():
        jobs = []
        for sub in ("CB55", "CS18", "CS863"):
            d = sorted((EV / sub / run).glob("diva_test_*"))
            if not d:
                print(name, sub, "missing"); continue
            jobs += [(sub, x) for x in sorted(d[0].glob("*.xml"))]
        with ProcessPoolExecutor(6) as ex:
            res = list(ex.map(page, jobs))
        tot = {k: sum(r[k] for r in res) for k in res[0]}
        print(f"{name:10s} pages {len(res)}  lines {sum(tot.values()):5d}  " + "  ".join(f"{k} {v}" for k, v in tot.items()), flush=True)
