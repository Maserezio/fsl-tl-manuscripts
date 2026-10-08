#!/usr/bin/env python3
"""All five FEST metrics for the U-DIADS-TL center-line rows of tab:udiads-center-line (2026-10-05).
Same models, cells, and settings as udiads_centerline.py ('center') and udiads_cl_crop_test.py; only the
full metric tuple (Pixel IU, Line IU, DR, RA, FM; page means) is kept instead of FM alone.
-> 99_evaluation/semantic_segmentation/center_line/udiads_ft/all_metrics.json"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, "99_evaluation/analysis")
import udiads_centerline as UC
HY = UC.HY
KEYS = ("PixelIU", "LineIU", "DR", "RA", "FM")


def mean5(rows):
    a = 100 * np.mean(np.asarray(rows, float), axis=0)
    return {k: round(float(v), 2) for k, v in zip(KEYS, a)}


def add_mean(d):
    d["Mean"] = {k: round(float(np.mean([d[s][k] for s in HY.SUBSETS])), 2) for k in KEYS}
    return d


res = {}
# U-Net mask cut by center-line cells (cells cached in udiads_ft/cells)
HY.OUT = UC.OUT
os.environ["HYBRID_MODE"] = "center"
jobs = [(s, f.stem) for s in HY.SUBSETS for f in sorted((HY.DATA / s / f"img-{s}" / "test").glob("*.jpg"))]
with ProcessPoolExecutor(4) as ex:
    rows = list(ex.map(HY.score, jobs))
res["center"] = add_mean({s: mean5([r[3] for r in rows if r[0] == s]) for s in HY.SUBSETS})
res["rq1_unet"] = add_mean({s: mean5([r[2] for r in rows if r[0] == s]) for s in HY.SUBSETS})
print("center", res["center"]["Mean"], "rq1", res["rq1_unet"]["Mean"], flush=True)

# Center-line boxes + crop U-Net
import cv2, torch
import udiads_cl_crop_tune as T
last = {}
_ev = T.B.evaluate_contiguous


def _keep(gt, canv, *a, **k):
    last["t"] = _ev(gt, canv, *a, **k)
    return last["t"]


T.B.evaluate_contiguous = _keep
U = T.U
res["clcrop"] = {}
for ms in ("Latin14396", "Latin2", "Syr341"):
    net = T.LF.build_model()
    net.load_state_dict(torch.load(T.LF.MODELS / f"UDIADS{ms}_cm.pt", map_location="cpu", weights_only=False)["model"])
    net = net.to(T.LF.DEV).eval()
    seg, _ = T.B.crop_base.load_segmenter(T.Q.REPO / f"80_models/instance_segmentation/rtdetr_bbox_unet/u-diads-tl/crop_seg_loss_ablation_components_1024x256/{ms}/tversky/best.pth", T.B.crop_base.DEVICE)
    rows = []
    for gp in sorted((U.DATA / ms / f"text-line-gt-{ms}/test").glob("*.png")):
        ib = cv2.imread(str(next((U.DATA / ms / f"img-{ms}/test").glob(gp.stem + ".*"))))
        gt = (cv2.imread(str(gp), 0) > 0).astype(np.uint8)
        bl = T.band_list(T.frags_of(net, cv2.cvtColor(ib, cv2.COLOR_BGR2RGB)), 2.0, 0.5, 3.0)
        T.score(ib, gt, bl, "box", seg, {})
        rows.append(last["t"])
    res["clcrop"][ms] = mean5(rows)
    print("clcrop", ms, res["clcrop"][ms], flush=True)
add_mean(res["clcrop"])
(UC.OUT / "all_metrics.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
