#!/usr/bin/env python3
"""Quick check (one test page per subset): center lines as the detector for the crop U-Net on U-DIADS-TL.
Center-line model fine-tuned on the subset (udiads_centerline.py, UDIADS<SUB>_cm.pt) -> one band per line
(graph grouping + traced ends + predicted heights, as the DIVA band decoder) -> box around the band, pad 15 ->
Tversky crop U-Net (threshold 0.5, largest component) -> instance map (lines top to bottom, first wins).
  clcrop       crop U-Net mask in the band's box
  clcrop_band  same, restricted to the band dilated by 0.5 x line height (no ink of the neighbouring line)
Compared with Mask R-CNN boxes + crop U-Net (fixed settings of the paired seam harness) and the RQ1 U-Net.
FEST/Zottin metric (contiguous implementation of maskrcnn_syr341_crop_refine.py).

    .venv/bin/python 50_modelling/semantic_segmentation/center_line/udiads_cl_crop.py
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "50_modelling").is_dir())  # repo root
sys.path[:0] = [str(HERE), str(REPO / "50_modelling/instance_segmentation/mask_rcnn"), str(REPO / "50_modelling/instance_segmentation/rtdetr_bbox_unet")]
import maskrcnn_syr341_crop_refine as B  # noqa: E402
import rq3_linefield as LF  # noqa: E402
import rq3_centre_lf as CL  # noqa: E402
import story_udiads as U  # noqa: E402

PAGES = {"Latin14396": None, "Latin2": "230", "Syr341": "031"}
PAD = 15
import os
CKPT = os.environ.get("CL_CKPT", "ft")


def bands(net, img):
    H, W = img.shape[:2]
    with torch.inference_mode():
        s1 = 1152 / max(H, W)
        cen, _, up, dn = LF.run_net(net, cv2.resize(img, (int(W * s1), int(H * s1)), interpolation=cv2.INTER_AREA))
        sel = cen > 0.5
        thick = float(np.median(up[sel] + dn[sel])) if sel.sum() > 50 else LF.T0
        s2 = min(s1 * LF.T0 / max(thick, 1.0), 3200 / max(H, W))
        cen, end, up, dn = LF.run_net(net, cv2.resize(img, (int(W * s2), int(H * s2)),
                                                      interpolation=cv2.INTER_AREA if s2 < 1 else cv2.INTER_LINEAR))
    frags = []
    for f in LF._fragments(cen, end, 0.4):
        ys_, xs_ = f["px"]
        frags.append({"xs": f["xs"] / s2, "cy": f["cy"] / s2, "n": len(f["xs"]),
                      "hu": float(np.percentile(up[ys_, xs_], 75)) / s2, "hd": float(np.percentile(dn[ys_, xs_], 75)) / s2})
    frg = [f for f in frags if len(f["xs"]) > 2]
    if not frg:
        return []
    pp = CL.frag_pitch(frg)
    lines = [l for l in (CL.group_row(frg, pp, gap_max=float(os.environ["CL_GAP"])) if os.environ.get("CL_GAP") else CL.group_graph(frg, pp)) if l["xs"][-1] - l["xs"][0] >= 0.5 * pp]
    out = []
    for l in sorted(lines, key=lambda q: q["cy"].mean()):
        hu, hd = CL.line_height(l)
        step = max(1, len(l["xs"]) // 60)
        x, y = l["xs"][::step], l["cy"][::step]
        out.append((np.concatenate([np.stack([x, y - hu], 1), np.stack([x, y + hd], 1)[::-1]]), hu + hd))
    return out


def main():
    rows = {}
    for ms, stem in PAGES.items():
        stems = sorted(p.stem for p in (U.DATA / ms / f"text-line-gt-{ms}/test").glob("*.png"))
        stem = stem or stems[0]
        img_bgr = cv2.imread(str(next((U.DATA / ms / f"img-{ms}/test").glob(stem + ".*"))))
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        gt = (cv2.imread(str(U.DATA / ms / f"text-line-gt-{ms}/test/{stem}.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        net = LF.build_model()
        net.load_state_dict(torch.load(LF.MODELS / (f"UDIADS{ms}_cm.pt" if CKPT == "ft" else "catmus_pretrain.pt"), map_location="cpu", weights_only=False)["model"])
        net = net.to(LF.DEV).eval()
        seg, _ = B.crop_base.load_segmenter(REPO / f"80_models/instance_segmentation/rtdetr_bbox_unet/u-diads-tl/crop_seg_loss_ablation_components_1024x256/{ms}/tversky/best.pth", B.crop_base.DEVICE)
        canv = {k: np.zeros((H, W), np.uint16) for k in ("clcrop", "clcrop_band")}
        nid = {k: 1 for k in canv}
        bl = bands(net, img)
        for poly, h in bl:
            x0, y0 = poly.min(0); x1, y1 = poly.max(0)
            X0, Y0 = max(0, int(x0) - PAD), max(0, int(y0) - PAD)
            X1, Y1 = min(W, int(np.ceil(x1)) + PAD), min(H, int(np.ceil(y1)) + PAD)
            if X1 - X0 < 4 or Y1 - Y0 < 4:
                continue
            ink = B.crop_probability(img_bgr[Y0:Y1, X0:X1], seg, 1024, 256) >= 0.5
            m = B.crop_base.connect_line(ink, 0.0).astype(bool)
            band = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
            cv2.fillPoly(band, [np.rint(poly - [X0, Y0]).astype(np.int32)], 1)
            k = max(3, int(0.5 * h) | 1)
            band = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
            for name, mk in (("clcrop", m), ("clcrop_band", ink & band)):
                reg = canv[name][Y0:Y1, X0:X1]
                wr = mk & (reg == 0)
                if wr.any():
                    reg[wr] = nid[name]; nid[name] += 1
        res = {k: B.evaluate_contiguous(gt, v)[4] for k, v in canv.items()}
        res["maskrcnn_crop"] = B.evaluate_contiguous(gt, U.labels("maskrcnn", ms, stem).astype(np.uint16))[4]
        res["unet_rq1"] = B.evaluate_contiguous(gt, U.labels("unet", ms, stem).astype(np.uint16))[4]
        rows[f"{ms}/{stem}"] = {k: round(100 * float(v), 1) for k, v in res.items()}
        print(ms, stem, f"{len(bl)} bands, {cv2.connectedComponents(gt)[0] - 1} GT lines", rows[f"{ms}/{stem}"], flush=True)
        for name in canv:
            cv2.imwrite(str(U.OUT / f"udiads_clcrop_{ms}_{stem}_{name}.png"), canv[name])
    (U.OUT / f"udiads_clcrop_{CKPT}{os.environ.get('CL_GAP', '')}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
