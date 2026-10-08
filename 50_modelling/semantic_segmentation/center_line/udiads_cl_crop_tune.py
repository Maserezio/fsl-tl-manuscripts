#!/usr/bin/env python3
"""Tune the center-line -> box -> crop U-Net pipeline of udiads_cl_crop.py on the U-DIADS-TL VALIDATION pages.
Network outputs (center-line fragments with heights) are cached once per page; every grouping setting is then
decoded and scored with the crop U-Net (crop probabilities cached per box).

    .venv/bin/python 50_modelling/semantic_segmentation/center_line/udiads_cl_crop_tune.py SUBSET [n_pages]
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import itertools, json, sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import udiads_cl_crop as Q  # noqa: E402
B, LF, CL, U = Q.B, Q.LF, Q.CL, Q.U

GRID = {"group": ["row"], "gap": [2.0, 3.0, 4.0, 6.0], "dy": [0.5, 0.7, 0.9], "minlen": [1.5, 3.0], "mode": ["box"]}


def frags_of(net, img):
    H, W = img.shape[:2]
    with torch.inference_mode():
        s1 = 1152 / max(H, W)
        cen, _, up, dn = LF.run_net(net, cv2.resize(img, (int(W * s1), int(H * s1)), interpolation=cv2.INTER_AREA))
        sel = cen > 0.5
        thick = float(np.median(up[sel] + dn[sel])) if sel.sum() > 50 else LF.T0
        s2 = min(s1 * LF.T0 / max(thick, 1.0), 3200 / max(H, W))
        cen, end, up, dn = LF.run_net(net, cv2.resize(img, (int(W * s2), int(H * s2)),
                                                      interpolation=cv2.INTER_AREA if s2 < 1 else cv2.INTER_LINEAR))
    out = []
    for f in LF._fragments(cen, end, 0.4):
        ys_, xs_ = f["px"]
        out.append({"xs": f["xs"] / s2, "cy": f["cy"] / s2, "n": len(f["xs"]),
                    "hu": float(np.percentile(up[ys_, xs_], 75)) / s2, "hd": float(np.percentile(dn[ys_, xs_], 75)) / s2})
    return [f for f in out if len(f["xs"]) > 2]


def band_list(frg, gap, dy, minlen):
    if not frg:
        return []
    pp = CL.frag_pitch(frg)
    lines = [l for l in CL.group_row(frg, pp, dy_max=dy, gap_max=gap) if l["xs"][-1] - l["xs"][0] >= minlen * pp]
    out = []
    for l in sorted(lines, key=lambda q: q["cy"].mean()):
        hu, hd = CL.line_height(l)
        step = max(1, len(l["xs"]) // 60)
        x, y = l["xs"][::step], l["cy"][::step]
        out.append((np.concatenate([np.stack([x, y - hu], 1), np.stack([x, y + hd], 1)[::-1]]), hu + hd))
    return out


def score(img_bgr, gt, bl, mode, seg, cache):
    H, W = gt.shape
    canv, nid = np.zeros((H, W), np.uint16), 1
    for poly, h in bl:
        x0, y0 = poly.min(0); x1, y1 = poly.max(0)
        X0, Y0 = max(0, int(x0) - Q.PAD), max(0, int(y0) - Q.PAD)
        X1, Y1 = min(W, int(np.ceil(x1)) + Q.PAD), min(H, int(np.ceil(y1)) + Q.PAD)
        if X1 - X0 < 4 or Y1 - Y0 < 4:
            continue
        key = (X0, Y0, X1, Y1)
        if key not in cache:
            cache[key] = B.crop_probability(img_bgr[Y0:Y1, X0:X1], seg, 1024, 256) >= 0.5
        ink = cache[key]
        if mode == "box":
            mk = B.crop_base.connect_line(ink, 0.0).astype(bool)
        else:
            band = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
            cv2.fillPoly(band, [np.rint(poly - [X0, Y0]).astype(np.int32)], 1)
            k = max(3, int(0.5 * h) | 1)
            mk = ink & (cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0)
        reg = canv[Y0:Y1, X0:X1]
        wr = mk & (reg == 0)
        if wr.any():
            reg[wr] = nid; nid += 1
    return B.evaluate_contiguous(gt, canv)[4]


def main():
    ms = sys.argv[1]
    npg = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    net = LF.build_model()
    net.load_state_dict(torch.load(LF.MODELS / f"UDIADS{ms}_cm.pt", map_location="cpu", weights_only=False)["model"])
    net = net.to(LF.DEV).eval()
    seg, _ = B.crop_base.load_segmenter(Q.REPO / f"80_models/instance_segmentation/rtdetr_bbox_unet/u-diads-tl/crop_seg_loss_ablation_components_1024x256/{ms}/tversky/best.pth", B.crop_base.DEVICE)
    gdir = next(U.DATA / ms / f"text-line-gt-{ms}" / n for n in ("validation", "val") if (U.DATA / ms / f"text-line-gt-{ms}" / n).is_dir())
    idir = next(U.DATA / ms / f"img-{ms}" / n for n in ("validation", "val") if (U.DATA / ms / f"img-{ms}" / n).is_dir())
    pages = []
    for gp in sorted(gdir.glob("*.png"))[:npg]:
        img_bgr = cv2.imread(str(next(idir.glob(gp.stem + ".*"))))
        gt = (cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        pages.append((img_bgr, gt, frags_of(net, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)), {}, cv2.connectedComponents(gt)[0] - 1))
    rows = []
    for gap, dy, minlen, mode in itertools.product(GRID["gap"], GRID["dy"], GRID["minlen"], GRID["mode"]):
        fms, nb = [], []
        for img_bgr, gt, frg, cache, ngt in pages:
            bl = band_list(frg, gap, dy, minlen)
            nb.append(len(bl) / max(ngt, 1))
            fms.append(score(img_bgr, gt, bl, mode, seg, cache))
        rows.append({"gap": gap, "dy": dy, "minlen": minlen, "mode": mode, "FM": round(100 * float(np.mean(fms)), 2), "bands_per_gt": round(float(np.mean(nb)), 2)})
        print(rows[-1], flush=True)
    best = max(rows, key=lambda r: r["FM"])
    print("BEST", best)
    (U.OUT / f"udiads_clcrop_tune_{ms}.json").write_text(json.dumps({"rows": rows, "best": best}, indent=1))


if __name__ == "__main__":
    main()
