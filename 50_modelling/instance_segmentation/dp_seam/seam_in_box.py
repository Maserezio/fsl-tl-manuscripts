#!/usr/bin/env python3
"""Seam carving inside the detection box (exploratory, DIVA-HisDB Task 2).

Same boxes (Mask R-CNN ConvNeXt-Tiny CATMuS 704, cached by contour_former.py cache SUB) and the same crop U-Net
(Tversky arm of the crop-loss study) as the Mask R-CNN + crop U-Net pipeline; only the polygon export differs.
Standard export: threshold the crop probability, keep the largest contour. Seam export: dynamic programming
finds an upper and a lower seam around the line on the probability map, maximizing the enclosed (P - thr) with
at most K rows of vertical change per 4-px column step; the polygon is the region between the seams, so it is
connected by construction, keeps detached ascenders/descenders when they pay off, and may overlap the
polygons of neighboring boxes.

    python seam_in_box.py SUB      probabilities, val grid (score, thr, K), test once
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "rtdetr_bbox_unet"))
import contour_former as C  # noqa: E402

STEP = 4
SCORES, THRS, KS = (0.5, 0.7, 0.9), (0.5, 0.7), (2, 8, 64)


def crop_prob(seg, crop_rgb):
    h0, w0 = crop_rgb.shape[:2]
    x = torch.from_numpy(cv2.resize(crop_rgb, (1024, 256))).permute(2, 0, 1).float().unsqueeze(0).to(C.DEV) / 255
    with torch.no_grad():
        p = torch.sigmoid(seg(x))[0, 0].float().cpu().numpy()
    return cv2.resize(p, (w0, h0))


def _dp(G, valid, K):
    """G (H, X) gains, valid mask; path t(x) maximizing sum G[t(x), x] with |t(x) - t(x-1)| <= K."""
    H, X = G.shape
    NEG = -1e9
    V = np.where(valid[:, 0], G[:, 0], NEG)
    back = np.zeros((H, X), np.int32)
    idx = np.arange(H)
    for x in range(1, X):
        best, arg = np.full(H, NEG), idx.copy()
        for d in range(-K, K + 1):                     # sliding-window max over the previous column
            src = idx + d
            ok = (src >= 0) & (src < H)
            cand = np.full(H, NEG); cand[ok] = V[src[ok]]
            better = cand > best
            best[better], arg[better] = cand[better], src[better]
        V = np.where(valid[:, x], G[:, x] + best, NEG)
        back[:, x] = arg
    t = np.zeros(X, np.int32); t[-1] = int(V.argmax())
    for x in range(X - 1, 0, -1):
        t[x - 1] = back[t[x], x]
    return t


def seam_polygon(P, thr, K):
    """P (H, W) crop probability -> polygon (n, 2) in crop coordinates, or None."""
    H, W = P.shape
    X = W // STEP
    if X < 2:
        return None
    Pc = P[:, :X * STEP].reshape(H, X, STEP).mean(axis=2)
    colmax = Pc.max(axis=0)
    on = np.where(colmax >= thr)[0]
    if len(on) < 2:
        return None
    a, b = on[0], on[-1] + 1                            # horizontal extent of the line
    Pc = Pc[:, a:b]
    X = b - a
    blur = cv2.GaussianBlur(Pc, (1, 0), sigmaX=0.1, sigmaY=max(1.0, H / 12))
    yc = blur.argmax(axis=0).astype(float)
    weak = Pc.max(axis=0) < thr                         # word gaps: interpolate the line center
    if weak.any() and (~weak).any():
        yc[weak] = np.interp(np.where(weak)[0], np.where(~weak)[0], yc[~weak])
    yc = np.round(cv2.medianBlur(yc.astype(np.float32).reshape(1, -1), 5).ravel()).astype(int).clip(0, H - 1)
    S = Pc - thr
    Cs = np.vstack([np.zeros((1, X)), np.cumsum(S, axis=0)])        # Cs[y] = sum S[:y]
    rows = np.arange(H)[:, None]
    cols = np.arange(X)
    top_gain = Cs[yc + 1, cols][None] - Cs[rows, cols[None]]         # sum S[t..yc]
    bot_gain = Cs[rows + 1, cols[None]] - Cs[yc, cols][None]         # sum S[yc..b]
    t = _dp(top_gain, rows <= yc[None], K)
    u = _dp(bot_gain, rows >= yc[None], K)
    xs = (a + np.arange(X)) * STEP
    top = [(x, y) for i, (x, y) in enumerate(zip(xs, t)) for x in (x, x + STEP)]
    bot = [(x, y + 1) for i, (x, y) in enumerate(zip(xs, u)) for x in (x, x + STEP)]
    return np.array(top + bot[::-1], np.int32)


def standard_polygon(P, thr):
    m = (P >= thr).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cs, key=cv2.contourArea)[:, 0] if cs else None


def main(sub):
    from evaluate import load_segm_model
    seg = load_segm_model(str(C.REPO / f"80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_loss_ablation_components_1024x256/{sub}/tversky/best.pth"),
                          "resnet34", "unet", str(C.DEV))
    det = json.loads((C.EVAL / f"{sub}_detections.json").read_text())
    polys = {}                                          # (split, stem) -> list of (score, {(mode, thr, K): polygon})
    for key, v in det.items():
        split, stem = key.split("/")
        img = cv2.cvtColor(cv2.imread(v["path"]), cv2.COLOR_BGR2RGB)
        Hh, Ww = img.shape[:2]
        region = np.array(v["region"], np.int32)
        out = []
        for (x0, y0, x1, y1), s in zip(v["boxes"], v["scores"]):
            if s < min(SCORES) or cv2.pointPolygonTest(region, (float((x0 + x1) / 2), float((y0 + y1) / 2)), False) < 0:
                continue
            cx0, cy0 = int(max(0, np.floor(x0) - C.PAD)), int(max(0, np.floor(y0) - C.PAD))
            cx1, cy1 = int(min(Ww, np.ceil(x1) + C.PAD)), int(min(Hh, np.ceil(y1) + C.PAD))
            P = crop_prob(seg, img[cy0:cy1, cx0:cx1])
            d = {}
            for thr in THRS:
                fg = (P >= thr).mean()
                if fg < 0.005 or fg > 0.95:             # same sanity filter as maskrcnn_crop_refine.py
                    continue
                p = standard_polygon(P, thr)
                d[("std", thr, 0)] = None if p is None else p + [cx0, cy0]
                for K in KS:
                    p = seam_polygon(P, thr, K)
                    d[("seam", thr, K)] = None if p is None else p + [cx0, cy0]
            out.append((s, d))
        polys[(split, stem)] = (out, v)
        print(key, len(out), flush=True)

    def run(split, mode, sc, thr, K):
        dd = C.EVAL / sub / f"seam_{split}_{mode}_s{sc:g}_t{thr:g}_k{K}"
        dd.mkdir(parents=True, exist_ok=True)
        for (sp, stem), (out, v) in polys.items():
            if sp == split:
                ps = [d.get((mode, thr, K)) for s, d in out if s >= sc]
                C.write_page(dd / f"{stem}.xml", v["size"], v["region"], [p.tolist() for p in ps if p is not None and len(p) >= 3])
        return C.score_dir(sub, split, dd)

    res = {}
    for mode, Ks in (("std", (0,)), ("seam", KS)):
        rows = []
        for sc in SCORES:
            for thr in THRS:
                for K in Ks:
                    r = run("val", mode, sc, thr, K)
                    rows.append((r["LinesFMeasure"], r["PixelIU"], sc, thr, K))
                    print(f"  [val] {mode} s={sc} t={thr} K={K} FM={100 * r['LinesFMeasure']:.2f}", flush=True)
        _, _, sc, thr, K = max(rows)
        r = run("test", mode, sc, thr, K)
        res[mode] = {"score": sc, "thr": thr, "K": K, **{k: round(100 * v, 2) for k, v in r.items()}}
        print(sub, mode, res[mode], flush=True)
    (C.EVAL / sub / "seam_summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
