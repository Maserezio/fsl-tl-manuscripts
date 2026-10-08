#!/usr/bin/env python3
"""Method figure of the dynamic-programming seam (DIVA-HisDB CB55 test page, Mask R-CNN ConvNeXt-Tiny
CATMuS boxes + Tversky crop U-Net, the CB55 operating points of seam_summary.json).

Picks the detected line whose largest-contour export loses the largest share of the thresholded crop
foreground and draws four aligned strips of that crop:
 (a) crop image, (b) crop U-Net probability with the estimated center row,
 (c) upper and lower DP paths and the region between them, (d) largest contour (std) vs seam polygon.
Writes graphics/dpseam_{a,b,c,d}.png into the thesis repo.
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json, sys
from pathlib import Path
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "50_modelling/instance_segmentation/dp_seam"))
sys.path.insert(0, str(REPO / "50_modelling/instance_segmentation/rtdetr_bbox_unet"))
import contour_former as C  # noqa: E402
import seam_in_box as S  # noqa: E402

OUT = Path.home() / "Thesis/fsl-tl-manuscripts-thesis/graphics"
SUB = "CB55"


def seam_parts(P, thr, K):
    """Same computation as seam_in_box.seam_polygon, but returns the center row and both paths."""
    H, W = P.shape
    X = W // S.STEP
    Pc = P[:, :X * S.STEP].reshape(H, X, S.STEP).mean(axis=2)
    on = np.where(Pc.max(axis=0) >= thr)[0]
    a, b = on[0], on[-1] + 1
    Pc = Pc[:, a:b]; X = b - a
    blur = cv2.GaussianBlur(Pc, (1, 0), sigmaX=0.1, sigmaY=max(1.0, H / 12))
    yc = blur.argmax(axis=0).astype(float)
    weak = Pc.max(axis=0) < thr
    if weak.any() and (~weak).any():
        yc[weak] = np.interp(np.where(weak)[0], np.where(~weak)[0], yc[~weak])
    yc = np.round(cv2.medianBlur(yc.astype(np.float32).reshape(1, -1), 5).ravel()).astype(int).clip(0, H - 1)
    Sg = Pc - thr
    Cs = np.vstack([np.zeros((1, X)), np.cumsum(Sg, axis=0)])
    rows, cols = np.arange(H)[:, None], np.arange(X)
    t = S._dp(Cs[yc + 1, cols][None] - Cs[rows, cols[None]], rows <= yc[None], K)
    u = S._dp(Cs[rows + 1, cols[None]] - Cs[yc, cols][None], rows >= yc[None], K)
    xs = (a + np.arange(X)) * S.STEP + S.STEP / 2
    return xs, yc, t, u


def main():
    summ = json.loads((C.EVAL / SUB / "seam_summary.json").read_text())
    thr_std, thr_seam, K, sc = summ["std"]["thr"], summ["seam"]["thr"], summ["seam"]["K"], summ["seam"]["score"]
    from evaluate import load_segm_model
    seg = load_segm_model(str(C.REPO / f"80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_loss_ablation_components_1024x256/{SUB}/tversky/best.pth"),
                          "resnet34", "unet", str(C.DEV))
    det = json.loads((C.EVAL / f"{SUB}_detections.json").read_text())
    best = None
    for key, v in det.items():
        if not key.startswith("test/"):
            continue
        img = cv2.cvtColor(cv2.imread(v["path"]), cv2.COLOR_BGR2RGB)
        Hh, Ww = img.shape[:2]
        region = np.array(v["region"], np.int32)
        for (x0, y0, x1, y1), s in zip(v["boxes"], v["scores"]):
            if s < sc or cv2.pointPolygonTest(region, (float((x0 + x1) / 2), float((y0 + y1) / 2)), False) < 0:
                continue
            cx0, cy0 = int(max(0, np.floor(x0) - C.PAD)), int(max(0, np.floor(y0) - C.PAD))
            cx1, cy1 = int(min(Ww, np.ceil(x1) + C.PAD)), int(min(Hh, np.ceil(y1) + C.PAD))
            crop = img[cy0:cy1, cx0:cx1]
            P = S.crop_prob(seg, crop)
            m = (P >= thr_std).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
            if n < 3:
                continue
            areas = st[1:, cv2.CC_STAT_AREA]
            lost = 1 - areas.max() / areas.sum()
            if crop.shape[1] > 600 and (best is None or lost > best[0]):
                best = (lost, key, crop, P)
    lost, key, crop, P = best
    print("chosen", key, "lost share %.3f" % lost, "crop", crop.shape)
    H, W = P.shape
    base = (0.6 * crop + 0.4 * 255).astype(np.uint8)
    # (a)
    cv2.imwrite(str(OUT / "dpseam_a.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    # (b) probability + center row
    xs, yc, t, u = seam_parts(P, thr_seam, K)
    pb = cv2.applyColorMap((np.clip(P, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    for i in range(0, len(xs), 3):
        cv2.circle(pb, (int(xs[i]), int(yc[i])), 2, (255, 255, 255), -1)
    cv2.imwrite(str(OUT / "dpseam_b.png"), pb)
    # (c) DP paths and region
    c = base.copy().astype(np.float32)
    poly = S.seam_polygon(P, thr_seam, K)
    layer = c.copy(); cv2.fillPoly(layer, [poly], (60, 160, 60)); c = 0.65 * c + 0.35 * layer
    c = c.astype(np.uint8)
    cv2.polylines(c, [np.stack([xs, t], 1).astype(np.int32)], False, (200, 30, 30), 2)
    cv2.polylines(c, [np.stack([xs, u + 1], 1).astype(np.int32)], False, (30, 60, 200), 2)
    cv2.imwrite(str(OUT / "dpseam_c.png"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
    # (d) largest contour vs seam polygon
    d = base.copy()
    m = (P >= thr_std).astype(np.uint8)
    lost_px = m.copy()
    std = S.standard_polygon(P, thr_std)
    keep = np.zeros_like(m); cv2.fillPoly(keep, [std.astype(np.int32)], 1)
    d[(m > 0) & (keep == 0)] = (220, 30, 30)
    cv2.polylines(d, [std.astype(np.int32)], True, (220, 120, 0), 2)
    cv2.polylines(d, [poly], True, (40, 140, 40), 2)
    cv2.imwrite(str(OUT / "dpseam_d.png"), cv2.cvtColor(d, cv2.COLOR_RGB2BGR))
    print("thr_std", thr_std, "thr_seam", thr_seam, "K", K)


if __name__ == "__main__":
    main()
