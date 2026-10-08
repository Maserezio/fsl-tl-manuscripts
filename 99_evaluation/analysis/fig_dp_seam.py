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
import json, os, sys
from pathlib import Path
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "50_modelling/instance_segmentation/dp_seam"))
sys.path.insert(0, str(REPO / "50_modelling/instance_segmentation/rtdetr_bbox_unet"))
import contour_former as C  # noqa: E402
import seam_in_box as S  # noqa: E402

OUT = Path.home() / "Thesis/fsl-tl-manuscripts-thesis/graphics"
SUB = os.environ.get("DP_SUB", "CB55")   # subset; DP_RANK picks the n-th worst line (default 0)


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
    cands = []
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
            if crop.shape[1] > 600:
                cands.append((lost, key, crop, P))
    cands.sort(key=lambda c: -c[0])
    for c in cands[:5]:
        print("candidate", c[1], "lost share %.3f" % c[0], "crop", c[2].shape)
    if os.environ.get("DP_SCAN") == "1":
        return
    lost, key, crop, P = cands[int(os.environ.get("DP_RANK", "0"))]
    print("chosen", key, "lost share %.3f" % lost, "crop", crop.shape)
    H, W = P.shape
    xs, yc, t, u = seam_parts(P, thr_seam, K)
    poly = S.seam_polygon(P, thr_seam, K)
    base = (0.6 * crop + 0.4 * 255).astype(np.uint8)
    # (a) problem: thresholded mask, largest contour kept (orange) and discarded parts (red)
    m = (P >= thr_std).astype(np.uint8)
    std = S.standard_polygon(P, thr_std)
    keep = np.zeros_like(m); cv2.fillPoly(keep, [std.astype(np.int32)], 1)
    a = base.copy()
    a[(m > 0) & (keep > 0)] = (235, 150, 40)
    a[(m > 0) & (keep == 0)] = (215, 35, 35)
    cv2.imwrite(str(OUT / "dpseam_a.png"), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    # (c) result: seam region (green) on the image
    c = base.copy().astype(np.float32)
    layer = c.copy(); cv2.fillPoly(layer, [poly], (60, 170, 60)); c = (0.6 * c + 0.4 * layer).astype(np.uint8)
    cv2.polylines(c, [poly], True, (30, 120, 30), 2)
    cv2.imwrite(str(OUT / "dpseam_c.png"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
    # (b) zoom on the widest word gap: P heatmap, 4-px columns, center rows, both DP paths
    Pc_max = np.array([P[:, int(x - S.STEP / 2):int(x + S.STEP / 2)].max() for x in xs])
    weak = Pc_max < thr_seam
    runs, start = [], None
    for k, w in enumerate(np.append(weak, False)):
        if w and start is None: start = k
        if not w and start is not None: runs.append((k - start, start, k)); start = None
    _, r0, r1 = max(runs)
    xc = int((xs[r0] + xs[r1 - 1]) / 2)
    half = 110
    zx0, zx1 = max(0, xc - half), min(W, xc + half)
    Z = 3
    heat = cv2.applyColorMap((np.clip(P[:, zx0:zx1], 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.resize(heat, ((zx1 - zx0) * Z, H * Z), interpolation=cv2.INTER_NEAREST)
    for x in range(0, zx1 - zx0, S.STEP):
        cv2.line(heat, (x * Z, 0), (x * Z, H * Z - 1), (90, 90, 90), 1)
    sel = (xs >= zx0) & (xs < zx1)
    zp = lambda xx, yy: np.stack([(xx - zx0) * Z, yy * Z], 1).astype(np.int32)
    def steps(yy):            # draw each 4-px column as a flat segment, joined by vertical steps
        pts = []
        for x, y in zip(xs[sel], yy[sel]):
            pts += [((x - S.STEP / 2 - zx0) * Z, y * Z), ((x + S.STEP / 2 - zx0) * Z, y * Z)]
        return np.array(pts, np.int32)
    cv2.polylines(heat, [steps(t)], False, (230, 40, 40), 3)
    cv2.polylines(heat, [steps(u + 1)], False, (60, 120, 255), 3)
    for x, y, w in zip(xs[sel], yc[sel], weak[sel]):
        cv2.circle(heat, (int((x - zx0) * Z), int(y * Z)), 4, (255, 255, 255) if not w else (255, 200, 0), -1)
    cv2.imwrite(str(OUT / "dpseam_b.png"), cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))
    # marker of the zoom window in (a) and (c)
    for f in ("dpseam_a.png", "dpseam_c.png"):
        im = cv2.imread(str(OUT / f)); cv2.rectangle(im, (zx0, 0), (zx1 - 1, H - 1), (0, 0, 0), 3); cv2.imwrite(str(OUT / f), im)
    print("zoom", zx0, zx1, "gap columns", r1 - r0)
    print("thr_std", thr_std, "thr_seam", thr_seam, "K", K)


if __name__ == "__main__":
    main()
