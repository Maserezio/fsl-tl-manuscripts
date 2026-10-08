#!/usr/bin/env python3
"""Method figure of the center-line model on a Phil. gr. 130 development page (protocol-A model trained on
ONB Cod. Syr. 1, WiSE-FT 0.5): (a) input at the second-pass processing scale, (b) predicted center map (red)
and line-end map (blue), (c) fragments and joined center lines, (d) nearest-center-line assignment of the
ink pixels. All panels show the same crop of the page at its native resolution; the network maps are
interpolated from the processing scale to it.

    .venv/bin/python 99_evaluation/analysis/story_centerline_method.py -> 99_evaluation/analysis/story_figures/cl_method_{a,b,c,d}.png
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rq3_linefield as LF  # noqa: E402
import rq3_centre_lf as CL  # noqa: E402

OUT = HERE / "story_figures"
PAGE_IDX = 0
CROP = (0.04, 0.33, 0.54, 0.42)       # x0, y0, x1, y1 as fractions of the page


def main():
    root = CL.DEV / "Phil_gr_130"
    coco = json.loads((root / "coco_instances/dev.json").read_text())
    im = sorted(coco["images"], key=lambda x: x["file_name"])[PAGE_IDX]
    img = np.asarray(Image.open(root / "images/dev" / im["file_name"]).convert("RGB"))
    H, W = img.shape[:2]
    net = LF.build_model()
    net.load_state_dict(torch.load(LF.MODELS / "ONB_A_w50.pt", map_location="cpu", weights_only=False)["model"])
    net = net.to(LF.DEV).eval()
    with torch.inference_mode():
        s1 = 1152 / max(H, W)
        cen, _, up, dn = LF.run_net(net, cv2.resize(img, (int(W * s1), int(H * s1)), interpolation=cv2.INTER_AREA))
        sel = cen > 0.5
        s2 = min(s1 * LF.T0 / max(float(np.median(up[sel] + dn[sel])), 1.0), 3200 / max(H, W))
        im2 = cv2.resize(img, (int(W * s2), int(H * s2)), interpolation=cv2.INTER_AREA if s2 < 1 else cv2.INTER_LINEAR)
        cen, end, up, dn = LF.run_net(net, im2)
    frags = [{"xs": f["xs"], "cy": f["cy"], "n": len(f["xs"])} for f in LF._fragments(cen, end, 0.4)]
    frags = [f for f in frags if len(f["xs"]) > 2]
    p = CL.frag_pitch(frags)
    lines = [l for l in CL.group_row(frags, p) if l["xs"][-1] - l["xs"][0] >= 0.5 * p]
    # everything is drawn on the native-resolution page crop; network maps are interpolated to it
    X0, Y0, X1, Y1 = (int(CROP[0] * W), int(CROP[1] * H), int(CROP[2] * W), int(CROP[3] * H))
    base = img[Y0:Y1, X0:X1].copy()
    ch, cw = base.shape[:2]
    up_map = lambda a: cv2.resize(a[int(Y0 * s2):int(Y1 * s2), int(X0 * s2):int(X1 * s2)], (cw, ch), interpolation=cv2.INTER_LINEAR)
    nat = lambda xs, ys: np.stack([xs / s2 - X0, ys / s2 - Y0], 1)
    lines_n = [nat(l["xs"], l["cy"]).astype(np.float32) for l in lines]
    inst = CL.CI.cells(lines_n, (ch, cw), p / s2)
    import rq3_merge_split as MS
    ink = MS.ink_mask(cv2.cvtColor(base, cv2.COLOR_RGB2GRAY))
    lw = max(2, int(round(0.08 * p / s2)))
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(base).save(OUT / "cl_method_a.png")
    light = (0.55 * base + 0.45 * 255).astype(np.float32)
    c, e = up_map(cen)[..., None], up_map(end)[..., None]
    b = light * (1 - c) + c * np.array([220, 30, 30])
    b = b * (1 - e) + e * np.array([30, 80, 230])
    Image.fromarray(b.clip(0, 255).astype(np.uint8)).save(OUT / "cl_method_b.png")
    pal = np.array([[214, 39, 40], [31, 119, 180], [44, 160, 44], [255, 127, 14], [148, 103, 189], [23, 190, 207]])
    cols = np.concatenate([[[0, 0, 0]], pal[np.arange(len(lines)) % len(pal)]])   # neighbors differ
    cimg = light.astype(np.uint8).copy()
    for f in frags:
        cv2.polylines(cimg, [nat(f["xs"], f["cy"]).astype(np.int32)], False, (90, 90, 90), 3 * lw)
    # predicted line height: vertical segments from center - up to center + down (processing scale),
    # drawn every ~3 pitches along each joined center line, alternating lines offset by half a step
    from scipy.ndimage import median_filter
    step = max(1, int(round(3 * p)))
    for i, l in enumerate(lines):
        xs, cy = np.asarray(l["xs"]), np.asarray(l["cy"], dtype=np.float32)
        yi = np.clip(np.round(cy).astype(int), 0, up.shape[0] - 1); xi = np.clip(xs.astype(int), 0, up.shape[1] - 1)
        u = median_filter(up[yi, xi].astype(np.float32), size=9); d_ = median_filter(dn[yi, xi].astype(np.float32), size=9)
        top, bot = nat(xs, cy - u), nat(xs, cy + d_)
        col = tuple(int(v) for v in cols[i + 1])
        for j in range((i % 2) * step // 2, len(xs), step):
            t, b_ = top[j].astype(int), bot[j].astype(int)
            cv2.line(cimg, tuple(t), tuple(b_), col, max(1, lw // 2))
            for q in (t, b_):
                cv2.line(cimg, (q[0] - 2 * lw, q[1]), (q[0] + 2 * lw, q[1]), col, max(1, lw // 2))
    for i, l in enumerate(lines_n):
        cv2.polylines(cimg, [l.astype(np.int32)], False, tuple(int(v) for v in cols[i + 1]), lw)
    Image.fromarray(cimg).save(OUT / "cl_method_c.png")
    d = light.astype(np.uint8).copy()
    m = ink & (inst > 0)
    d[m] = cols[inst[m]]
    Image.fromarray(d).save(OUT / "cl_method_d.png")
    x0, y0, x1, y1 = X0, Y0, X1, Y1
    print(im["file_name"], "s2", round(s2, 3), "crop", (x1 - x0, y1 - y0), "lines", len(lines), "pitch", round(p, 1))


if __name__ == "__main__":
    main()
