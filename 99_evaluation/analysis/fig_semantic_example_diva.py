#!/usr/bin/env python3
"""Figure 3.1, DIVA-HisDB row: U-Net postprocessing for polygon ground truth (predict_diva_seamcarve.py, preset
"unified", the setting of the thesis U-Net rows) on a test page with line FM 100% where the projection cuts are
visible. Uses the cached U-Net probabilities of the thesis model (ConvNeXt-Tiny, DINOv3 LVD-1689M) and its exported
polygons.
    .venv/bin/python 99_evaluation/analysis/fig_semantic_example_diva.py
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import shutil, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "50_modelling/semantic_segmentation/unet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict_diva_seamcarve as P  # noqa: E402
import fig_qualitative_v2 as Q  # noqa: E402
import overlap_lines_diva as O  # noqa: E402
from postproc import remove_small_objects  # noqa: E402

EV = ROOT / "99_evaluation/semantic_segmentation/unet/diva-hisdb"
RUN = "unet_tu-convnext_tiny.dinov3_lvd1689m_diva_{}"
U = P.PRESETS["unified"]
CLEAN = 200
WIN = (700, 1300)


def stages(sub, stem):
    prob = np.load(EV / "prob_cache" / RUN.format(sub) / f"{stem}.npy").astype(np.float32)
    thr = (prob > 0.5).astype(np.uint8)
    clean = remove_small_objects(thr, min_size=CLEAN)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (U["merge_width"], U["merge_height"]))
    merged = (cv2.morphologyEx(remove_small_objects(clean, CLEAN) * 255, cv2.MORPH_CLOSE, k) > 0)
    refined = P.separate_lines_projection(clean, merge_width=U["merge_width"], merge_height=U["merge_height"],
                                          line_sigma=U["line_sigma"], min_line_distance=U["min_line_distance"],
                                          peak_frac=U["peak_frac"], valley_ratio=U["valley_ratio"], min_area=CLEAN)
    removed = (thr > 0) & (clean == 0)
    cut = merged & (refined == 0)
    return prob, merged, removed, cut


def main():
    rows = []
    for sub in ("CB55", "CS18"):
        r = pd.read_csv(EV / "lines_eval" / RUN.format(sub) / "diva_csv" / "results.csv")
        for _, x in r[r.LinesFMeasure >= 0.9999].iterrows():
            stem = x.filename
            if not (EV / "prob_cache" / RUN.format(sub) / f"{stem}.npy").exists():
                continue
            prob, merged, removed, cut = stages(sub, stem)
            rows.append((int(cv2.connectedComponents(cut.astype(np.uint8))[0] - 1), sub, stem))
            print(sub, stem, "cut segments", rows[-1][0], flush=True)
    ncut, sub, stem = max(rows)
    prob, merged, removed, cut = stages(sub, stem)
    w = cv2.dilate(cut.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32) + removed
    ii = cv2.integral(w); H, W = w.shape; best, pos = -1, (0, 0)
    for y in range(0, H - WIN[0], 40):
        for x in range(0, W - WIN[1], 40):
            v = ii[y + WIN[0], x + WIN[1]] - ii[y, x + WIN[1]] - ii[y + WIN[0], x] + ii[y, x]
            if v > best:
                best, pos = v, (y, x)
    y0, x0 = pos; sl = (slice(y0, y0 + WIN[0]), slice(x0, x0 + WIN[1]))
    c = np.where(merged[..., None], np.uint8(30), np.uint8(255)).repeat(3, axis=2)
    c[removed] = (40, 40, 230)
    c[cut] = (230, 120, 0)
    cut_d = cv2.dilate(cut.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0     # cuts are 3 px; widen for print
    c[cut_d & ~removed] = (230, 120, 0)
    gtx, pix, imf = Q.files(sub, stem)
    img = cv2.cvtColor(cv2.imread(str(imf)), cv2.COLOR_BGR2RGB)
    fg = (cv2.imread(str(pix))[:, :, 0] & 0x08) > 0
    box = (x0, y0, x0 + WIN[1], y0 + WIN[0])
    polys = O.polys(EV / "lines_eval" / RUN.format(sub) / "pred_xml" / f"{stem}.xml")
    panels = {
        "a_prob": cv2.applyColorMap(255 - (np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_BONE)[sl],
        "b_merged": np.where(merged[..., None], np.uint8(30), np.uint8(255)).repeat(3, axis=2)[sl],
        "c_cuts": c[sl],
        "d_lines": cv2.cvtColor(Q.panel(img, fg, polys, box, max(2, WIN[1] // 450)), cv2.COLOR_RGB2BGR),
    }
    for k, v in panels.items():
        name = f"unetdiva_{k}.png"
        cv2.imwrite(str(Q.OUT / name), v); shutil.copy(Q.OUT / name, Q.THESIS / name)
    print("chosen", sub, stem, "cut segments on page", ncut, "window", box)


if __name__ == "__main__":
    main()
