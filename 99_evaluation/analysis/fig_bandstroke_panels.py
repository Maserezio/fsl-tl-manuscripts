#!/usr/bin/env python3
"""Center-line panels of the thesis figures redrawn with the bandseamstroke decoder (band + attached ink strokes split at the minimum-ink seams,
cl_bandstroke.py) of the thesis DIVA-HisDB center-line model (CATMuS pretraining, 20 pages); same crops as before.
    .venv/bin/python 99_evaluation/analysis/fig_bandstroke_panels.py
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import shutil, sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_qualitative_v2 as Q  # noqa: E402
import overlap_lines_diva as O  # noqa: E402

BS = str(Q.ROOT / "99_evaluation/semantic_segmentation/center_line/diva_cbad/pred_cm_cellseam/test/{}_full/bandseamstroke_inside")


def story(sub, stem, box, out):
    gtx, pix, imf = Q.files(sub, stem)
    img = cv2.cvtColor(cv2.imread(str(imf)), cv2.COLOR_BGR2RGB)
    fg = (cv2.imread(str(pix))[:, :, 0] & 0x08) > 0
    xml = Path(BS.format(sub)) / f"{stem}.xml"
    o = Q.panel(img, fg, O.polys(xml), box, max(2, int(round((box[2] - box[0]) / 450))))
    cv2.imwrite(str(Q.OUT / out), cv2.cvtColor(o, cv2.COLOR_RGB2BGR)); shutil.copy(Q.OUT / out, Q.THESIS / out)
    print(out, o.shape, "page FM %.2f" % Q.page_fm(sub, stem, xml))


def overlap(sub, stem, box, out):
    gtx, pix, imf = Q.files(sub, stem)
    x0, y0, x1, y1 = box
    img = cv2.cvtColor(cv2.imread(str(imf)), cv2.COLOR_BGR2RGB)
    fg = (cv2.imread(str(pix))[:, :, 0] & 0x08) > 0
    xml = Path(BS.format(sub)) / f"{stem}.xml"
    ps = sorted([p for p in O.polys(xml) if O.inter(O.bbox(p), box)], key=lambda p: p[:, 1].mean())
    base = (0.55 * img[y0:y1, x0:x1] + 0.45 * 255).astype(np.float32)
    cnt = np.zeros(base.shape[:2], np.uint8)
    for k, p in enumerate(ps):
        m = O.rast(p, box); cnt += m
        sel = m & fg[y0:y1, x0:x1]
        base[sel] = 0.35 * base[sel] + 0.65 * np.array(O.PAL[k % len(O.PAL)])
    twice = (cnt >= 2) & fg[y0:y1, x0:x1]
    base[twice] = (20, 20, 20)
    o = base.astype(np.uint8).copy()
    for k, p in enumerate(ps):
        cv2.polylines(o, [p - [x0, y0]], True, O.PAL[k % len(O.PAL)], 3)
    cv2.imwrite(str(Q.OUT / out), cv2.cvtColor(o, cv2.COLOR_RGB2BGR)); shutil.copy(Q.OUT / out, Q.THESIS / out)
    print(out, o.shape, "ink in two polygons", int(twice.sum()), "page FM %.2f" % Q.page_fm(sub, stem, xml))


def rq3(ds, stem, box, pred_dir, out, dest):
    """RQ3 panel at the crop of the thesis figure; written to dest (staging) only, not to the thesis folder."""
    gtx, pix, imf = Q.files(ds, stem)
    img = cv2.cvtColor(cv2.imread(str(imf)), cv2.COLOR_BGR2RGB)
    fg = (cv2.imread(str(pix))[:, :, 0] & 0x08) > 0
    xml = Path(pred_dir) / f"{stem}.xml"
    o = Q.panel(img, fg, O.polys(xml), box, max(2, int(round((box[2] - box[0]) / 450))))
    Path(dest).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(dest) / out), cv2.cvtColor(o, cv2.COLOR_RGB2BGR))
    print(out, o.shape, "page FM %.2f" % Q.page_fm(ds, stem, xml), flush=True)


if __name__ == "__main__" and sys.argv[1:2] == ["rq3"]:
    P = Q.ROOT / "99_evaluation/semantic_segmentation/center_line/pred_lf"
    dest = Q.ROOT / "99_evaluation/analysis/story_figures/bandstroke_staging"
    rq3("RASM", "RASM__Or 3366_0255", (500, 5150, 4200, 7450), P / "CATMuS_Z_test_h/row_1.0_bstroke/RASM",
        "q2_zs_rasm_3366_0255_centerline.png", dest)
    rq3("Phil_gr_130", "Phil_gr_130__Phil.gr.130_0090r", (380, 2900, 1300, 3700),
        P / "ONB_A_w50_test_h/row_1.0_bstroke/Phil_gr_130", "q2_phil_0090r_centerline.png", dest)
    # control: the thesis decoder on the re-cached predictions must reproduce the published panels
    rq3("RASM", "RASM__Or 3366_0255", (500, 5150, 4200, 7450), P / "CATMuS_Z_test_h/row_1.0/RASM",
        "control_rasm_row.png", dest)
elif __name__ == "__main__":
    story("CS18", "e-codices_csg-0018_096_max", (1000, 2330, 2650, 2715), "q2_diva_CS18_centerline.png")
    overlap("CB55", "e-codices_fmb-cb-0055_0099v_max", (2300, 4150, 3200, 4560), "overlap_cb55_0099v_centerline.png")
