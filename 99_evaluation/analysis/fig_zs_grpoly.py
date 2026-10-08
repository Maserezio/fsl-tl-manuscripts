#!/usr/bin/env python3
"""Figure 6.3 (d-f): zero-shot CATMuS Mask R-CNN and center-line polygons on GRPOLY-DB test page 0024
(5.81 predicted lines per ground-truth line, close to the collection mean 5.82).
    .venv/bin/python 99_evaluation/analysis/fig_zs_grpoly.py   -> thesis graphics q2_zs_grpoly_0024_{gt,maskrcnn,centerline}.png
"""
import shutil
from pathlib import Path

import cv2

import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import fig_qualitative_v2 as Q  # noqa: E402
import overlap_lines_diva as O  # noqa: E402

STEM, BOX = "GRPOLY___0024", (0, 330, 2276, 1350)
MR = Q.ROOT / "99_evaluation/instance_segmentation/mask_rcnn/cross_collection/GRPOLY_maskrcnn_convnext_tiny_catmus_1152_zeroshot_story_zeroshot/test_pred_xml"
CL = Q.ROOT / "99_evaluation/semantic_segmentation/center_line/pred_lf/CATMuS_Z_test/row_1.0/GRPOLY"

if __name__ == "__main__":
    gtx, pix, imf = Q.files("GRPOLY", STEM)
    img = cv2.cvtColor(cv2.imread(str(imf)), cv2.COLOR_BGR2RGB)
    fg = (cv2.imread(str(pix))[:, :, 0] & 0x08) > 0
    lw = max(2, int(round((BOX[2] - BOX[0]) / 450)))
    for name, xml in (("gt", gtx), ("maskrcnn", MR / f"{STEM}.xml"), ("centerline", CL / f"{STEM}.xml")):
        out = f"q2_zs_grpoly_0024_{name}.png"
        cv2.imwrite(str(Q.OUT / out), cv2.cvtColor(Q.panel(img, fg, O.polys(xml), BOX, lw), cv2.COLOR_RGB2BGR))
        shutil.copy(Q.OUT / out, Q.THESIS / out)
        print(out)
