#!/usr/bin/env python3
"""Method figure of the two-stage pipeline on the story page (DIVA-HisDB CS18 test page 096): RT-DETR boxes
(ConvNeXt-Tiny, CATMuS, validation-selected confidence and width filter) and the BCE crop U-Net masks, drawn on
the same native-resolution crop as the story figure.

    ../../.venv/bin/python story_twostage_boxes.py  -> 99_evaluation/analysis/story_figures/twostage_cs18_096_{boxes,masks}.png
"""
import os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import component_analysis_diva as C  # noqa: E402
from evaluate import _remove_overlapping_rows, load_segm_model, segment_crop  # noqa: E402
from rtdetr_load import load_detector  # noqa: E402
from transformers import AutoImageProcessor  # noqa: E402

SUB, STEM = "CS18", "e-codices_csg-0018_096_max"
BOX = (1000, 2170, 2760, 2950)
PAL = [(214, 39, 40), (31, 119, 180), (44, 160, 44), (255, 127, 14), (148, 103, 189), (23, 190, 207)]


def main():
    p = C.paths(SUB)
    thr, frac = C.OPERATING[SUB]
    det = load_detector(p["det"]).to(C.DEV).eval()
    proc = AutoImageProcessor.from_pretrained(p["det"])
    seg = load_segm_model(p["seg"], "resnet34", "unet", C.DEV)
    img = cv2.imread(os.path.join(p["img"], f"{STEM}.jpg"))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    region, _ = C.gt_lines(os.path.join(p["page"], f"{STEM}.xml"))
    raw = C.detect(det, proc, rgb, thr)
    kept = _remove_overlapping_rows(C.region_filter(raw, region, frac))
    boxes_img = (0.6 * rgb + 0.4 * 255).astype(np.uint8)
    masks_img = boxes_img.copy()
    kept = kept[np.argsort((kept[:, 1] + kept[:, 3]) / 2)]
    keep_ids = {tuple(b[:4]) for b in kept}
    for b in raw:                                   # dropped by the region / width filter: gray, dashed look
        if tuple(b[:4]) not in keep_ids:
            x1, y1, x2, y2 = (int(v) for v in b[:4])
            cv2.rectangle(boxes_img, (x1, y1), (x2, y2), (150, 150, 150), 3)
    for k, b in enumerate(kept):
        col = PAL[k % len(PAL)]
        x1, y1, x2, y2 = (int(v) for v in b[:4])
        cv2.rectangle(boxes_img, (x1, y1), (x2, y2), col, 5)
        mask = segment_crop(img[y1:y2, x1:x2], seg, 1024, 256, C.DEV, 0.5) > 0
        sub = masks_img[y1:y2, x1:x2].astype(np.float32)
        sub[mask] = 0.4 * sub[mask] + 0.6 * np.array(col)
        masks_img[y1:y2, x1:x2] = sub.astype(np.uint8)
        cv2.rectangle(masks_img, (x1, y1), (x2, y2), col, 2)
    out = os.path.join(C.REPO, "99_evaluation/analysis/story_figures")
    x0, y0, x1, y1 = BOX
    cv2.imwrite(os.path.join(out, "twostage_cs18_096_boxes.png"), cv2.cvtColor(boxes_img[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out, "twostage_cs18_096_masks.png"), cv2.cvtColor(masks_img[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
    print("raw", len(raw), "kept", len(kept))


if __name__ == "__main__":
    main()
