#!/usr/bin/env python3
"""Mask R-CNN counterpart of the RT-DETR method figure (story_twostage_boxes.py): DIVA-HisDB CS18 test page 096,
ConvNeXt-Tiny Mask R-CNN with CATMuS initialization (thesis model), same crop. Boxes of the exported detections
(colored; dropped detections are not drawn),
and the ROI masks restored to page resolution and clipped to the main-text region, as in write_prediction_xml.
CPU only.  .venv/bin/python 99_evaluation/analysis/fig_maskrcnn_cs18_096.py
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json, os, sys
from pathlib import Path
import cv2, numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "50_modelling/instance_segmentation/mask_rcnn"))
import maskrcnn_diva as M  # noqa: E402

SUB, STEM = "CS18", "e-codices_csg-0018_096_max"
BOX = (1000, 2170, 2760, 2950)          # same crop as twostage_cs18_096_{boxes,masks}.png
PAL = [(214, 39, 40), (31, 119, 180), (44, 160, 44), (255, 127, 14), (148, 103, 189), (23, 190, 207)]
RUN = REPO / f"80_models/instance_segmentation/mask_rcnn/diva-hisdb/{SUB}/maskrcnn_convnext_tiny_catmus_704"
EVAL = REPO / f"99_evaluation/instance_segmentation/mask_rcnn/diva-hisdb/{SUB}/maskrcnn_convnext_tiny_catmus_704"
GT = REPO / f"00_data/DIVA-HisDB/{SUB}/PAGE-gt-{SUB}-TASK-2/TASK-2/public-test"


def main():
    torch.set_num_threads(8)
    test_dir = next(EVAL.glob("diva_test_s*"))
    summary = json.loads((test_dir / "summary.json").read_text()) if (test_dir / "summary.json").exists() else {}
    score_thr = float(summary.get("score_threshold", float(test_dir.name.split("_s")[1].split("_m")[0])))
    mask_thr = 0.5
    data = M.DivaCocoLines(REPO / f"00_data/DIVA-HisDB/coco_task2_{SUB}", REPO / f"00_data/DIVA-HisDB/yolo_dataset_{SUB}/images",
                           "test", 704, augment=False)
    idx = next(i for i in range(len(data)) if Path(data[i][2]["path"]).stem == STEM)
    img_t, _, meta = data[idx]
    ck = torch.load(RUN / "best.pt", map_location="cpu", weights_only=True)
    model = M.build_model("convnext_tiny_catmus", 704, ck["args"]["mask_roi_size"], ck["args"]["mask_roi_width"])
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    with torch.no_grad():
        out = model([img_t])[0]
    ow, oh, nw, nh = int(meta["orig_w"]), int(meta["orig_h"]), int(meta["new_w"]), int(meta["new_h"])
    sx, sy = ow / nw, oh / nh
    region = M.region_polygon(GT / f"{STEM}.xml")
    region_mask = np.zeros((oh, ow), np.uint8)
    cv2.fillPoly(region_mask, [region.astype(np.int32)], 1)
    rgb = cv2.cvtColor(cv2.imread(str(next((REPO / f"00_data/DIVA-HisDB/{SUB}/img-{SUB}/img/public-test").glob(STEM + ".*")))),
                       cv2.COLOR_BGR2RGB)
    boxes_img = (0.6 * rgb + 0.4 * 255).astype(np.uint8)
    masks_img = boxes_img.copy()
    kept, dropped = [], []
    for box, score, mask in zip(out["boxes"].numpy(), out["scores"].numpy(), out["masks"][:, 0].numpy()):
        b = (int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy))
        if score < score_thr:
            dropped.append(b); continue
        restored = cv2.resize(mask[:nh, :nw], (ow, oh), interpolation=cv2.INTER_LINEAR)
        binary = (restored >= mask_thr).astype(np.uint8) * region_mask
        cs, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs or cv2.contourArea(max(cs, key=cv2.contourArea)) < 100:
            dropped.append(b); continue
        kept.append((b, binary.astype(bool)))
    kept.sort(key=lambda kb: (kb[0][1] + kb[0][3]) / 2)
    # dropped detections (69 of 100 on this page, mostly low scores) are not drawn: they would hide the page
    for k, (b, m) in enumerate(kept):
        col = PAL[k % len(PAL)]
        cv2.rectangle(boxes_img, b[:2], b[2:], col, 5)
        sub = masks_img.astype(np.float32)
        sub[m] = 0.4 * sub[m] + 0.6 * np.array(col)
        masks_img = sub.astype(np.uint8)
        cv2.rectangle(masks_img, b[:2], b[2:], col, 2)
    x0, y0, x1, y1 = BOX
    dst = REPO / "99_evaluation/analysis/story_figures"
    for name, im in (("boxes", boxes_img), ("masks", masks_img)):
        cv2.imwrite(str(dst / f"maskrcnn_cs18_096_{name}.png"), cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
    print("score_thr", score_thr, "detections", len(out["scores"]), "kept", len(kept), "dropped", len(dropped))


if __name__ == "__main__":
    main()
