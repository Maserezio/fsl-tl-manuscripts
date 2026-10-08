#!/usr/bin/env python3
"""Center lines -> box -> crop U-Net on all U-DIADS-TL test pages, grouping chosen on Syr341 validation pages
(udiads_cl_crop_tune.py: row grouping, gap 2.0, dy 0.5, min length 3 pitches, box mode, pad 15)."""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json, sys
import cv2, numpy as np, torch
sys.path.insert(0, "99_evaluation/analysis")
import udiads_cl_crop_tune as T
U = T.U
res = {}
for ms in ("Latin14396", "Latin2", "Syr341"):
    net = T.LF.build_model(); net.load_state_dict(torch.load(T.LF.MODELS / f"UDIADS{ms}_cm.pt", map_location="cpu", weights_only=False)["model"]); net = net.to(T.LF.DEV).eval()
    seg, _ = T.B.crop_base.load_segmenter(T.Q.REPO / f"80_models/instance_segmentation/rtdetr_bbox_unet/u-diads-tl/crop_seg_loss_ablation_components_1024x256/{ms}/tversky/best.pth", T.B.crop_base.DEVICE)
    fms = []
    for gp in sorted((U.DATA / ms / f"text-line-gt-{ms}/test").glob("*.png")):
        ib = cv2.imread(str(next((U.DATA / ms / f"img-{ms}/test").glob(gp.stem + ".*")))); gt = (cv2.imread(str(gp), 0) > 0).astype(np.uint8)
        bl = T.band_list(T.frags_of(net, cv2.cvtColor(ib, cv2.COLOR_BGR2RGB)), 2.0, 0.5, 3.0)
        fms.append(T.score(ib, gt, bl, "box", seg, {}))
    res[ms] = round(100 * float(np.mean(fms)), 2)
    print(ms, res[ms], flush=True)
res["Mean"] = round(float(np.mean(list(res.values()))), 2)
print(res)
(U.OUT / "udiads_clcrop_test.json").write_text(json.dumps(res, indent=1))
