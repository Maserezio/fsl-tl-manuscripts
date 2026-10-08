#!/usr/bin/env python3
"""Low-budget RQ3 mask experiments on retained Protocol-A checkpoints.

Idea A (inference only): overlapping mask pixels go to the instance with the highest
mask probability at that pixel instead of the most confident detection (the official
export paints masks in descending detection score).

Idea B (mask-head-only fine-tune): the frozen detector's mask head is fine-tuned on
the source training pages with extra RoIs made by stretching each GT box vertically,
so that the RoI contains neighbouring lines while the target mask stays the same line.

Development data are the *val* splits of every collection (no test page is used for
any decision). Settings are fixed in advance; nothing is tuned on the reported data.
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

for key in list(os.environ):
    if key.startswith("RQ3_"):
        os.environ.pop(key)
os.environ["RQ3_REPLACE_ANCHORS"] = "0"
os.environ.setdefault("DIVA_WORKERS", "6")

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models.detection.roi_heads import project_masks_on_boxes

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "50_modelling/instance_segmentation/mask_rcnn/cross_collection"))
import maskrcnn_rq3_loo as m

EVAL = ROOT / "99_evaluation/instance_segmentation/mask_rcnn/cross_collection"
CKPT = ROOT / "80_models/instance_segmentation/mask_rcnn/cross_collection/final_ab"
MATRIX = ROOT / "00_data/RQ3/matrix"
OUT = ROOT / "99_evaluation/analysis/rq3_cheap_exps"
COLLECTIONS = ["GRPOLY", "NorHand_v3", "ONB", "Phil_gr_130", "Pinkas", "RASAM", "RASM"]
SIZE = 1152
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


K = 3  # 3 = Protocol A (single source); 6 = Protocol B (LOCO-6, source = held-out collection)


def tag(source, variant=""):
    suffix = "_d1" if source == "NorHand_v3" and not variant and K == 3 else ""  # final NorHand draw
    return f"{source}_maskrcnn_convnext_tiny_catmus_1152_k{K}_final_ab_j1.6{suffix}{variant}"


def load(source, variant=""):
    """variant "" = retained final_ab checkpoint; otherwise a new run's best.pt (e.g. "_forge05")."""
    model = m.build_model("convnext_tiny_catmus", SIZE)
    m.configure_dense_inference(model, 300, replace_anchors=False)
    t = tag(source, variant)
    m.load_init_checkpoint(model, CKPT / f"{t}.pt" if not variant else CKPT.parent / t / "best.pt")
    summary = json.loads((EVAL / t / "summary.json").read_text())
    # Both exports drop detections below the source score threshold anyway; filtering them
    # inside the model only avoids pasting unused full-page masks (GPU memory).
    model.roi_heads.score_thresh = summary["score_threshold"]
    return model.to(DEV).eval(), summary["score_threshold"], summary["mask_threshold"]


def pixel_competition(output, meta, score_thr, mask_thr):
    """Idea A: per-pixel argmax of mask probability over the retained detections."""
    h, w = int(meta["new_h"]), int(meta["new_w"])
    keep = output["scores"] >= score_thr
    probs = output["masks"][keep, 0, :h, :w].float()
    canvas = np.zeros((h, w), np.uint16)
    if len(probs):
        best, arg = probs.max(0)
        lab = (arg + 1).cpu().numpy().astype(np.uint16)
        fg = (best >= mask_thr).cpu().numpy()
        canvas[fg] = lab[fg]
        # Relabel consecutively and drop instances that lost all pixels.
        ids = np.unique(canvas[canvas > 0])
        remap = np.zeros(len(probs) + 1, np.uint16)
        remap[ids] = np.arange(1, len(ids) + 1)
        canvas = remap[canvas]
    return cv2.resize(canvas, (int(meta["orig_w"]), int(meta["orig_h"])),
                      interpolation=cv2.INTER_NEAREST).astype(np.uint16, copy=False)


@torch.inference_mode()
def predict(model, collection, split, out_dir, st, mt, assigners):
    data = MATRIX / collection
    dataset, _ = m.make_loader(data, split, SIZE, False)
    for i in range(len(dataset)):
        img, _, meta = dataset[i]
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            out = model([img.to(DEV)])[0]
        out = {k: v.float() if v.is_floating_point() else v for k, v in out.items()}
        for name in assigners:
            inst = (m.prediction_to_instances(out, meta, st, mt) if name == "score"
                    else pixel_competition(out, meta, st, mt))
            folder = out_dir / name / collection
            folder.mkdir(parents=True, exist_ok=True)
            m.page_xml(meta, inst, folder / f"{Path(meta['path']).stem}.xml")


def score(out_dir, split, label):
    rows = []
    for folder in sorted(p for p in out_dir.glob("*/*") if p.is_dir()):
        per_page, metrics = m.diva_eval(MATRIX / folder.name, split, folder)
        rows.append({"run": label, "assign": folder.parent.name, "collection": folder.name,
                     "pages": len(per_page), **metrics})
        print(json.dumps(rows[-1]), flush=True)
    return rows


# ---------------------------------------------------------------- idea B
def stretched_rois(gt_boxes, n_per_gt, rng):
    """GT box stretched vertically by x1.5-3, target line kept inside, small x jitter."""
    rois, idx = [], []
    for j, (x1, y1, x2, y2) in enumerate(gt_boxes.tolist()):
        h, w = y2 - y1, x2 - x1
        for _ in range(n_per_gt):
            f = rng.uniform(1.5, 3.0)
            extra = (f - 1) * h
            top = extra * rng.uniform(0.2, 0.8)
            dx1, dx2 = rng.uniform(-.05, .05) * w, rng.uniform(-.05, .05) * w
            rois.append([max(0, x1 + dx1), max(0, y1 - top), min(SIZE, x2 + dx2), min(SIZE, y2 + extra - top)])
            idx.append(j)
    return torch.tensor(rois, dtype=torch.float32), torch.tensor(idx)


def finetune_mask_head(model, source, steps, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    rh = model.roi_heads
    params = list(rh.mask_head.parameters()) + list(rh.mask_predictor.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    dataset, _ = m.make_loader(MATRIX / source, "train", SIZE, True)  # same train aug as the recipe
    log = []
    model.eval()  # frozen BN/backbone behaviour; only the mask head gets gradients
    rh.mask_head.train()
    for step in range(steps):
        img, target, _ = dataset[step % len(dataset)]
        gtb = target["boxes"].to(DEV)
        if not len(gtb):
            continue
        with torch.no_grad(), torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            images, _ = model.transform([img.to(DEV)])
            feats = model.backbone(images.tensors)
        feats = {k: v.float() for k, v in feats.items()}
        # Half the RoIs are the ordinary positives (GT box with mild jitter), half stretched.
        sb, si = stretched_rois(gtb.cpu(), 2, rng)
        jit = gtb.cpu() + torch.randn(len(gtb), 4) * 0.05 * (gtb.cpu()[:, 2:] - gtb.cpu()[:, :2]).repeat(1, 2)
        rois = torch.cat([jit.clamp(0, SIZE), sb]).to(DEV)
        idx = torch.cat([torch.arange(len(gtb)), si]).to(DEV)
        mf = rh.mask_head(rh.mask_roi_pool(feats, [rois], images.image_sizes))
        logits = rh.mask_predictor(mf)[:, 1]
        tgt = project_masks_on_boxes(target["masks"].to(DEV), rois, idx, logits.shape[-1])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        log.append(float(loss))
        if step % 50 == 0:
            print(f"B step {step} loss {np.mean(log[-50:]):.4f}", flush=True)
    rh.mask_head.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", choices=["A", "B"], required=True)
    ap.add_argument("--protocol", choices=["A", "B"], default="A",
                    help="B: LOCO-6 models, each scored only on its held-out collection")
    ap.add_argument("--variant", default="", help='checkpoint variant suffix, e.g. "_forge05"')
    ap.add_argument("--sources", nargs="+", default=["ONB", "Phil_gr_130"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--targets", nargs="+", default=COLLECTIONS)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    global K
    K = 6 if args.protocol == "B" else 3
    rows = []
    result_csv = OUT / f"{'B_' if args.protocol == 'B' else ''}idea{args.idea}_{args.split}{args.variant}_{'_'.join(args.sources)}.csv"
    for source in args.sources:
        t0 = time.monotonic()
        model, st, mt = load(source, args.variant)
        label = source + args.variant
        if args.idea == "B":
            model = finetune_mask_head(model, source, args.steps, args.seed)
            label = f"{source}+B"
        out_dir = OUT / f"{'B_' if args.protocol == 'B' else ''}idea{args.idea}_{args.split}" / label
        for target in ([source] if args.protocol == "B" else args.targets):
            predict(model, target, args.split, out_dir, st, mt, ["score", "pixel"])
            print(f"{label} -> {target} predicted ({time.monotonic()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
        rows += score(out_dir, args.split, label)
        pd.DataFrame(rows).to_csv(result_csv, index=False)


if __name__ == "__main__":
    main()
