#!/usr/bin/env python3
"""Per-line best IoU of the U-DIADS-TL ground-truth lines (all 45 test pages) for the CATMuS ConvNeXt-Tiny
Mask R-CNN with native ROI masks vs. the same boxes refined by the crop U-Net; FEST match threshold 0.75.
    .venv/bin/python 99_evaluation/analysis/story_udiads_roi_iou.py  -> story_figures/udiads_roi_iou.{pdf,png}, thesis graphics
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import story_udiads as U, story_udiads_roi as R  # noqa: E402

def best_iou(gt, lab):
    n, comp = cv2.connectedComponents(gt)
    lab = lab.astype(np.int64); npred = int(lab.max()) + 1
    c = np.bincount(comp.ravel().astype(np.int64) * npred + lab.ravel(), minlength=n * npred).reshape(n, npred)
    inter = c[1:, 1:].astype(float); ga = c[1:].sum(1, keepdims=True); pa = c[:, 1:].sum(0, keepdims=True)
    iou = inter / np.maximum(ga + pa - inter, 1)
    return iou.max(1) if iou.shape[1] else np.zeros(n - 1)

res = {"roi": [], "crop": []}
for ms in ("Latin14396", "Latin2", "Syr341"):
    for p in sorted((U.DATA / ms / f"text-line-gt-{ms}/test").glob("*.png")):
        gt = U.gt(ms, p.stem)
        res["roi"].append(best_iou(gt, R.roi_labels(ms, p.stem)))
        res["crop"].append(best_iou(gt, U.labels("maskrcnn", ms, p.stem)))
res = {k: np.concatenate(v) for k, v in res.items()}
for k, v in res.items():
    print(k, len(v), "median", round(float(np.median(v)), 3), "matched@0.75", round(100 * float((v >= 0.75).mean()), 1),
          "in [0.6,0.75)", round(100 * float(((v >= 0.6) & (v < 0.75)).mean()), 1))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ink, muted, grid = "#2b2b29", "#6b6a63", "#e4e3dc"
fig, ax = plt.subplots(figsize=(5.2, 3.0))
bins = np.linspace(0, 1, 41)
for k, c, lbl in (("roi", "#eb6834", "Mask R-CNN ROI masks (28×28)"), ("crop", "#2a78d6", "Same boxes + BBox U-Net")):
    ax.hist(res[k], bins=bins, color=c, alpha=0.75, label=f"{lbl}: {100 * (res[k] >= 0.75).mean():.0f}% lines matched", edgecolor="white", linewidth=0.5)
ax.axvline(0.75, color=ink, lw=1.2, ls="--"); ax.text(0.35, ax.get_ylim()[1] * 0.8, "dashed: FEST threshold 0.75", ha="center", va="center", color=ink, fontsize=9)
ax.set_xlabel("Best IoU of a ground-truth line with one predicted instance", color=ink, fontsize=10)
ax.set_ylabel("Ground-truth lines", color=ink, fontsize=10)
ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(muted)
ax.tick_params(colors=muted, labelsize=9); ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, labelcolor=ink)
fig.tight_layout()
fig.savefig(U.OUT / "udiads_roi_iou.png", dpi=200)
fig.savefig(Path.home() / "Thesis/fsl-tl-manuscripts-thesis/graphics/udiads_roi_iou.pdf")
