"""Shared line-instance post-processing for both DIVA-HisDB and U-DIADS-TL.

`remove_small_objects` / `_seam` / `disconnect_components` are the actual
seam-carving algorithm and are dataset-agnostic -- both `predict_diva_seamcarve.py`
(DIVA, no baseline GT, no ARU-Net) and the U-DIADS pipeline import them from here
rather than keeping separate copies.

`arunet_baseline_prob` / `fuse_masks` / `build_fused` / `run_pipeline` are
U-DIADS-specific (DIVA has no baseline GT, so no ARU-Net fusion step there).

PER_MS_PARAMS holds the best validation-tuned disconnect params per U-DIADS subset.
"""
import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# best post-processing params per subset (from the val grid search in the notebook)
PER_MS_PARAMS = {
    "Latin14396": dict(line_sigma=4, min_line_distance=25, peak_frac=0.15, valley_ratio=0.4),
    "Latin2":     dict(line_sigma=4, min_line_distance=25, peak_frac=0.15, valley_ratio=0.4),
    "Syr341":     dict(line_sigma=4, min_line_distance=15, peak_frac=0.10, valley_ratio=0.5),
}


def remove_small_objects(binary, min_size=20):
    binary = binary.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out[labels == i] = 1
    return out


def _seam(cost):
    """Min-energy left->right seam via dynamic programming."""
    h, w = cost.shape
    M = np.empty((h, w))
    M[:, 0] = cost[:, 0]
    back = np.zeros((h, w), np.int32)
    for x in range(1, w):
        prev = M[:, x - 1]
        pm1 = np.empty(h); pm1[0] = np.inf; pm1[1:] = prev[:-1]
        pp1 = np.empty(h); pp1[-1] = np.inf; pp1[:-1] = prev[1:]
        stack = np.vstack([pm1, prev, pp1])
        idx = np.argmin(stack, axis=0)
        M[:, x] = cost[:, x] + stack[idx, np.arange(h)]
        back[:, x] = idx - 1
    seam = np.empty(w, np.int32)
    seam[-1] = int(np.argmin(M[:, -1]))
    for x in range(w - 1, 0, -1):
        seam[x - 1] = min(max(seam[x] + back[seam[x], x], 0), h - 1)
    return seam


def disconnect_components(mask, line_sigma=4, min_line_distance=25,
                          peak_frac=0.20, valley_ratio=0.6,
                          smooth_cost=2.0, cut_thickness=1, min_area=200):
    """Split components holding >1 text line (projection peaks) with seam cuts."""
    binary = (mask > 0).astype(np.uint8)
    result = binary.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        y = stats[i, cv2.CC_STAT_TOP]; x = stats[i, cv2.CC_STAT_LEFT]
        h = stats[i, cv2.CC_STAT_HEIGHT]; w = stats[i, cv2.CC_STAT_WIDTH]
        comp = (labels[y:y + h, x:x + w] == i).astype(np.float32)
        proj = gaussian_filter1d(comp.sum(1), line_sigma)
        if proj.max() == 0:
            continue
        peaks, _ = find_peaks(proj, distance=min_line_distance, height=proj.max() * peak_frac)
        if len(peaks) < 2:
            continue
        cost = gaussian_filter1d(comp, smooth_cost, axis=0) * w + 1e-3
        for r1, r2 in zip(peaks[:-1], peaks[1:]):
            v_row = r1 + int(np.argmin(proj[r1:r2 + 1]))
            if proj[v_row] > valley_ratio * min(proj[r1], proj[r2]):
                continue
            seam = _seam(cost[r1:r2 + 1, :]) + r1
            for xx in range(w):
                yy = seam[xx]
                result[y + max(0, yy - cut_thickness):y + yy + cut_thickness + 1, x + xx] = 0
    return result


def arunet_baseline_prob(img_rgb, arunet):
    """Raw ARU-Net baseline probability map (the expensive, cacheable step)."""
    with torch.no_grad():
        baseline_out = arunet(torch.from_numpy(img_rgb).unsqueeze(0).float())
    return torch.sigmoid(baseline_out[0, :, :, 0]).numpy().astype(np.float32)


def fuse_masks(baseline_prob, seg_prob, baseline_erode=8):
    """Fuse U-Net prob with the ARU-Net baseline -> pre-disconnect binary mask.

    baseline_erode = height of the vertical (1, N) erosion kernel applied to the
    thresholded baseline; larger N -> thinner baseline. Cheap (morphology only),
    so it can be swept without re-running ARU-Net.
    """
    baseline_bin = cv2.threshold((baseline_prob * 255).astype(np.uint8), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    pred_filt = remove_small_objects((seg_prob > 0.5).astype(np.uint8), min_size=50)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(baseline_erode))))
    thin = (cv2.erode(baseline_bin.astype(np.uint8), kernel, iterations=1) > 0).astype(np.uint8)
    fused = np.maximum(pred_filt, thin)
    return remove_small_objects(fused, min_size=500)


def build_fused(img_rgb, seg_prob, arunet, baseline_erode=8):
    """U-Net prob + ARU-Net baseline -> pre-disconnect binary mask (one-shot)."""
    return fuse_masks(arunet_baseline_prob(img_rgb, arunet), seg_prob, baseline_erode)


def run_pipeline(fused, params):
    """Disconnect + cleanup with given per-subset params."""
    refined = disconnect_components(fused, smooth_cost=2.0, cut_thickness=1, min_area=200, **params)
    return remove_small_objects(refined, min_size=200)
