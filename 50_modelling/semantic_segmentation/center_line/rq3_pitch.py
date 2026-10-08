"""Label-free, model-free line-pitch estimate of a page image (autocorrelation of ink profiles).

The page is fitted to 1152 px (as for the model). Ink is the darker Otsu side of a locally
normalised grey image. The page is cut into vertical strips (robust to slope); in each strip
the row-wise ink profile is autocorrelated and the first strong peak gives the pitch.
The page estimate is the median over strips with a clear peak.
"""
from __future__ import annotations
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)

import cv2
import numpy as np


def estimate_pitch(image_rgb: np.ndarray, size: int = 1152, strips: int = 8,
                   min_lag: int = 6, max_lag: int = 220) -> float | None:
    h, w = image_rgb.shape[:2]
    fit = min(size / w, size / h)
    g = cv2.cvtColor(cv2.resize(image_rgb, (max(1, round(w * fit)), max(1, round(h * fit))),
                                interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), 25)
    norm = np.clip(g / np.maximum(bg, 1), 0, 1.5)
    u8 = np.clip(norm * 170, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (u8 < t).astype(np.float32)
    H, W = ink.shape
    peaks = []
    for i in range(strips):
        x1, x2 = int(W * (0.1 + 0.8 * i / strips)), int(W * (0.1 + 0.8 * (i + 1) / strips))
        prof = ink[:, x1:x2].mean(1)
        prof = cv2.GaussianBlur(prof[:, None], (0, 0), 1.5)[:, 0]
        if prof.std() < 1e-3:
            continue
        p = prof - prof.mean()
        ac = np.correlate(p, p, "full")[len(p) - 1:]
        ac = ac / max(ac[0], 1e-9)
        lags = np.arange(len(ac))
        seg = (lags >= min_lag) & (lags <= min(max_lag, len(ac) - 2))
        cand = [l for l in lags[seg] if ac[l] > ac[l - 1] and ac[l] >= ac[l + 1] and ac[l] > 0.15]
        if cand:
            best = max(cand[:3], key=lambda l: ac[l])  # strongest of the first peaks
            peaks.append(best)
    return float(np.median(peaks)) if peaks else None


def label_pitch(coco_json: dict, size: int = 1152) -> float:
    """Median vertical distance between horizontally overlapping neighbouring line centroids
    of annotated (source) pages, on the fitted canvas."""
    by_img = {}
    for a in coco_json["annotations"]:
        pts = np.asarray(a["segmentation"][0], np.float32).reshape(-1, 2)
        by_img.setdefault(a["image_id"], []).append((pts[:, 1].mean(), pts[:, 0].min(), pts[:, 0].max()))
    gaps = []
    for im in coco_json["images"]:
        fit = min(size / im["width"], size / im["height"])
        lines = sorted(by_img.get(im["id"], []))
        for i, (cy, x1, x2) in enumerate(lines):
            for cy2, u1, u2 in lines[i + 1:]:
                if min(x2, u2) - max(x1, u1) > 0.3 * min(x2 - x1, u2 - u1):
                    gaps.append((cy2 - cy) * fit)
                    break
    return float(np.median(gaps))


def estimate_pitch_v2(image_rgb: np.ndarray, size: int = 1152, strips: int = 8,
                      min_lag: int = 6, max_lag: int = 260, ink_pct: float = 6.0,
                      min_corr: float = 0.05, min_strips: int = 3) -> float | None:
    """estimate_pitch with (1) ink = at most the darkest ink_pct % of the normalised page, so
    paper texture is not taken for ink; (2) a lower correlation floor but a consensus of at
    least min_strips strips; (3) a half-lag (harmonic) check."""
    h, w = image_rgb.shape[:2]
    fit = min(size / w, size / h)
    g = cv2.cvtColor(cv2.resize(image_rgb, (max(1, round(w * fit)), max(1, round(h * fit))),
                                interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    norm = g / np.maximum(cv2.GaussianBlur(g, (0, 0), 25), 1)
    u8 = np.clip(norm * 170, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (u8 < min(otsu, np.percentile(u8, ink_pct))).astype(np.float32)
    H, W = ink.shape
    peaks = []
    for i in range(strips):
        x1, x2 = int(W * (0.1 + 0.8 * i / strips)), int(W * (0.1 + 0.8 * (i + 1) / strips))
        prof = cv2.GaussianBlur(ink[:, x1:x2].mean(1)[:, None], (0, 0), 1.5)[:, 0]
        if prof.std() < 1e-4:
            continue
        p = prof - prof.mean()
        ac = np.correlate(p, p, "full")[len(p) - 1:]
        ac = ac / max(ac[0], 1e-9)
        hi = min(max_lag, len(ac) - 2)
        cand = [l for l in range(min_lag, hi) if ac[l] > ac[l - 1] and ac[l] >= ac[l + 1] and ac[l] > min_corr]
        if not cand:
            continue
        best = max(cand[:3], key=lambda l: ac[l])
        half = [l for l in cand if abs(l - best / 2) <= max(2, 0.1 * best / 2)]
        if half and ac[half[0]] > 0.5 * ac[best]:
            best = half[0]
        peaks.append(best)
    if len(peaks) < min_strips:
        return None
    med = float(np.median(peaks))
    agree = [p for p in peaks if abs(p - med) <= 0.2 * med]
    return float(np.median(agree)) if len(agree) >= min_strips else None
