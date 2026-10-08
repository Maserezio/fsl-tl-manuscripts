"""Band + attached strokes: the center-line band of each line, extended by the ink components that touch it.

A connected ink component joins the line whose band holds the largest share of its pixels if that share is at least
MIN_SHARE; components that touch no band (interlinear glosses, noise) and very tall components (initials, decoration)
are not added. Ascenders, descenders, and diacritics that belong to letters of the line are thereby kept, whereas the
space between the lines is not claimed as it is by the seam-bounded cells.
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import cv2
import numpy as np

MIN_SHARE = 0.2
MAX_HEIGHT = 2.5      # in line pitches


def band_strokes(bands, ink, pitch_px, exclusive=False):
    """bands: list of bool masks (one per line, same shape as ink) -> list of bool masks.
    exclusive: a stroke that joins one line is also cut out of the bands of the other lines, so that no ink
    lies in two regions (the descender of one line no longer stays inside the band of the next one)."""
    if not bands:
        return []
    lab = np.zeros(ink.shape, np.int32)
    for i, b in enumerate(bands):
        lab[b & (lab == 0)] = i + 1
    n, comp, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    nl = len(bands) + 1
    cnt = np.bincount(comp.ravel().astype(np.int64) * nl + lab.ravel(), minlength=n * nl).reshape(n, nl)
    area = cnt.sum(1)
    best = cnt[:, 1:].argmax(1) + 1
    share = cnt[np.arange(n), best] / np.maximum(area, 1)
    keep = (share >= MIN_SHARE) & (st[:, cv2.CC_STAT_HEIGHT] <= MAX_HEIGHT * pitch_px)
    keep[0] = False
    owner = np.where(keep, best, 0)[comp]
    k = np.ones((3, 3), np.uint8)
    if exclusive:
        out = []
        for i, b in enumerate(bands):
            mine = cv2.dilate((owner == i + 1).astype(np.uint8), k) > 0
            foreign = cv2.dilate(((owner > 0) & (owner != i + 1)).astype(np.uint8), k) > 0
            out.append((b & ~foreign) | mine)
        return out
    return [b | (cv2.dilate((owner == i + 1).astype(np.uint8), k) > 0) for i, b in enumerate(bands)]


def band_seam_strokes(bands, seamed, ink, pitch_px, min_touch=0.05):
    """Band + the ink of every component that touches some band (>= min_touch of its pixels), split between the
    lines by the seam-bounded cells; components that touch no band (interlinear glosses) are left out."""
    if not bands:
        return []
    anyband = np.zeros(ink.shape, bool)
    for b in bands:
        anyband |= b
    n, comp, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    touch = np.bincount(comp[anyband].ravel(), minlength=n) / np.maximum(st[:, cv2.CC_STAT_AREA], 1)
    keep = (touch >= min_touch) & (st[:, cv2.CC_STAT_HEIGHT] <= MAX_HEIGHT * pitch_px)
    keep[0] = False
    attached = keep[comp]
    k = np.ones((3, 3), np.uint8)
    return [b | (cv2.dilate((attached & (seamed == i + 1)).astype(np.uint8), k) > 0) for i, b in enumerate(bands)]
