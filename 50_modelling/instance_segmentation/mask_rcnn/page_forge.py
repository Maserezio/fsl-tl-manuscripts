"""PageForge: recompose the few annotated support pages into new pages with other line layouts.

Each training page is split into a background (text lines removed and filled in) and
per-line ink layers (page / background inside the line polygon, 1 elsewhere). A forged page
multiplies randomly chosen ink layers onto a support background with a random line pitch,
slope, curvature and ink strength. Letters keep their shape; only the layout changes, so
neighbouring lines can crowd into each other's boxes as on dense manuscripts. Polygons go
through the same displacement and remain exact labels (in the source's own convention).

Ranges are fixed a priori and are not fitted to any target collection.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

WORK_LONG_SIDE = 2048          # forging resolution (training later fits pages to <= 1843 px)
# Defaults reproduce the first PageForge runs; RQ3_FORGE_CFG (JSON) overrides any of them.
CFG = {
    "pitch": (1.0, 2.2),       # line pitch / median line thickness
    "slope": 6.0,              # page slope in degrees, plus +-0.4 deg per line
    "curve": 0.35,             # sinusoid amplitude / line thickness
    "gamma": (0.7, 1.4),       # ink strength: ratio ** gamma (>1 darker, <1 fainter)
    "scale": (0.85, 1.15),     # common rescaling of all lines on a forged page
    "deg_p": 0.0,              # probability to degrade a training page (forged or real)
    "stroke_p": 0.0,           # probability of stroke thinning/thickening on a forged page
}
CFG.update(json.loads(os.environ.get("RQ3_FORGE_CFG", "{}")))
PITCH_RANGE, SLOPE_DEG, CURVE_AMP = tuple(CFG["pitch"]), float(CFG["slope"]), float(CFG["curve"])
GAMMA_RANGE, LINE_SCALE = tuple(CFG["gamma"]), tuple(CFG["scale"])


def degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Document degradations on a float32 HxWx3 image in [0, 1] (the fitted training canvas).

    Each effect is drawn independently: faded ink, uneven illumination, stains, bleed-through
    (the page's own mirrored ink), blur, sensor noise and JPEG compression.
    """
    h, w = img.shape[:2]
    paper = np.percentile(img.reshape(-1, 3)[::97], 90, axis=0).astype(np.float32)
    if rng.random() < 0.5:  # fade: pull ink towards the paper colour
        a = rng.uniform(0.35, 0.85)
        img = paper - (paper - img) * a
    if rng.random() < 0.3:  # bleed-through
        ink = np.clip(1.0 - img / np.maximum(paper, 1e-3), 0, 1)
        ghost = cv2.GaussianBlur(ink[:, ::-1], (0, 0), rng.uniform(1.0, 2.5))
        img = img * (1.0 - rng.uniform(0.08, 0.3) * ghost)
    if rng.random() < 0.5:  # illumination field
        g = np.random.default_rng(rng.randrange(1 << 30)).uniform(0.7, 1.1, (4, 4, 1)).astype(np.float32)
        img = img * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)[..., None]
    if rng.random() < 0.3:  # stains
        stain = np.zeros((h, w), np.float32)
        for _ in range(rng.randint(1, 4)):
            c = (rng.randrange(w), rng.randrange(h))
            ax = (rng.randint(w // 30, w // 6), rng.randint(h // 30, h // 6))
            cv2.ellipse(stain, c, ax, rng.uniform(0, 180), 0, 360, rng.uniform(0.15, 0.45), -1)
        stain = cv2.GaussianBlur(stain, (0, 0), max(w, h) / 80)[..., None]
        tint = np.array([0.55, 0.45, 0.3], np.float32)  # brownish
        img = img * (1.0 - stain * (1.0 - tint))
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.3))
    if rng.random() < 0.3:
        img = img + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0.01, 0.04), img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    if rng.random() < 0.4:
        ok, enc = cv2.imencode(".jpg", (img * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 80)])
        img = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    return img


def _densify(pts: np.ndarray, step: float) -> np.ndarray:
    """Insert points along polygon edges so that a vertical warp bends straight edges."""
    out = []
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        n = max(1, int(math.ceil(np.hypot(*(b - a)) / step)))
        t = np.arange(n)[:, None] / n
        out.append(a + (b - a) * t)
    return np.concatenate(out)


def _thickness(pts: np.ndarray) -> float:
    """Mean thickness: polygon area / length (a min-area rectangle inflates curved lines)."""
    length = max(cv2.minAreaRect(pts.astype(np.float32))[1])
    return float(abs(cv2.contourArea(pts.astype(np.float32))) / max(length, 1.0))


class PageForge:
    def __init__(self, records, seed: int | None = None):
        self.backgrounds: list[np.ndarray] = []
        self.blocks: list[tuple[int, int, int, int]] = []
        self.lines: list[dict] = []
        self.bg_group: list[str] = []
        for path, _, annotations in records:
            self._add_page(Path(path), annotations)
        if not self.lines:
            raise ValueError("PageForge: no usable lines in the support pages")
        self.median_thickness = float(np.median([l["thickness"] for l in self.lines]))
        # Pages of different collections (LOCO roots, "<collection>__<page>") are never mixed on
        # one forged page; a single collection keeps the original sampling sequence unchanged.
        self.groups = sorted(set(self.bg_group))
        self.group_thickness = {g: float(np.median([l["thickness"] for l in self.lines if l["group"] == g]))
                                for g in self.groups}

    def _add_page(self, path: Path, annotations):
        group = path.stem.split("__")[0]
        image = np.asarray(Image.open(path).convert("RGB"))
        scale = WORK_LONG_SIDE / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
        h, w = image.shape[:2]
        polys = []
        for a in annotations:
            for seg in a.get("segmentation", []) if isinstance(a.get("segmentation"), list) else []:
                pts = np.asarray(seg, np.float32).reshape(-1, 2) * scale
                if len(pts) >= 3 and cv2.contourArea(pts) > 20:
                    polys.append(pts)
        if not polys:
            return
        union = np.zeros((h, w), np.uint8)
        for p in polys:
            cv2.fillPoly(union, [np.rint(p).astype(np.int32)], 1)
        # Background: grey closing (max filter) wipes out thin dark strokes everywhere, then blur.
        # Estimated at 512 px; unlike inpainting it never drags dark page borders into the text.
        thick = float(np.median([_thickness(p) for p in polys]))
        k = max(3, int(thick * 0.8) | 1)
        hole = cv2.dilate(union, np.ones((k, k), np.uint8))
        small = 512 / max(h, w)
        sm_img = cv2.resize(image, None, fx=small, fy=small, interpolation=cv2.INTER_AREA)
        ks = max(3, int(thick * small * 1.2) | 1)
        closed = cv2.morphologyEx(sm_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
        closed = cv2.GaussianBlur(closed, (0, 0), ks / 2)
        smooth = cv2.resize(closed, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        # Brightness of real paper relative to the (brighter) closing estimate, measured in a
        # ring around the text so that the cleaned text area matches its surroundings.
        ring = (cv2.dilate(hole, np.ones((4 * k, 4 * k), np.uint8)) > 0) & (hole == 0)
        rel = image / np.maximum(smooth, 1.0)
        level = np.median(rel[ring], axis=0) if ring.any() else np.ones(3, np.float32)
        paper = smooth * level
        # Parchment grain: the page's own high-frequency residual outside the text.
        clean = image.copy()
        clean[hole > 0] = paper[hole > 0]
        grain = image - cv2.GaussianBlur(image, (0, 0), 3)
        grain_std = float(grain[ring].std()) if ring.any() else 0.0
        noise = np.random.default_rng(0).normal(0, grain_std, clean.shape).astype(np.float32)
        clean[hole > 0] += noise[hole > 0]
        self.backgrounds.append(np.clip(clean, 0, 255))
        self.bg_group.append(group)
        ys, xs = np.nonzero(union)
        self.blocks.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
        # Ink layers relative to the local paper; each line is normalised to its own paper level
        # and feathered at the polygon edge, so the polygon itself leaves no visible band.
        ratio = image / np.maximum(smooth, 1.0)
        for p in polys:
            x1, y1 = np.floor(p.min(0)).astype(int)
            x2, y2 = np.ceil(p.max(0)).astype(int)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2 + 1), min(h, y2 + 1)
            if x2 - x1 < 8 or y2 - y1 < 4:
                continue
            local = p - [x1, y1]
            mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
            cv2.fillPoly(mask, [np.rint(local).astype(np.int32)], 1)
            if mask.sum() < 20:
                continue
            crop = ratio[y1:y2, x1:x2]
            level_line = np.percentile(crop[mask > 0], 80, axis=0)
            ink = np.clip(crop / np.maximum(level_line, 1e-3), 0.0, 1.0)
            feather = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.5)[..., None]
            layer = 1.0 - (1.0 - ink) * feather
            self.lines.append({"layer": layer.astype(np.float16), "poly": local, "thickness": _thickness(p), "group": group})

    def sample(self, rng: random.Random):
        """Return (PIL image, COCO-style annotations) of one forged page."""
        if len(self.groups) > 1:
            group = rng.choice(self.groups)
            bi = rng.choice([i for i, g in enumerate(self.bg_group) if g == group])
            pool = [i for i, l in enumerate(self.lines) if l["group"] == group]
            base_thickness = self.group_thickness[group]
        else:
            bi = rng.randrange(len(self.backgrounds))
            pool = list(range(len(self.lines)))
            base_thickness = self.median_thickness
        canvas = self.backgrounds[bi].copy()
        h, w = canvas.shape[:2]
        bx1, by1, bx2, by2 = self.blocks[bi]
        s = rng.uniform(*LINE_SCALE)
        thick = base_thickness * s
        pitch = rng.uniform(*PITCH_RANGE) * thick
        slope = math.radians(rng.uniform(-SLOPE_DEG, SLOPE_DEG))
        amp = rng.uniform(0, CURVE_AMP) * thick
        period = rng.uniform(0.6, 2.5) * max(bx2 - bx1, 1)
        phase = rng.uniform(0, 2 * math.pi)
        order = pool
        rng.shuffle(order)
        annotations, y, i = [], by1 + rng.uniform(0, pitch), 0
        while y < min(h - thick, by2 + pitch) and i < len(order) * 3:
            line = self.lines[order[i % len(order)]]
            i += 1
            layer = line["layer"].astype(np.float32)
            if s != 1.0:
                layer = cv2.resize(layer, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            poly = line["poly"] * s
            lh, lw = layer.shape[:2]
            if lw > w - 4:
                continue
            x0 = int(np.clip(bx1 + rng.uniform(-0.05, 0.05) * (bx2 - bx1), 0, w - lw))
            # Place the line's own centre on the current baseline position y.
            cy = float(poly[:, 1].mean())
            y0 = y - cy
            line_slope = slope + math.radians(rng.uniform(-0.4, 0.4))
            xs_abs = x0 + np.arange(lw)
            dy = np.tan(line_slope) * (xs_abs - bx1) + amp * np.sin(2 * math.pi * xs_abs / period + phase)
            dy = dy + y0
            top = int(math.floor(dy.min()))
            bot = int(math.ceil(dy.max())) + lh
            if top < 0 or bot > h:
                y += pitch
                continue
            # Warp: output row r at column c samples the layer row r - (dy[c] - top).
            rows = np.arange(bot - top, dtype=np.float32)[:, None] - (dy - top)[None, :].astype(np.float32)
            cols = np.broadcast_to(np.arange(lw, dtype=np.float32)[None, :], rows.shape)
            warped = cv2.remap(layer, cols, rows, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(1, 1, 1))
            warped = np.power(np.clip(warped, 1e-3, 1), rng.uniform(*GAMMA_RANGE))
            canvas[top:bot, x0:x0 + lw] *= warped
            dense = _densify(poly, step=max(4.0, thick / 3))
            cx = np.clip(np.rint(dense[:, 0]).astype(int), 0, lw - 1)
            out = np.stack([dense[:, 0] + x0, dense[:, 1] + dy[cx]], 1)
            annotations.append({"segmentation": [out.reshape(-1).round(1).tolist()], "iscrowd": 0})
            y += pitch * rng.uniform(0.92, 1.08)
        if CFG["stroke_p"] and rng.random() < CFG["stroke_p"]:
            # Stroke width change: grey dilation thins dark strokes, erosion thickens them.
            k = np.ones((rng.choice([2, 3]),) * 2, np.uint8)
            canvas = cv2.dilate(canvas, k) if rng.random() < 0.6 else cv2.erode(canvas, k)
        return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)), annotations
