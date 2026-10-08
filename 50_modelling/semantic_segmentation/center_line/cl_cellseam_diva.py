#!/usr/bin/env python3
"""Center-line model on DIVA-HisDB (thesis model: cBAD encoder, all 20 training pages): line regions bounded by
minimum-ink seams between neighbouring center lines instead of the predicted-height band, so that ascenders,
descenders, and diacritics outside the band stay with their line.
  band     thesis decoder (band of the predicted heights, clipped to the nearest-line cell)
  cell     nearest-center-line cell up to one line pitch, no band (control)
  cellseam cell with the mid-way boundary replaced by the minimum-ink seam (rq3_centre_lf.seam_refine), no band
    DIVA_TAG=cbad|cm .venv/bin/python 50_modelling/semantic_segmentation/center_line/cl_cellseam_diva.py val|test   (thesis models: cbad = cBAD encoder, cm = CATMuS)
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import os, sys
os.environ.setdefault("DIVA_TAG", "cbad"); os.environ.setdefault("DIVA_MODES", "full k1 k3 k5 k10 k15")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
import rq_diva_cbad as D
import cl_bandstroke as BS
D.SFX = f"_{D.TAG}_cellseam"


def decode(CL, pg, s, shape, reg):
    out = {f"{n}|inside": [] for n in ("band", "cell", "cellseam", "bandstroke", "bandseamstroke")}
    frg = [f for f in pg["frags"] if len(f["xs"]) > 2]
    if not frg:
        return out
    pp = CL.frag_pitch(frg)
    lines = [l for l in CL.group_graph(frg, pp) if l["xs"][-1] - l["xs"][0] >= 0.5 * pp]
    ink = CL.page_ink(pg, s)
    lines = CL.trace_ends(lines, pp, ink, s)
    cl = [np.stack([l["xs"] * s, l["cy"] * s], 1).astype(np.float32) for l in lines]
    if not cl:
        return out
    cells = CL.CI.cells(cl, shape, pp * s)
    seamed = CL.seam_refine(cells.copy(), cl, ink, pp * s)
    bands = []
    for i, l in enumerate(lines):
        hu, hd = CL.line_height(l)
        step = max(1, len(l["xs"]) // 60)
        x, y = l["xs"][::step] * s, l["cy"][::step] * s
        poly = np.concatenate([np.stack([x, y - hu * s], 1), np.stack([x, y + hd * s], 1)[::-1]])
        b = np.zeros(shape, np.uint8); cv2.fillPoly(b, [np.rint(poly).astype(np.int32)], 1)
        # horizontal extent of the line: cells must not run past the traced line ends
        span = np.zeros(shape, bool); span[:, max(0, int(x.min()) - 2):int(x.max()) + 3] = True
        bands.append((cells == i + 1) & (b > 0))
        for name, m in (("band", bands[-1]), ("cell", (cells == i + 1) & span),
                        ("cellseam", (seamed == i + 1) & span)):
            if m.any() and (m & reg).sum() >= 0.5 * m.sum():
                out[f"{name}|inside"].append(m & reg)
    for m in BS.band_strokes(bands, ink, pp * s):
        if m.any() and (m & reg).sum() >= 0.5 * m.sum():
            out["bandstroke|inside"].append(m & reg)
    for m in BS.band_seam_strokes(bands, seamed, ink, pp * s):
        if m.any() and (m & reg).sum() >= 0.5 * m.sum():
            out["bandseamstroke|inside"].append(m & reg)
    return out


D.decode = decode
D.evaluate(sys.argv[1], ["band|inside", "cell|inside", "cellseam|inside", "bandstroke|inside", "bandseamstroke|inside"])
