#!/usr/bin/env python3
"""LineField centre lines + ink assignment on the RQ3 dev pages (Protocol A sources).

  cache SRC...         run the LineField model of each source once per dev page (two-pass scale,
                       as rq3_linefield.infer_page) and store the centre-line fragments.
  eval  SRC... --variants G:C ...
                       group the fragments into lines (G = none | t0 | row), give every pixel to the
                       nearest line within C x the page pitch (rq3_centre_ink.cells), export and
                       score every dev target with DIVA.

Only the fragments' geometry is used; the height channels and polygon conventions are ignored.
Dev pages only (train+val of the target collections); no test page is read.
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import argparse, json, os, pickle, sys
from pathlib import Path

os.environ.setdefault("DIVA_WORKERS", "3")
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq3_centre_ink as CI  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT / "00_data/RQ3/dev"
OUT = ROOT / "99_evaluation/semantic_segmentation/center_line"
COLLS = ["GRPOLY", "NorHand_v3", "ONB", "Phil_gr_130", "Pinkas", "RASAM", "RASM"]


def targets(src, protocol, split):
    """(collection, data root, split) pairs: dev = all seven dev sets; test = the official test pages
    (Protocol A: every collection, the source's own test split from its training root; LOCO-k "Lk":
    the held-out collection only)."""
    if split == "dev":
        return [(c, DEV / c, "dev") for c in COLLS]
    if protocol == "Z":                      # zero-shot model: every collection's test pages
        return [(c, ROOT / ("00_data/RQ3/final_roots" if c == "NorHand_v3" else "00_data/RQ3/matrix") / c, "test")
                for c in COLLS]      # same roots as the thesis zero-shot table (run_rq3_improve.sh)
    if protocol == "A" or protocol.startswith("A_"):      # A_s43 etc.: other training seeds
        import rq3_aug_search as S
        return S.eval_targets(src, "A", "test")
    return [(src, ROOT / "00_data/RQ3/matrix" / src, "test")]


def cache_name(src, protocol, tag, split):
    # caches written after heights were added carry the "_h" suffix (CI_HEIGHTS=1)
    h = "_h" if os.environ.get("CI_HEIGHTS") == "1" else ""
    return f"{src}_{protocol}{tag}" + ("" if split == "dev" else f"_{split}") + h


def cache(src, tag="", split="dev", protocol="A"):
    import torch
    from PIL import Image
    import rq3_linefield as LF
    path = OUT / "cache" / f"{cache_name(src, protocol, tag, split)}.pkl"
    if path.exists():
        return
    ck = torch.load(LF.MODELS / f"{src}_{protocol}{tag}.pt", map_location="cpu", weights_only=False)
    net = LF.build_model(); net.load_state_dict(ck["model"]); net = net.to(LF.DEV).eval()
    pages = []
    with torch.inference_mode():
        for coll, root, sp in targets(src, protocol, split):
            coco = json.loads((root / f"coco_instances/{sp}.json").read_text())
            for im in coco["images"]:
                img = np.asarray(Image.open(root / "images" / sp / im["file_name"]).convert("RGB"))
                H, W = img.shape[:2]
                s1 = 1152 / max(H, W)
                im1 = cv2.resize(img, (max(8, int(W * s1)), max(8, int(H * s1))), interpolation=cv2.INTER_AREA)
                cen, _, up, dn = LF.run_net(net, im1)
                sel = cen > 0.5
                thick = float(np.median(up[sel] + dn[sel])) if sel.sum() > 50 else LF.T0
                s2 = min(s1 * LF.T0 / max(thick, 1.0) * float(os.environ.get("CI_S2_MULT", "1")), 3200 / max(H, W))
                im2 = cv2.resize(img, (max(8, int(W * s2)), max(8, int(H * s2))),
                                 interpolation=cv2.INTER_AREA if s2 < 1 else cv2.INTER_LINEAR)
                cen, end, up, dn = LF.run_net(net, im2)
                frags = []
                for f in LF._fragments(cen, end, float(os.environ.get("CI_CEN_THR", "0.4"))):
                    ys, xs = f["px"]
                    frags.append({"xs": f["xs"] / s2, "cy": f["cy"] / s2,
                                  "hu": float(np.percentile(up[ys, xs], 75)) / s2,
                                  "hd": float(np.percentile(dn[ys, xs], 75)) / s2, "n": len(f["xs"])})
                pages.append({"coll": coll, "root": str(root), "sp": sp, "im": im, "frags": frags, "s2": s2,
                              "T0": LF.T0 / s2})
            print(src, coll, "cached", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(pages, open(path, "wb"))


# ------------------------------------------------------------------ grouping (original pixel frame)
def frag_pitch(frags):
    return CI.pitch([np.stack([f["xs"], f["cy"]], 1) for f in frags if len(f["xs"]) > 2])


def group_t0(frags, t0):
    """rq3_linefield._merge with its thresholds (T0 = canonical thickness in original pixels)."""
    import rq3_linefield as LF
    fr = [{"xs": f["xs"], "cy": f["cy"], "px": (np.zeros(0, int), np.zeros(0, int))} for f in frags]
    old = LF.T0
    LF.T0 = t0
    try:
        return [{"xs": l["xs"], "cy": l["cy"]} for l in LF._merge(fr)]
    finally:
        LF.T0 = old


def blank_run(ink, x1, x2, y, half):
    """Longest run of ink-free columns between x1 and x2 in the band y +- half (working frame)."""
    H, W = ink.shape
    a, b = int(max(0, min(x1, x2))), int(min(W, max(x1, x2)))
    if b <= a:
        return 0
    band = ink[int(max(0, y - half)):int(min(H, y + half + 1)), a:b].any(0)
    run = best = 0
    for v in band:
        run = 0 if v else run + 1
        best = max(best, run)
    return best


def group_row(frags, p, dy_max=0.35, gap_max=3.0, ink=None, s=1.0, blank_max=None):
    """Join fragments left to right when the vertical offset at the joint is < dy_max x pitch and the
    horizontal gap < gap_max x pitch (overlaps up to half a pitch allowed); the offset is measured
    against the line's height extrapolated from its last pitch of length."""
    frags = sorted(frags, key=lambda f: f["xs"][0])
    used = [False] * len(frags)
    out = []
    for i, f in enumerate(frags):
        if used[i]:
            continue
        xs, cy = f["xs"], f["cy"]
        members = [f]
        used[i] = True
        while True:
            tail = xs >= xs[-1] - p
            slope = np.polyfit(xs[tail], cy[tail], 1)[0] if tail.sum() > 3 else 0.0
            slope = float(np.clip(slope, -0.2, 0.2))
            best, bd = None, None
            for j, g in enumerate(frags):
                if used[j]:
                    continue
                gap = g["xs"][0] - xs[-1]
                if gap < -0.5 * p or gap > gap_max * p:
                    continue
                pred = cy[-1] + slope * max(gap, 0)
                dy = abs(g["cy"][0] - pred)
                if blank_max is not None and ink is not None and gap > 0 and dy < dy_max * p and \
                        blank_run(ink, xs[-1] * s, g["xs"][0] * s, 0.5 * (cy[-1] + g["cy"][0]) * s, 0.3 * p * s) > blank_max * p * s:
                    continue      # an empty strip wider than blank_max pitches: column or margin gap
                if dy < dy_max * p and (bd is None or dy / p + 0.2 * gap / p < bd):
                    best, bd = j, dy / p + 0.2 * gap / p
            if best is None:
                break
            g = frags[best]
            used[best] = True
            members.append(g)
            keep = g["xs"] > xs[-1]
            xs, cy = np.concatenate([xs, g["xs"][keep]]), np.concatenate([cy, g["cy"][keep]])
        out.append({"xs": xs, "cy": cy, "members": members})
    return out


def snap_to_ink(lines, ink, p, win=0.4, iters=2):
    """Move every centre line to the ink centroid within +-win x pitch, column by column (smoothed over
    one pitch). Label-free; corrects a centre that follows the source's polygon rather than the ink."""
    H, W = ink.shape
    col = cv2.boxFilter(ink.astype(np.float32), -1, (max(1, int(p)) | 1, 1), normalize=True)   # horizontal smoothing
    yy = np.arange(H, dtype=np.float32)
    out = []
    for l in lines:
        xs, cy = l["xs"], l["cy"].astype(np.float32).copy()
        xi = np.clip(np.rint(xs).astype(int), 0, W - 1)
        for _ in range(iters):
            new = cy.copy()
            for k in range(0, len(xi), max(1, int(p / 8))):
                y1, y2 = int(max(0, cy[k] - win * p)), int(min(H, cy[k] + win * p + 1))
                w = col[y1:y2, xi[k]]
                if w.sum() > 0.5 * p * 0.05:
                    new[k] = float((w * yy[y1:y2]).sum() / w.sum())
                else:
                    new[k] = np.nan
            idx = np.arange(len(new))
            ok = ~np.isnan(new)
            if ok.sum() < 2:
                break
            new = np.interp(idx, idx[ok], new[ok])
            kern = max(1, int(p)) | 1
            pad = np.pad(new, kern // 2, mode="edge")
            cy = np.convolve(pad, np.ones(kern) / kern, mode="valid").astype(np.float32)[:len(xs)]
        out.append({"xs": xs, "cy": cy})
    return out


def page_ink(page, s):
    from PIL import Image
    import rq3_merge_split as MS
    im = page["im"]
    root = Path(page.get("root", DEV / page["coll"]))
    img = Image.open(root / "images" / page.get("sp", "dev") / im["file_name"]).convert("L")
    w, h = round(im["width"] * s), round(im["height"] * s)
    return MS.ink_mask(np.asarray(img.resize((w, h), Image.BILINEAR)))


def _cy_at(f, x):
    return np.interp(x, f["xs"], f["cy"])


def merge_stacked(frags, p, dy=0.4, ov=0.3):
    """Fragments that overlap horizontally and lie within dy x pitch of each other (two pieces of one
    line, e.g. a separate fragment for diacritics) are fused; cy is averaged where both exist."""
    n = len(frags)
    parent = list(range(n))
    find = lambda i: i if parent[i] == i else find(parent[i])
    for i in range(n):
        for j in range(i + 1, n):
            a, b = frags[i], frags[j]
            lo, hi = max(a["xs"][0], b["xs"][0]), min(a["xs"][-1], b["xs"][-1])
            if hi - lo <= ov * min(a["xs"][-1] - a["xs"][0], b["xs"][-1] - b["xs"][0]):
                continue
            xs = np.linspace(lo, hi, 16)
            if np.median(np.abs(_cy_at(a, xs) - _cy_at(b, xs))) < dy * p:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(frags[i])
    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append(g[0]); continue
        xs = np.unique(np.concatenate([f["xs"] for f in g]))
        cy = np.array([np.mean([_cy_at(f, x) for f in g if f["xs"][0] <= x <= f["xs"][-1]]) for x in xs])
        w = np.array([f.get("n", len(f["xs"])) for f in g], float)
        m = {"xs": xs, "cy": cy, "n": int(w.sum())}
        if all("hu" in f for f in g):
            m["hu"] = float(np.average([f["hu"] for f in g], weights=w)); m["hd"] = float(np.average([f["hd"] for f in g], weights=w))
        out.append(m)
    return out


def group_graph(frags, p, far=6.0, dy_far=0.5, quad=False, ink=None, s=1.0, veto=None):
    """merge_stacked, then the row joining of group_row; in addition, fragments up to `far` pitches
    apart are joined (vertical offset < dy_far x pitch from the extrapolated line) when no other
    fragment lies within half a pitch of the straight connection."""
    frags = sorted(merge_stacked(frags, p), key=lambda f: f["xs"][0])
    used = [False] * len(frags)
    out = []

    def unobstructed(x0, y0, x1, y1, skip):
        for k, h in enumerate(frags):
            if k in skip or h["xs"][-1] < x0 or h["xs"][0] > x1:
                continue
            xs = np.linspace(max(x0, h["xs"][0]), min(x1, h["xs"][-1]), 8)
            line = y0 + (y1 - y0) * (xs - x0) / max(x1 - x0, 1e-6)
            if np.any(np.abs(_cy_at(h, xs) - line) < 0.5 * p):
                return False
        return True

    for i, f in enumerate(frags):
        if used[i]:
            continue
        xs, cy, members = f["xs"], f["cy"], {i}
        used[i] = True
        while True:
            tail = xs >= xs[-1] - p
            slope = float(np.clip(np.polyfit(xs[tail], cy[tail], 1)[0], -0.2, 0.2)) if tail.sum() > 3 else 0.0
            qfit = None
            if quad:                       # curved continuation: parabola through the last two pitches
                t2 = xs >= xs[-1] - 2 * p
                if t2.sum() > 8:
                    qfit = np.polyfit(xs[t2] - xs[-1], cy[t2], 2)
                    qfit[0] = float(np.clip(qfit[0], -0.15 / p, 0.15 / p))
            best, bd = None, None
            for j, g in enumerate(frags):
                if used[j]:
                    continue
                gap = g["xs"][0] - xs[-1]
                if gap < -0.5 * p or gap > far * p:
                    continue
                pred = np.polyval(qfit, max(gap, 0)) if qfit is not None else cy[-1] + slope * max(gap, 0)
                dy = abs(g["cy"][0] - pred)
                near = gap <= 3.0 * p and dy < 0.35 * p
                if not near and not (dy < dy_far * p and unobstructed(xs[-1], cy[-1], g["xs"][0], g["cy"][0], members | {j})):
                    continue
                if veto is not None and ink is not None and gap > p and \
                        blank_run(ink, xs[-1] * s, g["xs"][0] * s, 0.5 * (cy[-1] + g["cy"][0]) * s, 0.3 * p * s) > veto * p * s:
                    continue               # an empty strip: column gutter, heading or hemistich break
                cost = dy / p + 0.2 * gap / p + (0 if near else 1.0)
                if bd is None or cost < bd:
                    best, bd = j, cost
            if best is None:
                break
            g = frags[best]
            used[best] = True
            members.add(best)
            keep = g["xs"] > xs[-1]
            xs, cy = np.concatenate([xs, g["xs"][keep]]), np.concatenate([cy, g["cy"][keep]])
        out.append({"xs": xs, "cy": cy, "members": [frags[k] for k in sorted(members)]})
    return out


def line_height(l):
    mem = l.get("members", [l])
    w = np.array([m.get("n", len(m["xs"])) for m in mem], float)
    return (float(np.average([m["hu"] for m in mem], weights=w)), float(np.average([m["hd"] for m in mem], weights=w)))


def overlap_masks(inst, lines, s, k):
    """Overlapping line regions: own cell united with the band center +- k x predicted heights."""
    masks = []
    for i, l in enumerate(lines):
        m = inst == i + 1
        if k > 0 and "hu" in l.get("members", [l])[0]:
            hu, hd = line_height(l)
            step = max(1, len(l["xs"]) // 40)
            x, y = l["xs"][::step] * s, l["cy"][::step] * s
            poly = np.concatenate([np.stack([x, y - k * hu * s], 1), np.stack([x, y + k * hd * s], 1)[::-1]])
            band = np.zeros(inst.shape, np.uint8)
            cv2.fillPoly(band, [np.rint(poly).astype(np.int32)], 1)
            m = m | (band > 0)
        masks.append(m)
    return masks


def page_xml_multi(path, image_name, W, H, masks, s):
    """PAGE XML with one (possibly overlapping) polygon per mask: largest contour, scaled to the page."""
    import xml.etree.ElementTree as ET
    import rq3_cheap_exps as ce
    m = ce.m
    ET.register_namespace("", m.PAGE_NS)
    root = ET.Element(f"{{{m.PAGE_NS}}}PcGts")
    page = ET.SubElement(root, f"{{{m.PAGE_NS}}}Page", {"imageFilename": Path(image_name).name,
                                                          "imageWidth": str(W), "imageHeight": str(H)})
    region = ET.SubElement(page, f"{{{m.PAGE_NS}}}TextRegion", {"id": "region_textline"})
    ET.SubElement(region, f"{{{m.PAGE_NS}}}Coords", {"points": f"0,0 {W},0 {W},{H} 0,{H}"})
    n = 0
    for mk in masks:
        cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)[:, 0, :] / s
        if len(c) < 3:
            continue
        line = ET.SubElement(region, f"{{{m.PAGE_NS}}}TextLine", {"id": f"line_{n}"})
        ET.SubElement(line, f"{{{m.PAGE_NS}}}Coords", {"points": " ".join(f"{int(round(x))},{int(round(y))}" for x, y in c)})
        n += 1
    ET.ElementTree(root).write(path, xml_declaration=True, encoding="utf-8")


def stroke_masks(inst, ink, n_lines, p=None, max_h=1.5, min_share=0.6, cut=None):
    """Line regions in which every connected ink stroke follows the line holding most of its pixels:
    the stroke is removed from the other cells and added to its majority line, so a descender that
    reaches into the neighbouring cell makes the two regions overlap (as in the ground truth)."""
    if cut is not None:        # RQ1-style disconnection: strokes touching two lines are cut along the seams
        ink = ink & ~cv2.dilate(cut.astype(np.uint8), np.ones((2, 1), np.uint8)).astype(bool)
    n, cc = cv2.connectedComponents(ink.astype(np.uint8), connectivity=4 if cut is not None else 8)
    lab = inst.astype(np.int64)
    counts = np.zeros((n, n_lines + 1), np.int64)
    np.add.at(counts, (cc.ravel(), lab.ravel()), 1)
    counts[:, 0] = 0
    owner = counts.argmax(1)
    owner[0] = 0
    if p is not None:          # only compact strokes that clearly belong to one line (tails, diacritics)
        tot = counts.sum(1)
        share = np.where(tot > 0, counts.max(1) / np.maximum(tot, 1), 0)
        stats = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=4 if cut is not None else 8)[2]
        tall = stats[:, cv2.CC_STAT_HEIGHT] > max_h * p
        owner[(share < min_share) | tall] = 0
    masks = [inst == i + 1 for i in range(n_lines)]
    stroke_owner = owner[cc]                          # per pixel: line that owns its stroke (0 = none)
    for i in range(n_lines):
        mine = stroke_owner == i + 1
        foreign = (cc > 0) & (stroke_owner != i + 1) & (stroke_owner > 0)   # owner 0: pixel cells kept
        masks[i] = (masks[i] & ~foreign) | mine
    return masks


def seam_refine(inst, cl, ink, p, seam_px=None):
    """Replace the mid-way boundary between vertically adjacent lines by a minimum-ink seam (dynamic
    programming, at most one pixel of vertical step per column) inside the band between them."""
    H, W = inst.shape
    energy = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), max(1.0, 0.08 * p))
    pairs = {}
    for x in range(W):
        on = [(np.interp(x, l[:, 0], l[:, 1]), i) for i, l in enumerate(cl) if l[0, 0] <= x <= l[-1, 0]]
        on.sort()
        for (ya, a), (yb, b) in zip(on, on[1:]):
            if yb - ya < 2.5 * p:
                pairs.setdefault((a, b), []).append(x)
    for (a, b), cols in pairs.items():
        cols = np.array(cols)
        for run in np.split(cols, np.where(np.diff(cols) > 1)[0] + 1):
            if len(run) < 3:
                continue
            ya = np.interp(run, cl[a][:, 0], cl[a][:, 1]); yb = np.interp(run, cl[b][:, 0], cl[b][:, 1])
            lo, hi = int(max(0, np.floor(ya.min()))), int(min(H - 1, np.ceil(yb.max())))
            if hi - lo < 3:
                continue
            ys = np.arange(lo, hi + 1)[:, None]
            band = (ys > ya[None] + 0.1 * p) & (ys < yb[None] - 0.1 * p)
            cost = np.where(band, energy[lo:hi + 1][:, run] + 0.02 * np.abs(ys - (ya + yb)[None] / 2) / p, np.inf)
            acc = cost[:, 0].copy(); back = np.zeros(cost.shape, np.int8)
            for c in range(1, cost.shape[1]):
                cand = np.stack([np.r_[np.inf, acc[:-1]], acc, np.r_[acc[1:], np.inf]])
                k = np.argmin(cand, 0); back[:, c] = k - 1
                acc = cost[:, c] + cand[k, np.arange(len(acc))]
            if not np.isfinite(acc).any():
                continue
            y = int(np.argmin(acc)); seam = np.empty(len(run), int)
            for c in range(len(run) - 1, -1, -1):
                seam[c] = y
                y = int(np.clip(y + int(back[y, c]), 0, len(acc) - 1))
            if seam_px is not None:
                seam_px[np.clip(seam + lo, 0, H - 1), run] = True
            for c, x in enumerate(run):
                y0, y1 = int(max(0, ya[c])), int(min(H, yb[c] + 1))
                col = inst[y0:y1, x]
                yy = np.arange(y0, y1) - lo
                sel = (col == a + 1) | (col == b + 1)
                col[sel & (yy <= seam[c])] = a + 1
                col[sel & (yy > seam[c])] = b + 1
    return inst


def extend_by_neighbours(lines, p, ink, s, max_ext=8.0, band=0.4, min_ink=0.3):
    """Nonlinear completion of line ends (curved page near the binding / clasp): a line that ends before
    its vertical neighbours is continued along the neighbours' shape, shifted by the local offset
    (average of the neighbour above and below where both exist). The continuation stops where the
    ink along it ends or where it would come within half a pitch of another line."""
    H, W = ink.shape
    out = []
    for i, L in enumerate(lines):
        xs, cy = L["xs"].astype(float), L["cy"].astype(float)
        for side in (1, -1):
            xe = xs[-1] if side == 1 else xs[0]
            near = xs >= xe - p if side == 1 else xs <= xe + p
            guides = []
            for j, N in enumerate(lines):
                if j == i:
                    continue
                if (side == 1 and N["xs"][-1] < xe + p) or (side == -1 and N["xs"][0] > xe - p):
                    continue
                if not (N["xs"][0] <= xe <= N["xs"][-1]):
                    continue
                d = float(np.median(cy[near] - np.interp(xs[near], N["xs"], N["cy"])))
                if 0.5 * p <= abs(d) <= 2.5 * p:
                    guides.append((abs(d), np.sign(d), N, d))
            if not guides:
                continue
            chosen = {}
            for ad, sg, N, d in sorted(guides, key=lambda g: g[0]):
                chosen.setdefault(sg, (N, d))          # nearest above and nearest below
            lim = min((N["xs"][-1] if side == 1 else N["xs"][0]) for N, _ in chosen.values())
            stop = xe + side * max_ext * p
            stop = min(stop, lim) if side == 1 else max(stop, lim)
            ext_x = np.arange(xe + side, stop, side * 1.0)
            if len(ext_x) < 2:
                continue
            ext_y = np.mean([np.interp(ext_x, N["xs"], N["cy"]) + d for N, d in chosen.values()], 0)
            keep = 0
            run_empty = 0
            for k, (x, y) in enumerate(zip(ext_x, ext_y)):
                if any(m is not L and m["xs"][0] <= x <= m["xs"][-1] and abs(np.interp(x, m["xs"], m["cy"]) - y) < 0.5 * p
                       for m in lines):
                    break
                xi, y0, y1 = int(x * s), int(max(0, (y - band * p) * s)), int(min(H, (y + band * p) * s + 1))
                has = 0 <= xi < W and y1 > y0 and ink[y0:y1, xi].any()
                run_empty = 0 if has else run_empty + 1
                if run_empty > 0.75 * p:
                    break
                if has:
                    keep = k + 1
            if keep < 2:
                continue
            ex, ey = ext_x[:keep], ext_y[:keep]
            if side == 1:
                xs, cy = np.r_[xs, ex], np.r_[cy, ey]
            else:
                xs, cy = np.r_[ex[::-1], xs], np.r_[ey[::-1], cy]
        out.append({**L, "xs": xs, "cy": cy})
    return out


def extend_absorb(lines, p, ink, s, max_ext=8.0, band=0.4, short=3.0, rounds=6):
    """extend_by_neighbours, but a short line (<= `short` pitches) lying on the continuation is absorbed
    into the extended line instead of stopping it (the curled end of a line near the binding that the
    detector returned as a separate fragment); the extension then continues from its far end."""
    H, W = ink.shape
    lines = [dict(l) for l in lines]
    alive = [True] * len(lines)
    span = lambda l: l["xs"][-1] - l["xs"][0]
    for i in range(len(lines)):
        if not alive[i]:
            continue
        for side in (1, -1):
            for _ in range(rounds):
                L = lines[i]
                xs, cy = L["xs"].astype(float), L["cy"].astype(float)
                xe = xs[-1] if side == 1 else xs[0]
                near = xs >= xe - p if side == 1 else xs <= xe + p
                chosen = {}
                for j, N in enumerate(lines):
                    if j == i or not alive[j] or span(N) <= short * p:
                        continue
                    if (side == 1 and N["xs"][-1] < xe + p) or (side == -1 and N["xs"][0] > xe - p) or not (N["xs"][0] <= xe <= N["xs"][-1]):
                        continue
                    d = float(np.median(cy[near] - np.interp(xs[near], N["xs"], N["cy"])))
                    if 0.5 * p <= abs(d) <= 2.5 * p and (np.sign(d) not in chosen or abs(d) < abs(chosen[np.sign(d)][1])):
                        chosen[np.sign(d)] = (N, d)
                if not chosen:
                    break
                lim = min((N["xs"][-1] if side == 1 else -N["xs"][0]) for N, _ in chosen.values())
                lim = lim if side == 1 else -lim
                stop = xe + side * max_ext * p
                stop = min(stop, lim) if side == 1 else max(stop, lim)
                ext_x = np.arange(xe + side, stop, side * 1.0)
                if len(ext_x) < 2:
                    break
                ext_y = np.mean([np.interp(ext_x, N["xs"], N["cy"]) + d for N, d in chosen.values()], 0)
                keep, run_empty, absorbed = 0, 0, None
                for k, (x, y) in enumerate(zip(ext_x, ext_y)):
                    hit = None
                    for j, m in enumerate(lines):
                        if j != i and alive[j] and m["xs"][0] <= x <= m["xs"][-1] and abs(np.interp(x, m["xs"], m["cy"]) - y) < 0.5 * p:
                            hit = j; break
                    if hit is not None:
                        if span(lines[hit]) <= short * p:
                            absorbed = hit; keep = k
                        break
                    xi, y0, y1 = int(x * s), int(max(0, (y - band * p) * s)), int(min(H, (y + band * p) * s + 1))
                    has = 0 <= xi < W and y1 > y0 and ink[y0:y1, xi].any()
                    run_empty = 0 if has else run_empty + 1
                    if run_empty > 0.75 * p:
                        break
                    if has:
                        keep = k + 1
                ex, ey = ext_x[:keep], ext_y[:keep]
                if absorbed is not None:
                    m = lines[absorbed]
                    sel = m["xs"] > xe if side == 1 else m["xs"] < xe
                    mx, my = m["xs"][sel].astype(float), m["cy"][sel].astype(float)
                    ex, ey = (np.r_[ex, mx], np.r_[ey, my]) if side == 1 else (np.r_[ex, mx[::-1]], np.r_[ey, my[::-1]])
                    alive[absorbed] = False
                if len(ex) < 2:
                    break
                if side == 1:
                    L["xs"], L["cy"] = np.r_[xs, ex], np.r_[cy, ey]
                else:
                    L["xs"], L["cy"] = np.r_[ex[::-1], xs], np.r_[ey[::-1], cy]
                if absorbed is None:
                    break
    return [l for l, a in zip(lines, alive) if a]


def trace_ends(lines, p, ink, s, max_ext=3.0, win=0.4, gain=0.5, max_slope=0.7, gap_stop=0.5):
    """Nonlinear continuation of every line end by tracing the ink: step along the current direction,
    pull the position towards the ink centroid in a window of +-win pitches, update the slope smoothly
    (strong bends allowed), stop after gap_stop pitches without ink or next to another line."""
    H, W = ink.shape
    inkf = ink.astype(np.float32)
    step = max(2.0, 0.2 * p)
    out = [dict(l) for l in lines]
    for i, L in enumerate(out):
        for side in (1, -1):
            xs, cy = L["xs"].astype(float), L["cy"].astype(float)
            seg = xs >= xs[-1] - p if side == 1 else xs <= xs[0] + p
            slope = float(np.clip(np.polyfit(xs[seg], cy[seg], 1)[0], -max_slope, max_slope)) if seg.sum() > 3 else 0.0
            x, y = (xs[-1], cy[-1]) if side == 1 else (xs[0], cy[0])
            px, py, empty = [], [], 0.0
            while abs(x - (xs[-1] if side == 1 else xs[0])) < max_ext * p:
                xn, yp = x + side * step, y + side * slope * step
                if any(j != i and m["xs"][0] <= xn <= m["xs"][-1] and abs(np.interp(xn, m["xs"], m["cy"]) - yp) < 0.5 * p
                       for j, m in enumerate(out)):
                    break
                a, b = sorted((int(x * s), int(xn * s)))
                y0, y1 = int(max(0, (yp - win * p) * s)), int(min(H, (yp + win * p) * s + 1))
                if a < 0 or b >= W or y1 <= y0:
                    break
                col = inkf[y0:y1, a:b + 1].sum(1)
                if col.sum() >= 0.05 * win * p * s:
                    yc = (np.arange(y0, y1)[:, None].ravel() @ col) / col.sum() / s
                    yn = (1 - gain) * yp + gain * yc
                    slope = float(np.clip(0.7 * slope + 0.3 * (yn - y) / (side * step), -max_slope, max_slope))
                    empty = 0.0
                    px.append(xn); py.append(yn)
                    y = yn
                else:
                    empty += step
                    if empty > gap_stop * p:
                        break
                    y = yp
                x = xn
            if px:
                if side == 1:
                    L["xs"], L["cy"] = np.r_[xs, px], np.r_[cy, py]
                else:
                    L["xs"], L["cy"] = np.r_[px[::-1], xs], np.r_[py[::-1], cy]
    return out


def page_instances(page, grouping, cut, min_len=0.5, snap=False, gap=3.0, blank=None, decoder="cells", seam=False, ovl=None, stroke=False, ext=False, quad=False, veto=None, bstroke=False):
    im = page["im"]
    H, W = im["height"], im["width"]
    frags = [f for f in page["frags"] if len(f["xs"]) > 2]
    if not frags:   # no center line found: empty prediction in the format the decoder returns
        return ([], min(1.0, CI.WORK_LONG / max(H, W))) if (stroke or ovl is not None) else np.zeros((H, W), np.uint16)
    p = frag_pitch(frags)
    s = min(1.0, CI.WORK_LONG / max(H, W))
    ink = page_ink(page, s) if (snap or seam or stroke or bstroke or ext or veto is not None or blank is not None) else None
    lines = (frags if grouping == "none" else group_t0(frags, page["T0"]) if grouping == "t0"
             else group_graph(frags, p, quad=quad, ink=ink, s=s, veto=veto) if grouping == "graph"
             else group_row(frags, p, gap_max=gap, ink=ink, s=s, blank_max=blank))
    lines = [l for l in lines if l["xs"][-1] - l["xs"][0] >= min_len * p]   # stray marks: their ink goes to a line
    if ext == "trace" and lines:
        lines = trace_ends(lines, p, ink, s)
    elif ext == "absorb" and lines:
        lines = extend_absorb(lines, p, ink, s)
    elif ext and lines:
        lines = extend_by_neighbours(lines, p, ink, s)
    shape = (round(H * s), round(W * s))
    if decoder == "hband":
        inst = np.zeros(shape, np.uint16)
        for i, l in enumerate(sorted(lines, key=lambda q: q["cy"].mean())):
            mem = l.get("members", [l])
            w = np.array([m.get("n", len(m["xs"])) for m in mem], float)
            hu = float(np.average([m["hu"] for m in mem], weights=w)) * s
            hd = float(np.average([m["hd"] for m in mem], weights=w)) * s
            step = max(1, len(l["xs"]) // 40)
            x, y = l["xs"][::step] * s, l["cy"][::step] * s
            poly = np.concatenate([np.stack([x, y - hu], 1), np.stack([x, y + hd], 1)[::-1]])
            mk = np.zeros(shape, np.uint8)
            cv2.fillPoly(mk, [np.rint(poly).astype(np.int32)], 1)
            inst[(mk > 0) & (inst == 0)] = i + 1
        return cv2.resize(inst, (W, H), interpolation=cv2.INTER_NEAREST)
    if snap and lines:
        scaled = [{"xs": l["xs"] * s, "cy": l["cy"] * s} for l in lines]
        lines = [{"xs": l["xs"] / s, "cy": l["cy"] / s} for l in snap_to_ink(scaled, ink, p * s)]
    cl = [np.stack([l["xs"] * s, l["cy"] * s], 1).astype(np.float32) for l in lines]
    inst = CI.cells(cl, shape, cut * p * s)
    seam_px = np.zeros(shape, bool) if (seam and stroke == "cut") else None
    if seam and cl:
        inst = seam_refine(inst, cl, ink, p * s, seam_px)
    if bstroke:                              # band of the predicted heights inside the cell + the ink strokes touching it
        import cl_bandstroke as BS
        bands = []
        for i, l in enumerate(lines):
            hu, hd = line_height(l)
            step = max(1, len(l["xs"]) // 60)
            x, y = l["xs"][::step] * s, l["cy"][::step] * s
            poly = np.concatenate([np.stack([x, y - hu * s], 1), np.stack([x, y + hd * s], 1)[::-1]])
            b = np.zeros(shape, np.uint8); cv2.fillPoly(b, [np.rint(poly).astype(np.int32)], 1)
            bands.append((inst == i + 1) & (b > 0))
        return BS.band_strokes(bands, ink, p * s), s
    if stroke:                               # whole strokes follow their majority line (overlapping regions)
        if stroke == "cut":    # seam-cut strokes; only the share rule, no height limit
            return stroke_masks(inst, ink, len(lines), p=p * s, max_h=1e9, cut=seam_px), s
        return stroke_masks(inst, ink, len(lines), p=(p * s if stroke == "safe" else None)), s
    if ovl is not None:                      # overlapping regions, returned at the working scale s
        return overlap_masks(inst, lines, s, ovl), s
    return cv2.resize(inst, (W, H), interpolation=cv2.INTER_NEAREST)


def evaluate(src, variants, tag="", split="dev", protocol="A"):
    import rq3_cheap_exps as ce
    name = cache_name(src, protocol, tag, split)
    pages = pickle.load(open(OUT / "cache" / f"{name}.pkl", "rb"))
    hsuf = "_h" if os.environ.get("CI_HEIGHTS") == "1" else ""
    res_path = OUT / (f"lf_{src}_A{tag}{hsuf}_dev.json" if split == "dev" and protocol == "A" else f"lf_{name}.json")
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for v in variants:
        if v in res:
            continue
        grouping, cut, *opt = v.split(":")
        snap = "snap" in opt
        gap = next((float(o[1:]) for o in opt if o.startswith("g")), 3.0)
        blank = next((float(o[1:]) for o in opt if o.startswith("b") and o[1:].replace(".", "", 1).isdigit()), None)
        decoder = "hband" if "hband" in opt else "cells"
        seam = "seam" in opt
        ovl = next((float(o[3:]) for o in opt if o.startswith("ovl")), None)
        stroke = "cut" if "stroke3" in opt else "safe" if "stroke2" in opt else ("stroke" in opt)
        ext, quad = ("trace" if "trace" in opt else "absorb" if "ext2" in opt else "ext" in opt), "q" in opt
        veto = next((float(o[4:]) for o in opt if o.startswith("veto")), None)
        bstroke = "bstroke" in opt
        res[v] = {}
        for coll in dict.fromkeys(pg["coll"] for pg in pages):
            work = OUT / "pred_lf" / name / v.replace(":", "_") / coll
            work.mkdir(parents=True, exist_ok=True)
            for pg in pages:
                if pg["coll"] != coll:
                    continue
                inst = page_instances(pg, grouping, float(cut), snap=snap, gap=gap, blank=blank, decoder=decoder, seam=seam, ovl=ovl, stroke=stroke, ext=ext, quad=quad, veto=veto, bstroke=bstroke)
                meta = {"path": pg["im"]["file_name"], "orig_w": pg["im"]["width"], "orig_h": pg["im"]["height"]}
                if ovl is not None or stroke or bstroke:
                    masks, sc = inst
                    page_xml_multi(work / f"{Path(pg['im']['file_name']).stem}.xml", pg["im"]["file_name"],
                                   pg["im"]["width"], pg["im"]["height"], masks, sc)
                else:
                    ce.m.page_xml(meta, inst, work / f"{Path(pg['im']['file_name']).stem}.xml")
            pg0 = next(pg for pg in pages if pg["coll"] == coll)
            per_page, metrics = ce.m.diva_eval(Path(pg0.get("root", DEV / coll)), pg0.get("sp", "dev"), work)
            res[v][coll] = {**metrics, "pages": {CI_page(r["filename"]): r["LinesFMeasure"] for r in per_page},
                            "flags": {"grouping": grouping, "cut": float(cut), "snap": snap, "gap_max": gap, "blank_max": blank, "decoder": decoder, "seam": seam, "overlap_k": ovl, "stroke": stroke, "extend": ext, "quad": quad, "veto": veto, "bstroke": bstroke, "min_len": 0.5, "model": f"{src}_{protocol}{tag}.pt", "split": split}}
            print(src + tag, v, coll, "FM %.1f" % (100 * metrics["LinesFMeasure"]), flush=True)
        res_path.write_text(json.dumps(res, indent=1))


def CI_page(name):
    import rq3_tts_eval as T
    return T.page_stem(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cache", "eval"])
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--variants", nargs="*", default=["none:1.0", "t0:1.0", "row:1.0", "row:0.75"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--protocol", default="A", help='"A" or LOCO-k "L1".."L3"')
    a = ap.parse_args()
    for s in a.sources:
        if a.mode == "cache":
            cache(s, a.tag, a.split, a.protocol)
        else:
            evaluate(s, a.variants, a.tag, a.split, a.protocol)


if __name__ == "__main__":
    main()
