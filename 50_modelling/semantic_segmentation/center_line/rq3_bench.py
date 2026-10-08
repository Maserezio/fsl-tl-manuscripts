#!/usr/bin/env python3
"""Fast RQ3 experiment bench on hard dev cases (post-processing ideas without the GPU).

  cache            run each bench model once on the hard dev pages (scale 1) and store the raw
                   detections: score, box, and the mask probabilities inside the mask's bbox.
  eval VARIANT...  rebuild instance maps from the cache with a post-processing variant, export
                   PAGE XML and score every page with the official DIVA evaluator.

Dev pages only (train+val of the target collections); no test page is read.
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json, os, sys, pickle, shutil
from pathlib import Path

os.environ.setdefault("DIVA_WORKERS", "12")
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq3_aug_search as S  # noqa: E402

OUT = S.OUT / "bench"
SOURCES = ["ONB", "RASAM", "NorHand_v3", "Pinkas", "GRPOLY"]
TARGETS = ["Phil_gr_130", "Pinkas", "NorHand_v3"]
MODEL = os.environ.get("BENCH_MODEL", "dense")


def cache():
    import torch
    import rq3_tts_eval as T
    from rq3_pitch import estimate_pitch
    from PIL import Image
    for src in SOURCES:
        path = OUT / "cache" / f"{src}_{MODEL}.pkl"
        if path.exists():
            continue
        m, ce, model, st, mt, _ = T.load_model(src, "A", MODEL)
        ref = T.source_ref_pitch(src, "A")
        pages = []
        for coll in TARGETS:
            if coll == src:
                continue
            root = S.ROOT / "00_data/RQ3/dev" / coll
            ds, _ = m.make_loader(root, "dev", 1152, False)
            for i in range(len(ds)):
                img, _, meta = ds[i]
                with torch.inference_mode(), torch.autocast(ce.DEV.type, dtype=torch.float16, enabled=ce.DEV.type == "cuda"):
                    o = model([img.to(ce.DEV)])[0]
                h, w = int(meta["new_h"]), int(meta["new_w"])
                dets = []
                for sc, box, mk in zip(o["scores"].float().cpu(), o["boxes"].float().cpu(), o["masks"][:, 0].float()):
                    if float(sc) < st:
                        continue
                    pm = mk[:h, :w]
                    ys, xs = torch.nonzero(pm > 0.05, as_tuple=True)
                    if len(ys) == 0:
                        continue
                    y1, y2, x1, x2 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
                    dets.append({"score": float(sc), "box": box.tolist(), "off": (y1, x1),
                                 "prob": (pm[y1:y2, x1:x2].cpu().numpy() * 255).astype(np.uint8)})
                pitch = estimate_pitch(np.asarray(Image.open(meta["path"]).convert("RGB")))
                gray = cv2.cvtColor((img[:, :h, :w].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                pages.append({"coll": coll, "meta": meta, "dets": dets, "pitch": pitch, "ref": ref,
                              "st": st, "mt": mt, "gray": gray})
        path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(pages, open(path, "wb"))
        print("cached", src, len(pages), "pages", flush=True)
        del model; torch.cuda.empty_cache()


# ------------------------------------------------------------------ assignment variants
def canvas_score(page, mt):
    h, w = page["gray"].shape
    canvas = np.zeros((h, w), np.uint16)
    for k, d in enumerate(sorted(page["dets"], key=lambda d: -d["score"])):
        y, x = d["off"]; p = d["prob"]
        fg = p >= mt * 255
        sub = canvas[y:y + p.shape[0], x:x + p.shape[1]]
        sub[fg & (sub == 0)] = k + 1
    return canvas


def canvas_pixel(page, mt):
    h, w = page["gray"].shape
    best = np.zeros((h, w), np.uint8); lab = np.zeros((h, w), np.uint16)
    for k, d in enumerate(page["dets"]):
        y, x = d["off"]; p = d["prob"]
        sb, sl = best[y:y + p.shape[0], x:x + p.shape[1]], lab[y:y + p.shape[0], x:x + p.shape[1]]
        win = p > sb
        sb[win] = p[win]; sl[win] = k + 1
    lab[best < mt * 255] = 0
    return lab


def split_by_ink(canvas, page, factor):
    """Cut instances taller than factor x page pitch at the ink-profile valleys inside them."""
    pitch = page["pitch"]
    fit = page["meta"]["new_w"] / page["meta"]["orig_w"]
    if not pitch or pitch != pitch:
        return canvas
    p_canvas = pitch * (1152 / max(page["meta"]["orig_w"], page["meta"]["orig_h"])) / fit * fit  # pitch is on the 1152 fit already
    p_canvas = pitch
    ink = page["gray"] < min(cv2.threshold(page["gray"], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0], 200)
    out = canvas.copy(); nxt = int(canvas.max()) + 1
    for lab in range(1, int(canvas.max()) + 1):
        ys, xs = np.nonzero(canvas == lab)
        if len(ys) < 50:
            continue
        y1, y2 = ys.min(), ys.max() + 1
        if (y2 - y1) < factor * p_canvas:
            continue
        # column-wise the line may slope: use the profile of ink inside this instance
        m = canvas[y1:y2] == lab
        prof = (ink[y1:y2] & m).sum(1).astype(np.float32)
        prof = cv2.GaussianBlur(prof[:, None], (0, 0), max(1.0, p_canvas / 6))[:, 0]
        n_lines = int(round((y2 - y1) / p_canvas))
        if n_lines < 2:
            continue
        cuts = []
        for j in range(1, n_lines):
            c = int(j * (y2 - y1) / n_lines)
            lo, hi = max(1, c - int(p_canvas / 3)), min(len(prof) - 1, c + int(p_canvas / 3))
            cuts.append(lo + int(np.argmin(prof[lo:hi])))
        bounds = [0] + sorted(cuts) + [y2 - y1]
        for a, b in zip(bounds[1:-1], bounds[2:]):
            seg = np.zeros_like(m); seg[a:b] = m[a:b]
            out[y1:y2][seg] = nxt; nxt += 1
    return out


def build(page, variant):
    mt = page["mt"]
    parts = variant.split("+")
    base = parts[0]
    for p in parts[1:]:
        if p.startswith("mt"):
            mt = float(p[2:])
    if base == "score":
        c = canvas_score(page, mt)
    elif base == "pixel":
        c = canvas_pixel(page, mt)
    elif base == "adaptive":
        dense = page["pitch"] and page["pitch"] == page["pitch"] and page["pitch"] < 0.6 * page["ref"]
        c = canvas_pixel(page, mt) if dense else canvas_score(page, mt)
    for p in parts[1:]:
        if p.startswith("split"):
            c = split_by_ink(c, page, float(p[5:]))
    return c


def evaluate(variant):
    import rq3_cheap_exps as ce
    m = ce.m
    rows = []
    for src in SOURCES:
        pages = pickle.load(open(OUT / "cache" / f"{src}_{MODEL}.pkl", "rb"))
        work = OUT / "work" / variant / src
        shutil.rmtree(work, ignore_errors=True)
        for pg in pages:
            meta = pg["meta"]
            c = build(pg, variant)
            inst = cv2.resize(c, (int(meta["orig_w"]), int(meta["orig_h"])), interpolation=cv2.INTER_NEAREST)
            d = work / pg["coll"]; d.mkdir(parents=True, exist_ok=True)
            m.page_xml(meta, inst, d / f"{Path(meta['path']).stem}.xml")
        for coll in sorted({p["coll"] for p in pages}):
            per_page, _ = m.diva_eval(S.ROOT / "00_data/RQ3/dev" / coll, "dev", work / coll)
            for r in per_page:
                rows.append({"variant": variant, "source": src, "target": coll,
                             "FM": float(r["LinesFMeasure"]) * 100 if r["LinesFMeasure"] not in ("", "NaN") else 0.0})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"eval_{MODEL}_{variant}.csv", index=False)
    return d


if __name__ == "__main__":
    if sys.argv[1] == "cache":
        cache()
    else:
        res = pd.concat([evaluate(v) for v in sys.argv[2:]])
        t = res.groupby(["variant", "target"]).FM.mean().unstack().round(1)
        t["mean"] = t.mean(1).round(1)
        print(t.to_string())
