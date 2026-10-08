#!/usr/bin/env python3
"""Controlled line-spacing perturbation (dev pages): FM and merge rate as the interline pitch shrinks.

  build   For dev pages of five collections, cut every text line out of the page with its ink (the
          PageForge cleaning: paper estimate by grey closing, per-line ink layer relative to the
          paper), and paste the lines back with their vertical distances to the first line scaled
          by f in FACTORS, x positions unchanged. GT polygons move with their lines. Writes one
          DIVA-ready root per f (images, page-gt, pixel-gt, coco_instances; split "dev").
  eval SRC...
          Mask R-CNN (ab, scale 1, score assignment) and the center-line model (CATMuS + WiSE-FT,
          row:1.0) of each Protocol-A source on every root, excluding pages of the source's own
          collection. FM per page and merge/split/miss rates (rq3_error_taxonomy.per_line).

f = 1.0 is the recomposed but unperturbed page (sanity reference). No test page is used.
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json, os, sys
from pathlib import Path

os.environ.setdefault("DIVA_WORKERS", "4")
import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "50_modelling").is_dir())  # repo root
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "50_modelling/instance_segmentation/mask_rcnn/cross_collection"))
DEV = ROOT / "00_data/RQ3/dev"
OUT = ROOT / "99_evaluation/semantic_segmentation/center_line/density"
FACTORS = [1.0, 0.85, 0.7, 0.55]
COLLS = ["ONB", "RASAM", "RASM", "NorHand_v3", "GRPOLY"]
PAGES_PER_COLL = 3
WORK = 2048


def page_layers(img_path, polys):
    """Clean paper background and per-line ink layers (PageForge._add_page, offsets kept)."""
    image = np.asarray(Image.open(img_path).convert("RGB")).astype(np.float32)
    h, w = image.shape[:2]
    union = np.zeros((h, w), np.uint8)
    for p in polys:
        cv2.fillPoly(union, [np.rint(p).astype(np.int32)], 1)
    from rq3_linefield import line_geometry
    th = [np.median(g[2] + g[3]) for g in (line_geometry(p, (h, w)) for p in polys) if g is not None]
    thick = float(np.median(th))
    k = max(3, int(thick * 0.8) | 1)
    hole = cv2.dilate(union, np.ones((k, k), np.uint8))
    small = 512 / max(h, w)
    sm = cv2.resize(image, None, fx=small, fy=small, interpolation=cv2.INTER_AREA)
    ks = max(3, int(thick * small * 1.2) | 1)
    closed = cv2.GaussianBlur(cv2.morphologyEx(sm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))), (0, 0), ks / 2)
    smooth = cv2.resize(closed, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    ring = (cv2.dilate(hole, np.ones((4 * k, 4 * k), np.uint8)) > 0) & (hole == 0)
    level = np.median((image / np.maximum(smooth, 1.0))[ring], axis=0) if ring.any() else np.ones(3, np.float32)
    clean = image.copy()
    clean[hole > 0] = (smooth * level)[hole > 0]
    ratio = image / np.maximum(smooth, 1.0)
    lines = []
    for p in polys:
        x1, y1 = np.floor(p.min(0)).astype(int); x2, y2 = np.ceil(p.max(0)).astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2 + 1), min(h, y2 + 1)
        if x2 - x1 < 8 or y2 - y1 < 4:
            continue
        mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
        cv2.fillPoly(mask, [np.rint(p - [x1, y1]).astype(np.int32)], 1)
        if mask.sum() < 20:
            continue
        crop = ratio[y1:y2, x1:x2]
        lvl = np.percentile(crop[mask > 0], 80, axis=0)
        ink = np.clip(crop / np.maximum(lvl, 1e-3), 0.0, 1.0)
        feather = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.5)[..., None]
        lines.append({"layer": 1.0 - (1.0 - ink) * feather, "x1": x1, "y1": y1, "poly": p,
                      "cy": float(p[:, 1].mean())})
    return clean, lines


def gt_xml(path, name, w, h, polys):
    pts = lambda p: " ".join(f"{int(round(x))},{int(round(y))}" for x, y in p)
    body = "".join(f'<TextLine id="line_{i}"><Coords points="{pts(p)}" /></TextLine>' for i, p in enumerate(polys))
    path.write_text("<?xml version='1.0' encoding='utf-8'?>\n"
                    '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">'
                    f'<Page imageFilename="{name}" imageWidth="{w}" imageHeight="{h}">'
                    f'<TextRegion id="region_textline"><Coords points="0,0 {w},0 {w},{h} 0,{h}" />{body}'
                    "</TextRegion></Page></PcGts>\n")


def build():
    from prepare_rq3_loo import write_pixel_gt
    from rq3_linefield import page_polys
    for f in FACTORS:
        root = OUT / f"f{f:.2f}"
        if (root / "coco_instances/dev.json").exists():
            continue
        for d in ("images/dev", "page-gt/dev", "pixel-gt", "coco_instances"):
            (root / d).mkdir(parents=True, exist_ok=True)
        images, anns, aid = [], [], 1
        for coll in COLLS:
            coco = json.loads((DEV / coll / "coco_instances/dev.json").read_text())
            by = page_polys(coco)
            ims = sorted(coco["images"], key=lambda x: x["file_name"])[:PAGES_PER_COLL]
            for im in ims:
                clean, lines = page_layers(DEV / coll / "images/dev" / im["file_name"], by[im["id"]])
                canvas = clean.copy()
                H, W = canvas.shape[:2]
                cy0 = min(l["cy"] for l in lines)
                polys = []
                for l in sorted(lines, key=lambda q: q["cy"]):
                    dy = int(round((l["cy"] - cy0) * (f - 1.0)))
                    lh, lw = l["layer"].shape[:2]
                    y1 = l["y1"] + dy
                    a, b = max(0, y1), min(H, y1 + lh)
                    if b <= a:
                        continue
                    canvas[a:b, l["x1"]:l["x1"] + lw] *= l["layer"][a - y1:b - y1]
                    polys.append(l["poly"] + [0, dy])
                name = f"{coll}__{Path(im['file_name']).stem}.jpg"
                Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(root / "images/dev" / name, quality=95)
                gt_xml(root / "page-gt/dev" / f"{Path(name).stem}.xml", name, W, H, polys)
                images.append({"id": len(images) + 1, "file_name": name, "width": W, "height": H, "collection": coll})
                for p in polys:
                    anns.append({"id": aid, "image_id": len(images), "category_id": 1, "iscrowd": 0,
                                 "segmentation": [p.ravel().round(1).tolist()]}); aid += 1
        payload = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "textline"}]}
        (root / "coco_instances/dev.json").write_text(json.dumps(payload))
        write_pixel_gt(payload, root / "images", "dev", root / "pixel-gt")
        print("built", root.name, len(images), "pages", flush=True)


def evaluate(sources):
    import torch
    import rq3_tts_eval as T
    import rq3_linefield as LF
    import rq3_centre_lf as CL
    import rq3_cheap_exps as ce
    import rq3_error_taxonomy as TX
    import rq3_merge_split as MS
    rows = []
    for src in sources:
        m, _, mr, st, mt, _ = T.load_model(src, "A", "base")
        net = LF.build_model()
        net.load_state_dict(torch.load(LF.MODELS / f"{src}_A_w50.pt", map_location="cpu", weights_only=False)["model"])
        net = net.to(LF.DEV).eval()
        for f in FACTORS:
            root = OUT / f"f{f:.2f}"
            coco = json.loads((root / "coco_instances/dev.json").read_text())
            by = {}
            for a in coco["annotations"]:
                by.setdefault(a["image_id"], []).append(np.asarray(a["segmentation"][0], np.float32).reshape(-1, 2))
            ds, _ = m.make_loader(root, "dev", 1152, False)
            work = {k: OUT / "pred" / src / f"f{f:.2f}" / k for k in ("maskrcnn", "center")}
            for w in work.values():
                w.mkdir(parents=True, exist_ok=True)
            maps = {}
            for i in range(len(ds)):
                img, _, meta = ds[i]
                im = next(x for x in coco["images"] if x["file_name"] == Path(meta["path"]).name)
                if im["collection"] == src:
                    continue
                inst_m, _ = T.predict_page(m, ce, mr, torch, img, meta, 1.0, st, mt, "score")
                m.page_xml(meta, inst_m, work["maskrcnn"] / f"{Path(im['file_name']).stem}.xml")
                rgb = np.asarray(Image.open(meta["path"]).convert("RGB"))
                H, W = rgb.shape[:2]
                with torch.inference_mode():
                    s1 = 1152 / max(H, W)
                    cen, _, up, dn = LF.run_net(net, cv2.resize(rgb, (int(W * s1), int(H * s1)), interpolation=cv2.INTER_AREA))
                    sel = cen > 0.5
                    thick = float(np.median(up[sel] + dn[sel])) if sel.sum() > 50 else LF.T0
                    s2 = min(s1 * LF.T0 / max(thick, 1.0), 3200 / max(H, W))
                    cen, end, up, dn = LF.run_net(net, cv2.resize(rgb, (int(W * s2), int(H * s2)),
                                                                  interpolation=cv2.INTER_AREA if s2 < 1 else cv2.INTER_LINEAR))
                frags = [{"xs": fr["xs"] / s2, "cy": fr["cy"] / s2} for fr in LF._fragments(cen, end, 0.4)]
                page = {"coll": im["collection"], "im": {"height": H, "width": W, "file_name": im["file_name"]},
                        "frags": frags, "T0": LF.T0 / s2, "root": str(root), "sp": "dev"}
                inst_c = CL.page_instances(page, "row", 1.0)
                m.page_xml({"path": im["file_name"], "orig_w": W, "orig_h": H}, inst_c, work["center"] / f"{Path(im['file_name']).stem}.xml")
                # merge statistics at the 1152 canvas
                cw, chh = int(meta["new_w"]), int(meta["new_h"]); fit = cw / W
                masks = []
                for p in by[im["id"]]:
                    mk = np.zeros((chh, cw), np.uint8); cv2.fillPoly(mk, [np.rint(p * fit).astype(np.int32)], 1); masks.append(mk > 0)
                gray = cv2.cvtColor(cv2.resize(rgb, (cw, chh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
                ink = MS.ink_mask(gray)
                inst_m_c = cv2.resize(inst_m.astype(np.uint16), (cw, chh), interpolation=cv2.INTER_NEAREST) if inst_m.shape != (chh, cw) else inst_m
                maps[im["file_name"]] = (im["collection"],
                                         [TX.per_line(c, masks, ink)[:3] for c in (inst_m_c, cv2.resize(inst_c, (cw, chh), interpolation=cv2.INTER_NEAREST))])
            for model, w in work.items():
                per_page, metrics = m.diva_eval(root, "dev", w)
                for r in per_page:
                    stem = T.page_stem(r["filename"])
                    fn = next(k for k in maps if Path(k).stem == stem)
                    coll, lst = maps[fn]
                    mg, sp, ms = lst[0 if model == "maskrcnn" else 1]
                    rows.append({"source": src, "f": f, "model": model, "collection": coll, "page": stem,
                                 "FM": float(r["LinesFMeasure"]) * 100 if r["LinesFMeasure"] not in ("", "NaN") else 0.0,
                                 "merged": float(np.mean(mg)), "split": float(np.mean(sp)), "missed": float(np.mean(ms))})
                print(src, f, model, "FM %.1f" % (100 * metrics["LinesFMeasure"]), flush=True)
        del mr, net; torch.cuda.empty_cache()
    import pandas as pd
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "results.csv", index=False)
    t = d.groupby(["model", "f"])[["FM", "merged", "split", "missed"]].mean()
    t[["merged", "split", "missed"]] *= 100
    print(t.round(1).unstack("model").to_string())


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        evaluate(sys.argv[2:])
