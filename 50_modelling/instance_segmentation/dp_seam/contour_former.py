#!/usr/bin/env python3
"""ContourFormer: a transformer boundary head for text lines (exploratory, DIVA-HisDB Task 2).

Each detected line box is cropped (pad 15 px, resized to 1024x256). A ConvNeXt-Tiny encoder (ImageNet-12k)
gives stride-8/16 feature tokens; a transformer decoder with N=128 column queries (learned embedding + sine
encoding of the column position) cross-attends to them, and every query regresses the line's upper and lower
boundary in its column plus a presence logit. The polygon is the upper boundary left to right and the lower
boundary right to left, so polygons of different lines may overlap and there is no mask grid.
Loss: presence BCE + L1 on the boundaries + (1 - soft ink IoU), the ink IoU of the evaluator.

Boxes come from the Mask R-CNN (ConvNeXt-Tiny, CATMuS, 704) of the thesis, score threshold selected on the
validation pages with the DIVA evaluator; the region filter of maskrcnn_crop_refine.py is applied.

    python contour_former.py cache SUB        detector boxes for val/test (score >= 0.3)
    python contour_former.py train SUB        train on the training pages, select on validation lines
    python contour_former.py eval SUB         select score threshold on val pages, score test pages once
"""
import json, math, random, sys, time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "50_modelling/instance_segmentation/mask_rcnn"))
sys.path.insert(0, str(REPO / "50_modelling/instance_segmentation/rtdetr_bbox_unet"))
sys.path.insert(0, str(REPO / "99_evaluation/analysis"))

DIVA = REPO / "00_data/DIVA-HisDB"
OUT = REPO / "80_models/instance_segmentation/dp_seam"
EVAL = REPO / "99_evaluation/instance_segmentation/dp_seam/diva-hisdb"
N, CW, CH, PAD = 128, 1024, 256, 15
SPLITS = {"train": "training", "val": "validation", "test": "public-test"}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- data
def polys(xml):
    out = []
    for el in ET.parse(xml).getroot().iter():
        if el.tag.endswith("TextLine"):
            c = next(ch for ch in el if ch.tag.endswith("Coords"))
            p = np.array([[int(float(v)) for v in xy.split(",")] for xy in c.get("points").split()], np.int32)
            if len(p) >= 3:
                out.append(p)
    return out


def page_paths(sub, split):
    s = SPLITS[split]
    gt = DIVA / sub / f"PAGE-gt-{sub}-TASK-2/TASK-2/{s}"
    # pages from the pixel GT: the CS18/CS863 XML folders also hold *_TEST.xml duplicates
    for px in sorted((DIVA / sub / f"pixel-level-gt-{sub}/pixel-level-gt/{s}").glob("*.png")):
        yield (px.stem, gt / f"{px.stem}.xml", next((DIVA / sub / f"img-{sub}/img/{s}").glob(px.stem + ".*")), px)


def line_store(sub, split):
    """Per GT line: a margin crop (scaled to <= 1300 px wide), its box inside it, and line/other-ink masks."""
    items = []
    for stem, xml, img_p, pix_p in page_paths(sub, split):
        img = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
        fg = (cv2.imread(str(pix_p))[:, :, 0] & 0x08) > 0
        H, W = fg.shape
        for p in polys(xml):
            x0, y0, x1, y1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max() + 1, p[:, 1].max() + 1
            w, h = x1 - x0, y1 - y0
            mx, my = int(0.04 * w) + 40, int(0.6 * h) + 40
            X0, Y0, X1, Y1 = max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)
            f = min(1.0, 1300 / (X1 - X0))
            size = (max(1, round((X1 - X0) * f)), max(1, round((Y1 - Y0) * f)))
            lm = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
            cv2.fillPoly(lm, [p - [X0, Y0]], 1)
            code = (lm * 2 + fg[Y0:Y1, X0:X1]).astype(np.uint8)      # bit1 polygon, bit0 ink
            items.append({"img": cv2.resize(img[Y0:Y1, X0:X1], size, interpolation=cv2.INTER_AREA),
                          "code": cv2.resize(code, size, interpolation=cv2.INTER_NEAREST),
                          "box": np.array([x0 - X0, y0 - Y0, x1 - X0, y1 - Y0], np.float32) * f, "f": f})
    return items


def make_sample(it, jitter):
    img, code = it["img"], it["code"]
    x0, y0, x1, y1 = it["box"]
    w, h = x1 - x0, y1 - y0
    if jitter:
        x0 += random.gauss(0, 0.015 * w); x1 += random.gauss(0, 0.015 * w)
        y0 += random.gauss(0, 0.07 * h); y1 += random.gauss(0, 0.07 * h)
    pad = PAD * it["f"]
    H, W = code.shape
    cx0, cy0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    cx1, cy1 = int(min(W, x1 + pad)), int(min(H, y1 + pad))
    crop = cv2.resize(img[cy0:cy1, cx0:cx1], (CW, CH), interpolation=cv2.INTER_LINEAR)
    c = cv2.resize(code[cy0:cy1, cx0:cx1], (CW, CH), interpolation=cv2.INTER_NEAREST)
    poly, ink = (c >> 1) & 1, c & 1
    flip = jitter and random.random() < 0.5
    if flip:
        crop, poly, ink = crop[:, ::-1], poly[:, ::-1], ink[:, ::-1]
    cols = poly.reshape(CH, N, CW // N).any(axis=2)                  # (CH, N)
    pres = cols.any(axis=0)
    top = np.where(pres, cols.argmax(0), 0) / CH
    bot = np.where(pres, CH - cols[::-1].argmax(0), 0) / CH
    L = (poly & ink).astype(np.float32)
    O = ((1 - poly) & ink).astype(np.float32)
    L = L.reshape(CH // 2, 2, N, CW // N).mean(axis=(1, 3))         # (128 rows, N) ink fractions
    O = O.reshape(CH // 2, 2, N, CW // N).mean(axis=(1, 3))
    if jitter:
        crop = np.clip(crop.astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-20, 20), 0, 255)
    x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float() / 255
    return x, torch.tensor(pres, dtype=torch.float32), torch.tensor(np.stack([top, bot], -1), dtype=torch.float32), \
        torch.from_numpy(L), torch.from_numpy(O)


# ---------------------------------------------------------------- model
def sine(pos, d):
    """pos: (...,) in [0,1] -> (..., d)"""
    i = torch.arange(d // 2, device=pos.device, dtype=torch.float32)
    freq = 1.0 / (100.0 ** (2 * i / d))
    a = pos[..., None] * 2 * math.pi * 10 * freq
    return torch.cat([a.sin(), a.cos()], -1)


class ContourFormer(nn.Module):
    def __init__(self, d=256, layers=4):
        super().__init__()
        self.enc = timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True, features_only=True, out_indices=(1, 2))
        self.proj = nn.ModuleList(nn.Conv2d(c, d, 1) for c in self.enc.feature_info.channels())
        self.level = nn.Parameter(torch.zeros(2, d))
        self.q = nn.Parameter(torch.randn(N, d) * 0.02)
        layer = nn.TransformerDecoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
        self.d = d

    def forward(self, x):
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        toks = []
        for i, f in enumerate(self.enc(x)):
            f = self.proj[i](f)
            B, C, H, W = f.shape
            ys = (torch.arange(H, device=f.device) + 0.5) / H
            xs = (torch.arange(W, device=f.device) + 0.5) / W
            pos = torch.cat([sine(ys[:, None].expand(H, W), C // 2), sine(xs[None].expand(H, W), C // 2)], -1)
            toks.append(f.flatten(2).transpose(1, 2) + pos.view(1, H * W, C) + self.level[i])
        mem = torch.cat(toks, 1)
        cx = (torch.arange(N, device=x.device) + 0.5) / N
        q = (self.q + sine(cx, self.d)).unsqueeze(0).expand(x.shape[0], -1, -1)
        o = self.head(self.dec(q, mem))
        t = torch.sigmoid(o[..., 1])
        b = t + (1 - t) * torch.sigmoid(o[..., 2])
        return o[..., 0], t, b


def soft_mask(logit, t, b, tau=2.0 / CH):
    y = (torch.arange(CH // 2, device=t.device) + 0.5) / (CH // 2)
    m = torch.sigmoid((y[None, :, None] - t[:, None]) / tau) * torch.sigmoid((b[:, None] - y[None, :, None]) / tau)
    return m * torch.sigmoid(logit)[:, None]                          # (B, rows, N)


def losses(out, pres, tb, L, O):
    logit, t, b = out
    lp = F.binary_cross_entropy_with_logits(logit, pres)
    w = pres.sum().clamp(min=1)
    l1 = ((t - tb[..., 0]).abs() * pres + (b - tb[..., 1]).abs() * pres).sum() / w
    M = soft_mask(logit, t, b)
    inter = (M * L).sum((1, 2))
    iou = inter / (L.sum((1, 2)) + (M * O).sum((1, 2))).clamp(min=1e-3)
    return lp + 5 * l1 + (1 - iou).mean(), iou.detach()


# ---------------------------------------------------------------- train
def train(sub, steps=4000, bs=8):
    random.seed(0); torch.manual_seed(0)
    t0 = time.time()
    tr, va = line_store(sub, "train"), line_store(sub, "val")
    print(f"{sub}: {len(tr)} train lines, {len(va)} val lines ({time.time() - t0:.0f}s)", flush=True)
    model = ContourFormer().to(DEV)
    enc = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": 1e-4}, {"params": rest, "lr": 3e-4}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 3e-4], total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    OUT.mkdir(parents=True, exist_ok=True)
    best, log = -1, []
    for step in range(1, steps + 1):
        model.train()
        batch = [make_sample(random.choice(tr), True) for _ in range(bs)]
        x, pres, tb, L, O = (torch.stack(v).to(DEV) for v in zip(*batch))
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
        loss, _ = losses([o.float() for o in out], pres, tb, L, O)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % 500 == 0 or step == steps:
            ious = validate(model, va)
            score = float((ious >= 0.75).mean() + ious.mean())
            log.append({"step": step, "loss": float(loss), "val_ink_iou": float(ious.mean()), "val_rec75": float((ious >= 0.75).mean())})
            print(f"  step {step} loss {loss:.3f} val ink IoU {ious.mean():.4f} rec@.75 {(ious >= .75).mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
            if score > best:
                best = score
                torch.save(model.state_dict(), OUT / f"{sub}.pt")
    (OUT / f"{sub}_log.json").write_text(json.dumps(log, indent=1))


@torch.no_grad()
def validate(model, items):
    """Hard ink IoU of the predicted region on GT boxes (no jitter), at the 1024x256 crop resolution."""
    model.eval()
    ious = []
    for i in range(0, len(items), 16):
        batch = [make_sample(it, False) for it in items[i:i + 16]]
        x, pres, tb, L, O = (torch.stack(v).to(DEV) for v in zip(*batch))
        with torch.autocast("cuda", dtype=torch.float16):
            logit, t, b = (o.float() for o in model(x))
        M = (soft_mask(logit, t, b, tau=1e-4) > 0.5).float()
        inter = (M * L).sum((1, 2)); un = L.sum((1, 2)) + (M * O).sum((1, 2))
        ious += (inter / un.clamp(min=1e-6)).tolist()
    return np.array(ious)


# ---------------------------------------------------------------- detection cache + evaluation
def cache(sub):
    from torch.utils.data import DataLoader
    from maskrcnn_diva import DivaCocoLines, build_model, collate, region_polygon
    run = REPO / f"80_models/instance_segmentation/mask_rcnn/diva-hisdb/{sub}/maskrcnn_convnext_tiny_catmus_704/best.pt"
    ck = torch.load(run, map_location="cpu", weights_only=True)
    cfg = ck["args"]
    model = build_model(cfg["arm"], cfg["image_size"], cfg.get("mask_roi_size", 14), cfg.get("mask_roi_width", 0),
                        cfg.get("roi_batch_size", 512), cfg.get("checkpoint_mask_head", False))
    model.load_state_dict(ck["model"]); model.eval().to(DEV)
    out = {}
    for split in ("val", "test"):
        ds = DivaCocoLines(DIVA / f"coco_task2_{sub}", DIVA / f"yolo_dataset_{sub}/images", split, cfg["image_size"], False)
        gt = DIVA / sub / f"PAGE-gt-{sub}-TASK-2/TASK-2/{SPLITS[split]}"
        for images, _, meta in DataLoader(ds, batch_size=1, num_workers=1, collate_fn=collate):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                o = model([images[0].to(DEV)])[0]
            m = meta[0]
            stem = Path(m["path"]).stem
            keep = o["scores"] >= 0.3
            boxes = o["boxes"][keep].float().cpu().numpy() / (m["new_w"] / m["orig_w"])
            region = region_polygon(gt / f"{stem}.xml")
            out[f"{split}/{stem}"] = {"boxes": boxes.tolist(), "scores": o["scores"][keep].float().cpu().tolist(),
                                      "region": region.tolist(), "size": [m["orig_w"], m["orig_h"]], "path": m["path"]}
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / f"{sub}_detections.json").write_text(json.dumps(out))
    print(sub, {s: sum(k.startswith(s) for k in out) for s in ("val", "test")})


@torch.no_grad()
def predict_page(model, img, boxes):
    """img RGB full page; boxes (n,4) -> list of polygons (page coordinates)."""
    H, W = img.shape[:2]
    crops, frames = [], []
    for x0, y0, x1, y1 in boxes:
        cx0, cy0 = int(max(0, x0 - PAD)), int(max(0, y0 - PAD))
        cx1, cy1 = int(min(W, x1 + PAD)), int(min(H, y1 + PAD))
        if cx1 - cx0 < 4 or cy1 - cy0 < 4:
            continue
        crops.append(torch.from_numpy(cv2.resize(img[cy0:cy1, cx0:cx1], (CW, CH))).permute(2, 0, 1).float() / 255)
        frames.append((cx0, cy0, cx1, cy1))
    res = []
    for i in range(0, len(crops), 16):
        x = torch.stack(crops[i:i + 16]).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            logit, t, b = (o.float().cpu().numpy() for o in model(x))
        for k in range(len(x)):
            cx0, cy0, cx1, cy1 = frames[i + k]
            on = np.where(logit[k] > 0)[0]
            if len(on) < 2:
                res.append(None); continue
            sx, sy = (cx1 - cx0) / N, cy1 - cy0
            top, bot = [], []
            for c in on:
                for xx in (c, c + 1):                                  # step boundary: both column edges
                    top.append((cx0 + xx * sx, cy0 + t[k, c] * sy))
                    bot.append((cx0 + xx * sx, cy0 + b[k, c] * sy))
            res.append(np.round(np.array(top + bot[::-1])).astype(int))
    return res


def write_page(path, size, region, polys_):
    ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    ET.register_namespace("", ns)
    root = ET.Element(f"{{{ns}}}PcGts")
    page = ET.SubElement(root, f"{{{ns}}}Page", {"imageFilename": path.stem + ".jpg", "imageWidth": str(size[0]), "imageHeight": str(size[1])})
    reg = ET.SubElement(page, f"{{{ns}}}TextRegion", {"id": "region_textline"})
    ET.SubElement(reg, f"{{{ns}}}Coords", {"points": " ".join(f"{x},{y}" for x, y in region)})
    for i, p in enumerate(polys_):
        ln = ET.SubElement(reg, f"{{{ns}}}TextLine", {"id": f"line_{i}"})
        ET.SubElement(ln, f"{{{ns}}}Coords", {"points": " ".join(f"{x},{y}" for x, y in p)})
        base = p[len(p) // 2:][::-1]                                   # lower boundary, left to right
        ET.SubElement(ln, f"{{{ns}}}Baseline", {"points": " ".join(f"{x},{y}" for x, y in base)})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def score_dir(sub, split, d):
    import story_diva_pages as S
    import shutil, subprocess, tempfile
    s = SPLITS[split]

    def one(xml):
        with tempfile.TemporaryDirectory() as cwd:
            shutil.copy(xml, Path(cwd) / xml.name)
            subprocess.run(["java", "-Djava.awt.headless=true", "-cp", f"/usr/share/openjfx/lib/*:{S.JAR}",
                            "ch.unifr.LineSegmentationEvaluatorTool",
                            "-igt", str(DIVA / sub / f"pixel-level-gt-{sub}/pixel-level-gt/{s}/{xml.stem}.png"),
                            "-xgt", str(DIVA / sub / f"PAGE-gt-{sub}-TASK-2/TASK-2/{s}/{xml.stem}.xml"),
                            "-xp", str(Path(cwd) / xml.name), "-csv"], cwd=cwd, capture_output=True, check=True)
            lines = (Path(cwd) / "results.csv").read_text().splitlines()
            head, vals = lines[0].split(","), lines[1].split(",")
            return dict(zip(head[1:], [float(v) for v in vals[-(len(head) - 1):]]))
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, sorted(d.glob("*.xml"))))
    keys = ["PixelIU", "LinesIU", "LinesRecall", "LinesPrecision", "LinesFMeasure"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def evaluate(sub, grid=(0.3, 0.5, 0.7, 0.9)):
    import cv2 as _cv2
    model = ContourFormer().to(DEV)
    model.load_state_dict(torch.load(OUT / f"{sub}.pt", map_location=DEV)); model.eval()
    det = json.loads((EVAL / f"{sub}_detections.json").read_text())
    preds = {}                                                         # (split, stem) -> (boxes, scores, polygons)
    for key, v in det.items():
        split, stem = key.split("/")
        img = _cv2.cvtColor(_cv2.imread(v["path"]), _cv2.COLOR_BGR2RGB)
        boxes = np.array(v["boxes"], np.float32).reshape(-1, 4)
        region = np.array(v["region"], np.int32)
        inside = [_cv2.pointPolygonTest(region, (float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)), False) >= 0 for b in boxes]
        boxes, scores = boxes[inside], np.array(v["scores"])[inside]
        preds[(split, stem)] = (scores, predict_page(model, img, boxes), v)

    def dump(split, thr):
        d = EVAL / sub / f"{split}_s{thr:g}"
        d.mkdir(parents=True, exist_ok=True)
        for (sp, stem), (scores, ps, v) in preds.items():
            if sp == split:
                write_page(d / f"{stem}.xml", v["size"], v["region"], [p for s, p in zip(scores, ps) if s >= thr and p is not None])
        return d
    rows = []
    for thr in grid:
        r = score_dir(sub, "val", dump("val", thr))
        rows.append((r["LinesFMeasure"], r["PixelIU"], thr)); print(f"  [val] s={thr} FM={100 * r['LinesFMeasure']:.2f}", flush=True)
    thr = max(rows)[2]
    r = score_dir(sub, "test", dump("test", thr))
    r["score_threshold"] = thr
    (EVAL / sub / "summary.json").write_text(json.dumps(r, indent=2))
    print(sub, "test", {k: round(100 * v, 2) if k != "score_threshold" else v for k, v in r.items()}, flush=True)


if __name__ == "__main__":
    cmd, sub = sys.argv[1], sys.argv[2]
    {"cache": cache, "train": train, "eval": evaluate}[cmd](sub)
