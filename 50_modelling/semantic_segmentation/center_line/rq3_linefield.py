#!/usr/bin/env python3
"""LineField: centre-line + line-height text-line segmentation (PERO/ParseNet-style) for RQ3.

Instead of box + mask (Mask R-CNN), a U-Net with the CATMuS ConvNeXt-Tiny encoder predicts per pixel
  0 centre line (thin polyline through the middle of each text line),
  1 line end points,
  2 height above the centre line, 3 height below it (px / T0, supervised on centre-line pixels only).
Targets are derived from each collection's own polygons, so the target's line convention is learned
from its training pages. Pages are rescaled so that the median line thickness is T0 px and then by a
random factor 2^N(0, SCALE_SIGMA) (PERO-style wide scale augmentation); training uses 512 px crops.
Inference runs twice: pass 1 estimates the median line thickness from the model's own height maps,
pass 2 processes the page rescaled to T0; polygons = centre line +- 75th-percentile heights.

    python 50_modelling/semantic_segmentation/center_line/rq3_linefield.py train SOURCE [--protocol A|B] [--steps N]
    python 50_modelling/semantic_segmentation/center_line/rq3_linefield.py eval SOURCE [--protocol A|B] [--split dev|test]
CPU-light by design: 2 torch threads, no loader workers, few DIVA workers.
"""
from __future__ import annotations
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)

import argparse, json, os, random, sys, time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("DIVA_WORKERS", "3")
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

torch.set_num_threads(2)
cv2.setNumThreads(1)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq3_aug_search as S  # noqa: E402

ROOT = S.ROOT
OUT = ROOT / "99_evaluation/analysis/rq3_linefield"
MODELS = ROOT / "80_models/semantic_segmentation/center_line"
CATMUS = ROOT / "80_models/instance_segmentation/mask_rcnn/catmus_pretrain/maskrcnn_convnext_tiny_704/best.pt"
TARGET = os.environ.get("LF_TARGET", "poly")   # "poly": annotated polygon extent; "ink": ink envelope
SUFFIX = "" if TARGET == "poly" else f"_{TARGET}"
T0 = float(os.environ.get("LF_T0", "24" if TARGET == "poly" else "16"))  # canonical line thickness (px)
SCALE_SIGMA = 0.6      # log2 std of the random rescaling around T0
CROP, BATCH, LR = 512, 4, 3e-4
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ model
def build_model():
    import segmentation_models_pytorch as smp
    import re
    net = smp.Unet("tu-convnext_tiny", encoder_weights=None, classes=4)
    if os.environ.get("LF_ENC") == "cbad":   # ConvNeXt-Tiny distilled from DINOv3 on unlabeled cBAD pages (RQ1/RQ2)
        sd = torch.load(ROOT / "80_models/instance_segmentation/mask_rcnn/cbad_distilled_backbones/convnext_tiny_cbad_dinov3_epoch1100.pt",
                        map_location="cpu", weights_only=True)
        enc = {"model." + re.sub(r"^(stem|stages)\.(\d+)", r"\1_\2", k): v for k, v in sd.items() if not k.startswith("head.")}
    else:
        ck = torch.load(CATMUS, map_location="cpu", weights_only=False)["model"]
        enc = {"model." + k[len("backbone.encoder."):]: v for k, v in ck.items() if k.startswith("backbone.encoder.")}
    net.encoder.load_state_dict(enc, strict=True)
    return net


# ------------------------------------------------------------------ targets
def page_polys(coco):
    by = defaultdict(list)
    for a in coco["annotations"]:
        for seg in a.get("segmentation", []):
            p = np.asarray(seg, np.float32).reshape(-1, 2)
            if len(p) >= 3:
                by[a["image_id"]].append(p)
    return by


def line_geometry(poly, shape):
    """Per-column centre / up / down of one polygon (in the given pixel frame)."""
    x1, y1 = np.floor(poly.min(0)).astype(int)
    x2, y2 = np.ceil(poly.max(0)).astype(int) + 1
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(shape[1], x2), min(shape[0], y2)
    if x2 - x1 < 3 or y2 - y1 < 2:
        return None
    m = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.fillPoly(m, [np.rint(poly - [x1, y1]).astype(np.int32)], 1)
    cols = np.where(m.any(0))[0]
    if len(cols) < 3:
        return None
    top = m[:, cols].argmax(0)
    bot = m.shape[0] - 1 - m[::-1, cols].argmax(0)
    cy = (top + bot) / 2.0
    return x1 + cols, y1 + cy, cy - top + 0.5, bot - cy + 0.5


def render_targets(polys, shape, scale):
    """Target maps for polygons given in original coordinates, rendered at `scale`."""
    h, w = shape
    cen = np.zeros((h, w), np.float32); end = np.zeros((h, w), np.float32)
    up = np.zeros((h, w), np.float32); dn = np.zeros((h, w), np.float32)
    for p in polys:
        g = line_geometry(p * scale, (h, w))
        if g is None:
            continue
        xs, cy, hu, hd = g
        k = max(3, int(len(xs) / 40) * 2 + 1)
        cy = cv2.GaussianBlur(cy.astype(np.float32)[:, None], (1, k), 0)[:, 0] if len(cy) > k else cy
        pts = np.stack([xs, cy], 1).round().astype(np.int32)
        th = max(1, int(round(0.08 * np.median(hu + hd))))
        cv2.polylines(cen, [pts], False, 1.0, th)
        hu_v, hd_v = float(np.percentile(hu, 75)), float(np.percentile(hd, 75))
        cv2.polylines(up, [pts], False, hu_v / T0, th)
        cv2.polylines(dn, [pts], False, hd_v / T0, th)
        r = max(2, th + 1)
        cv2.circle(end, tuple(pts[0]), r, 1.0, -1); cv2.circle(end, tuple(pts[-1]), r, 1.0, -1)
    return np.stack([cen, end, up, dn])


def ink_polys(img, polys):
    """Replace each annotated polygon by the envelope of the ink inside it (convention-free target).
    Ink = minority Otsu side inside the union of the polygons (same rule as the DIVA pixel GT)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    union = np.zeros(gray.shape, np.uint8)
    for p in polys:
        cv2.fillPoly(union, [np.rint(p).astype(np.int32)], 1)
    vals = gray[union > 0]
    if vals.size < 100:
        return polys
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (gray <= t)
    ink = dark if dark[union > 0].mean() <= 0.5 else ~dark
    out = []
    for p in polys:
        x1, y1 = np.floor(p.min(0)).astype(int); x2, y2 = np.ceil(p.max(0)).astype(int) + 1
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
        if x2 - x1 < 3 or y2 - y1 < 2:
            continue
        m = np.zeros((y2 - y1, x2 - x1), np.uint8)
        cv2.fillPoly(m, [np.rint(p - [x1, y1]).astype(np.int32)], 1)
        k = (m > 0) & ink[y1:y2, x1:x2]
        cols = np.where(k.any(0))[0]
        if len(cols) < 3:
            out.append(p); continue
        top = k[:, cols].argmax(0).astype(np.float32)
        bot = (k.shape[0] - 1 - k[::-1, cols].argmax(0)).astype(np.float32)
        # envelope: running percentile over a window of about one line height
        win = max(3, int(np.median(bot - top + 1)) * 2 + 1)
        pad = win // 2
        tp = np.pad(top, pad, mode="edge"); bp = np.pad(bot, pad, mode="edge")
        top_s = np.array([np.percentile(tp[i:i + win], 15) for i in range(len(cols))])
        bot_s = np.array([np.percentile(bp[i:i + win], 85) for i in range(len(cols))])
        xs = (cols + x1).astype(np.float32)
        step = max(1, len(xs) // 60)
        up_line = np.stack([xs[::step], top_s[::step] + y1], 1)
        lo_line = np.stack([xs[::step], bot_s[::step] + y1 + 1], 1)[::-1]
        out.append(np.concatenate([up_line, lo_line]).astype(np.float32))
    return out


def median_thickness(polys):
    t = []
    for p in polys:
        g = line_geometry(p, (int(p[:, 1].max()) + 2, int(p[:, 0].max()) + 2))
        if g is not None:
            t.append(np.median(g[2] + g[3]))
    return float(np.median(t)) if t else None


# ------------------------------------------------------------------ data
class Pages:
    def __init__(self, root, split, max_pages=None):
        coco = json.loads((root / f"coco_instances/{split}.json").read_text())
        polys = page_polys(coco)
        self.pages = []
        for im in sorted(coco["images"], key=lambda x: x["file_name"])[:max_pages] if max_pages else coco["images"]:
            ps = polys.get(im["id"], [])
            img = np.asarray(Image.open(root / "images" / split / im["file_name"]).convert("RGB"))
            if ps and TARGET == "ink":
                ps = ink_polys(img, ps)
            t = median_thickness(ps) if ps else None
            if not t:
                continue
            self.pages.append((img, ps, t))

    def sample(self, rng):
        img, polys, t = self.pages[rng.randrange(len(self.pages))]
        s = (T0 / t) * 2 ** rng.gauss(0, SCALE_SIGMA)
        s = min(s, 4096 / max(img.shape[:2]))
        h, w = max(8, int(img.shape[0] * s)), max(8, int(img.shape[1] * s))
        x0 = rng.randint(0, max(0, w - CROP)); y0 = rng.randint(0, max(0, h - CROP))
        # resize only the crop window (cheap)
        inv = 1.0 / s
        sx1, sy1 = int(x0 * inv), int(y0 * inv)
        sx2, sy2 = int(min(w, x0 + CROP) * inv) + 1, int(min(h, y0 + CROP) * inv) + 1
        part = img[sy1:sy2, sx1:sx2]
        cw, ch = min(w, x0 + CROP) - x0, min(h, y0 + CROP) - y0
        crop = cv2.resize(part, (cw, ch), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        shifted = [(p - [sx1, sy1]) for p in polys]
        tgt = render_targets(shifted, (ch, cw), cw / max(1, part.shape[1]))
        canvas = np.full((CROP, CROP, 3), 255, np.uint8); canvas[:ch, :cw] = crop
        t_can = np.zeros((4, CROP, CROP), np.float32); t_can[:, :ch, :cw] = tgt
        # photometric jitter
        a, b = rng.uniform(0.7, 1.3), rng.uniform(-25, 25)
        canvas = np.clip(canvas.astype(np.float32) * a + b, 0, 255)
        if rng.random() < 0.3:
            canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.3, 1.2))
        return canvas, t_can


def to_tensor(img):
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    return (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]


# ------------------------------------------------------------------ training
def dice(logit, t):
    p = torch.sigmoid(logit)
    return 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)


def lf_loss(y, t):
    m = t[:, 0]
    l_cen = F.binary_cross_entropy_with_logits(y[:, 0], t[:, 0], pos_weight=torch.tensor(5.0, device=y.device)) + dice(y[:, 0], t[:, 0])
    l_end = F.binary_cross_entropy_with_logits(y[:, 1], t[:, 1], pos_weight=torch.tensor(5.0, device=y.device)) + dice(y[:, 1], t[:, 1])
    l_h = ((F.smooth_l1_loss(y[:, 2], t[:, 2], reduction="none") + F.smooth_l1_loss(y[:, 3], t[:, 3], reduction="none")) * m).sum() / m.sum().clamp_min(1)
    return l_cen, l_end, l_h


def val_crops(colls, per_coll=12, pages_per_coll=3, seed=123):
    """Fixed validation crops from the dev pages of other collections (learning curves of transfer)."""
    rng = random.Random(seed)
    xs, ts = [], []
    for c in colls:
        pg = Pages(ROOT / "00_data/RQ3/dev" / c, "dev", max_pages=pages_per_coll)
        for _ in range(per_coll):
            x, t = pg.sample(rng); xs.append(x); ts.append(t)
    return xs, ts


def train(source, protocol, steps, seed=42, root=None, init=None, out=None):
    """root: data root with coco_instances/train.json (default: the RQ3 source); init: checkpoint to
    start from (e.g. the CATMuS-pretrained LineField); out: checkpoint path."""
    out = Path(out) if out else MODELS / f"{source}_{protocol}{SUFFIX}.pt"
    if out.exists():
        print("exists", out); return
    rng = random.Random(seed); torch.manual_seed(seed)
    pages = Pages(Path(root) if root else S.train_root(source, protocol), "train")
    net = build_model()
    if init:
        net.load_state_dict(torch.load(init, map_location="cpu", weights_only=False)["model"])
    net = net.to(DEV)
    enc = list(net.encoder.parameters()); enc_ids = {id(p) for p in enc}
    opt = torch.optim.AdamW([{"params": enc, "lr": LR / 3},
                             {"params": [p for p in net.parameters() if id(p) not in enc_ids], "lr": LR}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR / 3, LR], total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    vcolls = [c for c in os.environ.get("LF_VAL", "").split(",") if c and not (c == source and protocol.startswith("A"))]
    vx, vt = val_crops(vcolls) if vcolls else ([], [])
    curve = {"train": [], "val": [], "val_collections": vcolls, "encoder": os.environ.get("LF_ENC", "catmus_maskrcnn"),
             "init": str(init) if init else None}

    def val_loss():
        net.eval(); acc = []
        with torch.no_grad(), torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            for i in range(0, len(vx), 8):
                x = torch.stack([to_tensor(a) for a in vx[i:i + 8]]).to(DEV)
                t = torch.from_numpy(np.stack(vt[i:i + 8])).to(DEV)
                acc.append([float(v) for v in lf_loss(net(x).float(), t)])
        net.train()
        a = np.mean(acc, 0)
        return [float(a[0]), float(a[1]), float(a[2]), float(a[0] + 0.5 * a[1] + a[2])]

    net.train(); t0 = time.time(); hist = []
    for step in range(steps):
        xs, ts = zip(*(pages.sample(rng) for _ in range(BATCH)))
        x = torch.stack([to_tensor(i) for i in xs]).to(DEV)
        t = torch.from_numpy(np.stack(ts)).to(DEV)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            y = net(x)
        y = y.float()
        l_cen, l_end, l_h = lf_loss(y, t)
        loss = l_cen + 0.5 * l_end + l_h
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        hist.append([float(l_cen), float(l_end), float(l_h)])
        if (step + 1) % 50 == 0:
            a = np.mean(hist[-50:], 0)
            curve["train"].append([step + 1, float(a[0]), float(a[1]), float(a[2]), float(a[0] + 0.5 * a[1] + a[2])])
        if vx and (step % 250 == 0 or step == steps - 1):
            curve["val"].append([step + 1, *val_loss()])
        if step % 250 == 0 or step == steps - 1:
            print(f"{source} step {step} cen {np.mean([h[0] for h in hist[-250:]]):.3f} "
                  f"end {np.mean([h[1] for h in hist[-250:]]):.3f} h {np.mean([h[2] for h in hist[-250:]]):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "T0": T0, "steps": steps, "source": source, "protocol": protocol,
                "init": str(init) if init else None}, out)
    out.with_suffix(".curve.json").write_text(json.dumps(curve))
    print("saved", out, flush=True)


# ------------------------------------------------------------------ inference
@torch.inference_mode()
def run_net(net, img):
    h, w = img.shape[:2]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    pad = np.pad(img, ((0, ph), (0, pw), (0, 0)), constant_values=255)
    with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
        y = net(to_tensor(pad)[None].to(DEV))[0].float()
    y = y[:, :h, :w]
    return torch.sigmoid(y[0]).cpu().numpy(), torch.sigmoid(y[1]).cpu().numpy(), \
        (y[2] * T0).cpu().numpy(), (y[3] * T0).cpu().numpy()


def infer_page(net, img, fit_long=1152, max_long=3200, thr=0.4):
    """Two passes: estimate thickness at a nominal fit, then process at the canonical scale."""
    H, W = img.shape[:2]
    s1 = fit_long / max(H, W)
    im1 = cv2.resize(img, (max(8, int(W * s1)), max(8, int(H * s1))), interpolation=cv2.INTER_AREA)
    cen, _, up, dn = run_net(net, im1)
    sel = cen > 0.5
    thick = float(np.median(up[sel] + dn[sel])) if sel.sum() > 50 else T0
    s2 = min(s1 * T0 / max(thick, 1.0), max_long / max(H, W))
    im2 = cv2.resize(img, (max(8, int(W * s2)), max(8, int(H * s2))),
                     interpolation=cv2.INTER_AREA if s2 < 1 else cv2.INTER_LINEAR)
    cen, end, up, dn = run_net(net, im2)
    return polygons_from_maps(cen, end, up, dn, thr), s2


MERGE = os.environ.get("LF_MERGE", "0") == "1"   # join collinear centre-line fragments of one line


def _fragments(cen, end, thr):
    core = ((cen - 0.5 * end) > thr).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    frags = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 0.5 * T0:
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        xs = xs + x; ys = ys + y
        ux = np.unique(xs)
        cy = np.array([ys[xs == u].mean() for u in ux])
        frags.append({"xs": ux, "cy": cy, "px": (ys, xs)})
    return frags


def _merge(frags):
    """Greedy left-to-right join: gap <= 2.5 T0 horizontally, end heights within 0.5 T0."""
    frags = sorted(frags, key=lambda f: f["xs"][0])
    used = [False] * len(frags); out = []
    for i, f in enumerate(frags):
        if used[i]:
            continue
        cur = {"xs": f["xs"], "cy": f["cy"], "px": [f["px"]]}; used[i] = True
        changed = True
        while changed:
            changed = False
            best, bd = None, None
            for j, g in enumerate(frags):
                if used[j]:
                    continue
                gap = g["xs"][0] - cur["xs"][-1]
                if gap < -0.5 * T0 or gap > 2.5 * T0:
                    continue
                dy = abs(g["cy"][0] - cur["cy"][-1])
                if dy < 0.5 * T0 and (bd is None or gap + dy < bd):
                    best, bd = j, gap + dy
            if best is not None:
                g = frags[best]; used[best] = True
                keep = g["xs"] > cur["xs"][-1]
                cur["xs"] = np.concatenate([cur["xs"], g["xs"][keep]])
                cur["cy"] = np.concatenate([cur["cy"], g["cy"][keep]])
                cur["px"].append(g["px"]); changed = True
        out.append(cur)
    return out


def polygons_from_maps(cen, end, up, dn, thr):
    """Centre-line fragments (end points subtracted), optionally joined per line -> polygons."""
    frags = _fragments(cen, end, thr)
    lines = _merge(frags) if MERGE else [{"xs": f["xs"], "cy": f["cy"], "px": [f["px"]]} for f in frags]
    polys = []
    for ln in lines:
        ux, cy = ln["xs"], ln["cy"]
        if ux[-1] - ux[0] < 1.5 * T0:
            continue
        ys = np.concatenate([p[0] for p in ln["px"]]); xs = np.concatenate([p[1] for p in ln["px"]])
        hu = float(np.percentile(up[ys, xs], 75)); hd = float(np.percentile(dn[ys, xs], 75))
        if hu + hd < 3:
            continue
        step = max(1, len(ux) // 40)
        ux2, cy2 = ux[::step], cy[::step]
        top = np.stack([ux2, cy2 - hu], 1); bot = np.stack([ux2, cy2 + hd], 1)[::-1]
        polys.append(np.concatenate([top, bot]))
    return polys


def write_page_xml(path, image_name, w, h, polys):
    """Rasterise the polygons into an instance map and use the same PAGE export as the Mask R-CNN
    pipeline (maskrcnn_rq3_loo.page_xml), so every pipeline reaches the DIVA evaluator identically."""
    import rq3_cheap_exps as ce
    inst = np.zeros((h, w), np.uint16)
    for i, p in enumerate(sorted(polys, key=lambda q: q[:, 1].mean())):
        mk = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mk, [np.rint(p).astype(np.int32)], 1)
        inst[(mk > 0) & (inst == 0)] = i + 1
    meta = {"path": str(image_name), "orig_w": w, "orig_h": h}
    return ce.m.page_xml(meta, inst, path)


def evaluate(source, protocol, split):
    import rq3_cheap_exps as ce
    m = ce.m
    ck = torch.load(MODELS / f"{source}_{protocol}{SUFFIX}.pt", map_location="cpu", weights_only=False)
    net = build_model(); net.load_state_dict(ck["model"]); net = net.to(DEV).eval()
    res = {}
    for coll, root, sp in S.eval_targets(source, protocol, split):
        coco = json.loads((root / f"coco_instances/{sp}.json").read_text())
        work = OUT / "pred" / f"{source}_{protocol}{SUFFIX}_{split}" / coll
        work.mkdir(parents=True, exist_ok=True)
        for im in coco["images"]:
            img = np.asarray(Image.open(root / "images" / sp / im["file_name"]).convert("RGB"))
            polys, s2 = infer_page(net, img)
            polys = [p / s2 for p in polys]
            write_page_xml(work / f"{Path(im['file_name']).stem}.xml", im["file_name"], img.shape[1], img.shape[0], polys)
        _, metrics = m.diva_eval(root, sp, work)
        # every flag that changes the output is stored with the numbers (LF_MERGE once differed silently)
        res[coll] = {**metrics, "flags": {"LF_TARGET": TARGET, "LF_T0": T0, "LF_MERGE": MERGE,
                                          "checkpoint": f"{source}_{protocol}{SUFFIX}.pt"}}
        print(source, coll, "FM %.1f" % (100 * metrics["LinesFMeasure"]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"eval_{source}_{protocol}{SUFFIX}_{split}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("source")
    ap.add_argument("--protocol", default="A")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.source, a.protocol, a.steps)
    else:
        evaluate(a.source, a.protocol, a.split)
