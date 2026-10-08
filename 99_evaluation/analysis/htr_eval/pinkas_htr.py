#!/usr/bin/env python3
"""HTR-based evaluation of the RQ3 line polygons on the six Pinkas test pages (Hebrew / Western Yiddish cursive).

No public recogniser covers this script, so a small CRNN-CTC recogniser is trained on the line transcriptions of the
24 Pinkas pages that are not test pages (21 train, 3 validation; best epoch by validation CER). Hebrew is written
right to left: line images are mirrored horizontally so that CTC reads in the writing direction, and the texts stay
in logical order.

Page CER in reading order (lines sorted top to bottom) against
  (a) the Pinkas transcription of the ground-truth lines ("true" CER),
  (b) the recogniser output on the ground-truth polygons (segmentation-induced CER).

    HTR_DEVICE=cuda ../../.venv/bin/python pinkas_htr.py      (cwd 99_evaluation/analysis/htr_eval)
      -> pinkas_crnn.pt, pinkas_htr_pages.csv, pinkas_htr_texts.csv
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import fig_qualitative_v2 as Q  # noqa: E402
import overlap_lines_diva as O  # noqa: E402
import htr_common as H  # noqa: E402

SRC = Path.home() / "Thesis/RQ_3_datasets/Pinkas/pinkas_dataset_images_and_xmls"
TEST = {"Page 132_1", "Page 132_2", "Page 133_1", "Page 133_2", "Page 134_1", "Page 134_2"}
DEV = os.environ.get("HTR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HEIGHT, MAXW = 64, 1600
EV = Q.ROOT / "99_evaluation/instance_segmentation/mask_rcnn/cross_collection"
LF = Q.ROOT / "99_evaluation/semantic_segmentation/center_line/pred_lf"
PREDS = {"zs_maskrcnn": EV / "Pinkas_maskrcnn_convnext_tiny_catmus_1152_zeroshot_story_zeroshot/test_pred_xml",
         "zs_centerline": LF / "CATMuS_Z_test/row_1.0/Pinkas",
         "onb_centerline": LF / "ONB_A_w50_test/row_1.0/Pinkas"}


def page_lines(stem):
    r = ET.parse(SRC / f"{stem}.xml").getroot(); ns = r.tag.split("}")[0] + "}"
    out = []
    for tl in r.iter(ns + "TextLine"):
        c = tl.find(ns + "Coords")
        t = [u.text or "" for te in tl.findall(ns + "TextEquiv") for u in te.findall(ns + "Unicode")]
        if c is None or not t:
            continue
        p = np.array([[int(float(v)) for v in xy.split(",")] for xy in c.get("points").split()], np.int32)
        if len(p) >= 3 and t[0].strip():
            out.append((p, t[0].strip()))
    return out


def line_array(img, poly):
    a = np.asarray(H.line_image(img, poly).convert("L"))[:, ::-1]          # mirror: right-to-left script
    h, w = a.shape
    w2 = max(8, min(MAXW, round(w * HEIGHT / max(h, 1))))
    return cv2.resize(a, (w2, HEIGHT), interpolation=cv2.INTER_AREA)


class CRNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        def blk(i, o, pool):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True)] + ([nn.MaxPool2d(pool)] if pool else [])
        self.cnn = nn.Sequential(*blk(1, 64, 2), *blk(64, 128, 2), *blk(128, 256, None), *blk(256, 256, (2, 1)),
                                 *blk(256, 384, None), *blk(384, 384, (2, 1)), *blk(384, 384, (2, 1)))
        self.rnn = nn.LSTM(384 * 2, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(512, n)

    def forward(self, x):                      # x: B,1,64,W
        f = self.cnn(x)                         # B,384,2,W/4
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)
        return self.fc(self.rnn(f)[0])          # B,T,n


def augment(a):
    h, w = a.shape
    M = cv2.getAffineTransform(np.float32([[0, 0], [w, 0], [0, h]]),
                               np.float32([[random.uniform(-2, 2), random.uniform(-2, 2)], [w + random.uniform(-4, 4), random.uniform(-2, 2)],
                                           [random.uniform(-0.08, 0.08) * h, h + random.uniform(-2, 2)]]))
    a = cv2.warpAffine(a, M, (w, h), borderValue=255)
    a = np.clip(a.astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-25, 25), 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        a = cv2.GaussianBlur(a, (3, 3), 0)
    return a


def batches(items, bs, aug):
    idx = list(range(len(items)))
    if aug:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [items[j] for j in idx[i:i + bs]]
        arrs = [augment(a) if aug else a for a, _ in chunk]
        W = max(a.shape[1] for a in arrs)
        x = np.full((len(arrs), 1, HEIGHT, W), 255, np.uint8)
        for k, a in enumerate(arrs):
            x[k, 0, :, : a.shape[1]] = a
        yield torch.from_numpy(x).float().div(255.0).sub(0.5).div(0.5), [t for _, t in chunk]


def decode(logits, itos):
    out = []
    for seq in logits.argmax(-1).cpu().numpy():
        s, prev = [], 0
        for c in seq:
            if c != prev and c != 0:
                s.append(itos[c])
            prev = c
        out.append("".join(s))
    return out


def recognise(model, itos, arrs, bs=16):
    model.eval(); res = []
    with torch.no_grad():
        for i in range(0, len(arrs), bs):
            x, _ = next(batches([(a, "") for a in arrs[i:i + bs]], bs, False))
            res += decode(model(x.to(DEV)), itos)
    return res


def train():
    random.seed(0); torch.manual_seed(0)
    stems = sorted(p.stem for p in SRC.glob("*.xml") if p.stem not in TEST)
    val_stems = set(stems[-3:])
    data = {"train": [], "val": []}
    for st in stems:
        img = cv2.cvtColor(cv2.imread(str(SRC / f"{st}.jpg")), cv2.COLOR_BGR2RGB)
        for p, t in page_lines(st):
            data["val" if st in val_stems else "train"].append((line_array(img, p), t))
    chars = sorted({c for _, t in data["train"] + data["val"] for c in t})
    itos = ["<blank>"] + chars; stoi = {c: i for i, c in enumerate(itos)}
    print(f"train lines {len(data['train'])}, val lines {len(data['val'])}, charset {len(chars)}", flush=True)
    model = CRNN(len(itos)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    epochs = int(os.environ.get("PINKAS_EPOCHS", "120"))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * ((len(data["train"]) + 15) // 16))
    ctc = nn.CTCLoss(zero_infinity=True)
    best = (9.0, None)
    for ep in range(1, epochs + 1):
        model.train()
        for x, ts in batches(data["train"], 16, True):
            logits = model(x.to(DEV)).log_softmax(-1)          # B,T,n
            tgt = torch.tensor([stoi[c] for t in ts for c in t if c in stoi])
            lens = torch.tensor([sum(c in stoi for c in t) for t in ts])
            loss = ctc(logits.permute(1, 0, 2), tgt, torch.full((len(ts),), logits.shape[1], dtype=torch.long), lens)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step(); sched.step()
        if ep % 5 == 0 or ep == epochs:
            hyp = recognise(model, itos, [a for a, _ in data["val"]])
            cer = sum(H.cer(t, h) * len(t) for (_, t), h in zip(data["val"], hyp)) / sum(len(t) for _, t in data["val"])
            print(f"ep {ep} loss {loss.item():.3f} val CER {100 * cer:.2f}", flush=True)
            if cer < best[0]:
                best = (cer, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    torch.save({"model": best[1], "itos": itos, "val_cer": best[0]}, HERE / "pinkas_crnn.pt")
    print(f"best val CER {100 * best[0]:.2f}", flush=True)
    return model, itos


def order(ps):
    return sorted(range(len(ps)), key=lambda i: (round(ps[i][:, 1].mean() / 40), -ps[i][:, 0].mean()))


def evaluate(model, itos):
    rows, texts = [], []
    for st in sorted(TEST):
        img = cv2.cvtColor(cv2.imread(str(SRC / f"{st}.jpg")), cv2.COLOR_BGR2RGB)
        gl = page_lines(st)
        gp = [p for p, _ in gl]
        true_page = " ".join(gl[j][1] for j in order(gp))
        ref = recognise(model, itos, [line_array(img, p) for p in gp])
        ref_page = " ".join(ref[j] for j in order(gp))
        rows.append({"page": st, "model": "gt", "cer_true": 100 * H.cer(true_page, ref_page), "cer_seg": 0.0, "n": len(gp)})
        texts += [{"page": st, "model": "transcription", "text": true_page}, {"page": st, "model": "gt", "text": ref_page}]
        for key, d in PREDS.items():
            xml = d / f"Pinkas__{st}.xml"
            pp = O.polys(xml) if xml.exists() else []
            hyp = recognise(model, itos, [line_array(img, p) for p in pp]) if pp else []
            hyp_page = " ".join(hyp[i] for i in order(pp))
            rows.append({"page": st, "model": key, "cer_true": 100 * H.cer(true_page, hyp_page),
                         "cer_seg": 100 * H.cer(ref_page, hyp_page), "n": len(pp)})
            texts.append({"page": st, "model": key, "text": hyp_page})
            print(st, key, "lines", len(pp), "CER true %.1f seg %.1f" % (rows[-1]["cer_true"], rows[-1]["cer_seg"]), flush=True)
    pd.DataFrame(rows).to_csv(HERE / "pinkas_htr_pages.csv", index=False)
    pd.DataFrame(texts).to_csv(HERE / "pinkas_htr_texts.csv", index=False)
    print(pd.DataFrame(rows).groupby("model")[["cer_true", "cer_seg", "n"]].mean().round(2))


if __name__ == "__main__":
    missing = [k for k, d in PREDS.items() if not d.is_dir()]
    print("prediction dirs missing:", missing, flush=True)
    m, itos = train()
    evaluate(m, itos)
