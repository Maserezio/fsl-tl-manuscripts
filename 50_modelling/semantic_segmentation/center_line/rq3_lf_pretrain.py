#!/usr/bin/env python3
"""LineField pretrained on CATMuS (generic centre-line finder), then fine-tuned per RQ3 source.

  prepare            CATMuS medieval-segmentation train split -> pages downscaled so that the median
                     line thickness is at most 1.25 x T0 (keeps ~1300 pages in RAM), polygons scaled.
  pretrain STEPS     LineField from the CATMuS encoder init on the prepared pages.
  finetune SRC... [--steps N]
                     Protocol-A fine-tuning from the pretrained checkpoint on the source's k=3 pages;
                     saved as <SRC>_A_cm.pt next to the scratch models (tag "_cm").
  subset N           pretraining on a random subset of N prepared CATMuS pages (nested: the first N of one
                     seed-42 permutation), same number of steps; saved as catmus_pretrain_n<N>.pt.
  loco K HOLDOUT... [--init catmus|none]
                     Protocol B (LOCO-k): train on 00_data/RQ3/locok<K>/<HOLDOUT> (k pages from each of the
                     six other collections); saved as <HOLDOUT>_L<K>{_cm}.pt.

CATMuS is the same corpus that initialises every RQ3 Mask R-CNN; it contains no page of the seven
RQ3 collections (Latin-script medieval manuscripts only).
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import argparse, json, os, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq3_linefield as LF  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CAT = ROOT / "00_data/CATMuS/medieval-segmentation"
PREP = ROOT / "00_data/RQ3/catmus_linefield"
PRE_CKPT = LF.MODELS / ("catmus_pretrain.pt" if not os.environ.get("LF_ENC") else f"catmus_pretrain_{os.environ['LF_ENC']}.pt")


def prepare():
    Image.MAX_IMAGE_PIXELS = None
    coco = json.loads((CAT / "coco_instances/train.json").read_text())
    polys = LF.page_polys(coco)
    (PREP / "images/train").mkdir(parents=True, exist_ok=True)
    (PREP / "coco_instances").mkdir(parents=True, exist_ok=True)
    images, anns, aid = [], [], 1
    for n, im in enumerate(coco["images"]):
        ps = polys.get(im["id"], [])
        t = LF.median_thickness(ps) if ps else None
        if not t:
            continue
        s = min(1.0, 1.25 * LF.T0 / t)
        w, h = max(8, round(im["width"] * s)), max(8, round(im["height"] * s))
        name = f"{n:05d}.jpg"
        out = PREP / "images/train" / name
        if not out.exists():
            src = Image.open(CAT / "yolo_seg_dataset/images/train" / im["file_name"])
            src.draft("RGB", (w, h))                     # DCT-domain downscale: fast decode
            src.convert("RGB").resize((w, h), Image.LANCZOS).save(out, quality=92)
        images.append({"id": n + 1, "file_name": name, "width": w, "height": h, "source": im["file_name"]})
        for p in ps:
            anns.append({"id": aid, "image_id": n + 1, "category_id": 1, "iscrowd": 0,
                         "segmentation": [(p * [w / im["width"], h / im["height"]]).ravel().round(1).tolist()]})
            aid += 1
        if n % 100 == 0:
            print("prepared", n, flush=True)
    (PREP / "coco_instances/train.json").write_text(json.dumps(
        {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "textline"}]}))
    print("prepared", len(images), "pages,", len(anns), "lines", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "pretrain", "finetune", "loco", "subset"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--init", default="catmus", choices=["catmus", "none"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "pretrain":
        LF.train("CATMuS", "pre", int(a.args[0]) if a.args else 12000, root=PREP, out=PRE_CKPT)
    elif a.cmd == "subset":
        import random
        n = int(a.args[0])
        coco = json.loads((PREP / "coco_instances/train.json").read_text())
        order = sorted(im["id"] for im in coco["images"])
        random.Random(42).shuffle(order)
        keep = set(order[:n])
        root = PREP.parent / f"catmus_linefield_n{n}"
        (root / "coco_instances").mkdir(parents=True, exist_ok=True)
        if not (root / "images").exists():
            (root / "images").symlink_to(PREP / "images")
        (root / "coco_instances/train.json").write_text(json.dumps(
            {"images": [im for im in coco["images"] if im["id"] in keep],
             "annotations": [x for x in coco["annotations"] if x["image_id"] in keep],
             "categories": coco["categories"]}))
        LF.train("CATMuS", f"pre_n{n}", int(a.args[1]) if len(a.args) > 1 else 12000, root=root,
                 out=LF.MODELS / f"catmus_pretrain_n{n}.pt")
    elif a.cmd == "finetune":
        sfx = "" if a.seed == 42 else f"_s{a.seed}"          # seed 42 keeps the original file names
        for src in a.args:
            LF.train(src, f"A{sfx}", a.steps, seed=a.seed, init=PRE_CKPT, out=LF.MODELS / f"{src}_A{sfx}_cm.pt")
    else:
        k, tag = int(a.args[0]), "_cm" if a.init == "catmus" else ""
        sfx = "" if a.seed == 42 else f"_s{a.seed}"          # seed 42 keeps the original file names
        for h in a.args[1:]:
            LF.train(h, f"L{k}{sfx}", a.steps, seed=a.seed, root=ROOT / f"00_data/RQ3/locok{k}" / h,
                     init=PRE_CKPT if tag else None, out=LF.MODELS / f"{h}_L{k}{sfx}{tag}.pt")


if __name__ == "__main__":
    main()
