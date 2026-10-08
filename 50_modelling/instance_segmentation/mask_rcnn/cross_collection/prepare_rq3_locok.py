#!/usr/bin/env python3
"""RQ3 protocol B, revised: LOCO with k in {1, 2, 3} training pages per source collection.

For every held-out collection, the model is trained on k pages from each of the six other
collections (6, 12, or 18 pages). The page sets are nested (k=1 within k=2 within k=3):
all pages are protocol-A training pages, taken in one fixed random order per collection.
Validation keeps up to three validation pages per source collection, as for the original
LOCO-6; the held-out collection contributes
nothing to training or validation, and its test set is the shared protocol-A test set.
NorHand v3 uses draw d1 (as the "ab" protocol-A matrix). Seed 42.
Output: 00_data/RQ3/locok<k>/<holdout>.

    python 50_modelling/instance_segmentation/mask_rcnn/cross_collection/prepare_rq3_locok.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "00_data/RQ3"
DATASETS = ["Pinkas", "ONB", "RASAM", "RASM", "Phil_gr_130", "GRPOLY", "NorHand_v3"]
KS = (1, 2, 3)
VAL_PER_SOURCE = 3
SEED = 42


def root_of(ds: str) -> Path:
    return DATA / "norhand_draws/d1" if ds == "NorHand_v3" else DATA / "matrix" / ds


def load(ds: str, split: str) -> dict:
    return json.loads((root_of(ds) / f"coco_instances/{split}.json").read_text())


def main() -> None:
    rng = np.random.default_rng(SEED)
    order = {}  # per source: train page order and val page order (fixed once, nested budgets)
    for ds in DATASETS:
        train = [im["file_name"] for im in load(ds, "train")["images"]]
        val = [im["file_name"] for im in load(ds, "val")["images"]]
        order[ds] = (list(rng.permutation(train)), list(rng.permutation(val)))

    for k in KS:
        for holdout in DATASETS:
            out = DATA / f"locok{k}" / holdout
            if out.exists():
                shutil.rmtree(out)
            for kind in ("coco_instances", "images/train", "images/val", "page-gt/train",
                         "page-gt/val", "pixel-gt/val"):
                (out / kind).mkdir(parents=True)
            split_imgs = {"train": [], "val": []}
            split_anns = {"train": [], "val": []}
            manifest = {"train": [], "val": []}
            next_img = {"train": 1, "val": 1}
            next_ann = {"train": 1, "val": 1}
            for ds in DATASETS:
                if ds == holdout:
                    continue
                tr, va = order[ds]
                chosen = tr[:k] if k <= len(tr) else tr + va[: k - len(tr)]
                remaining_val = [v for v in va if v not in chosen][:VAL_PER_SOURCE]
                for split, names in (("train", chosen), ("val", remaining_val)):
                    for name in names:
                        src_split = "train" if name in tr else "val"
                        payload = load(ds, src_split)
                        im = next(i for i in payload["images"] if i["file_name"] == name)
                        new_id = next_img[split]
                        next_img[split] += 1
                        split_imgs[split].append({**im, "id": new_id})
                        for a in payload["annotations"]:
                            if a["image_id"] == im["id"]:
                                split_anns[split].append({**a, "id": next_ann[split], "image_id": new_id})
                                next_ann[split] += 1
                        stem = Path(name).stem
                        base = root_of(ds)
                        (out / "images" / split / name).symlink_to((base / "images" / src_split / name).resolve())
                        (out / "page-gt" / split / f"{stem}.xml").symlink_to(
                            (base / "page-gt" / src_split / f"{stem}.xml").resolve())
                        if split == "val":
                            pix = next((base / "pixel-gt" / src_split).glob(f"{stem}.*"))
                            (out / "pixel-gt/val" / pix.name).symlink_to(pix.resolve())
                        manifest[split].append({"dataset": ds, "page": stem, "from": src_split})
            cats = load(holdout, "test")["categories"]
            for split in ("train", "val"):
                (out / f"coco_instances/{split}.json").write_text(json.dumps(
                    {"images": split_imgs[split], "annotations": split_anns[split], "categories": cats}))
            hb = root_of(holdout)
            (out / "coco_instances/test.json").symlink_to((hb / "coco_instances/test.json").resolve())
            for kind in ("images", "page-gt", "pixel-gt"):
                (out / kind / "test").symlink_to((hb / kind / "test").resolve())
            test_pages = [{"dataset": holdout, "page": Path(i["file_name"]).stem}
                          for i in load(holdout, "test")["images"]]
            (out / "coco_instances/split_manifest.json").write_text(json.dumps({
                "holdout": holdout, "k": 6 * k, "pages_per_source": k, "seed": SEED,
                "selection": "nested: first k of the 3 protocol-A train pages in a fixed random order",
                "counts": {"train": len(manifest["train"]), "val": len(manifest["val"]),
                           "test": len(test_pages)},
                "splits": {**manifest, "test": test_pages}}, indent=2))
            print(f"LOCO k={k} {holdout}: train/val/test = {len(manifest['train'])}/"
                  f"{len(manifest['val'])}/{len(test_pages)}", flush=True)


if __name__ == "__main__":
    main()
