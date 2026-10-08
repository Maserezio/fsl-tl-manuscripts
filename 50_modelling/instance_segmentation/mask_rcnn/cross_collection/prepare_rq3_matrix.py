#!/usr/bin/env python3
"""RQ3 protocols A (7x7 transfer matrix) and B (leave-one-collection-out).

A: per collection, 3 random training pages, ~10% of the collection as validation (only
   for picking score/mask thresholds) and every other page as test. Collections with an
   official split (Pinkas, NorHand v3) draw train and validation from the official train
   part and keep the official test part. The test pages are fixed once and shared by
   every model of A and B.
B: for each held-out collection, one of the three A training pages from each of the six
   other collections (6 pages); validation is up to VAL_PER_SOURCE pages per source taken
   from those sources' A validation pages. The held-out test set is the A test set.

    python 50_modelling/instance_segmentation/mask_rcnn/cross_collection/prepare_rq3_matrix.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
from prepare_rq3_final_sweep import empty_coco, replace_symlink, write_split  # noqa: E402
from prepare_rq3_loo import write_pixel_gt  # noqa: E402
from prepare_rq3_zeroshot_random10 import collect  # noqa: E402

SEED = 42
K = 3
VAL_FRACTION = 0.1
VAL_PER_SOURCE = 3
OUT_A = REPO / "00_data/RQ3/matrix"
OUT_B = REPO / "00_data/RQ3/loco"
SRC = REPO / "00_data/RQ3_sources"
CAND = REPO / "00_data/RQ3_candidates"
RASM1 = Path("/home/artur/Thesis/RQ_3_datasets/rasm")

# Order fixes the rows/columns of the transfer matrix.
UNSPLIT = {
    "Pinkas": None,
    "ONB": [(SRC / "ONB_Cod_Syr_1/page", SRC / "ONB_Cod_Syr_1/images")],
    "RASAM": [(SRC / "RASAM/page", SRC / "RASAM/images")],
    "RASM": [(RASM1, RASM1), (CAND / "RASM_part2_extracted", CAND / "RASM_part2_extracted")],
    "Phil_gr_130": [(SRC / "Phil_gr_130/page", SRC / "Phil_gr_130/images")],
    "GRPOLY": [(CAND / "GRPOLY_Handwritten/data", CAND / "GRPOLY_Handwritten/data")],
    "NorHand_v3": None,
}
OFFICIAL = {
    "Pinkas": SRC / "Pinkas_official",
    "NorHand_v3": SRC / "NorHand_v3_selected",
}
DATASETS = list(UNSPLIT)


def n_val(total: int) -> int:
    return max(1, round(VAL_FRACTION * total))


def split_dataset(name: str, rng) -> dict[str, list]:
    if name in OFFICIAL:
        root = OFFICIAL[name]
        pool = collect(root / "train/page", root / "train/images")
        test = collect(root / "test/page", root / "test/images")
        order = rng.permutation(len(pool))
        v = n_val(len(pool) + len(test))
        return {"train": [pool[i] for i in order[:K]],
                "val": [pool[i] for i in order[K:K + v]], "test": test,
                "policy": "official test; train/val drawn from official train"}
    items = [it for xml_root, img_root in UNSPLIT[name] for it in collect(xml_root, img_root)]
    order = rng.permutation(len(items))
    v = n_val(len(items))
    return {"train": [items[i] for i in order[:K]],
            "val": [items[i] for i in order[K:K + v]],
            "test": [items[i] for i in order[K + v:]],
            "policy": f"{K} train / {VAL_FRACTION:.0%} val / rest test"}


def page_list(dataset, items):
    return [{"dataset": dataset, "page": it[1].stem} for it in items]


def write_root(root: Path, splits: dict[str, list]) -> dict[str, int]:
    """splits maps split -> [(dataset, item)]; val/test also get DIVA pixel GT."""
    if root.exists():
        shutil.rmtree(root)
    counts = {}
    for split in ("train", "val", "test"):
        entries = splits.get(split, [])
        counts[split] = len(entries)
        if not entries:
            (root / "coco_instances").mkdir(parents=True, exist_ok=True)
            empty_coco(root / f"coco_instances/{split}.json")
            continue
        payload = write_split(entries, root, split)
        if split != "train":
            write_pixel_gt(payload, root / "images", split, root / "pixel-gt")
    return counts


def main() -> None:
    parts = {}
    for name in DATASETS:
        # One generator per dataset: adding or reordering collections leaves the others intact.
        parts[name] = split_dataset(name, np.random.default_rng(SEED))

    for name, s in parts.items():
        root = OUT_A / name
        counts = write_root(root, {k: [(name, it) for it in s[k]] for k in ("train", "val", "test")})
        manifest = {"dataset": name, "holdout": name, "k": K, "seed": SEED,
                    "split_policy": s["policy"], "selection": "uniform random", "counts": counts,
                    "splits": {k: page_list(name, s[k]) for k in ("train", "val", "test")}}
        (root / "coco_instances/split_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[A] {name}: {counts}", flush=True)

    for holdout in DATASETS:
        rng = np.random.default_rng(SEED)
        train, val = [], []
        for src in DATASETS:
            if src == holdout:
                continue
            train.append((src, parts[src]["train"][int(rng.integers(K))]))
            val += [(src, it) for it in parts[src]["val"][:VAL_PER_SOURCE]]
        root = OUT_B / holdout
        counts = write_root(root, {"train": train, "val": val})
        # The held-out test set is the protocol-A test set of that collection, shared on disk.
        shared = OUT_A / holdout
        for sub in ("images/test", "page-gt/test", "pixel-gt/test", "coco_instances/test.json"):
            replace_symlink(shared / sub, root / sub)
        counts["test"] = len(parts[holdout]["test"])
        manifest = {"holdout": holdout, "k": len(train), "seed": SEED,
                    "selection": "1 of the 3 protocol-A train pages per source collection",
                    "pool_size": len(train), "counts": counts,
                    "splits": {"train": [{"dataset": d, "page": it[1].stem} for d, it in train],
                               "val": [{"dataset": d, "page": it[1].stem} for d, it in val],
                               "test": page_list(holdout, parts[holdout]["test"])}}
        (root / "coco_instances/split_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[B] LOCO {holdout}: {counts}", flush=True)


if __name__ == "__main__":
    main()
