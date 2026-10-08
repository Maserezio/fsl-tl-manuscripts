#!/usr/bin/env python3
"""Prepare the final five-dataset RQ3 LOO sweep with shared random-10 tests."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
from prepare_rq3_loo import farthest_point_order, resnet18_features, write_page_gt, write_pixel_gt  # noqa: E402
from prepare_rq3_zeroshot_random10 import SOURCES, collect  # noqa: E402

SEED = 42
CAP = 30
SHOTS = (3, 5, 10)
METHODS = ("pca", "random")
SHARED = REPO / "00_data/RQ3/final_shared_test"


def hardlink_or_copy(source: Path, target: Path):
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def write_split(items, root: Path, split: str):
    image_dir = root / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    payload = {"images": [], "annotations": [],
               "categories": [{"id": 1, "name": "TextLine", "supercategory": "text"}]}
    ann_id = 1
    for image_id, (dataset, item) in enumerate(items, start=1):
        _, source, (width, height), polygons, rotated = item
        suffix = source.suffix.lower() if source.suffix.lower() in {".jpg", ".png"} else ".png"
        name = f"{dataset}__{source.stem}{suffix}"
        target = image_dir / name
        if rotated:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(str(target), image)
        else:
            hardlink_or_copy(source, target)
        payload["images"].append({"id": image_id, "file_name": name, "width": width,
                                  "height": height, "dataset": dataset})
        for poly in polygons:
            xs, ys = zip(*poly)
            payload["annotations"].append({
                "id": ann_id, "image_id": image_id, "category_id": 1, "iscrowd": 0,
                "segmentation": [[v for xy in poly for v in xy]],
                "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                "area": float(cv2.contourArea(np.asarray(poly, dtype=np.float32))),
            })
            ann_id += 1
        page_dir = root / "page-gt" / split
        page_dir.mkdir(parents=True, exist_ok=True)
        write_page_gt(polygons, (width, height), name, page_dir / f"{Path(name).stem}.xml")
    coco = root / "coco_instances"
    coco.mkdir(parents=True, exist_ok=True)
    (coco / f"{split}.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def empty_coco(path: Path):
    path.write_text(json.dumps({"images": [], "annotations": [],
                                "categories": [{"id": 1, "name": "TextLine"}]}))


def replace_symlink(target: Path, link: Path):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.symlink_to(target)


def main():
    rng = np.random.default_rng(SEED)
    datasets = {name: collect(*roots) for name, roots in SOURCES.items()}
    for name, items in datasets.items():
        print(f"{name}: {len(items)} clean pages", flush=True)

    capped = {}
    for name, items in datasets.items():
        index = rng.choice(len(items), min(CAP, len(items)), replace=False)
        capped[name] = [items[i] for i in sorted(index.tolist())]

    # Compute each source image feature once; each holdout PCA is then very cheap.
    feature_items = [(name, item) for name in sorted(capped) for item in capped[name]]
    features = resnet18_features([item[1] for _, item in feature_items],
                                 "cuda" if torch.cuda.is_available() else "cpu")
    feature_of = {(name, item[1]): feat for (name, item), feat in zip(feature_items, features)}

    for holdout in datasets:
        shared = SHARED / holdout
        test_rng = np.random.default_rng(SEED)
        chosen = test_rng.choice(len(datasets[holdout]), 10, replace=False)
        test_items = [(holdout, datasets[holdout][i]) for i in sorted(chosen.tolist())]
        test_payload = write_split(test_items, shared, "test")
        write_pixel_gt(test_payload, shared / "images", "test", shared / "pixel-gt")

        pool = [(name, item) for name in sorted(capped) if name != holdout for item in capped[name]]
        pool_features = np.stack([feature_of[(name, item[1])] for name, item in pool])
        scaled = StandardScaler().fit_transform(pool_features)
        pca_order = farthest_point_order(PCA(n_components=2, random_state=SEED).fit_transform(scaled))
        random_order = np.random.default_rng(SEED).permutation(len(pool)).tolist()

        for method, order in (("pca", pca_order), ("random", random_order)):
            for k in SHOTS:
                root = REPO / f"00_data/RQ3/final_loo_{holdout}_k{k}_{method}"
                payload = write_split([pool[i] for i in order[:k]], root, "train")
                empty_coco(root / "coco_instances/val.json")
                replace_symlink(shared / "images/test", root / "images/test")
                replace_symlink(shared / "page-gt/test", root / "page-gt/test")
                replace_symlink(shared / "pixel-gt/test", root / "pixel-gt/test")
                replace_symlink(shared / "coco_instances/test.json", root / "coco_instances/test.json")
                manifest = {
                    "holdout": holdout, "k": k, "seed": SEED, "cap": CAP,
                    "selection": "PCA max-min" if method == "pca" else "uniform random",
                    "pool_size": len(pool),
                    "splits": {
                        "train": [{"dataset": pool[i][0], "page": pool[i][1][1].stem}
                                  for i in order[:k]],
                        "val": [],
                        "test": [{"dataset": holdout, "page": item[1].stem}
                                 for _, item in test_items],
                    },
                }
                (root / "coco_instances/split_manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8")
                print(f"prepared {root.name}: {len(payload['images'])} train / 10 test", flush=True)


if __name__ == "__main__":
    main()
