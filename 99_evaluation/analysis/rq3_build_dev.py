#!/usr/bin/env python3
"""Development split for the RQ3 augmentation search: train + val pages of each collection.

A Protocol-A model trained on collection S never sees the train or val pages of another
collection T, so T's train+val pages are an unbiased development set for S -> T transfer.
Test pages are never touched. Pixel GT for train pages is generated with the same rule as
the existing val/test GT (Otsu minority side inside the union of line polygons).

Output: 00_data/RQ3/dev/<collection>/{coco_instances/dev.json, images/dev, page-gt/dev,
pixel-gt/dev}, with symlinks to the existing files wherever possible.
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "50_modelling/instance_segmentation/mask_rcnn/cross_collection"))
from prepare_rq3_loo import write_pixel_gt  # noqa: E402

MATRIX = ROOT / "00_data/RQ3/matrix"
DEV = ROOT / "00_data/RQ3/dev"
VAL_CAP = {"RASAM": 12}


def link(src: Path, dst: Path):
    if not dst.exists():
        os.symlink(src.resolve(), dst)


def main():
    for coll in sorted(p.name for p in MATRIX.iterdir() if p.is_dir()):
        src, out = MATRIX / coll, DEV / coll
        for d in ("coco_instances", "images/dev", "page-gt/dev", "pixel-gt/dev"):
            (out / d).mkdir(parents=True, exist_ok=True)
        images, annotations, next_img, next_ann = [], [], 1, 1
        for split in ("train", "val"):
            payload = json.loads((src / f"coco_instances/{split}.json").read_text())
            remap = {}
            if split == "val" and coll in VAL_CAP:  # large, easy collection: keep dev evaluation cheap
                keep = {im["id"] for im in sorted(payload["images"], key=lambda x: x["file_name"])[:VAL_CAP[coll]]}
                payload = {**payload, "images": [im for im in payload["images"] if im["id"] in keep],
                           "annotations": [x for x in payload["annotations"] if x["image_id"] in keep]}
            for im in payload["images"]:
                remap[im["id"]] = next_img
                images.append({**im, "id": next_img})
                next_img += 1
                stem = Path(im["file_name"]).stem
                link(src / "images" / split / im["file_name"], out / "images/dev" / im["file_name"])
                link(src / "page-gt" / split / f"{stem}.xml", out / "page-gt/dev" / f"{stem}.xml")
                if split == "val":
                    link(src / "pixel-gt/val" / f"{stem}.png", out / "pixel-gt/dev" / f"{stem}.png")
            for a in payload["annotations"]:
                annotations.append({**a, "id": next_ann, "image_id": remap[a["image_id"]]})
                next_ann += 1
            if split == "train":
                write_pixel_gt(payload, src / "images", "train", out / "pixel-gt-train-tmp")
        for png in (out / "pixel-gt-train-tmp/train").glob("*.png"):
            png.rename(out / "pixel-gt/dev" / png.name)
        (out / "pixel-gt-train-tmp/train").rmdir()
        (out / "pixel-gt-train-tmp").rmdir()
        (out / "coco_instances/dev.json").write_text(json.dumps(
            {"images": images, "annotations": annotations, "categories": payload["categories"]}))
        n = {d: len(list((out / d).iterdir())) for d in ("images/dev", "page-gt/dev", "pixel-gt/dev")}
        print(coll, len(images), "pages", len(annotations), "lines", n)


if __name__ == "__main__":
    main()
