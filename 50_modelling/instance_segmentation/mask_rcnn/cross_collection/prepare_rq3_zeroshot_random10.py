#!/usr/bin/env python3
"""Prepare a fixed-seed, ten-page zero-shot sanity check for the RQ3 collections."""
from __future__ import annotations

import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
from prepare_rq3_loo import write_page_gt, write_pixel_gt  # noqa: E402

SEED = 42
OUT = REPO / "00_data/RQ3/zeroshot_random10"
PINKAS = Path("/home/artur/Thesis/RQ_3_datasets/Pinkas/pinkas_dataset_images_and_xmls")

SOURCES = {
    "Pinkas": (PINKAS, PINKAS),
    "ONB": (REPO / "00_data/RQ3_sources/ONB_Cod_Syr_1/page",
            REPO / "00_data/RQ3_sources/ONB_Cod_Syr_1/images"),
    "RASAM": (REPO / "00_data/RQ3_sources/RASAM/page",
              REPO / "00_data/RQ3_sources/RASAM/images"),
    "eNDP": (REPO / "00_data/RQ3_candidates/eNDP/data/HTR_Ground-Truth/page_xml",
             REPO / "00_data/RQ3_candidates/eNDP/data/HTR_Ground-Truth/images"),
    "EarlyModernGerman": (REPO / "00_data/RQ3_candidates/Early_Modern_German",
                          REPO / "00_data/RQ3_candidates/Early_Modern_German"),
}


def parse_page(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    ns_uri = root.tag.split("}")[0].strip("{")
    ns = {"p": ns_uri}
    page = root.find(".//p:Page", ns)
    width, height = int(page.get("imageWidth")), int(page.get("imageHeight"))
    name = page.get("imageFilename")
    polygons = []
    invalid = 0
    for line in root.findall(".//p:TextLine", ns):
        coords = line.find("p:Coords", ns)
        try:
            points = [tuple(map(float, p.split(","))) for p in coords.get("points").split()]
        except (AttributeError, TypeError, ValueError):
            invalid += 1
            continue
        if len(points) < 3:
            invalid += 1
        else:
            polygons.append(points)
    return name, (width, height), polygons, invalid


def image_index(root: Path):
    suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    index = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            index.setdefault(path.stem.casefold(), []).append(path)
    return index


def nearest_image(xml_path: Path, filename: str | None, index):
    stem = Path(filename).stem if filename else xml_path.stem
    candidates = index.get(stem.casefold(), [])
    if not candidates:
        return None
    # This resolves duplicate German folio names to the image in the same manuscript.
    return min(candidates, key=lambda p: len(set(p.parts) ^ set(xml_path.parts)))


def collect(xml_root: Path, image_root: Path):
    index = image_index(image_root)
    items = []
    for xml_path in sorted(xml_root.rglob("*.xml")):
        try:
            filename, xml_size, polygons, invalid = parse_page(xml_path)
        except (ET.ParseError, AttributeError, TypeError, ValueError):
            continue
        image_path = nearest_image(xml_path, filename, index)
        if image_path is None or invalid or not polygons:
            continue
        try:
            with Image.open(image_path) as opened:
                iw, ih = opened.size
        except OSError:
            continue
        xscale, yscale = iw / xml_size[0], ih / xml_size[1]
        # RASAM IIIF downloads are sometimes uniformly rescaled. German has 18 pages
        # stored with width/height transposed; rotate those before scaling polygons.
        rotate = (iw, ih) == (xml_size[1], xml_size[0]) and xml_size[0] != xml_size[1]
        if rotate:
            iw, ih = xml_size
            xscale, yscale = 1.0, 1.0
        scaled = [[(x * xscale, y * yscale) for x, y in poly] for poly in polygons]
        items.append((xml_path, image_path, (iw, ih), scaled, rotate))
    return items


def empty_coco(path: Path):
    path.write_text(json.dumps({"images": [], "annotations": [],
                                "categories": [{"id": 1, "name": "TextLine"}]}))


def main():
    rng = np.random.default_rng(SEED)
    coco_dir = OUT / "coco_instances"
    image_dir = OUT / "images/test"
    gt_dir = OUT / "page-gt/test"
    for path in (coco_dir, image_dir, gt_dir):
        path.mkdir(parents=True, exist_ok=True)
    empty_coco(coco_dir / "train.json")
    empty_coco(coco_dir / "val.json")

    payload = {"images": [], "annotations": [],
               "categories": [{"id": 1, "name": "TextLine", "supercategory": "text"}]}
    manifest = {"holdout": "RQ3-random10", "k": 0, "seed": SEED,
                "selection": "10 uniformly random clean pages per dataset", "pool_size": 0,
                "splits": {"train": [], "val": [], "test": []}}
    ann_id = 1
    image_id = 1
    for dataset, (xml_root, images_root) in SOURCES.items():
        candidates = collect(xml_root, images_root)
        chosen = rng.choice(len(candidates), size=min(10, len(candidates)), replace=False)
        for index in sorted(chosen.tolist()):
            xml_path, source_image, (width, height), polygons, rotated = candidates[index]
            suffix = source_image.suffix.lower() if source_image.suffix.lower() in {".jpg", ".png"} else ".png"
            name = f"{dataset}__{source_image.stem}{suffix}"
            target = image_dir / name
            if rotated:
                image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(str(target), image)
            elif suffix == ".png" and source_image.suffix.lower() != ".png":
                image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
                cv2.imwrite(str(target), image)
            else:
                shutil.copy2(source_image, target)
            write_page_gt(polygons, (width, height), name, gt_dir / f"{Path(name).stem}.xml")
            payload["images"].append({"id": image_id, "file_name": name,
                                      "width": width, "height": height, "dataset": dataset})
            for poly in polygons:
                xs, ys = zip(*poly)
                payload["annotations"].append({
                    "id": ann_id, "image_id": image_id, "category_id": 1, "iscrowd": 0,
                    "segmentation": [[v for point in poly for v in point]],
                    "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                    "area": float(cv2.contourArea(np.asarray(poly, dtype=np.float32))),
                })
                ann_id += 1
            manifest["splits"]["test"].append({"dataset": dataset, "page": source_image.stem})
            image_id += 1
        print(f"{dataset}: chose {len(chosen)} of {len(candidates)} clean pages")

    (coco_dir / "test.json").write_text(json.dumps(payload), encoding="utf-8")
    write_pixel_gt(payload, OUT / "images", "test", OUT / "pixel-gt")
    (coco_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"prepared {len(payload['images'])} pages and {len(payload['annotations'])} lines in {OUT}")


if __name__ == "__main__":
    main()
