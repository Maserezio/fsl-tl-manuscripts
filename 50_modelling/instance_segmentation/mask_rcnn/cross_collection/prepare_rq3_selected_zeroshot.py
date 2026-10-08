#!/usr/bin/env python3
"""Prepare fixed train/val/test partitions for the seven selected RQ3 datasets."""
from __future__ import annotations

import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "50_modelling/common"))
from audit_remote_page_zip import RangeFile  # noqa: E402
from prepare_rq3_loo import write_pixel_gt  # noqa: E402
from prepare_rq3_zeroshot_random10 import collect  # noqa: E402
from prepare_rq3_final_sweep import empty_coco, write_split  # noqa: E402

OUT = REPO / "00_data/RQ3/selected_splits"
PINKAS = Path("/home/artur/Thesis/RQ_3_datasets/Pinkas")
RASM1 = Path("/home/artur/Thesis/RQ_3_datasets/rasm")
RASM2 = REPO / "00_data/RQ3_candidates/RASM_part2_extracted"
SEED = 42


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def norhand_roots() -> dict[str, tuple[Path, Path]]:
    root = REPO / "00_data/RQ3_sources/NorHand_v3_selected"
    record = json.loads((REPO / "00_data/RQ3_candidates/NorHand_v3_audit/record.json").read_text())
    entry = next(f for f in record["files"] if f["key"] == "norhand_v3.zip")
    remote = RangeFile(entry["links"]["self"], entry["size"])
    rng = np.random.default_rng(SEED)
    with zipfile.ZipFile(remote) as archive:
        names = archive.namelist()
        train_xml = sorted(n for n in names if n.startswith("train/page/") and n.endswith(".xml"))
        picked = rng.choice(len(train_xml), 30, replace=False)
        selections = {"train": [train_xml[i] for i in sorted(picked.tolist())],
                      "test": sorted(n for n in names if n.startswith("test/page/") and n.endswith(".xml"))}
        for split, members in selections.items():
            images, pages = root / split / "images", root / split / "page"
            images.mkdir(parents=True, exist_ok=True); pages.mkdir(parents=True, exist_ok=True)
            image_by_stem = {Path(n).stem: n for n in names if n.startswith(f"{split}/images/")
                             and n.lower().endswith((".jpg", ".png", ".tif", ".tiff"))}
            for index, xml_name in enumerate(members, 1):
                image_name = image_by_stem[Path(xml_name).stem]
                for member, target in ((xml_name, pages / Path(xml_name).name),
                                       (image_name, images / Path(image_name).name)):
                    if not target.exists():
                        target.write_bytes(archive.read(member))
                if index % 25 == 0:
                    print(f"NorHand {split}: {index}/{len(members)}", flush=True)
    return {s: (root / s / "page", root / s / "images") for s in ("train", "test")}


def phil_root() -> tuple[Path, Path]:
    audit = REPO / "00_data/RQ3_candidates/Phil_gr_130_audit"
    root = REPO / "00_data/RQ3_sources/Phil_gr_130"
    images, pages = root / "images", root / "page"
    images.mkdir(parents=True, exist_ok=True); pages.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((audit / "manifest.json").read_text())
    session = requests.Session()
    for xml, canvas in zip(sorted(audit.glob("sample_*.xml"))[:18], manifest["items"][183:201]):
        name = Path(ET.parse(xml).getroot().find(".//{*}Page").get("imageFilename"))
        link(xml, pages / f"{name.stem}.xml")
        target = images / f"{name.stem}.jpg"
        if not target.exists():
            service = canvas["items"][0]["items"][0]["body"]["service"][0]["id"]
            response = session.get(f"{service}/full/max/0/default.jpg", timeout=180)
            response.raise_for_status(); target.write_bytes(response.content)
    return pages, images


def pinkas_roots() -> dict[str, tuple[Path, Path]]:
    source = PINKAS / "pinkas_dataset_images_and_xmls"
    root = REPO / "00_data/RQ3_sources/Pinkas_official"
    result = {}
    for split, listing in (("train", "train_set.txt"), ("test", "test_set.txt")):
        images, pages = root / split / "images", root / split / "page"
        images.mkdir(parents=True, exist_ok=True); pages.mkdir(parents=True, exist_ok=True)
        for value in (PINKAS / listing).read_text().splitlines():
            xml = source / value.strip()
            link(xml, pages / xml.name); link(xml.with_suffix(".jpg"), images / xml.with_suffix(".jpg").name)
        result[split] = (pages, images)
    return result


def random_split(items: list) -> dict[str, list]:
    order = np.random.default_rng(SEED).permutation(len(items))
    n_train, n_val = int(0.7 * len(items)), int(0.1 * len(items))
    return {"train": [items[i] for i in order[:n_train]],
            "val": [items[i] for i in order[n_train:n_train + n_val]],
            "test": [items[i] for i in order[n_train + n_val:]]}


def main() -> None:
    pinkas, norhand = pinkas_roots(), norhand_roots()
    official = {"Pinkas": {s: collect(*pinkas[s]) for s in ("train", "test")},
                "NorHand_v3": {s: collect(*norhand[s]) for s in ("train", "test")}}
    raw = {
        "ONB": (REPO / "00_data/RQ3_sources/ONB_Cod_Syr_1/page", REPO / "00_data/RQ3_sources/ONB_Cod_Syr_1/images"),
        "RASAM": (REPO / "00_data/RQ3_sources/RASAM/page", REPO / "00_data/RQ3_sources/RASAM/images"),
        "Phil_gr_130": phil_root(),
        "GRPOLY": (REPO / "00_data/RQ3_candidates/GRPOLY_Handwritten/data", REPO / "00_data/RQ3_candidates/GRPOLY_Handwritten/data"),
    }
    partitions = {name: random_split(collect(*roots)) for name, roots in raw.items()}
    partitions["RASM"] = random_split(collect(RASM1, RASM1) + collect(RASM2, RASM2))
    for name, values in official.items():
        partitions[name] = {"train": values["train"], "val": [], "test": values["test"]}

    for dataset, splits in partitions.items():
        root = OUT / dataset
        if root.exists(): shutil.rmtree(root)
        counts = {}
        for split in ("train", "val", "test"):
            if splits[split]:
                payload = write_split([(dataset, item) for item in splits[split]], root, split)
                if split == "test": write_pixel_gt(payload, root / "images", split, root / "pixel-gt")
            else:
                (root / "coco_instances").mkdir(parents=True, exist_ok=True)
                empty_coco(root / f"coco_instances/{split}.json")
            counts[split] = len(splits[split])
        manifest = {"dataset": dataset, "seed": SEED,
                    "split_policy": "official" if dataset in official else "70/10/20 page-level",
                    "counts": counts}
        (root / "coco_instances/split_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(dataset, counts, flush=True)


if __name__ == "__main__": main()
