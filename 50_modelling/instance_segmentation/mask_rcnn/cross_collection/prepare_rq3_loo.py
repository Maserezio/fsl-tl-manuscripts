#!/usr/bin/env python3
"""Leave-one-dataset-out pool for the RQ3 pilot: train on the other collections.

Held-out dataset (default CHAMDoc) supplies only the test pages. The k training
pages and k validation pages come from the remaining collections, drawn from a
shared pool with the PCA max-min rule of RQ2 (ResNet-18 features -> PCA(2) ->
greedy farthest-point). Each source dataset contributes at most --cap pages to the
pool, so a large collection cannot crowd out the small ones.

All ground truth is read from PAGE XML TextLine/Coords and written as COCO
polygons; DIVA-style pixel ground truth for the evaluator is written for the
held-out split only (the source splits are scored with the same evaluator, so
they get one too).

    python 50_modelling/instance_segmentation/mask_rcnn/cross_collection/prepare_rq3_loo.py --holdout CHAMDoc -k 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import sys
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "50_modelling/instance_segmentation/rtdetr_bbox_unet"))
from dataset import polygon_to_baseline  # noqa: E402

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
RQ3 = Path("/home/artur/Thesis/RQ_3_datasets")
SEED = 42

# dataset -> (xml glob, image directory resolver)
SOURCES = {
    "CHAMDoc": (RQ3 / "CHAMDoc/xml_gt/*.xml", lambda x: RQ3 / "CHAMDoc/Full-clean-Inscription" / f"{x.stem}.png"),
    "Pinkas": (RQ3 / "Pinkas/pinkas_dataset_images_and_xmls/*.xml",
               lambda x: x.with_suffix(".jpg")),
    "LeafOCR-Line": (RQ3 / "LeafOCR-Line/PAGE-gt/train/*.xml",
                     lambda x: RQ3 / "LeafOCR-Line/Image/image-train" / f"{x.stem}.jpg"),
    "VML-AHTE": (RQ3 / "ahte_dataset/ahte_train_ground_xml/*.xml",
                 lambda x: RQ3 / "ahte_dataset/ahte_train_binary_images" / f"{x.stem}.png"),
    # Bozen (READ 2016): 350 colour pages of Bolzano council minutes with Transkribus
    # line polygons; the largest homogeneous Latin-script source on disk.
    "Bozen": (RQ3 / "Bozen/Training/page/*.xml",
              lambda x: RQ3 / "Bozen/Training/Images" / f"{x.stem}.JPG"),
}


def pages(dataset):
    pattern, image_of = SOURCES[dataset]
    out = []
    for xml_path in sorted(Path(pattern.parent).glob(pattern.name)):
        image = image_of(xml_path)
        if image.exists():
            out.append((xml_path, image))
    return out


def page_lines(xml_path):
    root = ET.parse(xml_path).getroot()
    ns = {"p": root.tag.split("}")[0].strip("{")}
    page = root.find(".//p:Page", ns)
    size = (int(page.get("imageWidth")), int(page.get("imageHeight")))
    polygons = []
    for line in root.findall(".//p:TextLine", ns):
        coords = line.find("p:Coords", ns)
        if coords is None or not coords.get("points"):
            continue
        pts = [tuple(float(v) for v in q.split(",")) for q in coords.get("points").split()]
        if len(pts) >= 3:
            polygons.append(pts)
    return size, polygons


def resnet18_features(paths, device):
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    prep = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    out = []
    with torch.inference_mode():
        for p in paths:
            image = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            out.append(net(prep(image).unsqueeze(0).to(device))[0].cpu().numpy())
    return np.stack(out)


def farthest_point_order(points):
    remaining = list(range(len(points)))
    start = int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    order, remaining = [start], [i for i in remaining if i != start]
    while remaining:
        d = np.min(np.linalg.norm(points[remaining][:, None] - points[order][None], axis=2), axis=1)
        pick = remaining[int(np.argmax(d))]
        order.append(pick)
        remaining.remove(pick)
    return order


def ink_is_bright(image, polygons):
    """True when the annotated lines are brighter than their surroundings."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.float32).round().astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inside = mask > 0
    if not inside.any():
        return False
    # Ink is the minority class inside a line; report which side of the split it is on.
    return int(((gray > threshold) & inside).sum()) < int(((gray <= threshold) & inside).sum())


def write_coco(items, coco_dir, image_dir, split, normalise_polarity=False):
    (image_dir / split).mkdir(parents=True, exist_ok=True)
    payload = {"images": [], "annotations": [],
               "categories": [{"id": 1, "name": "TextLine", "supercategory": "text"}]}
    ann_id = 1
    for image_id, (xml_path, image_path, dataset) in enumerate(items, start=1):
        (w, h), polys = page_lines(xml_path)
        name = f"{dataset.replace('/', '_')}__{image_path.stem}{image_path.suffix}"
        target = image_dir / split / name
        if not target.exists():
            if normalise_polarity:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if ink_is_bright(image, polys):
                    image = 255 - image
                cv2.imwrite(str(target), image)
            else:
                shutil.copy2(image_path, target)
        payload["images"].append({"id": image_id, "file_name": name, "width": w, "height": h,
                                  "dataset": dataset})
        # The DIVA evaluator needs the ground truth next to the prediction, under the
        # same stem as the copied image. Its PAGE parser reads no polygons from files
        # that carry PrintSpace/ReadingOrder before the regions (Transkribus exports
        # such as Bozen), so the lines are rewritten into the same minimal shape the
        # predictions use: one region, TextLine Coords plus a derived Baseline.
        gt_dir = coco_dir.parent / "page-gt" / split
        gt_dir.mkdir(parents=True, exist_ok=True)
        write_page_gt(polys, (w, h), name, gt_dir / f"{Path(name).stem}.xml")
        for poly in polys:
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            payload["annotations"].append({
                "id": ann_id, "image_id": image_id, "category_id": 1, "iscrowd": 0,
                "segmentation": [[c for xy in poly for c in xy]],
                "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                "area": float(cv2.contourArea(np.array(poly, dtype=np.float32))),
            })
            ann_id += 1
    (coco_dir / f"{split}.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def write_page_gt(polygons, size, image_name, out_path):
    """Minimal PAGE file: one region covering the page, one TextLine per polygon."""
    ET.register_namespace("", PAGE_NS)
    ET.register_namespace("xsi", XSI_NS)
    root = ET.Element(f"{{{PAGE_NS}}}PcGts",
                      {f"{{{XSI_NS}}}schemaLocation": f"{PAGE_NS} {PAGE_NS}/pagecontent.xsd"})
    metadata = ET.SubElement(root, f"{{{PAGE_NS}}}Metadata")
    ET.SubElement(metadata, f"{{{PAGE_NS}}}Creator").text = "RQ3 ground-truth normalisation"
    width, height = size
    page = ET.SubElement(root, f"{{{PAGE_NS}}}Page", {
        "imageFilename": image_name, "imageWidth": str(width), "imageHeight": str(height)})
    region = ET.SubElement(page, f"{{{PAGE_NS}}}TextRegion", {"id": "region_textline", "custom": "0"})
    ET.SubElement(region, f"{{{PAGE_NS}}}Coords",
                  {"points": f"0,0 {width},0 {width},{height} 0,{height}"})
    for index, poly in enumerate(polygons):
        pts = [(int(round(x)), int(round(y))) for x, y in poly]
        line = ET.SubElement(region, f"{{{PAGE_NS}}}TextLine", {"id": f"line_{index}"})
        ET.SubElement(line, f"{{{PAGE_NS}}}Coords",
                      {"points": " ".join(f"{x},{y}" for x, y in pts)})
        baseline = polygon_to_baseline(pts)
        if len(baseline) >= 2:
            ET.SubElement(line, f"{{{PAGE_NS}}}Baseline",
                          {"points": " ".join(f"{x},{y}" for x, y in baseline)})
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def write_pixel_gt(payload, image_dir, split, out_root):
    (out_root / split).mkdir(parents=True, exist_ok=True)
    by_image = {}
    for a in payload["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a["segmentation"][0])
    for info in payload["images"]:
        image = cv2.imread(str(image_dir / split / info["file_name"]), cv2.IMREAD_GRAYSCALE)
        lines = np.zeros(image.shape, dtype=np.uint8)
        for flat in by_image.get(info["id"], []):
            pts = np.array(flat, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
            cv2.fillPoly(lines, [pts], 1)
        # Ink is dark on the RGB collections and bright on the inverted binarisations.
        # Inside an annotated line it is always the minority class of the Otsu split,
        # which holds for tight polygons and for Transkribus-style bands alike.
        threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inside = lines > 0
        dark = (image <= threshold) & inside
        bright = (image > threshold) & inside
        ink = dark if dark.sum() <= bright.sum() else bright
        gt = np.ones(image.shape, dtype=np.uint8)
        gt[ink] = 8
        out = np.zeros((*image.shape, 3), dtype=np.uint8)
        out[..., 0] = gt
        # PNG only: the evaluator reads the class id out of the blue channel, and JPEG
        # compression turns {1, 8} into a smear of neighbouring values.
        cv2.imwrite(str(out_root / split / f"{Path(info['file_name']).stem}.png"), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="CHAMDoc", choices=list(SOURCES))
    ap.add_argument("-k", type=int, default=3, help="training pages drawn from the source pool")
    ap.add_argument("--val-pages", type=int, default=5,
                    help="validation pages, taken from the END of the selection order so the same "
                         "pages serve every k and the k-curve stays comparable")
    ap.add_argument("--cap", type=int, default=30, help="max pages per source dataset in the pool")
    ap.add_argument("--exclude", nargs="*", default=[], choices=list(SOURCES),
                    help="datasets kept out of the experiment entirely (neither source nor target)")
    ap.add_argument("--normalise-polarity", action="store_true", dest="normalise_polarity",
                    help="store every page as dark ink on a light background, inverting the "
                         "collections distributed as inverted binarisations (CHAMDoc)")
    ap.add_argument("--root-name", default=None,
                    help="output directory name under 00_data/RQ3 (default: derived from the flags)")
    ap.add_argument("--test-cap", type=int, default=30,
                    help="max held-out pages to score (the Java evaluator costs ~20 s per page)")
    args = ap.parse_args()

    suffix = "".join(f"_no{d.replace('/', '_').replace('-', '')}" for d in sorted(args.exclude))
    name = args.root_name or f"loo_{args.holdout.replace('/', '_')}_k{args.k}{suffix}"
    root = REPO / "00_data/RQ3" / name
    coco_dir, image_dir, pixel_dir = root / "coco_instances", root / "images", root / "pixel-gt"
    coco_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    pool = []
    for dataset in SOURCES:
        if dataset == args.holdout or dataset in args.exclude:
            continue
        items = pages(dataset)
        if len(items) > args.cap:
            idx = rng.choice(len(items), args.cap, replace=False)
            items = [items[i] for i in sorted(idx)]
        pool += [(x, i, dataset) for x, i in items]
        print(f"pool += {dataset}: {len(items)} pages")
    features = StandardScaler().fit_transform(resnet18_features([i for _, i, _ in pool], device))
    order = farthest_point_order(PCA(n_components=2, random_state=SEED).fit_transform(features))
    if args.k + args.val_pages > len(pool):
        raise SystemExit(f"pool has {len(pool)} pages, need k + val_pages = {args.k + args.val_pages}")
    # Validation is fixed across k and stratified over the source datasets (round robin
    # through each dataset's own selection order), so the k-curve stays comparable and
    # the thresholds are not tuned on a single collection.
    by_dataset = {}
    for rank, i in enumerate(order):
        by_dataset.setdefault(pool[i][2], []).append(i)
    val_index, cursor = [], 0
    while len(val_index) < args.val_pages:
        for dataset in sorted(by_dataset):
            picks = by_dataset[dataset]
            if cursor < len(picks) and len(val_index) < args.val_pages:
                val_index.append(picks[-(cursor + 1)])
        cursor += 1
    train_index = [i for i in order if i not in set(val_index)][:args.k]
    train_items = [pool[i] for i in train_index]
    val_items = [pool[i] for i in val_index]
    held = pages(args.holdout)
    if len(held) > args.test_cap:
        idx = np.random.default_rng(SEED).choice(len(held), args.test_cap, replace=False)
        held = [held[i] for i in sorted(idx)]
    test_items = [(x, i, args.holdout) for x, i in held]

    manifest = {"holdout": args.holdout, "excluded": sorted(args.exclude),
                "normalise_polarity": bool(args.normalise_polarity),
                "k": args.k, "val_pages": args.val_pages, "cap": args.cap, "seed": SEED,
                "selection": "pca_max_distance (greedy farthest-point in PCA-2D of ResNet-18 features)",
                "pool_size": len(pool), "splits": {}}
    for split, items in (("train", train_items), ("val", val_items), ("test", test_items)):
        payload = write_coco(items, coco_dir, image_dir, split, args.normalise_polarity)
        write_pixel_gt(payload, image_dir, split, pixel_dir)
        manifest["splits"][split] = [{"dataset": d, "page": i.stem} for _, i, d in items]
        print(f"{split}: {len(payload['images'])} pages, {len(payload['annotations'])} lines "
              f"({', '.join(sorted({d for _, _, d in items}))})")
    (coco_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["splits"]["train"] + manifest["splits"]["val"], indent=2))


if __name__ == "__main__":
    main()
