"""Build COCO detection data from the TASK-2 PAGE ground truth.

The existing coco_dataset_<subset>/ was converted from the full PAGE annotation, which
also marks interlinear glosses as TextLines. The metrics, however, come from TASK-2,
which annotates only the main text block. On CB55 the two coincide inside the text
region (283 == 283 on the test split) so the mismatch never showed; on CS18 the ratio
is 6:1 (166 COCO objects per page against 27 TASK-2 lines) and the detector spends its
capacity on glosses it is then penalised for finding.

Training on TASK-2 makes the detector's target identical to what is measured.
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET

from PIL import Image

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
SPLITS = {"train": "training", "val": "validation", "test": "public-test"}


def page_lines(xml_path):
    root = ET.parse(xml_path).getroot()
    out = []
    for line in root.findall(f".//{{{NS}}}TextLine"):
        coords = line.find(f"{{{NS}}}Coords")
        if coords is None:
            continue
        pts = [tuple(map(int, xy.split(","))) for xy in coords.attrib["points"].split()]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out.append((pts, min(xs), min(ys), max(xs), max(ys)))
    return out


def build(subset, data_root, out_root):
    gt_root = os.path.join(data_root, subset, f"PAGE-gt-{subset}-TASK-2", "TASK-2")
    img_root = os.path.join(data_root, subset, f"img-{subset}", "img")
    os.makedirs(out_root, exist_ok=True)

    for coco_split, diva_split in SPLITS.items():
        gt_dir = os.path.join(gt_root, diva_split)
        img_dir = os.path.join(img_root, diva_split)
        if not os.path.isdir(gt_dir) or not os.path.isdir(img_dir):
            print(f"[skip] {subset}/{diva_split}: missing directory")
            continue

        images, annotations = [], []
        # "<stem>_TEST.xml" duplicates the plain file; keeping both would double-count.
        stems = sorted({os.path.splitext(f)[0].replace("_TEST", "")
                        for f in os.listdir(gt_dir) if f.endswith(".xml")})
        for image_id, stem in enumerate(stems, 1):
            xml_path = os.path.join(gt_dir, f"{stem}.xml")
            if not os.path.exists(xml_path):
                continue
            img_path = os.path.join(img_dir, f"{stem}.jpg")
            if not os.path.exists(img_path):
                print(f"[warn] no image for {stem}")
                continue
            with Image.open(img_path) as im:
                width, height = im.size
            images.append(dict(id=image_id, file_name=f"{stem}.jpg",
                               width=width, height=height))
            for pts, x1, y1, x2, y2 in page_lines(xml_path):
                annotations.append(dict(
                    id=len(annotations) + 1, image_id=image_id, category_id=1,
                    bbox=[x1, y1, x2 - x1, y2 - y1], area=(x2 - x1) * (y2 - y1),
                    iscrowd=0,
                    segmentation=[[c for p in pts for c in p]]))

        path = os.path.join(out_root, f"{coco_split}.json")
        json.dump(dict(images=images, annotations=annotations,
                       categories=[dict(id=1, name="TextLine", supercategory="text")]),
                  open(path, "w"))
        per_page = len(annotations) / max(len(images), 1)
        print(f"  {subset}/{coco_split}: {len(images)} страниц, {len(annotations)} строк "
              f"({per_page:.0f} на страницу) -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["CB55", "CS18", "CS863"])
    ap.add_argument("--data-root", default="00_data/DIVA-HisDB")
    args = ap.parse_args()
    for s in args.subsets:
        build(s, args.data_root, os.path.join(args.data_root, f"coco_task2_{s}"))
