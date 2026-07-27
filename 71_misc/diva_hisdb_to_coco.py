import os
import json
import xml.etree.ElementTree as ET
from PIL import Image

NS = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}

CATEGORIES = [
    {"id": 1, "name": "TextLine", "supercategory": "text"},
]


def parse_xml_polygons(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons = []
    for tl in root.findall(".//ns:TextLine", NS):
        coords = tl.find("ns:Coords", NS)
        if coords is None:
            continue
        pts = []
        for xy in coords.attrib["points"].split():
            x, y = map(int, xy.split(","))
            pts.append((x, y))
        polygons.append(pts)
    return polygons


def polygon_to_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    xmin, ymin = min(xs), min(ys)
    xmax, ymax = max(xs), max(ys)
    w = xmax - xmin
    h = ymax - ymin
    return [xmin, ymin, w, h]


def polygon_to_segmentation(poly):
    return [coord for pt in poly for coord in pt]


def convert_split(img_dir, xml_dir, split_name):
    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    img_files = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".tif"))
    )

    for img_name in img_files:
        base = os.path.splitext(img_name)[0]
        xml_path = os.path.join(xml_dir, base + ".xml")
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(xml_path):
            print(f"[SKIP] Missing XML for {img_name}")
            continue

        with Image.open(img_path) as img:
            w, h = img.size

        images.append({
            "id": img_id,
            "file_name": img_name,
            "width": w,
            "height": h,
        })

        polygons = parse_xml_polygons(xml_path)
        for poly in polygons:
            bbox = polygon_to_bbox(poly)
            seg = polygon_to_segmentation(poly)
            area = bbox[2] * bbox[3]

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,
                "segmentation": [seg],
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

        print(f"[OK] {img_name}: {len(polygons)} annotations")
        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    print(f"Split '{split_name}': {len(images)} images, {len(annotations)} annotations")
    return coco


def main():
    base_dir = "../00_data/DIVA-HisDB"
    subsets = ["CB55", "CS18", "CS863"]

    splits = {
        "train": "training",
        "val": "validation",
        "test": "public-test",
    }

    for subset in subsets:
        output_dir = os.path.join(base_dir, f"coco_dataset_{subset}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n=== Converting subset {subset} ===")

        for coco_split, diva_split in splits.items():
            img_dir = os.path.join(base_dir, subset, f"img-{subset}", "img", diva_split)
            xml_dir = os.path.join(base_dir, subset, f"PAGE-gt-{subset}", "PAGE-gt", diva_split)

            if not os.path.isdir(img_dir):
                print(f"[WARN] Image dir not found: {img_dir}")
                continue
            if not os.path.isdir(xml_dir):
                print(f"[WARN] XML dir not found: {xml_dir}")
                continue

            coco = convert_split(img_dir, xml_dir, coco_split)

            out_path = os.path.join(output_dir, f"{coco_split}.json")
            with open(out_path, "w") as f:
                json.dump(coco, f, indent=2)
            print(f"Saved {out_path}\n")

    print("All subset conversions completed.")


if __name__ == "__main__":
    main()
