"""Convert U-DIADS-TL binary text-line GT to COCO instance annotations.

U-DIADS has no PAGE-XML/polygons — only a binary text-line mask per page. Each
connected component is one text-line instance; its exact mask is stored as RLE.
Outputs coco_dataset_<manuscript_lower>/{train,val,test}.json (same layout as the
DIVA coco datasets, so detectron2's register_coco_instances works directly).

Usage:
  python udiads_to_coco.py --manuscript Latin14396
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CATEGORIES = [{"id": 1, "name": "TextLine", "supercategory": "text"}]
IMG_ALIAS = {"train": ["train", "training"], "val": ["val", "validation"], "test": ["test"]}
GT_ALIAS = {"train": ["training", "train"], "val": ["validation", "val"], "test": ["test"]}
EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _resolve(base: Path, aliases):
    for a in aliases:
        if (base / a).exists():
            return base / a
    return None


def convert_split(img_dir: Path, gt_dir: Path, min_area: int):
    images, annotations = [], []
    img_id, ann_id = 1, 1
    stems = sorted({p.stem for p in img_dir.iterdir() if p.suffix.lower() in EXTS})
    for stem in stems:
        gp = gt_dir / f"{stem}.png"
        ip = next((img_dir / f"{stem}{e}" for e in EXTS if (img_dir / f"{stem}{e}").exists()), None)
        if ip is None or not gp.exists():
            continue
        gt = (cv2.imread(str(gp)).sum(axis=-1) > 0).astype(np.uint8)
        H, W = gt.shape
        images.append({"id": img_id, "file_name": ip.name, "width": W, "height": H})
        n, labels, stats, _ = cv2.connectedComponentsWithStats(gt, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            comp = (labels == i).astype(np.uint8)
            # polygon segmentation (detectron2's COCO LSJ mapper needs mask_format='polygon')
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segs = []
            for c in contours:
                pts = c.reshape(-1, 2)
                if len(pts) >= 3:                       # need >=3 points for a valid polygon
                    segs.append([float(v) for v in pts.flatten()])
            if not segs:
                continue
            x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                          int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [x, y, w, h], "area": float(stats[i, cv2.CC_STAT_AREA]),
                "segmentation": segs, "iscrowd": 0,
            })
            ann_id += 1
        img_id += 1
    return {"images": images, "annotations": annotations, "categories": CATEGORIES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", required=True)
    ap.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "00_data" / "U-DIADS-TL"),
                    dest="data_root")
    ap.add_argument("--min-area", type=int, default=50, dest="min_area")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ms = args.manuscript
    root = Path(args.data_root)
    img_base = root / ms / f"img-{ms}"
    gt_base = root / ms / f"text-line-gt-{ms}"
    out = Path(args.out) if args.out else root / f"coco_dataset_{ms.lower()}"
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        img_dir = _resolve(img_base, IMG_ALIAS[split])
        gt_dir = _resolve(gt_base, GT_ALIAS[split])
        if img_dir is None or gt_dir is None:
            print(f"[skip] {split}: img={img_dir} gt={gt_dir}")
            continue
        coco = convert_split(img_dir, gt_dir, args.min_area)
        (out / f"{split}.json").write_text(json.dumps(coco))
        print(f"{split:5s}: {len(coco['images'])} images, {len(coco['annotations'])} lines "
              f"(img_root={img_dir})")
    print(f"DONE -> {out}")


if __name__ == "__main__":
    main()
