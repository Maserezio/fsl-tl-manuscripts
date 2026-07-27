"""DIVA-HisDB TASK-2 -> YOLO-seg CROP dataset (one padded line crop = one image, one polygon).

Stage-2 training data for the 2-stage pipeline: each GT TextLine's padded bbox is cropped
from the page; the label is that line's polygon in crop coordinates (normalized).

  python diva_hisdb_to_yolo_seg_crops.py --manuscript CB55 --pad 15
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

_NS = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
SPLITS = {"train": "training", "val": "validation"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default="CB55")
    ap.add_argument("--pad", type=int, default=15)
    args = ap.parse_args()
    MS, PAD = args.manuscript, args.pad
    repo = Path(__file__).resolve().parents[1]
    base = repo / "00_data" / "DIVA-HisDB" / MS
    out = repo / "00_data" / "DIVA-HisDB" / f"yolo_seg_crops_{MS}"

    n_crops = 0
    for split, subdir in SPLITS.items():
        img_dir = base / f"img-{MS}" / "img" / subdir
        xml_dir = base / f"PAGE-gt-{MS}-TASK-2" / "TASK-2" / subdir
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for imgp in sorted(img_dir.glob("*.jpg")):
            xml = xml_dir / f"{imgp.stem}.xml"
            if not xml.exists():
                continue
            img = cv2.imread(str(imgp))
            H, W = img.shape[:2]
            root = ET.parse(str(xml)).getroot()
            for j, tl in enumerate(root.findall(".//ns:TextLine", _NS)):
                co = tl.find("ns:Coords", _NS)
                if co is None:
                    continue
                pts = np.array([tuple(map(int, xy.split(","))) for xy in co.attrib["points"].split()])
                if len(pts) < 3:
                    continue
                x0 = max(0, pts[:, 0].min() - PAD); y0 = max(0, pts[:, 1].min() - PAD)
                x1 = min(W, pts[:, 0].max() + PAD); y1 = min(H, pts[:, 1].max() + PAD)
                cw, ch = x1 - x0, y1 - y0
                if cw < 10 or ch < 10:
                    continue
                crop = img[y0:y1, x0:x1]
                stem = f"{imgp.stem}_l{j:03d}"
                cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), crop)
                rel = (pts - [x0, y0]) / [cw, ch]
                rel = np.clip(rel, 0, 1)
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in rel)
                (out / "labels" / split / f"{stem}.txt").write_text(f"0 {coords}\n")
                n_crops += 1
        print(f"{split}: {len(list((out / 'images' / split).glob('*.jpg')))} crops")

    (out / "dataset.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: textline\n")
    print(f"DONE: {n_crops} line crops -> {out}")


if __name__ == "__main__":
    main()
