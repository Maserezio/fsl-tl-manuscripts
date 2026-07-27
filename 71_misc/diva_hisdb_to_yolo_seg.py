"""DIVA-HisDB TASK-2 PAGE-XML -> YOLO segmentation dataset (full pages, polygon labels).

Mirrors the existing yolo_seg_dataset_CS18/CS863 convention:
  yolo_seg_dataset_<MS>/{dataset.yaml, images/{train,val,test}/*.jpg, labels/{train,val,test}/*.txt}
Label line: "0 x1 y1 x2 y2 ..." (class textline, polygon normalized to page size).

  python diva_hisdb_to_yolo_seg.py --manuscript CB55
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

_NS = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
SPLITS = {"train": "training", "val": "validation", "test": "public-test"}


def page_polys(xml_path):
    root = ET.parse(str(xml_path)).getroot()
    page = root.find(".//ns:Page", _NS)
    W, H = int(page.attrib["imageWidth"]), int(page.attrib["imageHeight"])
    polys = []
    for tl in root.findall(".//ns:TextLine", _NS):
        co = tl.find("ns:Coords", _NS)
        if co is None:
            continue
        pts = [tuple(map(int, xy.split(","))) for xy in co.attrib["points"].split()]
        if len(pts) >= 3:
            polys.append(pts)
    return polys, W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default="CB55")
    args = ap.parse_args()
    MS = args.manuscript
    repo = Path(__file__).resolve().parents[1]
    base = repo / "00_data" / "DIVA-HisDB" / MS
    out = repo / "00_data" / "DIVA-HisDB" / f"yolo_seg_dataset_{MS}"

    n_pages, n_lines = 0, 0
    for split, subdir in SPLITS.items():
        img_dir = base / f"img-{MS}" / "img" / subdir
        xml_dir = base / f"PAGE-gt-{MS}-TASK-2" / "TASK-2" / subdir
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in sorted(img_dir.glob("*.jpg")):
            xml = xml_dir / f"{img.stem}.xml"
            if not xml.exists():
                print(f"[warn] no XML for {img.stem}, skipping")
                continue
            polys, W, H = page_polys(xml)
            lines = []
            for pts in polys:
                coords = " ".join(f"{min(max(x / W, 0), 1):.6f} {min(max(y / H, 0), 1):.6f}" for x, y in pts)
                lines.append(f"0 {coords}")
            (out / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")
            shutil.copy(img, out / "images" / split / img.name)
            n_pages += 1
            n_lines += len(lines)
        print(f"{split}: {len(list((out / 'images' / split).glob('*.jpg')))} pages")

    (out / "dataset.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: textline\n")
    print(f"DONE: {n_pages} pages, {n_lines} line polygons -> {out}")


if __name__ == "__main__":
    main()
