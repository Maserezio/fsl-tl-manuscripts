#!/usr/bin/env python3
"""CATMuS Medieval pages for pretraining the BBox U-Net (train_crops_loss_ablation_2stage.py --family catmus).

Every COCO page is resized to a long side of LONG_SIDE pixels (the crops are resized to 1024x256 anyway, and the
crop dataset reads the full page for every line) and written next to a PAGE XML file with one TextLine per COCO
line polygon, scaled to the resized page. File stems are "<manuscript>__<page>", unique across manuscripts.

    python 50_modelling/instance_segmentation/rtdetr_bbox_unet/prepare_catmus_crops.py [--limit N]
      -> 00_data/CATMuS/crops_pagexml/{train,val}/<stem>.jpg + <stem>.xml
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from xml.sax.saxutils import quoteattr

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
COCO = REPO / "00_data/CATMuS/medieval-segmentation/coco_instances"
IMAGES = REPO / "00_data/CATMuS/medieval-segmentation/yolo_seg_dataset/images"
OUT = REPO / "00_data/CATMuS/crops_pagexml"
NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def page_xml(name, w, h, polys):
    lines = "\n".join(
        f'      <TextLine id="l{i}"><Coords points="{" ".join(f"{x},{y}" for x, y in p)}"/></TextLine>'
        for i, p in enumerate(polys))
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<PcGts xmlns="{NS}">\n'
            f'  <Page imageFilename={quoteattr(name)} imageWidth="{w}" imageHeight="{h}">\n'
            f'    <TextRegion id="r0"><Coords points="0,0 {w},0 {w},{h} 0,{h}"/>\n{lines}\n'
            f'    </TextRegion>\n  </Page>\n</PcGts>\n')


def one(job):
    split, file_name, segs, long_side = job
    stem = file_name.replace("/", "__").rsplit(".", 1)[0]
    out_img, out_xml = OUT / split / f"{stem}.jpg", OUT / split / f"{stem}.xml"
    if out_img.exists() and out_xml.exists():
        return 0
    img = cv2.imread(str(IMAGES / split / file_name), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(IMAGES / split / file_name)
    h, w = img.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    polys = []
    for seg in segs:
        for ring in seg:
            p = np.rint(np.asarray(ring, np.float64).reshape(-1, 2) * s).astype(int)
            if len(p) >= 3:
                polys.append(p.tolist())
    cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    out_xml.write_text(page_xml(out_img.name, img.shape[1], img.shape[0], polys))
    return len(polys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-side", type=int, default=2400)
    ap.add_argument("--limit", type=int, default=0, help="pages per split (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = []
    for split in ("train", "val"):
        (OUT / split).mkdir(parents=True, exist_ok=True)
        coco = json.loads((COCO / f"{split}.json").read_text())
        segs = {}
        for an in coco["annotations"]:
            segs.setdefault(an["image_id"], []).append(an["segmentation"])
        images = coco["images"][: a.limit] if a.limit else coco["images"]
        jobs += [(split, im["file_name"], segs.get(im["id"], []), a.long_side) for im in images]
    with ProcessPoolExecutor(a.workers) as ex:
        n = sum(ex.map(one, jobs, chunksize=4))
    print(f"{len(jobs)} pages, {n} new line polygons -> {OUT}")


if __name__ == "__main__":
    main()
