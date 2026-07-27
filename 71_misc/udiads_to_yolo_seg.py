"""U-DIADS-TL text-line GT -> YOLO segmentation dataset (full pages, polygon labels).

GT PNGs are binary line-band masks; each connected component = one text-line instance.
Output mirrors the DIVA yolo_seg convention:
  yolo_seg_dataset_udiads_<ms>/{dataset.yaml, images/{train,val,test}/, labels/...}
Label line: "0 x1 y1 x2 y2 ..." (normalized outer-contour polygon per component).

  python udiads_to_yolo_seg.py --manuscript Latin14396
  python udiads_to_yolo_seg.py --all
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

SPLITS = {"train": ("train", "training"), "val": ("val", "validation"), "test": ("test", "test")}
MIN_AREA = 100          # drop speck components (pages are ~2016x1344)


def first_existing(base, names):
    for n in names:
        p = base / n
        if p.exists():
            return p
    return None


def convert(ms: str, repo: Path):
    base = repo / "00_data" / "U-DIADS-TL" / ms
    out = repo / "00_data" / "U-DIADS-TL" / f"yolo_seg_dataset_udiads_{ms.lower()}"
    n_pages = n_lines = 0
    for split, cand in SPLITS.items():
        img_dir = first_existing(base / f"img-{ms}", cand)
        gt_dir = first_existing(base / f"text-line-gt-{ms}", cand)
        if img_dir is None or gt_dir is None:
            print(f"[warn] {ms}/{split}: missing dirs, skipping")
            continue
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for imgp in sorted(img_dir.iterdir()):
            if imgp.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            gtp = gt_dir / f"{imgp.stem}.png"
            if not gtp.exists():
                print(f"[warn] no GT for {ms}/{split}/{imgp.stem}")
                continue
            gt = cv2.imread(str(gtp))
            H, W = gt.shape[:2]
            binary = (gt.sum(axis=-1) > 0).astype(np.uint8)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            lines = []
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
                    continue
                comp = (labels == i).astype(np.uint8)
                cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                c = max(cnts, key=cv2.contourArea)
                approx = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2)
                if len(approx) < 3:
                    continue
                pts = np.clip(approx / [W, H], 0, 1)
                lines.append("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts))
            (out / "labels" / split / f"{imgp.stem}.txt").write_text("\n".join(lines) + "\n")
            img_out = out / "images" / split / imgp.name
            if not img_out.exists():
                img_out.symlink_to(imgp.resolve())
            n_pages += 1
            n_lines += len(lines)
        print(f"{ms}/{split}: {len(list((out / 'images' / split).iterdir()))} pages")
    (out / "dataset.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: textline\n")
    print(f"{ms} DONE: {n_pages} pages, {n_lines} line polygons -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    subsets = ["Latin14396", "Latin2", "Syr341"] if args.all or not args.manuscript else [args.manuscript]
    for ms in subsets:
        convert(ms, repo)


if __name__ == "__main__":
    main()
