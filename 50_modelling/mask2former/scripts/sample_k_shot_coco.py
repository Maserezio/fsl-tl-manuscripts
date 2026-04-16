"""
Sample a k-shot subset from a COCO annotation file.

Page selection uses the same strategy as 60_allspark / 70_simple:
  grayscale_variance  (default) — pick K pages with highest pixel variance
  random              — random subset (seeded)
  precomputed         — read stems from a pre-computed diversity file

Usage examples
--------------
# Grayscale-variance selection (default)
python sample_k_shot_coco.py \\
    --input-json full.json --output-json k5.json \\
    --k 5 --images-dir path/to/train/images/

# Random selection
python sample_k_shot_coco.py \\
    --input-json full.json --output-json k5.json \\
    --k 5 --images-dir path/to/train/images/ --method random --seed 42

# Pre-computed diversity file
python sample_k_shot_coco.py \\
    --input-json full.json --output-json k5.json \\
    --k 5 --images-dir path/to/train/images/ \\
    --method grayscale_variance --precomputed path/to/diverse_images.txt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from few_shot_sampler import select_labeled_pages


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-json",   required=True, help="Source COCO JSON")
    p.add_argument("--output-json",  required=True, help="Output k-shot COCO JSON")
    p.add_argument("--k",            type=int, required=True, help="Number of pages to sample")
    p.add_argument("--images-dir",   required=True,
                   help="Directory of training images (used for diversity selection)")
    p.add_argument("--method",       default="grayscale_variance",
                   choices=["grayscale_variance", "random",
                            "pca_max_distance", "pca_centroid",
                            "ica_max_distance", "ica_centroid"],
                   help="Page selection strategy (default: grayscale_variance)")
    p.add_argument("--precomputed",  default=None,
                   help="Path to pre-computed diversity file (overrides on-the-fly selection)")
    p.add_argument("--seed",         type=int, default=42,
                   help="Random seed (used when method=random, default: 42)")
    return p.parse_args()


def load_coco(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_subset(coco: dict, selected_stems: list) -> dict:
    stem_set = set(selected_stems)
    selected_images = [
        img for img in coco["images"]
        if Path(img["file_name"]).stem in stem_set
    ]
    selected_ids = {img["id"] for img in selected_images}
    return {
        "info":        coco.get("info", {}),
        "licenses":    coco.get("licenses", []),
        "categories":  coco.get("categories", []),
        "images":      selected_images,
        "annotations": [a for a in coco.get("annotations", [])
                        if a["image_id"] in selected_ids],
        **{k: v for k, v in coco.items()
           if k in ("type",) and k not in ("info", "licenses", "categories",
                                            "images", "annotations")},
    }


def main():
    args = parse_args()

    coco = load_coco(args.input_json)
    if not coco.get("images"):
        raise ValueError(f"No images found in {args.input_json}")

    selected_stems = select_labeled_pages(
        img_dir=args.images_dir,
        k=args.k,
        method=args.method,
        precomputed_path=args.precomputed,
        seed=args.seed,
    )

    subset = build_subset(coco, selected_stems)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=True, indent=2)

    print(f"Wrote {len(subset['images'])} images / "
          f"{len(subset['annotations'])} annotations → {out}")
    print("Selected stems:")
    for stem in selected_stems:
        print(f"  {stem}")


if __name__ == "__main__":
    main()
