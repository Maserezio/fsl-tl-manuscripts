"""Resolve every (subset, method, k) cell to its page set, and dedupe the training jobs.

Different selection methods often land on the same pages -- and at k=15 out of 20 any
two selections must share at least 10 -- so training one model per cell would repeat
work. This writes a manifest with two parts:

    jobs   one entry per UNIQUE (subset, page set): what actually gets trained
    cells  all 90 (subset, method, k) cells, each pointing at its job

The run script trains jobs; the reporting step expands jobs back out to cells.

    python 50_modelling/common/shot_selection_plan.py
    -> 99_evaluation/summaries/rq2/shot_selection_manifest.json
"""

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "50_modelling/common"))
from few_shot_sampler import select_labeled_pages  # noqa: E402

DATA = REPO / "00_data/DIVA-HisDB"
SEL_DIR = DATA / "shot_selection"
OUT = REPO / "99_evaluation/summaries/rq2/shot_selection_manifest.json"

SUBSETS = ("CB55", "CS18", "CS863")
K_VALUES = (1, 3, 5, 10, 15)
METHODS = ("grayscale_variance", "pca_max_distance", "pca_centroid",
           "ica_max_distance", "ica_centroid", "random")
RANDOM_SEED = 42


def pages_for(subset, method, k):
    img_dir = DATA / subset / f"img-{subset}" / "img" / "training"
    # random is generated from a seed; everything else is read from the precomputed file
    # (PCA/ICA need ResNet18 features over the split and cannot be redone here).
    precomputed = None if method == "random" else str(
        SEL_DIR / f"diva_{subset.lower()}_{k}_diverse_images.txt")
    return sorted(select_labeled_pages(img_dir=str(img_dir), k=k, method=method,
                                       precomputed_path=precomputed, seed=RANDOM_SEED))


def main():
    jobs, cells = {}, []
    for subset in SUBSETS:
        for k in K_VALUES:
            for method in METHODS:
                pages = pages_for(subset, method, k)
                if len(pages) != k:
                    raise SystemExit(f"{subset}/{method}/k={k}: {len(pages)} pages, want {k}")
                digest = hashlib.md5((subset + "|" + ",".join(pages)).encode()).hexdigest()[:8]
                job_id = f"{subset}_k{k}_{digest}"
                jobs.setdefault(job_id, dict(job=job_id, subset=subset, k=k, pages=pages))
                cells.append(dict(subset=subset, method=method, k=k, job=job_id))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(jobs=list(jobs.values()), cells=cells), indent=1))

    print(f"{len(cells)} cells -> {len(jobs)} unique training jobs "
          f"({len(cells) - len(jobs)} skipped as duplicates)")
    for subset in SUBSETS:
        per_k = []
        for k in K_VALUES:
            n = len({c["job"] for c in cells if c["subset"] == subset and c["k"] == k})
            per_k.append(f"k={k}:{n}/{len(METHODS)}")
        print(f"  {subset:6} " + "  ".join(per_k))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
