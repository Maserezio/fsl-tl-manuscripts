"""
Shared few-shot page selection for DIVA-HisDB.

Imported by all models (60_allspark, 70_simple, 104_dino_det, 105_dino_det,
2stage) so that k-shot sampling is consistent across experiments.

Three selection modes
---------------------
precomputed   : reads from an existing text file produced by the diversity-
                selection notebooks (diva_cb55_{k}_diverse_images.txt).
grayscale_var : selects the K training pages with the highest grayscale
                variance — a simple proxy for layout diversity.
random        : random subset (for ablation / baselines).

Precomputed file format (produced by the existing notebooks)
------------------------------------------------------------
  Grayscale variance method:
    Image 1: e-codices_fmb-cb-0055_0099v_max.jpg (variance: 7333.80)

  PCA max distance method:
    Image 0: e-codices_fmb-cb-0055_0098v_max.jpg
  ...

The function parses the block for the requested method and returns K stems.
"""

import os
import random
import re
from typing import List, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Precomputed file parser
# ---------------------------------------------------------------------------

_METHOD_HEADERS = {
    "grayscale_variance": "Grayscale variance method",
    "pca_max_distance":   "PCA max distance method",
    "pca_centroid":       "PCA centroid method",
    "ica_max_distance":   "ICA max distance method",
    "ica_centroid":       "ICA centroid method",
}


def parse_selection_file(
    path: str,
    method: str = "grayscale_variance",
    k: int = 1,
) -> List[str]:
    """
    Parse a pre-computed diverse-images text file and return K stems.

    Parameters
    ----------
    path   : path to e.g. diva_cb55_1_diverse_images.txt
    method : one of the keys in _METHOD_HEADERS
    k      : number of images to return (file may contain fewer → returns all)
    """
    header = _METHOD_HEADERS.get(method, method)
    stems: List[str] = []
    inside = False

    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if header in line:
                inside = True
                continue
            if inside:
                if line == "" or (line.strip() and not line.startswith(" ")):
                    if line.strip():
                        inside = False
                    continue
                m = re.search(r"Image\s+\d+:\s+(\S+\.jpg)", line)
                if m:
                    stems.append(os.path.splitext(m.group(1))[0])

    if not stems:
        raise ValueError(
            f"No entries found for method '{method}' in {path}.\n"
            f"Available headers: {list(_METHOD_HEADERS.values())}"
        )
    return stems[:k]


# ---------------------------------------------------------------------------
# On-the-fly grayscale-variance selection
# ---------------------------------------------------------------------------

def _grayscale_variance(img_path: str) -> float:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(np.var(img.astype(np.float32)))


def select_by_grayscale_variance(img_dir: str, k: int) -> List[str]:
    """Select K training pages with the highest grayscale variance."""
    entries = [
        (f, _grayscale_variance(os.path.join(img_dir, f)))
        for f in os.listdir(img_dir) if f.lower().endswith(".jpg")
    ]
    entries.sort(key=lambda x: x[1], reverse=True)
    return [os.path.splitext(f)[0] for f, _ in entries[:k]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_labeled_pages(
    img_dir: str,
    k: int,
    method: str = "grayscale_variance",
    precomputed_path: Optional[str] = None,
    seed: int = 42,
) -> List[str]:
    """
    Return a list of K labeled page stems for the few-shot setup.

    Parameters
    ----------
    img_dir           : directory containing training images (*.jpg)
    k                 : number of labeled pages
    method            : "grayscale_variance" | "random" | PCA/ICA key
                        (PCA/ICA require precomputed_path)
    precomputed_path  : if provided, parse this file instead of re-computing
    seed              : random seed used when method="random"
    """
    if precomputed_path is not None and os.path.exists(precomputed_path):
        return parse_selection_file(precomputed_path, method=method, k=k)

    if method == "grayscale_variance":
        return select_by_grayscale_variance(img_dir, k)
    elif method == "random":
        all_stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir) if f.lower().endswith(".jpg")
        )
        rng = random.Random(seed)
        return rng.sample(all_stems, min(k, len(all_stems)))
    else:
        raise ValueError(
            f"Unknown selection method '{method}'. "
            f"Provide precomputed_path for PCA/ICA methods."
        )
