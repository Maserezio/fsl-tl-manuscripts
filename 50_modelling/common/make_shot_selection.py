"""Pick the k labeled pages for every DIVA subset, by every selection method.

Writes the text format few_shot_sampler.parse_selection_file already reads, one file
per (subset, k):

    00_data/DIVA-HisDB/shot_selection/diva_<subset>_<k>_diverse_images.txt

WHY THIS EXISTS INSTEAD OF THE NOTEBOOK
50_modelling/common/diva_cb55_2d_analysis.ipynb computes the same five methods, but it looks for a
`public-train` directory that does not exist in any subset (the split is called
`training`). Its `train_images` therefore comes out empty and it silently falls back to
`all_images`, which at that point holds only `public-test`. Run as-is it selects TEST
pages -- pages that are not even in the training split. This script reads `training/`.

The algorithms match the notebook exactly:
  grayscale_variance  top-k by np.var(gray)
  pca_max_distance    ResNet18 features -> StandardScaler -> PCA(2), then a brute-force
                      search over combinations for the set with the largest MINIMUM
                      pairwise distance (a max-min spread, not top-k by distance)
  pca_centroid        top-k by distance from the mean in PCA-2D
  ica_max_distance    as pca_max_distance but FastICA(2, random_state=42)
  ica_centroid        top-k by distance from the mean in ICA-2D
`random` is not written here -- few_shot_sampler generates it on the fly from a seed.

Features are extracted once per subset and reused by every k and both decompositions.
Brute force is affordable at these sizes: C(20,5)=15504, worst case C(20,10)=184756.
"""

import hashlib
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "50_modelling/common"))
from few_shot_sampler import _METHOD_HEADERS  # noqa: E402

DATA = REPO / "00_data/DIVA-HisDB"
OUT_DIR = DATA / "shot_selection"
SUBSETS = ("CB55", "CS18", "CS863")
K_VALUES = (1, 3, 5, 10, 15)
RESIZE = (720, 720)
SEED = 42
# FastICA's defaults (max_iter=200, tol=1e-4) do not converge on CS863 -- the result is
# then just wherever the iteration stopped, which makes the ica_* selections arbitrary
# rather than reproducible. Measured: 200 and 2000 both fail, 10000 at tol=1e-5 converges.
ICA_MAX_ITER = 10000
ICA_TOL = 1e-5
# Order matters only for readability; parse_selection_file finds blocks by header.
METHODS = ("grayscale_variance", "pca_max_distance", "pca_centroid",
           "ica_max_distance", "ica_centroid")


def train_dir(subset):
    return DATA / subset / f"img-{subset}" / "img" / "training"


def resnet18_features(paths, device):
    """512-d penultimate ResNet18 activations, one row per image."""
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net = torch.nn.Sequential(*list(net.children())[:-1]).to(device).eval()
    prep = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    rows = []
    with torch.no_grad():
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"unreadable image: {p}")
            x = prep(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[None].to(device)
            rows.append(net(x).flatten().cpu().numpy())
    del net
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.stack(rows)


def max_min_spread(points, k):
    """Indices of the k points whose smallest pairwise distance is largest."""
    n = len(points)
    if k >= n:
        return list(range(n))
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    best, best_score = None, -1.0
    for combo in itertools.combinations(range(n), k):
        idx = np.array(combo)
        sub = dist[np.ix_(idx, idx)]
        # Ignore the zero diagonal when taking the minimum.
        score = sub[~np.eye(k, dtype=bool)].min() if k > 1 else 0.0
        if score > best_score:
            best, best_score = combo, score
    return list(best)


def furthest_from_centroid(points, k):
    d = np.linalg.norm(points - points.mean(axis=0), axis=1)
    return list(np.argsort(d)[-k:][::-1])


def selections_for(subset, device):
    """-> {k: {method: [filename, ...]}} for one subset."""
    paths = sorted(p for p in train_dir(subset).iterdir()
                   if p.suffix.lower() == ".jpg")
    if not paths:
        raise SystemExit(f"no .jpg under {train_dir(subset)}")
    print(f"[{subset}] {len(paths)} training pages", flush=True)

    grays = np.array([float(np.var(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                                   .astype(np.float32))) for p in paths])
    scaled = StandardScaler().fit_transform(resnet18_features(paths, device))
    pca2 = PCA(n_components=2).fit_transform(scaled)
    ica2 = FastICA(n_components=2, random_state=SEED,
                   max_iter=ICA_MAX_ITER, tol=ICA_TOL).fit_transform(scaled)

    out = {}
    for k in K_VALUES:
        picks = {
            "grayscale_variance": list(np.argsort(grays)[-k:][::-1]),
            "pca_max_distance": max_min_spread(pca2, k),
            "pca_centroid": furthest_from_centroid(pca2, k),
            "ica_max_distance": max_min_spread(ica2, k),
            "ica_centroid": furthest_from_centroid(ica2, k),
        }
        out[k] = {m: [paths[i].name for i in idx] for m, idx in picks.items()}
    return out, {p.name for p in paths}


def write_file(subset, k, per_method):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"diva_{subset.lower()}_{k}_diverse_images.txt"
    lines = []
    for method in METHODS:
        lines.append(f"{_METHOD_HEADERS[method]}:")
        for i, name in enumerate(per_method[method]):
            lines.append(f"  Image {i}: {name}")
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}\n")
    fingerprints = {}
    for subset in SUBSETS:
        per_k, valid = selections_for(subset, device)
        for k, per_method in per_k.items():
            for method, names in per_method.items():
                if len(names) != min(k, len(valid)):
                    raise RuntimeError(f"{subset}/{method}/k={k}: got {len(names)} pages")
                unknown = set(names) - valid
                if unknown:
                    raise RuntimeError(f"{subset}/{method}/k={k} selected pages outside "
                                       f"the training split: {sorted(unknown)}")
                fingerprints[(subset, method, k)] = hashlib.md5(
                    ",".join(sorted(names)).encode()).hexdigest()[:8]
            path = write_file(subset, k, per_method)
            print(f"  k={k:<2} -> {path.name}", flush=True)

    # How much duplicated training the run can skip: methods often agree, and at k=15
    # of 20 any two selections must share at least 10 pages.
    print("\nunique page sets per (subset, k):")
    for subset in SUBSETS:
        row = []
        for k in K_VALUES:
            n = len({fingerprints[(subset, m, k)] for m in METHODS})
            row.append(f"k={k}:{n}/{len(METHODS)}")
        print(f"  {subset:6} " + "  ".join(row))
    total = len(set(fingerprints.values()))
    print(f"\n{total} unique selections out of {len(fingerprints)} "
          f"(random adds {len(SUBSETS) * len(K_VALUES)} more, generated at train time)")


if __name__ == "__main__":
    main()
