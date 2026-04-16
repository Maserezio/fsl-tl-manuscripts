"""
DIVA-HisDB datasets for supervised few-shot training with supervoxel maps.

DIVA-HisDB pixel-level GT colour encoding
------------------------------------------
    R=0,   B=1  → background
    R=0,   B=8  → **main text body** (text lines)  ← target class
    R=0,   B=4  → decoration
    R=0,   B=12 → marginalia
    R=128, B=*  → don't-care (boundary) pixels — excluded from loss

Dataset layout
--------------
  <data_root>/DIVA-HisDB/<manuscript>/
    img-<manuscript>/img/{training,validation,public-test}/*.jpg
    pixel-level-gt-<manuscript>/pixel-level-gt/{training,validation,public-test}/*.png
    PAGE-gt-<manuscript>/PAGE-gt/{training,validation,public-test}/*.xml

SLIC superpixels
----------------
Each item includes a supervoxel (superpixel) map computed on the cropped,
augmented RGB image via SLIC. The supervoxel loss uses this map to enforce
intra-superpixel prediction consistency without requiring additional labels.
"""

import os
import random
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Image / mask loaders
# ---------------------------------------------------------------------------

def _load_image(path: str) -> np.ndarray:
    """Load image as RGB uint8 [H, W, 3]."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_mask(gt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pixel-level GT PNG.

    Returns
    -------
    mask      : uint8 [H, W] — 1 = text line, 0 = background/other
    dont_care : bool  [H, W] — True where loss is excluded (R=128)
    """
    gt_rgb    = np.array(Image.open(gt_path).convert("RGB"))
    mask      = (gt_rgb[:, :, 2] == 8).astype(np.uint8)   # blue ch == 8
    dont_care = (gt_rgb[:, :, 0] == 128)                   # red  ch == 128
    return mask, dont_care


def _parse_xml_polygons(xml_path: str) -> list:
    """Parse PAGE-XML TextLine polygons. Returns list of polygon arrays."""
    _NS = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons = []
    for tl in root.findall(".//ns:TextLine", _NS):
        coords = tl.find("ns:Coords", _NS)
        if coords is None:
            continue
        pts = np.array(
            [(int(x), int(y)) for x, y in
             (p.split(",") for p in coords.attrib["points"].split())],
            dtype=np.int32,
        )
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons


def _load_boundary_mask(xml_path: str, H: int, W: int) -> np.ndarray:
    """
    Parse XML polygons → instance map → boundary mask.

    Returns bool [H, W] — True at inter-instance boundary pixels (where
    adjacent text-line instances touch) plus a 3-px-wide polygon contour band.
    """
    polygons = _parse_xml_polygons(xml_path)
    if not polygons:
        return np.zeros((H, W), dtype=bool)

    instance_map = np.zeros((H, W), dtype=np.int32)
    for i, poly in enumerate(polygons, start=1):
        cv2.fillPoly(instance_map, [poly], color=i)

    boundary = np.zeros((H, W), dtype=bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(instance_map, dy, axis=0), dx, axis=1)
        boundary |= (instance_map > 0) & (shifted > 0) & (instance_map != shifted)

    for poly in polygons:
        contour_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.polylines(contour_mask, [poly], isClosed=True, color=1, thickness=3)
        boundary |= contour_mask.astype(bool)

    return boundary


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _random_crop(
    img: np.ndarray,
    mask: np.ndarray,
    dc: np.ndarray,
    crop_size: int,
    boundary: Optional[np.ndarray] = None,
) -> tuple:
    H, W = img.shape[:2]
    y0 = random.randint(0, max(0, H - crop_size))
    x0 = random.randint(0, max(0, W - crop_size))
    y1, x1 = y0 + crop_size, x0 + crop_size

    def _pad(arr, fill=0):
        ph = max(0, y1 - H)
        pw = max(0, x1 - W)
        if ph or pw:
            if arr.ndim == 3:
                arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), constant_values=fill)
            else:
                arr = np.pad(arr, ((0, ph), (0, pw)), constant_values=fill)
        return arr

    img  = _pad(img)
    mask = _pad(mask)
    dc   = _pad(dc.astype(np.uint8), fill=1).astype(bool)

    result = (img[y0:y1, x0:x1], mask[y0:y1, x0:x1], dc[y0:y1, x0:x1])
    if boundary is not None:
        boundary = _pad(boundary.astype(np.uint8), fill=0).astype(bool)
        result   = result + (boundary[y0:y1, x0:x1],)
    return result


def _color_jitter(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img *= random.uniform(0.80, 1.20)                    # brightness
    mean = img.mean()
    img  = (img - mean) * random.uniform(0.80, 1.20) + mean  # contrast
    return np.clip(img, 0, 255).astype(np.uint8)


def _rotate(
    img: np.ndarray,
    mask: np.ndarray,
    dc: np.ndarray,
    angle: float,
    boundary: Optional[np.ndarray] = None,
) -> tuple:
    H, W = img.shape[:2]
    M    = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    img  = cv2.warpAffine(img,  M, (W, H), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    dc   = cv2.warpAffine(dc.astype(np.uint8), M, (W, H),
                           flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)
    result = (img, mask, dc)
    if boundary is not None:
        boundary = cv2.warpAffine(
            boundary.astype(np.uint8), M, (W, H),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(bool)
        result = result + (boundary,)
    return result


def _compute_superpixels(
    img_rgb: np.ndarray,     # [H, W, 3] uint8, already-cropped
    n_segments: int = 200,
    compactness: float = 10.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Run SLIC on the augmented RGB crop.

    Returns int32 [H, W] — integer superpixel ID per pixel, starting from 0.
    Computed on the raw (unnormalised) pixel values so that ink/background
    colour contrast directly guides the superpixel boundaries.
    """
    if n_segments <= 0:
        return np.zeros(img_rgb.shape[:2], dtype=np.int32)

    from skimage.segmentation import slic

    segments = slic(
        img_rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1,
    )
    return segments.astype(np.int32)


def _to_tensor(img: np.ndarray, crop_size: int) -> torch.Tensor:
    """Resize-if-needed, ImageNet-normalise, return [3, H, W] float."""
    if img.shape[0] != crop_size or img.shape[1] != crop_size:
        img = cv2.resize(img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    for c in range(3):
        t[c] = (t[c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
    return t


# ---------------------------------------------------------------------------
# Single dataset
# ---------------------------------------------------------------------------

class DIVADataset(Dataset):
    """
    Dataset for one DIVA-HisDB split.

    Each item is a tuple:
        (img_t, mask_t, dc_t, bnd_t, sp_t)

    img_t  : [3, H, W] float, ImageNet-normalised
    mask_t : [1, H, W] float {0, 1} — text-line pixels
    dc_t   : [1, H, W] bool         — don't-care pixels (excluded from loss)
    bnd_t  : [1, H, W] float {0, 1} — inter-instance boundary pixels
    sp_t   : [H, W]    int64        — SLIC superpixel IDs (0-indexed)

    For unlabeled pages (gt_dir=None) mask_t, bnd_t are zeros and dc_t is
    all-False; sp_t is still computed so the supervoxel loss can apply.
    """

    def __init__(
        self,
        img_dir: str,
        gt_dir: Optional[str],
        stems: List[str],
        crop_size: Optional[int] = 448,
        augment: bool = True,
        xml_dir: Optional[str] = None,
        sv_n_segments: int = 200,
        sv_compactness: float = 10.0,
        sv_sigma: float = 1.0,
    ):
        self.img_dir        = img_dir
        self.gt_dir         = gt_dir
        self.xml_dir        = xml_dir
        self.stems          = stems
        self.crop_size      = crop_size
        self.augment        = augment
        self.sv_n_segments  = sv_n_segments
        self.sv_compactness = sv_compactness
        self.sv_sigma       = sv_sigma

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img  = _load_image(os.path.join(self.img_dir, stem + ".jpg"))
        H, W = img.shape[:2]

        has_gt = self.gt_dir is not None
        if has_gt:
            mask, dc = _load_mask(os.path.join(self.gt_dir, stem + ".png"))
        else:
            mask = np.zeros((H, W), dtype=np.uint8)
            dc   = np.zeros((H, W), dtype=bool)

        boundary = None
        if self.xml_dir is not None:
            xml_path = os.path.join(self.xml_dir, stem + ".xml")
            if os.path.exists(xml_path):
                boundary = _load_boundary_mask(xml_path, H, W)

        # --- Random crop --------------------------------------------------
        if self.crop_size is not None:
            if boundary is not None:
                img, mask, dc, boundary = _random_crop(
                    img, mask, dc, self.crop_size, boundary)
            else:
                img, mask, dc = _random_crop(img, mask, dc, self.crop_size)

        # --- Augmentation -------------------------------------------------
        if self.augment and random.random() < 0.5:   # horizontal flip
            img      = img[:, ::-1, :].copy()
            mask     = mask[:, ::-1].copy()
            dc       = dc[:, ::-1].copy()
            if boundary is not None:
                boundary = boundary[:, ::-1].copy()

        if self.augment:
            img = _color_jitter(img)

        if self.augment and random.random() < 0.5:   # small rotation
            angle = random.uniform(-5, 5)
            if boundary is not None:
                img, mask, dc, boundary = _rotate(img, mask, dc, angle, boundary)
            else:
                img, mask, dc = _rotate(img, mask, dc, angle)

        # --- Superpixel map (computed on augmented, unnormalised crop) ----
        sp = _compute_superpixels(
            img,
            n_segments=self.sv_n_segments,
            compactness=self.sv_compactness,
            sigma=self.sv_sigma,
        )

        # --- To tensors ---------------------------------------------------
        if self.crop_size is not None:
            img_t = _to_tensor(img, self.crop_size)
        else:
            t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
            for c in range(3):
                t[c] = (t[c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
            img_t = t

        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        dc_t   = torch.from_numpy(dc.astype(np.uint8)).unsqueeze(0).bool()
        bnd_t  = (
            torch.from_numpy(boundary.astype(np.uint8)).unsqueeze(0).float()
            if boundary is not None else torch.zeros_like(mask_t)
        )
        sp_t = torch.from_numpy(sp).long()   # [H, W]

        return img_t, mask_t, dc_t, bnd_t, sp_t


# ---------------------------------------------------------------------------
# Semi-supervised paired dataset (labeled + unlabeled per step)
# ---------------------------------------------------------------------------

class SemiSupervisedDIVADataset(Dataset):
    """
    Returns (labeled_item, unlabeled_item) pairs.

    labeled_item  : (img_l, mask_l, dc_l, bnd_l, sp_l)
    unlabeled_item: (img_u, mask_u, dc_u, bnd_u, sp_u)  — mask/bnd are zeros

    The supervoxel loss on unlabeled items provides self-supervised spatial
    regularisation without requiring any labels on those pages.

    Labeled items are cycled with replacement when k < total training pages.
    """

    def __init__(self, labeled_ds: DIVADataset, unlabeled_ds: DIVADataset):
        self.labeled   = labeled_ds
        self.unlabeled = unlabeled_ds

    def __len__(self) -> int:
        return len(self.unlabeled)

    def __getitem__(self, idx: int):
        u_item = self.unlabeled[idx]
        l_idx  = idx % len(self.labeled)
        l_item = self.labeled[l_idx]
        return l_item, u_item


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_splits(
    data_root: str,
    manuscript: str,
    labeled_stems: List[str],
    crop_size: int = 448,
    sv_n_segments: int = 200,
    sv_compactness: float = 10.0,
    sv_sigma: float = 1.0,
) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    """
    Build datasets for training and evaluation.

    Returns (train_paired, val_ds, test_ds, train_l_only)
    where train_paired is a SemiSupervisedDIVADataset (for supervoxel on
    unlabeled) and train_l_only is the plain labeled-only dataset.
    """
    base     = os.path.join(data_root, manuscript)
    img_base = os.path.join(base, f"img-{manuscript}", "img")
    gt_base  = os.path.join(base, f"pixel-level-gt-{manuscript}", "pixel-level-gt")
    xml_base = os.path.join(base, f"PAGE-gt-{manuscript}-TASK-2", "TASK-2")

    def img_dir(split):  return os.path.join(img_base, split)
    def gt_dir(split):   return os.path.join(gt_base,  split)
    def xml_dir(split):
        d = os.path.join(xml_base, split)
        return d if os.path.isdir(d) else None

    def _all_stems(split):
        d = img_dir(split)
        return sorted(os.path.splitext(f)[0] for f in os.listdir(d)
                      if f.lower().endswith(".jpg"))

    sv_kw = dict(
        sv_n_segments=sv_n_segments,
        sv_compactness=sv_compactness,
        sv_sigma=sv_sigma,
    )

    all_train       = _all_stems("training")
    unlabeled_stems = [s for s in all_train if s not in set(labeled_stems)]

    train_l = DIVADataset(
        img_dir("training"), gt_dir("training"),
        labeled_stems, crop_size=crop_size, augment=True,
        xml_dir=xml_dir("training"), **sv_kw,
    )
    train_u = DIVADataset(
        img_dir("training"), gt_dir=None,
        stems=unlabeled_stems if unlabeled_stems else all_train,
        crop_size=crop_size, augment=True, **sv_kw,
    )
    train_paired = SemiSupervisedDIVADataset(train_l, train_u)

    val_ds = DIVADataset(
        img_dir("validation"), gt_dir("validation"),
        _all_stems("validation"), crop_size=None, augment=False,
        xml_dir=xml_dir("validation"), **sv_kw,
    )
    test_ds = DIVADataset(
        img_dir("public-test"), gt_dir("public-test"),
        _all_stems("public-test"), crop_size=None, augment=False,
        xml_dir=xml_dir("public-test"), **sv_kw,
    )
    return train_paired, val_ds, test_ds, train_l
