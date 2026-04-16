import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# XML / PAGE utilities
# ---------------------------------------------------------------------------

def parse_xml_polygons(xml_path: str) -> List[List[Tuple[int, int]]]:
    ns = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons = []
    for tl in root.findall(".//ns:TextLine", ns):
        coords = tl.find("ns:Coords", ns)
        if coords is None:
            continue
        pts = [(int(x), int(y)) for x, y in
               (p.split(",") for p in coords.attrib["points"].split())]
        polygons.append(pts)
    return polygons


def polygon_to_bbox(poly: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_to_mask(poly: List[Tuple[int, int]], H: int, W: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.array(poly, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_to_polygon(mask: np.ndarray, ox: int, oy: int):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    return [(int(p[0][0] + ox), int(p[0][1] + oy)) for p in cnt]


def polygon_to_baseline(poly: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    poly = np.array(poly)
    y_max = poly[:, 1].max()
    base = poly[poly[:, 1] >= y_max - 2]
    base = base[np.argsort(base[:, 0])]
    return [(int(x), int(y)) for x, y in base]


def create_page_xml(img_path: str, W: int, H: int):
    root = ET.Element("PcGts", {
        "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation":
            "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 "
            "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"
    })
    meta = ET.SubElement(root, "Metadata")
    ET.SubElement(meta, "Creator").text = "Arthur XML Generator"
    ET.SubElement(meta, "Created").text = datetime.now().isoformat()
    ET.SubElement(meta, "LastChange").text = datetime.now().isoformat()
    page = ET.SubElement(root, "Page", {
        "imageFilename": os.path.basename(img_path),
        "imageWidth": str(W),
        "imageHeight": str(H),
    })
    region = ET.SubElement(page, "TextRegion", {"id": "region_textline"})
    ET.SubElement(region, "Coords", {"points": f"0,0 {W},0 {W},{H} 0,{H}"})
    return root, region


# ---------------------------------------------------------------------------
# Segmentation dataset — directory-based (full / standard splits)
# ---------------------------------------------------------------------------

class TextLineSegDataset(Dataset):
    """Crops text-line bboxes from page images and returns (crop, mask) pairs.

    Args:
        img_dir:       directory with page images (.jpg / .png / .tif)
        xml_dir:       directory with PAGE-XML ground-truth (same stem as images)
        pad:           pixel padding around each bounding box crop
        resize:        (W, H) tuple to resize every crop, or None to keep original
        include_stems: optional set of page stems to restrict to (for k-shot)
    """

    def __init__(self, img_dir: str, xml_dir: str, pad: int = 15,
                 resize: Tuple[int, int] = None,
                 include_stems: set = None):
        self.samples = []
        self.pad = pad
        self.resize = resize

        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".png", ".tif")):
                continue
            stem = os.path.splitext(fname)[0]
            if include_stems is not None and stem not in include_stems:
                continue
            img_path = os.path.join(img_dir, fname)
            xml_path = os.path.join(xml_dir, stem + ".xml")
            if not os.path.exists(xml_path):
                continue
            for poly in parse_xml_polygons(xml_path):
                self.samples.append((img_path, poly, polygon_to_bbox(poly)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, poly, bbox = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1 - self.pad), max(0, y1 - self.pad)
        x2, y2 = min(W, x2 + self.pad), min(H, y2 + self.pad)

        crop = img[y1:y2, x1:x2]
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        poly_shifted = np.array(poly) - np.array([x1, y1])
        cv2.fillPoly(mask, [poly_shifted.astype(np.int32)], 1)

        if self.resize is not None:
            crop = cv2.resize(crop, self.resize)
            mask = cv2.resize(mask, self.resize, interpolation=cv2.INTER_NEAREST)

        crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return crop, mask


