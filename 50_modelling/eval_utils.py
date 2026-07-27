# Evaluation utilities for manuscript text line detection and segmentation
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
from shapely.geometry import Polygon

try:
    from .geometry_utils import match_score, polygon_to_mask
except ImportError:
    # For standalone usage
    def match_score(mask_gt, mask_pred):
        inter = np.logical_and(mask_gt, mask_pred).sum()
        union = np.logical_or(mask_gt, mask_pred).sum()
        if union == 0:
            return 0.0
        return inter / union

    def polygon_to_mask(poly, height, width):
        import cv2

        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
        return mask

try:
    from .xml_utils import parse_xml_polygons
except ImportError:
    def parse_xml_polygons(xml_path: str):
        ns = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        tree = ET.parse(xml_path)
        root = tree.getroot()
        polygons = []
        for tl in root.findall(".//ns:TextLine", ns):
            coords = tl.find("ns:Coords", ns)
            if coords is None:
                continue
            pts = []
            for xy in coords.attrib["points"].split():
                x, y = map(int, xy.split(","))
                pts.append((x, y))
            if len(pts) >= 3:
                polygons.append(pts)
        return polygons

# NOTE: hscp_polygon_eval / hscp_polygon_eval_diva were removed (2026-07-08).
# Line-level DR/RA/FM are now reported via the official DIVA Java evaluator
# (LinesRecall/LinesPrecision/LinesFMeasure in its results.csv; Simistira et al. 2017,
# tool: Alberti et al. 2017). See run_diva_evaluator in predict_diva_seamcarve.py.

def diva_ink_mask(pixel_gt_path, text_bit: int = 0x8) -> np.ndarray:
    """Foreground (main-text) ink mask from a DIVA-HisDB pixel-level GT PNG.

    DIVA encodes the main text body in the blue-channel bit ``0x8``. Returns a
    uint8 (H, W) mask of the text pixels.
    """
    import cv2
    pg = cv2.imread(str(pixel_gt_path))
    return ((pg[:, :, 0].astype(int) & text_bit) > 0).astype(np.uint8)


def calculate_iou_polygon(poly_a_coords: np.ndarray, poly_b_coords: np.ndarray) -> float:
    """
    Calculate IoU between two polygons using Shapely.
    
    Args:
        poly_a_coords, poly_b_coords: Polygon coordinates
    
    Returns:
        IoU score between 0 and 1
    """
    try:
        a = Polygon(poly_a_coords)
        b = Polygon(poly_b_coords)
        if not a.is_valid: a = a.buffer(0)
        if not b.is_valid: b = b.buffer(0)
        intersection = a.intersection(b).area
        union = a.union(b).area
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0
