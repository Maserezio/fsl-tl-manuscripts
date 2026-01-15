# Evaluation utilities for manuscript text line detection and segmentation
import numpy as np
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

def hscp_polygon_eval(gt_xml: str, pred_xml: str, page_height: int, page_width: int,
                     iou_thresh: float = 0.75) -> Dict[str, float]:
    """
    Evaluate polygon predictions using Detection Rate (DR), Recognition Accuracy (RA), and F-Measure (FM).
    
    Args:
        gt_xml: Path to ground truth XML
        pred_xml: Path to prediction XML
        page_height, page_width: Page dimensions
        iou_thresh: IoU threshold for matching
    
    Returns:
        Dictionary with DR, RA, FM, M (matches), N_gt, N_pred
    """
    from .xml_utils import parse_xml_polygons
    
    gt_polys = parse_xml_polygons(gt_xml)
    pred_polys = parse_xml_polygons(pred_xml)
    
    N1 = len(gt_polys)     # GT lines
    N2 = len(pred_polys)   # Pred lines
    
    if N1 == 0 or N2 == 0:
        return {
            "DR": 0.0,
            "RA": 0.0,
            "FM": 0.0,
            "M": 0,
            "N_gt": N1,
            "N_pred": N2
        }
    
    gt_masks = [
        polygon_to_mask(p, page_height, page_width)
        for p in gt_polys
    ]
    pred_masks = [
        polygon_to_mask(p, page_height, page_width)
        for p in pred_polys
    ]
    
    used_gt = set()
    used_pred = set()
    M = 0
    
    # Greedy one-to-one matching
    for i, gt_mask in enumerate(gt_masks):
        best_j = -1
        best_score = 0.0
        
        for j, pr_mask in enumerate(pred_masks):
            if j in used_pred:
                continue
            score = match_score(gt_mask, pr_mask)
            if score > best_score:
                best_score = score
                best_j = j
        
        if best_score >= iou_thresh and best_j >= 0:
            M += 1
            used_gt.add(i)
            used_pred.add(best_j)
    
    DR = M / N1 if N1 > 0 else 0.0
    RA = M / N2 if N2 > 0 else 0.0
    FM = (2 * DR * RA) / (DR + RA) if (DR + RA) > 0 else 0.0
    
    return {
        "DR": round(DR, 2),
        "RA": round(RA, 2),
        "FM": round(FM, 2),
        "M": M,
        "N_gt": N1,
        "N_pred": N2
    }

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

def hscp_pixel_eval(gt_mask: np.ndarray, pred_mask: np.ndarray,
                   iou_thresh: float = 0.75) -> Dict[str, float]:
    """
    Evaluate pixel-level predictions using connected components.
    
    Args:
        gt_mask: Ground truth binary mask
        pred_mask: Predicted binary mask
        iou_thresh: IoU threshold for matching components
    
    Returns:
        Dictionary with DR, RA, FM, M (matches), N_gt, N_pred
    """
    import cv2
    
    # Find connected components in GT mask
    gt_num_labels, gt_labels = cv2.connectedComponents(gt_mask.astype(np.uint8))
    
    # Find connected components in predicted mask
    pred_num_labels, pred_labels = cv2.connectedComponents(pred_mask.astype(np.uint8))
    
    N_gt = gt_num_labels - 1  # Exclude background (label 0)
    N_pred = pred_num_labels - 1  # Exclude background (label 0)
    
    if N_gt == 0 or N_pred == 0:
        return {
            "DR": 0.0,
            "RA": 0.0,
            "FM": 0.0,
            "M": 0,
            "N_gt": N_gt,
            "N_pred": N_pred
        }
    
    # Create individual component masks
    gt_component_masks = []
    for i in range(1, gt_num_labels):  # Skip background
        component_mask = (gt_labels == i).astype(np.uint8)
        gt_component_masks.append(component_mask)
    
    pred_component_masks = []
    for i in range(1, pred_num_labels):  # Skip background
        component_mask = (pred_labels == i).astype(np.uint8)
        pred_component_masks.append(component_mask)
    
    # Match components using greedy approach
    used_gt = set()
    used_pred = set()
    M = 0
    
    for i, gt_comp_mask in enumerate(gt_component_masks):
        best_j = -1
        best_score = 0.0
        
        for j, pred_comp_mask in enumerate(pred_component_masks):
            if j in used_pred:
                continue
            score = match_score(gt_comp_mask, pred_comp_mask)
            if score > best_score:
                best_score = score
                best_j = j
        
        if best_score >= iou_thresh and best_j >= 0:
            M += 1
            used_gt.add(i)
            used_pred.add(best_j)
    
    DR = M / N_gt if N_gt > 0 else 0.0
    RA = M / N_pred if N_pred > 0 else 0.0
    FM = (2 * DR * RA) / (DR + RA) if (DR + RA) > 0 else 0.0
    
    return {
        "DR": round(DR, 2),
        "RA": round(RA, 2),
        "FM": round(FM, 2),
        "M": M,
        "N_gt": N_gt,
        "N_pred": N_pred
    }

TRAIN_IMG_DIR = "./data/all/CB55/img-CB55/img/training/"
TRAIN_XML_DIR = "./data/all/CB55/PAGE-gt-CB55/PAGE-gt/training/"

VAL_IMG_DIR = "./data/all/CB55/img-CB55/img/validation/"
VAL_XML_DIR = "./data/all/CB55/PAGE-gt-CB55/PAGE-gt/validation/"

TEST_IMG_DIR = "./data/all/CB55/img-CB55/img/public-test/"
TEST_XML_DIR = "./data/all/CB55/PAGE-gt-CB55/PAGE-gt/public-test/"