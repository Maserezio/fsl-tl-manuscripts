import cv2
import numpy as np

from utils.helpers import parse_xml_polygons, polygon_iou


def hscp_pixel_eval(gt_img, pred_img, threshold=0.75):
    """
    HSCP evaluation on pixel images (no conversions).
    
    Parameters:
    - gt_img: 2D numpy array, 0=background, >0=line
    - pred_img: 2D numpy array, 0=background, >0=line
    - threshold: IoU threshold for one-to-one matching
    
    Returns:
    - DR: Detection Rate
    - RA: Recognition Accuracy
    - FM: F-measure
    """

    # --- 1. Extract connected components (lines) ---
    _, gt_cc = cv2.connectedComponents(gt_img)
    _, pred_cc = cv2.connectedComponents(pred_img)

    # Extract masks for each line
    gt_lines = [(gt_cc == i) for i in range(1, gt_cc.max() + 1)]
    pred_lines = [(pred_cc == i) for i in range(1, pred_cc.max() + 1)]

    N, M = len(gt_lines), len(pred_lines)
    if N == 0 or M == 0:
        return 0.0, 0.0, 0.0

    # --- 2. Compute IoU for each line pair ---
    match_scores = np.zeros((M, N), dtype=np.float32)
    for i, pred_mask in enumerate(pred_lines):
        for j, gt_mask in enumerate(gt_lines):
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            if union > 0:
                match_scores[i, j] = intersection / union

    # --- 3. Greedy one-to-one matching ---
    o2o_matches = 0
    match_scores_copy = match_scores.copy()
    while True:
        i, j = np.unravel_index(np.argmax(match_scores_copy), match_scores_copy.shape)
        if match_scores_copy[i, j] < threshold:
            break
        o2o_matches += 1
        match_scores_copy[i, :] = -1
        match_scores_copy[:, j] = -1

    # --- 4. Compute HSCP metrics ---
    DR = o2o_matches / N
    RA = o2o_matches / M
    FM = (2 * DR * RA / (DR + RA)) if (DR + RA) > 0 else 0.0

    return DR, RA, FM

# --- HSCP XML evaluation ---
def hscp_xml_eval(gt_xml, pred_xml, threshold=0.75):
    """HSCP evaluation for XML polygons: returns DR, RA, FM."""
    gt_polys = parse_xml_polygons(gt_xml) # import from helpers.py
    pred_polys = parse_xml_polygons(pred_xml)
    
    N, M = len(gt_polys), len(pred_polys)
    if N == 0 or M == 0:
        return 0.0, 0.0, 0.0

    # MatchScore matrix
    match_scores = [[polygon_iou(pred, gt) for gt in gt_polys] for pred in pred_polys]

    # Greedy one-to-one matching
    o2o_matches = 0
    match_scores = np.array(match_scores)
    match_scores_copy = match_scores.copy()

    while True:
        i, j = np.unravel_index(np.argmax(match_scores_copy), match_scores_copy.shape)
        if match_scores_copy[i, j] < threshold:
            break
        o2o_matches += 1
        match_scores_copy[i, :] = -1
        match_scores_copy[:, j] = -1

    DR = o2o_matches / N
    RA = o2o_matches / M
    FM = (2 * DR * RA) / (DR + RA) if (DR + RA) > 0 else 0.0

    return DR, RA, FM
