import numpy as np
import cv2

from utils.helpers import parse_xml_polygons

def diva_pixel_eval(gt_img, pred_img, threshold=0.75):
    """
    DIVA evaluation on pixel images.
    
    Parameters:
    - gt_img: 2D numpy array, 0=background, >0=line
    - pred_img: 2D numpy array, 0=background, >0=line
    - threshold: minimum pixel precision & recall to consider a line matched
    
    Returns:
    - pixel_iou: pixel-level IoU
    - line_iou: line-level IoU
    """

    # --- 1. Pixel IoU ---
    gt_bin = (gt_img > 0)
    pred_bin = (pred_img > 0)

    tp = np.logical_and(gt_bin, pred_bin).sum()
    fp = np.logical_and(~gt_bin, pred_bin).sum()
    fn = np.logical_and(gt_bin, ~pred_bin).sum()
    pixel_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # --- 2. Extract connected components for line-level IoU ---
    _, gt_cc = cv2.connectedComponents(gt_img)
    _, pred_cc = cv2.connectedComponents(pred_img)

    gt_lines = [(gt_cc == i) for i in range(1, gt_cc.max() + 1)]
    pred_lines = [(pred_cc == i) for i in range(1, pred_cc.max() + 1)]

    N, M = len(gt_lines), len(pred_lines)
    if N == 0 or M == 0:
        return pixel_iou, 0.0

    # --- 3. Compute precision & recall for each predicted line vs GT ---
    gt_matched = set()
    pred_matched = set()

    for i, pred_mask in enumerate(pred_lines):
        for j, gt_mask in enumerate(gt_lines):
            if j in gt_matched:
                continue
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            pred_area = pred_mask.sum()
            gt_area = gt_mask.sum()
            precision = intersection / pred_area if pred_area > 0 else 0.0
            recall = intersection / gt_area if gt_area > 0 else 0.0
            if precision >= threshold and recall >= threshold:
                gt_matched.add(j)
                pred_matched.add(i)
                break

    # --- 4. Compute line IoU ---
    tp_lines = len(gt_matched)
    fp_lines = M - len(pred_matched)
    fn_lines = N - len(gt_matched)
    line_iou = tp_lines / (tp_lines + fp_lines + fn_lines) if (tp_lines + fp_lines + fn_lines) > 0 else 0.0

    return pixel_iou, line_iou

# --- DIVA XML evaluation ---
def diva_xml_eval(gt_xml, pred_xml, threshold=0.75):
    """
    DIVA evaluation for XML polygons.
    Returns Pixel IoU (PIU) and Line IoU (LIU).
    Automatically determines mask size from polygons.
    """
    gt_polys = parse_xml_polygons(gt_xml)
    pred_polys = parse_xml_polygons(pred_xml)

    if not gt_polys or not pred_polys:
        return 0.0, 0.0

    # Determine mask size from all polygons
    all_points = [pt for poly in gt_polys + pred_polys for pt in poly]
    max_x = max(p[0] for p in all_points) + 1
    max_y = max(p[1] for p in all_points) + 1
    mask_shape = (max_y, max_x)

    # Rasterize polygons
    gt_mask = np.zeros(mask_shape, dtype=bool)
    for poly in gt_polys:
        gt_mask |= polygon_to_mask(poly, mask_shape)

    pred_mask = np.zeros(mask_shape, dtype=bool)
    for poly in pred_polys:
        pred_mask |= polygon_to_mask(poly, mask_shape)

    # Pixel IoU
    TP = np.logical_and(gt_mask, pred_mask).sum()
    FP = np.logical_and(~gt_mask, pred_mask).sum()
    FN = np.logical_and(gt_mask, ~pred_mask).sum()
    pixel_iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    # Line IoU
    matched_gt = set()
    matched_pred = set()
    for i, pred in enumerate(pred_polys):
        pred_mask_line = polygon_to_mask(pred, mask_shape)
        for j, gt in enumerate(gt_polys):
            if j in matched_gt:
                continue
            gt_mask_line = polygon_to_mask(gt, mask_shape)
            intersection = np.logical_and(pred_mask_line, gt_mask_line).sum()
            precision = intersection / pred_mask_line.sum() if pred_mask_line.sum() > 0 else 0
            recall = intersection / gt_mask_line.sum() if gt_mask_line.sum() > 0 else 0
            if precision >= threshold and recall >= threshold:
                matched_gt.add(j)
                matched_pred.add(i)
                break

    TP_lines = len(matched_gt)
    FP_lines = len(pred_polys) - len(matched_pred)
    FN_lines = len(gt_polys) - len(matched_gt)
    line_iou = TP_lines / (TP_lines + FP_lines + FN_lines) if (TP_lines + FP_lines + FN_lines) > 0 else 0.0

    return pixel_iou, line_iou