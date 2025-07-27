import cv2
import numpy as np
import os
from tqdm import tqdm

def evaluate_text_line_segmentation(gt_dir, pred_dir):
    pixel_iou_list, line_iou_list = [], []
    dr_list, ra_list, fm_list = [], [], []

    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])

    if not gt_files or not pred_files:
        raise ValueError("No PNG images found in directories")

    for gt_file in tqdm(gt_files, desc="Evaluating images"):
        if gt_file not in pred_files:
            print(f"Warning: Missing prediction for {gt_file}")
            continue

        gt_img = cv2.imread(os.path.join(gt_dir, gt_file), cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(os.path.join(pred_dir, gt_file), cv2.IMREAD_GRAYSCALE)

        if gt_img.shape != pred_img.shape:
            pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        gt_bin, pred_bin = (gt_img > 0), (pred_img > 0)

        tp_pixel = np.logical_and(gt_bin, pred_bin).sum()
        fp_pixel = np.logical_and(np.logical_not(gt_bin), pred_bin).sum()
        fn_pixel = np.logical_and(gt_bin, np.logical_not(pred_bin)).sum()

        pixel_iou = tp_pixel / (tp_pixel + fp_pixel + fn_pixel) if (tp_pixel + fp_pixel + fn_pixel) > 0 else 0.0
        pixel_iou_list.append(pixel_iou)

        _, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(gt_img)
        _, pred_labels, pred_stats, _ = cv2.connectedComponentsWithStats(pred_img)

        gt_boxes = [(*stat[0:4], stat[4]) for stat in gt_stats[1:]]
        pred_boxes = [(*stat[0:4], stat[4]) for stat in pred_stats[1:]]

        grid_size = 50
        spatial_index = {}
        for idx, (x, y, w, h, _) in enumerate(gt_boxes):
            for i in range(y // grid_size, (y + h) // grid_size + 1):
                for j in range(x // grid_size, (x + w) // grid_size + 1):
                    spatial_index.setdefault((i, j), []).append(idx)

        tp = 0
        matched_gt, matched_pred = set(), set()

        for pred_idx, (x_p, y_p, w_p, h_p, area_p) in enumerate(pred_boxes):
            candidates = set()
            for i in range(y_p // grid_size, (y_p + h_p) // grid_size + 1):
                for j in range(x_p // grid_size, (x_p + w_p) // grid_size + 1):
                    if (i, j) in spatial_index:
                        candidates.update(spatial_index[(i, j)])

            for gt_idx in candidates:
                if gt_idx in matched_gt:
                    continue
                x_g, y_g, w_g, h_g, area_g = gt_boxes[gt_idx]

                ix1, iy1 = max(x_p, x_g), max(y_p, y_g)
                ix2, iy2 = min(x_p + w_p, x_g + w_g), min(y_p + h_p, y_g + h_g)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                roi = (slice(iy1, iy2), slice(ix1, ix2))
                pred_mask = (pred_labels[roi] == pred_idx + 1)
                gt_mask = (gt_labels[roi] == gt_idx + 1)
                intersection = np.logical_and(pred_mask, gt_mask).sum()

                precision = intersection / area_p
                recall = intersection / area_g

                if precision > 0.75 and recall > 0.75:
                    tp += 1
                    matched_gt.add(gt_idx)
                    matched_pred.add(pred_idx)
                    break

        n_gt, n_pred = len(gt_boxes), len(pred_boxes)
        fp = n_pred - len(matched_pred)
        fn = n_gt - len(matched_gt)

        line_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dr = tp / n_gt if n_gt > 0 else 0.0
        ra = tp / n_pred if n_pred > 0 else 0.0
        fm = 2 * dr * ra / (dr + ra) if (dr + ra) > 0 else 0.0

        line_iou_list.append(line_iou)
        dr_list.append(dr)
        ra_list.append(ra)
        fm_list.append(fm)

    print("\nFinal Evaluation Metrics:")
    print(f"Pixel IoU: {np.mean(pixel_iou_list):.4f}")
    print(f"Line IoU: {np.mean(line_iou_list):.4f}")
    print(f"Detection Rate (DR): {np.mean(dr_list):.4f}")
    print(f"Recognition Accuracy (RA): {np.mean(ra_list):.4f}")
    print(f"F-Measure (FM): {np.mean(fm_list):.4f}")
