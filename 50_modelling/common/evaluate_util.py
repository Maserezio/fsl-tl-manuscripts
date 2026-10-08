"""
@author: Silvia Zottin
Evaluation code for FEST 2025: few shot text line segmentation ICDAR 2025 competition.

"""

from sklearn.metrics import precision_score, recall_score
import os
import os.path
import numpy as np
import cv2


def calculate_IoU(component1, component2):
    intersection = np.logical_and(component1, component2).sum()
    union = np.logical_or(component1, component2).sum()
    if union == 0:
        return 0
    return intersection / union


def find_best_matches(gt_components, pred_components, num_gt, num_pred):
    matches = []
    matches_all = []
    for gt_label in range(1, num_gt):
        best_iou = 0
        best_match = -1
        gt_mask = (gt_components == gt_label)

        for pred_label in range(1, num_pred):
            pred_mask = (pred_components == pred_label)
            iou = calculate_IoU(gt_mask, pred_mask)

            if iou > best_iou:
                best_iou = iou
                best_match = pred_label

            if iou >= 0.75:
                matches_all.append((gt_label, pred_label))

        if best_match != -1:
            matches.append((gt_label, best_match))

    return matches, matches_all


def calculate_pixel_and_line_IU(gt_components, pred_components, matches, threshold=0.75):
    TP, FP, FN = 0, 0, 0
    CL, ML, EL = 0, 0, 0

    for gt_label, pred_label in matches:
        gt_mask = (gt_components == gt_label)
        pred_mask = (pred_components == pred_label)

        tp = np.logical_and(gt_mask, pred_mask).sum()
        fp = np.logical_and(np.logical_not(gt_mask), pred_mask).sum()
        fn = np.logical_and(gt_mask, np.logical_not(pred_mask)).sum()

        TP += tp
        FP += fp
        FN += fn

        precision = precision_score(np.array(gt_mask).flatten(), np.array(pred_mask).flatten(), zero_division=0)
        recall = recall_score(np.array(gt_mask).flatten(), np.array(pred_mask).flatten(), zero_division=0)

        if precision >= threshold and recall >= threshold:
            CL += 1
        elif recall < threshold:
            ML += 1
        elif precision < threshold:
            EL += 1

    pixel_IU = 0 if (TP + FP + FN) == 0 else TP / (TP + FP + FN)
    line_IU = 0 if (CL + ML + EL) == 0 else CL / (CL + ML + EL)

    return pixel_IU, line_IU


def evaluate_metrics(gt_img, pred_img):
    num_gt, gt_components = cv2.connectedComponents(gt_img)
    num_pred = len(np.unique(pred_img))

    matches, matches_all = find_best_matches(gt_components, pred_img, num_gt, num_pred)
    pixel_IU, line_IU = calculate_pixel_and_line_IU(gt_components, pred_img, matches)

    N1 = num_gt - 1
    N2 = num_pred - 1

    M = len(matches_all)

    DR = 0 if N1 == 0 else M / N1
    RA = 0 if N1 == 0 else M / N2
    FM = 0 if DR + RA == 0 else 2 * (DR * RA) / (DR + RA)

    return pixel_IU, line_IU, DR, RA, FM


def udiads_textline_evaluate(result_directory, gt_directory):
    """
    Evaluate the results provided by the files in result_directory with respect
    to the ground truth information given by the files in gt_directory.
    """

    # Check whether result_directory and gt_directory are directories
    if not os.path.isdir(result_directory):
        print("The result folder is not a directory")
        return

    if not os.path.isdir(gt_directory):
        print("The gt folder is not a directory")
        return

    pixel_list = []
    line_list = []
    DR_list = []
    RA_list = []
    FM_list = []
    # For each file of the ground truth directory read the result
    for f in sorted(os.listdir(gt_directory)):
        pred_path = os.path.join(result_directory, f)
        pred_path = os.path.splitext(pred_path)[0] + ".png"
        img = cv2.imread(pred_path)
        # color-->label mapping (if needed)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        flat_img = img_rgb.reshape(-1, 3)
        unique_colors, unique_indices = np.unique(flat_img, axis=0, return_inverse=True)
        label_map = unique_indices.reshape(h, w).astype(np.uint8)

        mask_path = os.path.join(gt_directory, f)
        mask_path = os.path.splitext(mask_path)[0] + ".png"
        ground_truth_bin = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # calculating IoU based metrics
        print("Calculating metrics...")
        pixel_IU, line_IU, DR, RA, FM = evaluate_metrics(ground_truth_bin, label_map)
        print("Pixel IU: ", pixel_IU)
        print("Line IU: ", line_IU)
        print("Detection Rate: ", DR)
        print("Recognition Accuracy: ", RA)
        print("F-measure: ", FM)
        pixel_list.append(pixel_IU)
        line_list.append(line_IU)
        DR_list.append(DR)
        RA_list.append(RA)
        FM_list.append(FM)

    return np.mean(pixel_list), np.mean(line_list), np.mean(DR_list), np.mean(RA_list), np.mean(FM_list)
