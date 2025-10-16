import numpy as np

def pra_pixel(gt_img, pred_img):
    """
    Compute Precision, Recall, Accuracy, and F1 on two binary images.

    Parameters:
    - gt_img: 2D numpy array, 0=background, >0=foreground (ground truth)
    - pred_img: 2D numpy array, 0=background, >0=foreground (prediction)

    Returns:
    - precision, recall, accuracy, f1
    """
    # Binarize
    gt = (gt_img > 0)
    pred = (pred_img > 0)

    TP = np.logical_and(gt, pred).sum()
    TN = np.logical_and(~gt, ~pred).sum()
    FP = np.logical_and(~gt, pred).sum()
    FN = np.logical_and(gt, ~pred).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, accuracy, f1
