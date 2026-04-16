import cv2
import numpy as np


def mask_to_polygon(mask, epsilon_ratio=0.0025):
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 3:
        return None

    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True)
    if polygon.shape[0] < 3:
        polygon = contour
    return [(int(point[0][0]), int(point[0][1])) for point in polygon]


def sort_masks_top_to_bottom(masks):
    ordering = []
    for index, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            ordering.append((np.inf, np.inf, index))
            continue
        ordering.append((float(ys.mean()), float(xs.mean()), index))
    ordering.sort()
    return [index for _, _, index in ordering]