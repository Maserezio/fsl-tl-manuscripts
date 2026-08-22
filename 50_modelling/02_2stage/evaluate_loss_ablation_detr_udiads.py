#!/usr/bin/env python3
"""Evaluate U-DIADS crop-loss arms with validation-selected RF-/RT-DETR.

Detector confidence is selected from validation box F1 only, then test boxes
are held fixed across BCE, Tversky, and SuperVoxel segmenters.  Every padded
native crop is resized to 1024x256 before U-Net inference; the probability map
is resized back to the native crop before thresholding and instance stitching.
The output uses the repository's official U-DIADS/Zottin implementation.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "71_misc"))
from evaluate_util import evaluate_metrics as evaluate_metrics_reference  # noqa: E402

ARMS = ("bce", "tversky", "supervoxel")
DETECTORS = {"rf_detr": "RF-DETR", "rt_detr": "RT-DETR"}
# Determinism. The Zottin metric matches at a hard IoU >= 0.75, so a few ULPs of
# difference in the segmenter's logits flip individual lines across the threshold.
# Two renders of the same Latin2 checkpoint with identical detections disagreed by
# up to 0.074 FM on a page and 0.007 at the subset mean; TF32 and cuDNN's timing-based
# algorithm choice were the source. Off, so a re-render reproduces its numbers.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def evaluate_metrics(gt_img, pred_img, threshold=0.75):
    """Vectorized equivalent of ``71_misc/evaluate_util.py:evaluate_metrics``.

    The official implementation materializes a full-page boolean mask for every
    GT/prediction pair.  A label contingency table gives exactly the same
    intersections, unions, best matches, and FEST/Zottin scores without that
    quadratic image-scanning cost.
    """
    num_gt, gt_components = cv2.connectedComponents(gt_img)
    unique_pred, pred_components = np.unique(pred_img, return_inverse=True)
    pred_components = pred_components.reshape(pred_img.shape)
    num_pred = len(unique_pred)
    n_gt = num_gt - 1
    n_pred = num_pred - 1
    if n_gt == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    contingency = np.bincount(
        (gt_components.ravel().astype(np.int64) * num_pred + pred_components.ravel()),
        minlength=num_gt * num_pred,
    ).reshape(num_gt, num_pred)
    intersections = contingency[1:, 1:].astype(np.float64)
    gt_area = contingency[1:, :].sum(axis=1).astype(np.float64)
    pred_area = contingency[:, 1:].sum(axis=0).astype(np.float64)
    unions = gt_area[:, None] + pred_area[None, :] - intersections
    ious = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)

    # The reference counts every pair above threshold for DR/RA, then uses each
    # GT component's single best non-empty match for pixel and line IoU.
    matched_all = int((ious >= threshold).sum())
    if n_pred:
        best_pred = np.argmax(ious, axis=1)
        best_iou = ious[np.arange(n_gt), best_pred]
        matched_gt = np.flatnonzero(best_iou > 0)
    else:
        best_pred = np.empty((n_gt,), dtype=np.int64)
        matched_gt = np.empty((0,), dtype=np.int64)

    if len(matched_gt):
        chosen_pred = best_pred[matched_gt]
        tp = intersections[matched_gt, chosen_pred]
        fp = pred_area[chosen_pred] - tp
        fn = gt_area[matched_gt] - tp
        pixel_denominator = float((tp + fp + fn).sum())
        pixel_iu = 0.0 if pixel_denominator == 0 else float(tp.sum() / pixel_denominator)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        line_iu = float(((precision >= threshold) & (recall >= threshold)).mean())
    else:
        pixel_iu = line_iu = 0.0

    detection_rate = matched_all / n_gt
    recognition_accuracy = 0.0 if n_pred == 0 else matched_all / n_pred
    f_measure = (
        0.0
        if detection_rate + recognition_accuracy == 0
        else 2 * detection_rate * recognition_accuracy / (detection_rate + recognition_accuracy)
    )
    return pixel_iu, line_iu, detection_rate, recognition_accuracy, f_measure


def image_paths(directory: Path):
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in EXTENSIONS)


def component_boxes(mask_path: Path, min_area=50):
    gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    n, _, stats, _ = cv2.connectedComponentsWithStats((gray > 0).astype(np.uint8), 8)
    boxes = []
    for label_id in range(1, n):
        if stats[label_id, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h = (
            int(stats[label_id, cv2.CC_STAT_LEFT]),
            int(stats[label_id, cv2.CC_STAT_TOP]),
            int(stats[label_id, cv2.CC_STAT_WIDTH]),
            int(stats[label_id, cv2.CC_STAT_HEIGHT]),
        )
        boxes.append((x, y, x + w, y + h))
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def box_iou(one, many):
    if not len(many):
        return np.empty((0,), dtype=np.float32)
    top_left = np.maximum(one[:2], many[:, :2])
    bottom_right = np.minimum(one[2:], many[:, 2:])
    wh = np.maximum(0.0, bottom_right - top_left)
    intersection = wh[:, 0] * wh[:, 1]
    area_one = np.prod(np.maximum(0.0, one[2:] - one[:2]))
    area_many = np.prod(np.maximum(0.0, many[:, 2:] - many[:, :2]), axis=1)
    return intersection / (area_one + area_many - intersection + 1e-9)


def deduplicate(boxes, scores, threshold):
    kept = []
    for index in np.argsort(-scores):
        if all(float(box_iou(boxes[index], boxes[[old]])[0]) < threshold for old in kept):
            kept.append(int(index))
    return np.asarray(kept, dtype=np.int64)


def detection_counts(predicted, truth, iou_threshold=0.5):
    matched_truth = set()
    true_positive = 0
    for box in predicted:
        ious = box_iou(box, truth)
        order = np.argsort(-ious)
        match = next(
            (int(index) for index in order if ious[index] >= iou_threshold and int(index) not in matched_truth),
            None,
        )
        if match is not None:
            matched_truth.add(match)
            true_positive += 1
    return true_positive, len(predicted) - true_positive, len(truth) - true_positive


@torch.no_grad()
def raw_detections(model_dir, paths):
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = AutoModelForObjectDetection.from_pretrained(str(model_dir)).to(DEVICE).eval()
    detections = {}
    for path in paths:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        side = max(width, height)
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        result = processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=[(side, side)]
        )[0]
        boxes = result["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = result["scores"].detach().cpu().numpy().astype(np.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        valid = (
            (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
            & np.isfinite(boxes).all(axis=1)
        )
        detections[path.stem] = (boxes[valid], scores[valid])
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return detections


def select_confidence(raw, val_paths, gt_dir, dedup_iou, values):
    rows = []
    for confidence in values:
        tp = fp = fn = 0
        counts = []
        for path in val_paths:
            boxes, scores = raw[path.stem]
            selected = np.where(scores >= confidence)[0]
            boxes_here, scores_here = boxes[selected], scores[selected]
            keep = deduplicate(boxes_here, scores_here, dedup_iou)
            predicted = boxes_here[keep]
            truth = component_boxes(gt_dir / f"{path.stem}.png")
            one_tp, one_fp, one_fn = detection_counts(predicted, truth)
            tp += one_tp
            fp += one_fp
            fn += one_fn
            counts.append(len(predicted))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append(
            {
                "confidence": float(confidence),
                "box_precision@0.5": precision,
                "box_recall@0.5": recall,
                "box_F1@0.5": f1,
                "mean_boxes": float(np.mean(counts)),
            }
        )
    frame = pd.DataFrame(rows)
    best = frame.sort_values(["box_F1@0.5", "confidence"], ascending=False).iloc[0]
    return float(best.confidence), frame


def finalized_detections(raw, paths, confidence, dedup_iou):
    result = {}
    for path in paths:
        boxes, scores = raw[path.stem]
        selected = np.where(scores >= confidence)[0]
        boxes_here, scores_here = boxes[selected], scores[selected]
        keep = deduplicate(boxes_here, scores_here, dedup_iou)
        # keep remains score-descending, which defines deterministic overlap ownership
        result[path.stem] = (boxes_here[keep], scores_here[keep])
    return result


def load_segmenter(path: Path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = smp.Unet(
        encoder_name=checkpoint.get("backbone", "resnet34"),
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def connect_line(mask, close_fraction):
    if not mask.any():
        return mask.astype(np.uint8)
    width = max(3, int(round(mask.shape[1] * close_fraction)))
    if width % 2 == 0:
        width += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 3))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    if count <= 1:
        return closed
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


@torch.no_grad()
def segment_crop(crop_bgr, model, crop_w, crop_h, threshold, close_fraction):
    native_h, native_w = crop_bgr.shape[:2]
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    tensor = (
        torch.from_numpy(np.ascontiguousarray(resized))
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .to(DEVICE)
        / 255.0
    )
    probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    # Critical geometry step: return from 1024x256 to the detector crop's native size.
    probability = cv2.resize(
        probability, (native_w, native_h), interpolation=cv2.INTER_LINEAR
    )
    return connect_line(probability >= threshold, close_fraction)


def predict_instances(page_bgr, boxes, model, args):
    height, width = page_bgr.shape[:2]
    instance = np.zeros((height, width), dtype=np.uint16)
    next_id = 1
    for box in boxes:
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, y1 = max(0, x1 - args.pad), max(0, y1 - args.pad)
        x2, y2 = min(width, x2 + args.pad), min(height, y2 + args.pad)
        if x2 <= x1 or y2 <= y1:
            continue
        crop_mask = segment_crop(
            page_bgr[y1:y2, x1:x2],
            model,
            args.crop_w,
            args.crop_h,
            args.segmenter_threshold,
            args.close_fraction,
        ).astype(bool)
        if not crop_mask.any():
            continue
        region = instance[y1:y2, x1:x2]
        writable = crop_mask & (region == 0)
        if not writable.any():
            continue
        region[writable] = next_id
        next_id += 1
    return instance


def colorize(instance):
    output = np.zeros((*instance.shape, 3), dtype=np.uint8)
    for label in np.unique(instance):
        if label == 0:
            continue
        rng = np.random.default_rng(int(label))
        output[instance == label] = rng.integers(40, 256, size=3, dtype=np.uint8)
    return output


def evaluate_combination(detector, arm, detections, test_paths, gt_dir, seg_root, out_root, args):
    model, checkpoint = load_segmenter(seg_root / arm / "best.pth", DEVICE)
    combination_dir = out_root / f"{detector}_{arm}"
    instance_dir = combination_dir / "instance"
    overlay_dir = combination_dir / "overlay"
    instance_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in test_paths:
        page = cv2.imread(str(path), cv2.IMREAD_COLOR)
        boxes, _ = detections[path.stem]
        instance = predict_instances(page, boxes, model, args)
        gt = cv2.imread(str(gt_dir / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        values = evaluate_metrics(gt, instance.astype(np.int32))
        row = {
            "page": path.stem,
            "Pixel_IU": values[0],
            "Line_IU": values[1],
            "DR": values[2],
            "RA": values[3],
            "FM": values[4],
            "boxes": len(boxes),
            "predicted_instances": int(instance.max()),
        }
        rows.append(row)
        cv2.imwrite(str(instance_dir / f"{path.stem}.png"), instance)
        overlay = cv2.addWeighted(page, 0.62, colorize(instance), 0.38, 0.0)
        cv2.imwrite(str(overlay_dir / f"{path.stem}.png"), overlay)
        print(
            f"[{detector}/{arm}] {path.stem}: boxes={len(boxes)} instances={instance.max()} "
            f"Pixel_IU={values[0]:.3f} Line_IU={values[1]:.3f} FM={values[4]:.3f}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(combination_dir / "per_page_zottin.csv", index=False)
    summary = {
        "detector": detector,
        "loss": arm,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "n_pages": len(frame),
        "mean_boxes": float(frame.boxes.mean()),
        **{metric: float(frame[metric].mean()) for metric in ("Pixel_IU", "Line_IU", "DR", "RA", "FM")},
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="Latin14396", choices=("Latin14396", "Latin2", "Syr341"))
    # No choices=: the backbone matrix names runs rtdetr_<backbone>_<init>, not
    # just rf_detr/rt_detr.
    parser.add_argument("--detectors", nargs="+", default=tuple(DETECTORS))
    parser.add_argument("--detector-root", default=None, dest="detector_root")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS,
                        help="which segmenter losses to score (default: all three)")
    parser.add_argument("--out-name", default=None, dest="out_name")
    parser.add_argument("--crop-width", type=int, default=1024, dest="crop_w")
    parser.add_argument("--crop-height", type=int, default=256, dest="crop_h")
    parser.add_argument("--pad", type=int, default=15)
    parser.add_argument("--segmenter-threshold", type=float, default=0.5, dest="segmenter_threshold")
    parser.add_argument("--dedup-iou", type=float, default=0.9, dest="dedup_iou")
    parser.add_argument("--close-fraction", type=float, default=0.04, dest="close_fraction")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    subset_root = REPO / "00_data" / "U-DIADS-TL" / args.subset
    image_base = subset_root / f"img-{args.subset}"
    gt_base = subset_root / f"text-line-gt-{args.subset}"
    # Split directories are not named consistently across subsets: Latin14396 has
    # img-*/val while Latin2 and Syr341 have img-*/validation (the GT side uses
    # "validation" everywhere). Resolve instead of assuming.
    def _split_dir(base, *candidates):
        for c in candidates:
            if (base / c).is_dir():
                return base / c
        raise FileNotFoundError(f"none of {candidates} under {base}")

    val_paths = image_paths(_split_dir(image_base, "val", "validation"))
    test_paths = image_paths(_split_dir(image_base, "test", "public-test"))
    val_gt = _split_dir(gt_base, "validation", "val")
    test_gt = _split_dir(gt_base, "test", "public-test")
    # --detector-root points at a directory whose subdirectories are runs, each
    # holding best_model/. Defaults to the original rf_detr/rt_detr layout; the
    # backbone matrix lives under detection/rtdetr_hf/<subset>/ instead.
    detector_root = (
        Path(args.detector_root) if args.detector_root
        else REPO / "80_models/02_2stage/u-diads-tl/detection"
             / f"hf_detr_{args.subset.lower()}_square_geometry"
    )
    seg_root = REPO / "80_models/02_2stage/u-diads-tl/crop_seg_loss_ablation_components_1024x256" / args.subset
    out_root = (REPO / "99_evaluation/02_2stage/u-diads-tl"
                / (args.out_name or f"{args.subset.lower()}_loss_ablation_detr_1024x256"))
    out_root.mkdir(parents=True, exist_ok=True)
    completed_path = out_root / "zottin_metrics.csv"
    if completed_path.exists() and not args.force:
        completed = pd.read_csv(completed_path)
        expected = {(detector, arm) for detector in args.detectors for arm in args.arms}
        present = set(zip(completed.get("detector", []), completed.get("loss", [])))
        if expected.issubset(present):
            print(completed.to_string(index=False), flush=True)
            print(f"already complete -> {completed_path}", flush=True)
            return

    confidence_values = np.round(np.arange(0.01, 0.92, 0.02), 2)
    all_test_detections = {}
    threshold_rows = []
    for detector in args.detectors:
        model_dir = detector_root / detector / "best_model"
        if not model_dir.exists():
            raise FileNotFoundError(model_dir)
        print(f"[{detector}] validation confidence sweep", flush=True)
        validation_raw = raw_detections(model_dir, val_paths)
        confidence, sweep = select_confidence(
            validation_raw, val_paths, val_gt, args.dedup_iou, confidence_values
        )
        sweep.insert(0, "detector", detector)
        sweep.to_csv(out_root / f"{detector}_validation_confidence_sweep.csv", index=False)
        best = sweep.loc[sweep["confidence"] == confidence].iloc[0].to_dict()
        threshold_rows.append(best)
        print(f"[{detector}] selected confidence={confidence:.2f}, val box F1={best['box_F1@0.5']:.4f}", flush=True)
        test_raw = raw_detections(model_dir, test_paths)
        all_test_detections[detector] = finalized_detections(
            test_raw, test_paths, confidence, args.dedup_iou
        )
    pd.DataFrame(threshold_rows).to_csv(out_root / "validation_selected_thresholds.csv", index=False)

    summaries = []
    for detector in args.detectors:
        for arm in args.arms:
            summaries.append(
                evaluate_combination(
                    detector,
                    arm,
                    all_test_detections[detector],
                    test_paths,
                    test_gt,
                    seg_root,
                    out_root,
                    args,
                )
            )
            pd.DataFrame(summaries).to_csv(out_root / "zottin_metrics.csv", index=False)
    report = pd.DataFrame(summaries)
    report.to_csv(out_root / "zottin_metrics.csv", index=False)
    metadata = {
        "subset": args.subset,
        "detector_root": str(detector_root),
        "segmenter_root": str(seg_root),
        "resize_before_prediction": [args.crop_w, args.crop_h],
        "resize_probability_back_to_native_crop": True,
        "segmenter_threshold": args.segmenter_threshold,
        "detector_threshold_selection": "validation micro box F1 at IoU 0.5",
        "dedup_iou": args.dedup_iou,
        "close_fraction": args.close_fraction,
    }
    (out_root / "evaluation_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\n" + report.to_string(index=False), flush=True)
    print(f"DONE -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
