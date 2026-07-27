import argparse
import os
from argparse import Namespace

import yaml

from evaluate import cmd_predict, cmd_score_diva, cmd_score_hscp, cmd_score_udiads
from train import resolve_yolo_best_weight, train_segm_full, train_yolo


def _resolve_paths(value, base_dir: str):
    if isinstance(value, dict):
        return {key: _resolve_paths(item, base_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, base_dir) for item in value]
    if isinstance(value, str) and not os.path.isabs(value):
        if value.startswith(".") or "/" in value:
            return os.path.normpath(os.path.join(base_dir, value))
    return value


def load_experiment(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _resolve_paths(cfg, os.path.dirname(os.path.abspath(config_path)))


def resolve_segm_best_weight(segm_cfg: dict) -> str:
    candidate = os.path.join(segm_cfg["out_dir"], "best.pth")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Could not find trained segmentation weights: {candidate}")
    return candidate


def build_predict_args(exp_cfg: dict) -> Namespace:
    detection = exp_cfg["detection"]
    segmentation = exp_cfg["segmentation"]
    prediction = exp_cfg["prediction"]

    yolo_variant = prediction.get("yolo_variant") or detection.get("selected_variant")
    yolo_weights = prediction.get("yolo_weights")
    if not yolo_weights or yolo_weights == "auto":
        yolo_weights = resolve_yolo_best_weight(detection, yolo_variant)

    segm_weights = prediction.get("segm_weights")
    if not segm_weights or segm_weights == "auto":
        segm_weights = resolve_segm_best_weight(segmentation)

    return Namespace(
        dataset_family=exp_cfg["dataset_family"],
        yolo_weights=yolo_weights,
        segm_weights=segm_weights,
        test_img_dir=prediction["test_img_dir"],
        test_xml_dir=prediction.get("test_xml_dir"),
        out_xml_dir=prediction.get("out_xml_dir"),
        out_instance_dir=prediction.get("out_instance_dir"),
        out_binary_dir=prediction.get("out_binary_dir"),
        out_overlay_dir=prediction.get("out_overlay_dir"),
        encoder=segmentation.get("encoder", "resnet34"),
        arch=segmentation.get("architecture", "unet"),
        resize_w=prediction.get("resize_w", segmentation.get("resize", [1080, 128])[0]),
        resize_h=prediction.get("resize_h", segmentation.get("resize", [1080, 128])[1]),
        bin_thresh=prediction.get("bin_thresh", 0.1),
        conf=prediction.get("conf", 0.5),
        crop_pad=prediction.get("crop_pad", segmentation.get("pad", 15)),
        native_segm=prediction.get("native_segm", False),
        close_frac=prediction.get("close_frac", 0.04),
    )


def build_score_args(exp_cfg: dict, mode: str) -> Namespace:
    evaluation = exp_cfg["evaluation"][mode]
    return Namespace(
        mode=mode,
        pred_xml_dir=evaluation.get("pred_xml_dir"),
        gt_xml_dir=evaluation.get("gt_xml_dir"),
        test_img_dir=evaluation.get("test_img_dir"),
        iou_thresh=evaluation.get("iou_thresh", 0.75),
        gt_pixel_dir=evaluation.get("gt_pixel_dir"),
        gt_page_dir=evaluation.get("gt_page_dir"),
        img_dir=evaluation.get("img_dir"),
        pred_instance_dir=evaluation.get("pred_instance_dir"),
        gt_mask_dir=evaluation.get("gt_mask_dir"),
    )


def run_evaluation(exp_cfg: dict):
    if exp_cfg["dataset_family"] == "diva":
        if "hscp" in exp_cfg.get("evaluation", {}):
            cmd_score_hscp(build_score_args(exp_cfg, "hscp"))
        if "diva" in exp_cfg.get("evaluation", {}):
            cmd_score_diva(build_score_args(exp_cfg, "diva"))
        return

    if "udiads" in exp_cfg.get("evaluation", {}):
        cmd_score_udiads(build_score_args(exp_cfg, "udiads"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument(
        "action",
        choices=["train-det", "train-segm", "predict", "evaluate", "all"],
        help="Pipeline stage to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    exp_cfg = load_experiment(args.config)

    if args.action in {"train-det", "all"}:
        train_yolo(exp_cfg["detection"])

    if args.action in {"train-segm", "all"}:
        train_segm_full(exp_cfg["segmentation"])

    if args.action in {"predict", "all"}:
        cmd_predict(build_predict_args(exp_cfg))

    if args.action in {"evaluate", "all"}:
        run_evaluation(exp_cfg)


if __name__ == "__main__":
    main()