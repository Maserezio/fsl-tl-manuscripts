import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dino_det.utils.runtime import setup_project_imports

setup_project_imports()

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.evaluation import print_csv_format

import dino_det
from dino_det.data_loader import register_text_line_coco_dataset
from dino_det.engine import DinoMask2FormerTrainer


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 + Mask2Former on a COCO split")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--dataset-images", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser


def main(args):
    cfg = get_cfg()
    dino_det.add_dino_mask2former_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    dino_det.apply_mask2former_mode(cfg)
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = args.weights
    cfg.OUTPUT_DIR = args.output_dir

    if args.device == "cpu":
        cfg.SOLVER.AMP.ENABLED = False

    dataset_name = register_text_line_coco_dataset("text_line_eval", args.dataset_json, args.dataset_images)
    cfg.DATASETS.TEST = (dataset_name,)
    cfg.DATASETS.TRAIN = ()
    cfg.freeze()

    model = DinoMask2FormerTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    results = DinoMask2FormerTrainer.test(cfg, model)
    print_csv_format(results)


if __name__ == "__main__":
    main(build_argument_parser().parse_args())