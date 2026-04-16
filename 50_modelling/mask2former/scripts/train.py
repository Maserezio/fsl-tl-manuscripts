import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELLING_ROOT = Path(__file__).resolve().parents[2]
for p in (str(PROJECT_ROOT), str(MODELLING_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from few_shot_sampler import select_labeled_pages

from dino_det.utils.runtime import setup_project_imports

setup_project_imports()

from detectron2.config import get_cfg
from detectron2.engine import launch
from detectron2.utils.logger import setup_logger

import dino_det
from dino_det.data_loader import prepare_hf_instance_dataset, register_text_line_coco_dataset
from dino_det.engine import DinoMask2FormerTrainer


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Train DINOv2 + Mask2Former text-line segmentation")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--train-json")
    parser.add_argument("--train-images")
    parser.add_argument("--val-json")
    parser.add_argument("--val-images")
    parser.add_argument("--hf-dataset", help="Optional Hugging Face dataset name, e.g. m3/balloon")
    parser.add_argument("--hf-train-split", default="train")
    parser.add_argument("--hf-val-split", default="validation")
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--hf-output-dir", help="Where the downloaded dataset is exported as local COCO")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k-shot", type=int, default=None,
                        help="If set, subsample --train-json to K pages using diversity selection")
    parser.add_argument("--k-shot-method", default="grayscale_variance",
                        choices=["grayscale_variance", "random",
                                 "pca_max_distance", "pca_centroid",
                                 "ica_max_distance", "ica_centroid"])
    parser.add_argument("--k-shot-precomputed", default=None,
                        help="Path to pre-computed diversity file")
    parser.add_argument("--k-shot-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser


def apply_k_shot_to_coco(train_json: str, train_images: str, args) -> str:
    """Subsample train_json to k pages using select_labeled_pages().
    Writes a temp JSON and returns its path."""
    stems = select_labeled_pages(
        img_dir=train_images,
        k=args.k_shot,
        method=args.k_shot_method,
        precomputed_path=args.k_shot_precomputed,
        seed=args.k_shot_seed,
    )
    with open(train_json, encoding="utf-8") as f:
        coco = json.load(f)

    stem_set = set(stems)
    selected_images = [img for img in coco["images"]
                       if Path(img["file_name"]).stem in stem_set]
    selected_ids = {img["id"] for img in selected_images}
    subset = {**coco,
              "images": selected_images,
              "annotations": [a for a in coco.get("annotations", [])
                               if a["image_id"] in selected_ids]}

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(subset, tmp, ensure_ascii=True)
    tmp.close()
    print(f"[k-shot] method={args.k_shot_method} k={args.k_shot} "
          f"→ {len(selected_images)} pages, {len(subset['annotations'])} annotations")
    return tmp.name


def resolve_dataset_paths(args):
    if args.hf_dataset:
        hf_output_dir = args.hf_output_dir or os.path.join(args.output_dir, "hf_dataset")
        return prepare_hf_instance_dataset(
            dataset_name=args.hf_dataset,
            output_root=hf_output_dir,
            train_split=args.hf_train_split,
            val_split=args.hf_val_split,
            cache_dir=args.hf_cache_dir,
        )

    if not args.train_json or not args.train_images:
        raise ValueError("Provide --train-json and --train-images, or use --hf-dataset.")

    train_json = args.train_json
    if args.k_shot is not None:
        train_json = apply_k_shot_to_coco(train_json, args.train_images, args)

    return {
        "train_json": train_json,
        "train_images": args.train_images,
        "val_json": args.val_json,
        "val_images": args.val_images,
    }


def setup_cfg(args):
    dataset_paths = resolve_dataset_paths(args)

    cfg = get_cfg()
    dino_det.add_dino_mask2former_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    dino_det.apply_mask2former_mode(cfg)
    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.DEVICE = args.device

    if args.device == "cpu":
        cfg.SOLVER.AMP.ENABLED = False

    train_name = register_text_line_coco_dataset("text_line_train", dataset_paths["train_json"], dataset_paths["train_images"])
    cfg.DATASETS.TRAIN = (train_name,)

    if dataset_paths.get("val_json") and dataset_paths.get("val_images"):
        val_name = register_text_line_coco_dataset("text_line_val", dataset_paths["val_json"], dataset_paths["val_images"])
        cfg.DATASETS.TEST = (val_name,)
    else:
        cfg.DATASETS.TEST = ()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    setup_logger(output=cfg.OUTPUT_DIR, name="dino_det")
    trainer = DinoMask2FormerTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    arguments = build_argument_parser().parse_args()
    num_gpus = 0 if arguments.device == "cpu" else arguments.num_gpus
    launch(
        main,
        num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(arguments,),
    )