import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dino_det.utils.runtime import setup_project_imports

setup_project_imports()

from detectron2.config import get_cfg

import dino_det
from dino_det.engine import FullImagePredictor
from dino_det.pagexml import write_pagexml
from dino_det.utils import collect_image_paths, mask_to_baseline, mask_to_polygon, sort_masks_top_to_bottom


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Run full-image text-line inference and export PAGE XML")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser


def load_cfg(args):
    cfg = get_cfg()
    dino_det.add_dino_mask2former_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    dino_det.apply_mask2former_mode(cfg)
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze()
    return cfg


def draw_overlay(image_bgr, text_lines):
    overlay = image_bgr.copy()
    rng = np.random.RandomState(42)
    colors = [tuple(int(c) for c in rng.randint(80, 255, size=3)) for _ in text_lines]

    for line, color in zip(text_lines, colors):
        pts = np.array(line["coords"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], color)

    blended = cv2.addWeighted(image_bgr, 0.55, overlay, 0.45, 0)

    for line, color in zip(text_lines, colors):
        pts = np.array(line["coords"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(blended, [pts], isClosed=True, color=color, thickness=2)
        score_text = f"{line['score']:.2f}"
        tx, ty = pts[0, 0]
        cv2.putText(blended, score_text, (tx, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return blended


def main(args):
    cfg = load_cfg(args)
    predictor = FullImagePredictor(cfg)
    image_paths = collect_image_paths(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictions = predictor(image_rgb)
        instances = predictions["instances"].to("cpu")

        masks = instances.pred_masks.numpy().astype(np.uint8) if instances.has("pred_masks") else np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
        order = sort_masks_top_to_bottom(masks)
        text_lines = []

        for rank, index in enumerate(order):
            mask = masks[index]
            polygon = mask_to_polygon(mask)
            if polygon is None:
                continue
            baseline = mask_to_baseline(mask)
            if len(baseline) < 2:
                continue
            text_lines.append({
                "coords": polygon,
                "baseline": baseline,
                "score": float(instances.scores[index]),
            })
            if args.save_masks:
                cv2.imwrite(str(output_dir / f"{image_path.stem}_mask_{rank:04d}.png"), mask * 255)

        write_pagexml(
            image_path=image_path,
            image_size=(image_rgb.shape[1], image_rgb.shape[0]),
            text_lines=text_lines,
            output_path=output_dir / f"{image_path.stem}.xml",
        )

        vis = draw_overlay(image_bgr, text_lines)
        vis_path = output_dir / f"{image_path.stem}_overlay.png"
        cv2.imwrite(str(vis_path), vis)
        print(f"Saved overlay: {vis_path}")


if __name__ == "__main__":
    main(build_argument_parser().parse_args())