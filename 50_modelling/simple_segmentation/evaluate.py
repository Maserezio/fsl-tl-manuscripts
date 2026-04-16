import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
import xml.etree.ElementTree as ET
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.diva_dataset import _IMAGENET_MEAN, _IMAGENET_STD
from models import UnifiedSegmenter


def _normalize(img_rgb: np.ndarray, crop_size: int) -> torch.Tensor:
    img = cv2.resize(img_rgb, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1)
    for channel in range(3):
        tensor[channel] = (tensor[channel] - _IMAGENET_MEAN[channel]) / _IMAGENET_STD[channel]
    return tensor.unsqueeze(0)


def sliding_window_inference(
    model: UnifiedSegmenter,
    img_rgb: np.ndarray,
    crop_size: int,
    stride: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = True,
) -> np.ndarray:
    if stride is None:
        stride = crop_size // 2

    height, width = img_rgb.shape[:2]
    prob_sum = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)

    def _gauss_window(size: int) -> np.ndarray:
        axis = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(axis, axis)
        return np.exp(-(xx ** 2 + yy ** 2) / 0.5).astype(np.float32)

    gauss = _gauss_window(crop_size)
    ys = list(range(0, max(1, height - crop_size + 1), stride))
    xs = list(range(0, max(1, width - crop_size + 1), stride))
    if height > crop_size and ys[-1] + crop_size < height:
        ys.append(height - crop_size)
    if width > crop_size and xs[-1] + crop_size < width:
        xs.append(width - crop_size)

    for y0 in ys:
        for x0 in xs:
            y1, x1 = y0 + crop_size, x0 + crop_size
            patch = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            hy = min(height, y1) - y0
            hx = min(width, x1) - x0
            patch[:hy, :hx] = img_rgb[y0 : y0 + hy, x0 : x0 + hx]

            tensor = _normalize(patch, crop_size).to(device)
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                    prob = model.predict(tensor)
            prob_np = prob[0, 0].cpu().float().numpy()
            prob_sum[y0 : y0 + hy, x0 : x0 + hx] += prob_np[:hy, :hx] * gauss[:hy, :hx]
            count_map[y0 : y0 + hy, x0 : x0 + hx] += gauss[:hy, :hx]

    return prob_sum / np.maximum(count_map, 1e-6)


def _create_page_xml(img_filename: str, width: int, height: int, creator: str) -> Tuple[ET.Element, ET.Element]:
    namespace = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    root = ET.Element(
        "PcGts",
        {
            "xmlns": namespace,
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": f"{namespace} {namespace}/pagecontent.xsd",
        },
    )
    meta = ET.SubElement(root, "Metadata")
    ET.SubElement(meta, "Creator").text = creator
    ET.SubElement(meta, "Created").text = datetime.now().isoformat()
    ET.SubElement(meta, "LastChange").text = datetime.now().isoformat()

    page = ET.SubElement(root, "Page", {"imageFilename": img_filename, "imageWidth": str(width), "imageHeight": str(height)})
    region = ET.SubElement(page, "TextRegion", {"id": "r_text"})
    ET.SubElement(region, "Coords", {"points": f"0,0 {width},0 {width},{height} 0,{height}"})
    return root, region


def prob_map_to_pagexml(
    prob_map: np.ndarray,
    img_path: str,
    output_xml: str,
    creator: str,
    threshold: float = 0.5,
    min_area: int = 200,
) -> int:
    height, width = prob_map.shape
    binary = (prob_map > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    root, region = _create_page_xml(os.path.basename(img_path), width, height, creator)
    line_count = 0

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        component_mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue

        epsilon = 0.005 * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        points = " ".join(f"{point[0][0]},{point[0][1]}" for point in approx)

        line_id = f"tl_{line_count:04d}"
        text_line = ET.SubElement(region, "TextLine", {"id": line_id})
        ET.SubElement(text_line, "Coords", {"points": points})

        x, y, box_w, box_h = (
            stats[label_id, cv2.CC_STAT_LEFT],
            stats[label_id, cv2.CC_STAT_TOP],
            stats[label_id, cv2.CC_STAT_WIDTH],
            stats[label_id, cv2.CC_STAT_HEIGHT],
        )
        baseline_y = y + box_h - 1
        ET.SubElement(text_line, "Baseline", {"points": f"{x},{baseline_y} {x + box_w},{baseline_y}"})
        line_count += 1

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return line_count


def run_diva_evaluator(
    pred_xml_dir: str,
    gt_pixel_dir: str,
    gt_xml_dir: str,
    img_dir: str,
    out_csv_dir: str,
    java_cp: str = "/usr/share/openjfx/lib/*:/home/artur/Thesis/DIVA_Line_Segmentation_Evaluator/out/artifacts/LineSegmentationEvaluator.jar",
    java_main: str = "ch.unifr.LineSegmentationEvaluatorTool",
) -> None:
    os.makedirs(out_csv_dir, exist_ok=True)
    pred_xmls = sorted(filename for filename in os.listdir(pred_xml_dir) if filename.endswith(".xml"))

    for xml_name in tqdm(pred_xmls, desc="DIVA evaluator"):
        stem = os.path.splitext(xml_name)[0]
        pred_xml = os.path.join(pred_xml_dir, xml_name)
        gt_xml = os.path.join(gt_xml_dir, stem + ".xml")
        gt_png = os.path.join(gt_pixel_dir, stem + ".png")
        overlap = os.path.join(img_dir, stem + ".jpg")
        out_csv = os.path.join(out_csv_dir, stem + ".csv")

        if not os.path.exists(gt_xml) or not os.path.exists(gt_png):
            print(f"[WARN] Missing GT for {stem}, skipping.")
            continue

        cmd = ["java", "-cp", java_cp, java_main, "-igt", gt_png, "-xgt", gt_xml, "-xp", pred_xml, "-overlap", overlap, "-csv"]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            with open(out_csv, "w") as handle:
                handle.write(result.stdout)
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] Evaluator failed on {stem}: {exc.stderr[:200]}")
        except FileNotFoundError:
            print("[WARN] Java not found; skipping DIVA evaluator.")
            break


def summarise_csv_results(csv_dir: str) -> dict:
    import csv

    metrics = {"DR": [], "RA": [], "FM": []}
    for filename in os.listdir(csv_dir):
        if not filename.endswith(".csv"):
            continue
        with open(os.path.join(csv_dir, filename)) as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for key in metrics:
                    if key in row:
                        try:
                            metrics[key].append(float(row[key]))
                        except ValueError:
                            pass
    return {key: (sum(values) / len(values) if values else float("nan")) for key, values in metrics.items()}


def main(
    cfg_path: str,
    checkpoint: str,
    split: str = "public-test",
    threshold: float = 0.5,
    min_area: int = 200,
    run_eval: bool = True,
):
    with open(cfg_path) as handle:
        cfg = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    manuscript = dcfg["manuscript"]

    base = os.path.join(dcfg["data_root"], manuscript)
    img_dir = os.path.join(base, f"img-{manuscript}", "img", split)
    gt_xml_dir = os.path.join(base, f"PAGE-gt-{manuscript}", "PAGE-gt", split)
    gt_pix_dir = os.path.join(base, f"pixel-level-gt-{manuscript}", "pixel-level-gt", split)

    run_name = f"{manuscript}_{os.path.splitext(os.path.basename(cfg_path))[0]}_eval"
    project_root = str(Path(__file__).resolve().parent)
    out_dir = os.path.join(project_root, "results", run_name)
    xml_out_dir = os.path.join(out_dir, "pred_xml")
    csv_out_dir = os.path.join(out_dir, "diva_csv")
    os.makedirs(xml_out_dir, exist_ok=True)

    model = UnifiedSegmenter(cfg).to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data["state_dict"])
    model.eval()
    print(
        f"Loaded checkpoint from {checkpoint} "
        f"(epoch {checkpoint_data.get('epoch', '?')}, val_iou={checkpoint_data.get('val_iou', '?')})"
    )

    img_files = sorted(filename for filename in os.listdir(img_dir) if filename.lower().endswith(".jpg"))
    total_lines = 0

    for filename in tqdm(img_files, desc=f"Inference [{split}]"):
        stem = os.path.splitext(filename)[0]
        img_rgb = cv2.cvtColor(cv2.imread(os.path.join(img_dir, filename)), cv2.COLOR_BGR2RGB)
        prob_map = sliding_window_inference(
            model,
            img_rgb,
            crop_size=tcfg["crop_size"],
            device=device,
            use_amp=tcfg.get("amp", True),
        )
        xml_path = os.path.join(xml_out_dir, stem + ".xml")
        total_lines += prob_map_to_pagexml(
            prob_map,
            img_path=os.path.join(img_dir, filename),
            output_xml=xml_path,
            creator=model.creator_name,
            threshold=threshold,
            min_area=min_area,
        )

    print(f"Generated XML for {len(img_files)} pages, total {total_lines} lines.")

    if run_eval:
        run_diva_evaluator(
            pred_xml_dir=xml_out_dir,
            gt_pixel_dir=gt_pix_dir,
            gt_xml_dir=gt_xml_dir,
            img_dir=img_dir,
            out_csv_dir=csv_out_dir,
        )
        summary = summarise_csv_results(csv_out_dir)
        print("\n=== DIVA Evaluation Results ===")
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}")
        print(f"  XML outputs : {xml_out_dir}")
        print(f"  CSV outputs : {csv_out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="public-test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_area", type=int, default=200)
    parser.add_argument("--no_eval", action="store_true")
    arguments = parser.parse_args()

    main(
        cfg_path=arguments.config,
        checkpoint=arguments.checkpoint,
        split=arguments.split,
        threshold=arguments.threshold,
        min_area=arguments.min_area,
        run_eval=not arguments.no_eval,
    )
