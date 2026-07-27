"""DIVA-HisDB text-line pipeline: U-Net prob map -> seam carving -> PAGE XML.

Unlike the U-DIADS ensemble, DIVA-HisDB has no baseline GT, so there is no
ARU-Net fusion step. The text-line instances are recovered purely from the
U-Net main-text-body probability map:

    1. Sliding-window U-Net inference            -> prob map  [H, W]
    2. Threshold + cleanup                        -> foreground binary
    3. Component-based seam carving               -> disconnected text lines
    4. Clip to GT main TextRegion polygon         -> drop marginalia/decoration
    5. Connected components -> polygon + baseline -> PAGE XML (TASK-2 schema)
    6. DIVA Line Segmentation Evaluator (Java)    -> DR / RA / FM

The PAGE XML export mirrors the object-detection converter
(`50_modelling/evaluate_mask2former_archive.py::write_page_xml`): one
`TextRegion` carrying the GT main-text polygon, one `TextLine` per component
with an approximated `Coords` polygon and a bbox-bottom `Baseline`.
"""

import argparse
import csv
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.diva_dataset import _resolve_image_path, _resolve_layout
from evaluate import sliding_window_inference
from models import build_model
from postproc import remove_small_objects, _seam, disconnect_components


PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
ET.register_namespace("", PAGE_NS)
ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")


# ---------------------------------------------------------------------------
# Seam carving: remove_small_objects / _seam / disconnect_components are
# dataset-agnostic and imported from postproc.py (shared with U-DIADS-TL).
# separate_lines_projection below is DIVA-specific (no seam carving).
# ---------------------------------------------------------------------------

def separate_lines_projection(
    mask: np.ndarray,
    merge_width: int = 49,
    merge_height: int = 5,
    line_sigma: float = 8,
    min_line_distance: int = 30,
    peak_frac: float = 0.20,
    valley_ratio: float = 0.5,
    cut_thickness: int = 1,
    min_area: int = 200,
) -> np.ndarray:
    """Simple horizontal-projection line separation (no seam carving).

    The U-Net foreground is fragmented (characters/words as separate blobs), so:

      1. denoise (drop specks),
      2. **merge** fragments into solid line bands with a wide horizontal close
         (``merge_width`` x ``merge_height``) — a small height keeps adjacent
         lines apart, the width bridges word gaps so one line = one component,
      3. for each merged component, find line centres as peaks of the smoothed
         row projection and cut a *straight* horizontal line at the valley
         between consecutive lines, but only at a genuine gap (``valley_ratio``),
      4. denoise again.

    Step 2 is the key difference from seam carving, which never merged the
    fragments and so left hundreds of blobs. Mirrors the object-detection
    exporter's morphological close.
    """
    binary = remove_small_objects((mask > 0).astype(np.uint8), min_area)

    if merge_width > 1 or merge_height > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, merge_width), max(1, merge_height)))
        binary = (cv2.morphologyEx(binary * 255, cv2.MORPH_CLOSE, kernel) > 0).astype(np.uint8)

    result = binary.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue

        y = stats[i, cv2.CC_STAT_TOP]
        x = stats[i, cv2.CC_STAT_LEFT]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        comp = (labels[y:y + h, x:x + w] == i).astype(np.float32)

        proj = gaussian_filter1d(comp.sum(1), line_sigma)
        if proj.max() == 0:
            continue

        peaks, _ = find_peaks(proj, distance=min_line_distance, height=proj.max() * peak_frac)
        if len(peaks) < 2:
            continue

        for r1, r2 in zip(peaks[:-1], peaks[1:]):
            v_row = r1 + int(np.argmin(proj[r1:r2 + 1]))
            if proj[v_row] > valley_ratio * min(proj[r1], proj[r2]):
                continue
            result[y + max(0, v_row - cut_thickness):y + v_row + cut_thickness + 1, x:x + w] = 0

    return remove_small_objects(result, min_area)


# ---------------------------------------------------------------------------
# PAGE XML export (region-clipped, mirrors object-detection write_page_xml)
# ---------------------------------------------------------------------------

def load_main_region_polygon(xml_path: Path) -> Optional[list]:
    """Read the first TextRegion polygon from a DIVA TASK-2 GT XML."""
    if not xml_path.exists():
        return None
    ns = {"ns": PAGE_NS}
    root = ET.parse(xml_path).getroot()
    region = root.find(".//ns:TextRegion", ns)
    if region is None:
        return None
    coords = region.find("ns:Coords", ns)
    if coords is None:
        return None
    points = [tuple(map(int, xy.split(","))) for xy in coords.attrib["points"].split()]
    return points if len(points) >= 3 else None


def create_page_xml(image_name: str, width: int, height: int, creator: str, region_points=None):
    root = ET.Element(
        f"{{{PAGE_NS}}}PcGts",
        {f"{{http://www.w3.org/2001/XMLSchema-instance}}schemaLocation": f"{PAGE_NS} {PAGE_NS}/pagecontent.xsd"},
    )
    meta = ET.SubElement(root, f"{{{PAGE_NS}}}Metadata")
    ET.SubElement(meta, f"{{{PAGE_NS}}}Creator").text = creator
    ET.SubElement(meta, f"{{{PAGE_NS}}}Created").text = datetime.now().isoformat()
    ET.SubElement(meta, f"{{{PAGE_NS}}}LastChange").text = datetime.now().isoformat()
    page = ET.SubElement(
        root,
        f"{{{PAGE_NS}}}Page",
        {"imageFilename": image_name, "imageWidth": str(width), "imageHeight": str(height)},
    )
    region = ET.SubElement(page, f"{{{PAGE_NS}}}TextRegion", {"id": "region_textline", "custom": "0"})
    if region_points is None:
        region_points = [(0, 0), (width, 0), (width, height), (0, height)]
    ET.SubElement(region, f"{{{PAGE_NS}}}Coords", {"points": " ".join(f"{x},{y}" for x, y in region_points)})
    return ET.ElementTree(root), region


def mask_to_polygon(component_mask: np.ndarray, approx_ratio: float,
                    dilate_size: int = 0, shape: str = "rect"):
    """Component -> `TextLine` `Coords` points + bottom-edge `Baseline`.

    ``shape="rect"`` (default) emits the axis-aligned bounding rectangle — a
    clean 4-corner box per line that matches the roughly rectangular DIVA GT
    line polygons and tends to score higher line IoU than a jagged contour.
    ``shape="polygon"`` keeps the filled, approximated outer contour.

    ``dilate_size`` grows the line first so the box/polygon encloses the full
    glyph height (incl. ascenders/descenders the U-Net trims).
    """
    mask = (component_mask > 0).astype(np.uint8)
    if dilate_size and dilate_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        mask = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None, None

    x, y, width, height = cv2.boundingRect(contour)
    baseline = [(x, y + height - 1), (x + width, y + height - 1)]

    if shape == "rect":
        points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        return points, baseline

    epsilon = approx_ratio * cv2.arcLength(contour, closed=True)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    if len(approx) < 3:
        return None, None
    points = [(int(p[0][0]), int(p[0][1])) for p in approx]
    return points, baseline


def write_page_xml(
    refined: np.ndarray,
    image_name: str,
    output_xml: Path,
    region_points,
    creator: str,
    approx_ratio: float,
    min_area: int,
    dilate_size: int = 0,
    shape: str = "rect",
    min_aspect: float = 0.0,
    width_frac: float = 0.0,
) -> int:
    """Line binary mask -> PAGE XML. Returns number of text lines written.

    Each kept component becomes one slightly dilated `TextLine` element
    (``shape`` "rect" bounding box or "polygon" contour). Non-line components
    are dropped: below ``min_area`` (specks), aspect ratio w/h < ``min_aspect``
    (square/tall decoration blobs), or width < ``width_frac`` x the median line
    width (short fragments). ``min_aspect``/``width_frac`` = 0 disables them.
    """
    height, width = refined.shape
    tree, region = create_page_xml(image_name, width, height, creator, region_points)

    merged_mask = (refined > 0).astype(np.uint8) * 255
    if region_points is not None:
        region_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(region_mask, [np.array(region_points, dtype=np.int32)], 255)
        merged_mask = cv2.bitwise_and(merged_mask, region_mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged_mask, connectivity=8)

    # Pre-filter components to line-shaped boxes, then drop short fragments
    # relative to the median line width.
    candidates = []
    for label_id in range(1, num_labels):
        if int(stats[label_id, cv2.CC_STAT_AREA]) < min_area:
            continue
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if min_aspect > 0 and w / max(h, 1) < min_aspect:
            continue
        candidates.append((label_id, w))
    if width_frac > 0 and candidates:
        median_w = float(np.median([w for _, w in candidates]))
        candidates = [(lid, w) for lid, w in candidates if w >= width_frac * median_w]

    kept = 0
    for label_id, _w in candidates:
        component_mask = (labels == label_id).astype(np.uint8) * 255
        polygon, baseline = mask_to_polygon(component_mask, approx_ratio,
                                            dilate_size=dilate_size, shape=shape)
        if polygon is None:
            continue
        text_line = ET.SubElement(region, f"{{{PAGE_NS}}}TextLine", {"id": f"textline_{kept}", "custom": "0"})
        ET.SubElement(text_line, f"{{{PAGE_NS}}}Coords", {"points": " ".join(f"{x},{y}" for x, y in polygon)})
        ET.SubElement(text_line, f"{{{PAGE_NS}}}Baseline", {"points": " ".join(f"{x},{y}" for x, y in baseline)})
        text_equiv = ET.SubElement(text_line, f"{{{PAGE_NS}}}TextEquiv")
        ET.SubElement(text_equiv, f"{{{PAGE_NS}}}Unicode").text = ""
        kept += 1

    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return kept


# ---------------------------------------------------------------------------
# DIVA Java evaluator
# ---------------------------------------------------------------------------

# Columns of interest from the DIVA Line Segmentation Evaluator results.csv.
_DIVA_METRIC_KEYS = [
    "LinesIU", "LinesFMeasure", "LinesRecall", "LinesPrecision",
    "PixelIU", "PixelFMeasure", "PixelPrecision", "PixelRecall",
]


def run_diva_evaluator(
    pred_xml_dir: Path,
    gt_pixel_dir: str,
    gt_xml_dir: str,
    img_dir: str,
    out_csv_dir: Path,
    java_cp: str,
    java_main: str,
) -> dict:
    """Run the Java evaluator per page and average the per-page results.csv.

    The tool appends one row per call to ``results.csv`` inside the prediction
    XML directory (the parent of ``-xp``), so we clear any stale file first and
    read it back afterwards — same contract as the object-detection converter.
    """
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    results_csv = pred_xml_dir / "results.csv"
    if results_csv.exists():
        results_csv.unlink()

    pred_xmls = sorted(p for p in os.listdir(pred_xml_dir) if p.endswith(".xml"))
    for xml_name in tqdm(pred_xmls, desc="DIVA evaluator"):
        stem = os.path.splitext(xml_name)[0]
        gt_xml = os.path.join(gt_xml_dir, stem + ".xml")
        gt_png = os.path.join(gt_pixel_dir, stem + ".png")
        if not (os.path.exists(gt_xml) and os.path.exists(gt_png)):
            print(f"[WARN] missing GT for {stem}, skipping")
            continue
        overlap = _resolve_image_path(img_dir, stem)
        cmd = ["java", "-cp", java_cp, java_main,
               "-igt", gt_png, "-xgt", gt_xml,
               "-xp", str(pred_xml_dir / xml_name), "-overlap", overlap, "-csv"]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] evaluator failed on {stem}: {exc.stderr[:200]}")
            continue
        except FileNotFoundError:
            print("[WARN] java not found; skipping DIVA evaluator")
            return {}

    if not results_csv.exists():
        return {key: float("nan") for key in _DIVA_METRIC_KEYS}

    # keep a copy alongside the other outputs
    (out_csv_dir / "results.csv").write_text(results_csv.read_text())
    metrics = {key: [] for key in _DIVA_METRIC_KEYS}
    with results_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for key in metrics:
                try:
                    value = float(row[key])
                except (KeyError, TypeError, ValueError):
                    continue
                if not np.isnan(value):
                    metrics[key].append(value)

    return {key: (sum(v) / len(v) if v else float("nan")) for key, v in metrics.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_data_root(data_root: str) -> str:
    path = Path(data_root)
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parents[1] / path).resolve())


_SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val",
                  "validation": "val", "test": "test", "public-test": "test"}


# Named post-processing presets. Any explicitly passed CLI flag overrides its preset value.
#   unified    — the DEFAULT: filled contour polygons (the "diva_poly" recipe; CB55 test LineIU 0.947,
#                LineF 0.971 with the 200ep resnet34). CS18 prefers --merge-width 101 --valley-ratio 0.35.
#   playground — dilated bounding boxes (rect) instead of filled polygons; good Line IU too.
PRESETS = {
    "unified":    dict(method="projection", merge_width=75, merge_height=13, line_sigma=8.0,
                       min_line_distance=30, peak_frac=0.20, valley_ratio=0.5, min_area=800,
                       dilate_size=0, shape="polygon", min_aspect=1.5, width_frac=0.3, approx_ratio=0.001),
    "playground": dict(method="projection", merge_width=75, merge_height=11, line_sigma=8.0,
                       min_line_distance=30, peak_frac=0.20, valley_ratio=0.35, min_area=800,
                       dilate_size=11, shape="rect", min_aspect=1.5, width_frac=0.3, approx_ratio=0.001),
}


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None,
                        help="training yaml; optional — falls back to the cfg embedded in the checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manuscript", default=None, help="override cfg manuscript")
    parser.add_argument("--preset", choices=list(PRESETS), default="unified",
                        help="named post-processing preset (explicit flags below override it)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=None)
    parser.add_argument("--dilate-size", type=int, default=None,
                        help="grow each line by this ellipse kernel before boxing (0 = off)")
    parser.add_argument("--shape", choices=["rect", "polygon"], default=None,
                        help="TextLine geometry: axis-aligned rectangle or contour polygon")
    parser.add_argument("--min-aspect", type=float, default=None,
                        help="drop boxes with width/height below this (decoration blobs); 0=off")
    parser.add_argument("--width-frac", type=float, default=None,
                        help="drop boxes narrower than this fraction of median line width; 0=off")
    parser.add_argument("--approx-ratio", type=float, default=0.001)
    parser.add_argument("--clean-min-size", type=int, default=200,
                        help="remove components smaller than this before/after line separation "
                             "(keep at 200: raising it deletes word bits BEFORE the merge and breaks lines)")
    parser.add_argument("--method", choices=["projection", "seam"], default=None,
                        help="line separation: simple horizontal projection or seam carving")
    # line-separation params (preset-controlled unless given)
    parser.add_argument("--line-sigma", type=float, default=None)
    parser.add_argument("--min-line-distance", type=int, default=None)
    parser.add_argument("--peak-frac", type=float, default=None)
    parser.add_argument("--valley-ratio", type=float, default=None)
    parser.add_argument("--merge-width", type=int, default=None,
                        help="horizontal close width to merge line fragments (projection method)")
    parser.add_argument("--merge-height", type=int, default=None,
                        help="horizontal close height; keep small so lines stay apart")
    parser.add_argument("--no-region-clip", action="store_true",
                        help="do not clip predictions to the GT main TextRegion")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--java-cp",
        default="/usr/share/openjfx/lib/*:/home/artur/Thesis/DIVA_Line_Segmentation_Evaluator/out/artifacts/LineSegmentationEvaluator.jar",
    )
    parser.add_argument("--java-main", default="ch.unifr.LineSegmentationEvaluatorTool")
    args = parser.parse_args()

    # fill preset values for every flag the user did not pass explicitly
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    print(f"[preset={args.preset}] method={args.method} shape={args.shape} "
          f"merge={args.merge_width}x{args.merge_height} ls={args.line_sigma} mld={args.min_line_distance} "
          f"pf={args.peak_frac} vr={args.valley_ratio} min_area={args.min_area} dilate={args.dilate_size}")

    if args.config:
        with open(args.config) as handle:
            cfg = yaml.safe_load(handle)
    else:                                   # checkpoints embed their full training cfg
        cfg = torch.load(args.checkpoint, map_location="cpu").get("cfg")
        if cfg is None:
            raise SystemExit("checkpoint has no embedded cfg — pass --config")
    manuscript = args.manuscript or cfg["data"]["manuscript"]
    split_key = _SPLIT_ALIASES[args.split.lower()]
    data_root = _resolve_data_root(cfg["data"]["data_root"])

    split_dirs = _resolve_layout(data_root, manuscript, "diva")
    img_dir = split_dirs[split_key]["img"]
    gt_pix_dir = split_dirs[split_key]["gt"]
    gt_xml_dir = split_dirs[split_key]["xml"]
    if gt_xml_dir is None:
        raise FileNotFoundError(f"No TASK-2 PAGE XML GT for {manuscript} {split_key}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt.get("cfg", cfg)   # build the checkpoint's own arch/encoder
    model = build_model(model_cfg).to(device)
    state = ckpt["state_dict"]
    # back-compat: old SMPUNet stored the net as self.unet; current one as self.net
    state = {("net." + k[len("unet."):] if k.startswith("unet.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, "
          f"val_iou={ckpt.get('val_iou', '?')}) — {manuscript} / {split_key}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        run_name = f"diva_seamcarve_{manuscript}_{Path(args.config).stem}_{split_key}"
        out_dir = repo_root / "99_evaluation" / "01_simple_segmentation" / "diva-hisdb" / "simple_segmentation" / run_name
    xml_dir = out_dir / "pred_xml"
    xml_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(f for f in os.listdir(img_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")))
    total_lines = 0
    for filename in tqdm(img_files, desc=f"Inference [{manuscript}/{split_key}]"):
        stem = os.path.splitext(filename)[0]
        img_bgr = cv2.imread(_resolve_image_path(img_dir, stem))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        prob_map = sliding_window_inference(
            model, img_rgb, crop_size=cfg["training"]["crop_size"],
            device=device, use_amp=cfg["training"].get("amp", True),
        )

        fused = remove_small_objects((prob_map > args.threshold).astype(np.uint8), min_size=args.clean_min_size)
        if args.method == "projection":
            refined = separate_lines_projection(
                fused,
                merge_width=args.merge_width,
                merge_height=args.merge_height,
                line_sigma=args.line_sigma,
                min_line_distance=args.min_line_distance,
                peak_frac=args.peak_frac,
                valley_ratio=args.valley_ratio,
                min_area=args.clean_min_size,
            )
        else:
            refined = remove_small_objects(
                disconnect_components(
                    fused,
                    line_sigma=args.line_sigma,
                    min_line_distance=args.min_line_distance,
                    peak_frac=args.peak_frac,
                    valley_ratio=args.valley_ratio,
                    min_area=args.clean_min_size,
                ),
                min_size=args.clean_min_size,
            )

        region_points = None if args.no_region_clip else load_main_region_polygon(Path(gt_xml_dir) / f"{stem}.xml")
        total_lines += write_page_xml(
            refined, filename, xml_dir / f"{stem}.xml", region_points,
            creator=model.creator_name, approx_ratio=args.approx_ratio,
            min_area=args.min_area, dilate_size=args.dilate_size, shape=args.shape,
            min_aspect=args.min_aspect, width_frac=args.width_frac,
        )

    print(f"Wrote {len(img_files)} PAGE XML files ({total_lines} text lines) -> {xml_dir}")

    if not args.no_eval:
        summary = run_diva_evaluator(
            xml_dir, gt_pix_dir, gt_xml_dir, img_dir, out_dir / "diva_csv",
            java_cp=args.java_cp, java_main=args.java_main,
        )
        with (out_dir / "diva_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
        print("\n=== DIVA Line Segmentation Evaluation ===")
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}")
        print(f"  XML : {xml_dir}")


if __name__ == "__main__":
    main()
