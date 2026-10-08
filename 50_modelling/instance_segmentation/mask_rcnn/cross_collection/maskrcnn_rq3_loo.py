#!/usr/bin/env python3
"""RQ3 pilot: leave-one-dataset-out k-shot Mask R-CNN, scored with the DIVA evaluator.

Same recipe as the DIVA-HisDB matrix (704 px, 30 epochs, lr 1e-3 / 1e-4 backbone,
AdamW + cosine, AMP) and the same CATMuS-initialised ConvNeXt-Tiny arm. Training and
validation pages come from the source collections (prepare_rq3_loo.py, PCA max-min);
validation only picks the score and mask thresholds, and every page of the held-out
dataset is scored once with the official DIVA Line Segmentation Evaluator on exported
PAGE XML. No target-domain fine-tuning takes place.

    python 50_modelling/instance_segmentation/mask_rcnn/cross_collection/maskrcnn_rq3_loo.py --data-root 00_data/RQ3/loo_CHAMDoc_k3
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "50_modelling/instance_segmentation/mask_rcnn"))
sys.path.insert(0, str(REPO / "50_modelling/instance_segmentation/rtdetr_bbox_unet"))

from dataset import mask_to_polygon, polygon_to_baseline  # noqa: E402
from maskrcnn_diva import (  # noqa: E402
    PAGE_NS,
    XSI_NS,
    DivaCocoLines,
    build_model,
    collate,
    load_init_checkpoint,
    resolve_init_checkpoint,
    serializable_args,
    train_one_epoch,
)
from maskrcnn_diva import SCALE_JITTER  # noqa: E402  FIXED: recorded in the summary
from maskrcnn_udiads import (  # noqa: E402
    POST_LOG, POST_MERGE, POST_MIN_AREA, POST_SHAPE_REGION,
    configure_dense_inference, prediction_to_instances,
)

JAR = Path.home() / "Thesis/DIVA_Line_Segmentation_Evaluator/out/artifacts/LineSegmentationEvaluator.jar"
SCORE_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
MASK_GRID = (0.3, 0.5, 0.7)
# Cheaper grid for sweeps (--fast-grid): the extremes of both axes.
FAST_SCORE_GRID = (0.3, 0.5, 0.7, 0.9)
FAST_MASK_GRID = (0.3, 0.5)
DIVA_WORKERS = int(os.environ.get("DIVA_WORKERS", "6"))
# FIXED: RQ3 robustness ablation. Every default reproduces the previous runs; the ablation
# overrides them through the environment so that each run records its own configuration.
# (a) False keeps the CATMuS-pretrained RPN (ratios 0.05-1.0, sizes 32-512) instead of a new one.
REPLACE_ANCHORS = os.environ.get("RQ3_REPLACE_ANCHORS", "1") == "1"
# (c) ROI size of the mask branch (Torchvision default 14).
MASK_ROI_SIZE = int(os.environ.get("RQ3_MASK_ROI", "14"))
# (d) box-NMS IoU values added to the source-validation grid; empty = keep the model default.
NMS_GRID = tuple(float(v) for v in os.environ.get("RQ3_NMS_GRID", "").split(",") if v) or (None,)
# (e) frozen ConvNeXt stages (stem included whenever > 0) and early stopping on source val FM.
FREEZE_STAGES = int(os.environ.get("RQ3_FREEZE_STAGES", "0"))
ES_PATIENCE = int(os.environ.get("RQ3_ES_PATIENCE", "0"))   # 0 = train all epochs, keep the last
ES_EVAL_EVERY = 5                                            # epochs between validation checks
ES_SCORE, ES_MASK = 0.5, 0.5                                 # fixed thresholds for the checks
METRICS = ["PixelIU", "LinesIU", "LinesRecall", "LinesPrecision", "LinesFMeasure"]
# Label-free scale normalisation at inference. A first pass estimates the median line
# height of the page from the model's own predictions; if it exceeds the median line
# height of the *training* pages, the page is shrunk by ref/est (at most to 0.4x) and
# predicted again. Pages are never enlarged: the model input size is fixed. No target
# annotation is used. RQ3_SCALE_REF_ROOT names the training data root when a checkpoint
# is evaluated on another collection (default: the --data-root itself).
SCALE_NORM = os.environ.get("RQ3_SCALE_NORM", "0") == "1"
SCALE_NORM_MIN = 0.4
SCALE_NORM_SKIP = 0.9          # ratios above this leave the page unchanged
SCALE_REF = None               # set in main(): median training line height on the canvas


def median_line_height_train(root: Path, size: int) -> float:
    """Median short side of the min-area rectangle of each training line, on the canvas."""
    coco = json.loads((root / "coco_instances/train.json").read_text())
    fit = {im["id"]: min(size / im["width"], size / im["height"]) for im in coco["images"]}
    heights = []
    for a in coco["annotations"]:
        for poly in a.get("segmentation", []):
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if len(pts) >= 3:
                heights.append(min(cv2.minAreaRect(pts)[1]) * fit[a["image_id"]])
    return float(np.median(heights))


def median_line_height_pred(instances: np.ndarray, fit: float) -> float | None:
    heights = []
    for label in range(1, int(instances.max()) + 1):
        ys, xs = np.nonzero(instances == label)
        if len(xs) < 20:
            continue
        pts = np.stack([xs, ys], 1).astype(np.float32)
        heights.append(min(cv2.minAreaRect(pts)[1]) * fit)
    return float(np.median(heights)) if heights else None


def make_loader(data_root, split, size, augment):
    data = DivaCocoLines(data_root / "coco_instances", data_root / "images", split, size, augment)
    return data, DataLoader(data, batch_size=1, shuffle=augment, num_workers=1,
                            pin_memory=True, collate_fn=collate)


def page_xml(meta, instances, out_path):
    ET.register_namespace("", PAGE_NS)
    ET.register_namespace("xsi", XSI_NS)
    root = ET.Element(f"{{{PAGE_NS}}}PcGts",
                      {f"{{{XSI_NS}}}schemaLocation": f"{PAGE_NS} {PAGE_NS}/pagecontent.xsd"})
    md = ET.SubElement(root, f"{{{PAGE_NS}}}Metadata")
    ET.SubElement(md, f"{{{PAGE_NS}}}Creator").text = "Mask R-CNN 3-shot (RQ3 pilot)"
    now = datetime.now(timezone.utc).isoformat()
    ET.SubElement(md, f"{{{PAGE_NS}}}Created").text = now
    ET.SubElement(md, f"{{{PAGE_NS}}}LastChange").text = now
    page = ET.SubElement(root, f"{{{PAGE_NS}}}Page", {
        "imageFilename": Path(meta["path"]).name,
        "imageWidth": str(int(meta["orig_w"])), "imageHeight": str(int(meta["orig_h"]))})
    region = ET.SubElement(page, f"{{{PAGE_NS}}}TextRegion", {"id": "region_textline"})
    h, w = int(meta["orig_h"]), int(meta["orig_w"])
    ET.SubElement(region, f"{{{PAGE_NS}}}Coords",
                  {"points": f"0,0 {w},0 {w},{h} 0,{h}"})
    written = 0
    for label in range(1, int(instances.max()) + 1):
        mask = (instances == label).astype(np.uint8) * 255
        if mask.sum() == 0:
            continue
        polygon = mask_to_polygon(mask, 0, 0)
        if polygon is None or len(polygon) < 3:
            continue
        baseline = polygon_to_baseline(polygon)
        if len(baseline) < 2:
            continue
        line = ET.SubElement(region, f"{{{PAGE_NS}}}TextLine", {"id": f"line_{written}"})
        ET.SubElement(line, f"{{{PAGE_NS}}}Coords",
                      {"points": " ".join(f"{x},{y}" for x, y in polygon)})
        ET.SubElement(line, f"{{{PAGE_NS}}}Baseline",
                      {"points": " ".join(f"{x},{y}" for x, y in baseline)})
        written += 1
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return written


@torch.inference_mode()
def export_split(model, loader, device, split, out_dir, score_threshold, mask_threshold):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    counts = []
    size = loader.dataset.size
    assert not loader.dataset.augment, "evaluation loader must not augment"
    for images, _, metadata in loader:
        meta = metadata[0]
        # FIXED: the evaluation input is the unchanged fit of the page into size x size.
        fit = min(size / meta["orig_w"], size / meta["orig_h"])
        assert tuple(images[0].shape) == (3, size, size)
        assert (meta["new_w"], meta["new_h"]) == (max(1, round(meta["orig_w"] * fit)),
                                                  max(1, round(meta["orig_h"] * fit)))
        with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model([images[0].to(device)])[0]
        instances = prediction_to_instances(output, meta, score_threshold, mask_threshold)
        if SCALE_NORM and SCALE_REF is not None:
            est = median_line_height_pred(instances, fit)
            ratio = 1.0 if est is None else max(SCALE_NORM_MIN, min(1.0, SCALE_REF / est))
            if ratio < SCALE_NORM_SKIP:
                content = images[0][:, :meta["new_h"], :meta["new_w"]]
                h2, w2 = max(1, round(meta["new_h"] * ratio)), max(1, round(meta["new_w"] * ratio))
                small = torch.nn.functional.interpolate(content[None], size=(h2, w2), mode="bilinear",
                                                        align_corners=False, antialias=True)[0]
                canvas = torch.ones_like(images[0])
                canvas[:, :h2, :w2] = small
                meta2 = {**meta, "new_h": h2, "new_w": w2}
                with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    output = model([canvas.to(device)])[0]
                instances = prediction_to_instances(output, meta2, score_threshold, mask_threshold)
        counts.append(page_xml(meta, instances, out_dir / f"{Path(meta['path']).stem}.xml"))
    return counts


def _diva_page(data_root, split, xml_path, overlap):
    """Score one page in its own directory. The evaluator writes results.csv (and a
    visualisation PNG) next to the prediction, so a copy of it is scored there."""
    stem = xml_path.stem
    with tempfile.TemporaryDirectory(dir=xml_path.parent) as cwd:
        local = Path(cwd) / xml_path.name
        shutil.copy2(xml_path, local)
        run = subprocess.run([
            "java", "-Djava.awt.headless=true", "-cp", f"/usr/share/openjfx/lib/*:{JAR}",
            "ch.unifr.LineSegmentationEvaluatorTool",
            "-igt", str(next((data_root / "pixel-gt" / split).glob(f"{stem}.*"))),
            "-xgt", str(data_root / "page-gt" / split / f"{stem}.xml"),
            "-xp", str(local),
            # The overlap/visualisation images are several hundred MB per split and are
            # not needed for the metrics; writing them ran the machine out of memory.
            *(["-overlap", str(next((data_root / "images" / split).glob(f"{stem}.*")))] if overlap else []),
            "-csv",
        ], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if run.returncode:
            raise RuntimeError(f"{stem}: {run.stderr[-400:]}")
        return (Path(cwd) / "results.csv").read_text().splitlines()


def diva_eval(data_root, split, xml_dir, overlap=False):
    results = xml_dir / "results.csv"
    if results.exists():
        results.unlink()
    pages = sorted(xml_dir.glob("*.xml"))
    # One JVM per page is CPU-bound and independent, so pages are scored concurrently.
    with ThreadPoolExecutor(max_workers=DIVA_WORKERS) as pool:
        outputs = list(pool.map(lambda p: _diva_page(data_root, split, p, overlap), pages))
    header = next((o[0] for o in outputs if o), "")
    results.write_text("\n".join([header] + [line for o in outputs for line in o[1:]]) + "\n")
    # The evaluator does not quote filenames. Some German folio names contain commas,
    # so DictReader shifts every metric column. The five metrics are always the final
    # five comma-separated fields; recover them from the right in that case.
    rows = []
    with results.open(newline="") as fh:
        header = next(fh, "").rstrip("\r\n").split(",")
        for line in fh:
            fields = line.rstrip("\r\n").split(",")
            if len(fields) == len(header):
                rows.append(dict(zip(header, fields)))
            elif len(fields) >= len(header):
                value_count = len(header) - 1
                rows.append({"filename": ",".join(fields[:-value_count]),
                             **dict(zip(header[1:], fields[-value_count:]))})
    rows = list({r["filename"]: r for r in rows}.values())
    # A page with no matched line leaves NaN in the matched-pixel columns; those pages
    # still count for the line metrics, so the mean skips the missing values only.
    return rows, {m: float(np.nanmean([float(r[m]) for r in rows])) for m in METRICS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True,
                        help="prepare_rq3_loo.py output, e.g. 00_data/RQ3/loo_CHAMDoc_k3")
    parser.add_argument("--arm", default="convnext_tiny_catmus")
    parser.add_argument("--image-size", type=int, default=704)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--preserve-rpn", action="store_true",
                        help="retain the pretrained proposal head during fine-tuning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero-shot", action="store_true", dest="zero_shot",
                        help="no fine-tuning at all: score the CATMuS pretrain as it is "
                             "(thresholds are still picked on the source validation pages)")
    parser.add_argument("--skip-test", action="store_true",
                        help="stop after the validation grid (sweeping shots/initialisations)")
    parser.add_argument("--fast-grid", action="store_true", help="use FAST_*_GRID thresholds")
    parser.add_argument("--fixed-score", type=float, default=None,
                        help="skip validation calibration and use this score threshold")
    parser.add_argument("--fixed-mask", type=float, default=None,
                        help="skip validation calibration and use this mask threshold")
    parser.add_argument("--fixed-nms", type=float, default=None,  # FIXED: (d)
                        help="box-NMS IoU selected on the source validation pages")
    parser.add_argument("--clean-predictions", action="store_true",
                        help="remove exported PAGE predictions after metrics are saved")
    parser.add_argument("--clean-checkpoint", action="store_true",
                        help="remove the trained checkpoint after metrics are saved")
    parser.add_argument("--eval-checkpoint", type=Path, default=None,
                        help="load a completed run checkpoint and perform evaluation only")
    parser.add_argument("--result-suffix", default="",
                        help="append a unique suffix to output paths for evaluation diagnostics")
    args = parser.parse_args()
    if (args.fixed_score is None) != (args.fixed_mask is None):
        parser.error("--fixed-score and --fixed-mask must be supplied together")
    score_grid = FAST_SCORE_GRID if args.fast_grid else SCORE_GRID
    mask_grid = FAST_MASK_GRID if args.fast_grid else MASK_GRID

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root if args.data_root.is_absolute() else REPO / args.data_root
    manifest = json.loads((data_root / "coco_instances/split_manifest.json").read_text())
    global SCALE_REF
    if SCALE_NORM:
        ref_root = Path(os.environ.get("RQ3_SCALE_REF_ROOT", str(data_root)))
        ref_root = ref_root if ref_root.is_absolute() else REPO / ref_root
        SCALE_REF = median_line_height_train(ref_root, args.image_size)
        print(f"scale normalisation: reference line height {SCALE_REF:.1f}px from {ref_root}", flush=True)
    tag = f"{data_root.name}_maskrcnn_{args.arm}_{args.image_size}"
    tag += "_zeroshot" if args.zero_shot else f"_k{manifest['k']}"
    if args.result_suffix:
        tag += f"_{args.result_suffix}"
    run_dir = REPO / "80_models/instance_segmentation/mask_rcnn/cross_collection" / tag
    eval_dir = REPO / "99_evaluation/instance_segmentation/mask_rcnn/cross_collection" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    fixed_thresholds = args.fixed_score is not None
    if fixed_thresholds and args.zero_shot:
        train_data = []
        train_loader = None
        val_loader = None
    else:
        train_data, train_loader = make_loader(data_root, "train", args.image_size, True)
        val_loader = None if fixed_thresholds else make_loader(
            data_root, "val", args.image_size, False)[1]
    _, test_loader = make_loader(data_root, "test", args.image_size, False)
    print(f"pages train/val/test = {len(train_data)}/{0 if val_loader is None else len(val_loader)}/"
          f"{len(test_loader)}", flush=True)

    model = build_model(args.arm, args.image_size, mask_roi_size=MASK_ROI_SIZE)  # FIXED: (c)
    init = resolve_init_checkpoint(args.arm, None)
    if init is not None:
        load_init_checkpoint(model, init)
    # The dense-inference helper swaps in a fresh RPN head with U-DIADS anchors. That is
    # fine when the run fine-tunes it, but in zero-shot mode it would discard the
    # pretrained proposal head and predict nothing, so the anchors are left alone.
    configure_dense_inference(model, args.max_detections,  # FIXED: (a) REPLACE_ANCHORS
                              replace_anchors=REPLACE_ANCHORS and not (args.zero_shot or args.preserve_rpn))
    default_nms = model.roi_heads.nms_thresh
    if FREEZE_STAGES:  # FIXED: (e) freeze the ConvNeXt stem and the first stages
        frozen = ("stem_",) + tuple(f"stages_{i}." for i in range(FREEZE_STAGES))
        for name, param in model.backbone.encoder.named_parameters():
            if name.startswith(frozen):
                param.requires_grad_(False)
    if args.eval_checkpoint is not None:
        checkpoint = (args.eval_checkpoint if args.eval_checkpoint.is_absolute()
                      else REPO / args.eval_checkpoint)
        load_init_checkpoint(model, checkpoint)
        print(f"evaluation-only reload: {checkpoint}", flush=True)
    model.to(device)

    if args.zero_shot:
        print("zero-shot: the pretrained weights are scored without fine-tuning", flush=True)
    backbone = list(model.backbone.parameters())
    ids = {id(p) for p in backbone}
    other = [p for p in model.parameters() if id(p) not in ids]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.lr_backbone},
                                   {"params": other, "lr": args.lr}],
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    best_state, best_fm, stale, best_epoch = None, -1.0, 0, None
    for epoch in range(1, (0 if args.zero_shot or args.eval_checkpoint is not None
                           else args.epochs) + 1):
        loss, _ = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        print(f"epoch={epoch:03d} loss={loss:.4f}", flush=True)
        # FIXED: (e) best-epoch selection / early stopping on source-validation FM.
        if ES_PATIENCE and (epoch % ES_EVAL_EVERY == 0 or epoch == args.epochs):
            es_loader = make_loader(data_root, "val", args.image_size, False)[1]
            d = eval_dir / "_es_val"
            export_split(model, es_loader, device, "val", d, ES_SCORE, ES_MASK)
            fm = diva_eval(data_root, "val", d)[1]["LinesFMeasure"]
            shutil.rmtree(d, ignore_errors=True)
            print(f"[es] epoch={epoch} val FM={fm:.4f}", flush=True)
            if fm > best_fm:
                best_fm, stale, best_epoch = fm, 0, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= ES_PATIENCE:
                    print(f"[es] stop at epoch {epoch}; best epoch {best_epoch}", flush=True)
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    if not args.zero_shot and args.eval_checkpoint is None:
        torch.save({"model": model.state_dict(), "args": serializable_args(args),
                    "epoch": args.epochs}, run_dir / "best.pt")

    # Thresholds on validation only.
    if fixed_thresholds:
        best = {"score_threshold": args.fixed_score, "mask_threshold": args.fixed_mask,
                "nms_threshold": args.fixed_nms, **{m: None for m in METRICS}}
        print(f"fixed thresholds: score={args.fixed_score} mask={args.fixed_mask}", flush=True)
    else:
        grid = []
        for nms in NMS_GRID:  # FIXED: (d) box NMS in the grid; None = model default
          model.roi_heads.nms_thresh = default_nms if nms is None else nms
          for score in score_grid:
            for mask_t in mask_grid:
                d = eval_dir / f"val_s{score:g}_m{mask_t:g}"
                export_split(model, val_loader, device, "val", d, score, mask_t)
                _, mean = diva_eval(data_root, "val", d)
                grid.append({"score_threshold": score, "mask_threshold": mask_t,
                             "nms_threshold": nms, **mean})
                print(f"[val] nms={nms} s={score} m={mask_t} FM={mean['LinesFMeasure']:.4f} "
                      f"PIU={mean['PixelIU']:.4f}", flush=True)
        best = max(grid, key=lambda r: (r["LinesFMeasure"], r["PixelIU"]))
        with (eval_dir / "validation_grid.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(grid[0]))
            writer.writeheader()
            writer.writerows(grid)
        print(f"[val] selected {best}", flush=True)
    # FIXED: the ablation configuration of this run, recorded in every summary.
    config = {"replace_anchors": REPLACE_ANCHORS, "scale_jitter": SCALE_JITTER,
              "mask_roi_size": MASK_ROI_SIZE, "nms_grid": list(NMS_GRID),
              "freeze_stages": FREEZE_STAGES, "es_patience": ES_PATIENCE,
              "best_epoch": best_epoch, "post_merge": POST_MERGE,
              "post_min_area": POST_MIN_AREA, "post_shape_region": POST_SHAPE_REGION}
    if args.skip_test:
        summary = {
            "holdout": manifest.get("holdout", manifest.get("dataset")), "arm": args.arm,
            "shots": 0 if args.zero_shot else manifest["k"], "zero_shot": bool(args.zero_shot),
            "image_size": args.image_size, "epochs": args.epochs,
            "score_threshold": best["score_threshold"], "mask_threshold": best["mask_threshold"],
            "nms_threshold": best.get("nms_threshold"), "config": config,
            "validation": {m: best[m] for m in METRICS}, "test": None,
        }
        (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    nms = best.get("nms_threshold")  # FIXED: (d)
    model.roi_heads.nms_thresh = default_nms if nms is None else nms
    POST_LOG.clear()
    test_dir = eval_dir / "test_pred_xml"
    counts = export_split(model, test_loader, device, "test", test_dir,
                          best["score_threshold"], best["mask_threshold"])
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rows, mean = diva_eval(data_root, "test", test_dir)
    summary = {
        "holdout": manifest.get("holdout", manifest.get("dataset")), "arm": args.arm,
            "shots": 0 if args.zero_shot else manifest["k"], "zero_shot": bool(args.zero_shot),
        "train_pages": manifest.get("splits", {}).get("train", []),
        "pool_size": manifest.get("pool_size", 0),
        "selection": manifest.get("selection", manifest.get("split_policy")),
        "image_size": args.image_size, "epochs": args.epochs,
        "preserve_rpn": bool(args.preserve_rpn),
        "score_threshold": best["score_threshold"], "mask_threshold": best["mask_threshold"],
        "nms_threshold": best.get("nms_threshold"),
        "config": config,  # FIXED: ablation configuration
        "evaluator": "DIVA Line Segmentation Evaluator",
        "test_pages": len(rows), "mean_predicted_lines": float(np.mean(counts)),
        "validation": None if fixed_thresholds else {m: best[m] for m in METRICS},
        "test": mean,
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if POST_LOG:  # FIXED: per-page counts of every post-processing rule
        with (eval_dir / "postprocess_log.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(POST_LOG[0]))
            writer.writeheader()
            writer.writerows(POST_LOG)
    if args.clean_predictions:
        shutil.rmtree(test_dir, ignore_errors=True)
    if args.clean_checkpoint and not args.zero_shot:
        shutil.rmtree(run_dir, ignore_errors=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
