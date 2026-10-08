#!/usr/bin/env python3
"""DP seam for the RT-DETR + BBox U-Net pipeline in the page-count study (CATMuS RT-DETR, random selection).

For one cell (subset, k) the pipeline of predict_and_eval_rtdetr_diva.py is reproduced exactly: same detector,
confidence and width fraction picked on the validation pages (pick_threshold), same k-shot BBox U-Net, same box
filters. Only the export of each box mask differs:
  std            largest contour of the thresholded crop probability (the pipeline; reproduction check)
  seam_t{t}_k{K} dynamic-programming seam of seam_core.seam_polygon on the crop probability
The seam setting (t in {0.5, 0.7}, K in {2, 8, 64}) is selected on the validation pages by mean line FM and applied
once to the test pages.

    .venv/bin/python 99_evaluation/analysis/rtdetr_kshot_seam.py SUB K      (K = 20: the full-partition RQ1 model)
    -> 99_evaluation/analysis/rtdetr_kshot_seam/<SUB>_k<K>/result.json
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import importlib, json, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "99_evaluation/analysis/rtdetr_kshot_seam"
JAR = Path.home() / "Thesis/DIVA_Line_Segmentation_Evaluator/out/artifacts/LineSegmentationEvaluator.jar"
GRID = [(t, K) for t in (0.5, 0.7) for K in (2, 8, 64)]
SPLITS = {"val": "validation", "test": "public-test"}


def job_of(sub, k):
    m = json.loads((REPO / "99_evaluation/summaries/rq2/shot_selection_manifest.json").read_text())
    return next(c["job"] for c in m["cells"] if c["method"] == "random" and c["subset"] == sub and c["k"] == k)


def setup(sub, k):
    os.environ.update(SUBSET=sub, ARM="tversky", TUNE_THRESHOLD_ON_VAL="1")
    if k == 20:
        os.environ["RUN_NAME"] = "rtdetr_convnext_tiny_catmus"
    else:
        job = job_of(sub, k)
        os.environ.update(RUN_NAME=f"sel_catmus_{job}", K_SHOT=str(k),
                          SEG_KSHOT_ROOT=f"80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/{sub}/{job}/k{k}/tversky")
    sys.path[:0] = [str(REPO / "50_modelling/instance_segmentation/rtdetr_bbox_unet"), str(REPO / "50_modelling/instance_segmentation/dp_seam")]
    return importlib.import_module("predict_and_eval_rtdetr_diva"), importlib.import_module("seam_core")


def diva_metrics(sub, split, stem, xml):
    d = REPO / "00_data/DIVA-HisDB" / sub
    with tempfile.TemporaryDirectory() as cwd:
        shutil.copy(xml, Path(cwd) / f"{stem}.xml")
        subprocess.run(["java", "-Djava.awt.headless=true", "-cp", f"/usr/share/openjfx/lib/*:{JAR}",
                        "ch.unifr.LineSegmentationEvaluatorTool",
                        "-igt", str(d / f"pixel-level-gt-{sub}/pixel-level-gt/{SPLITS[split]}/{stem}.png"),
                        "-xgt", str(d / f"PAGE-gt-{sub}-TASK-2/TASK-2/{SPLITS[split]}/{stem}.xml"),
                        "-xp", str(Path(cwd) / f"{stem}.xml"), "-csv"], cwd=cwd, capture_output=True, text=True, check=True)
        head, vals = (Path(cwd) / "results.csv").read_text().splitlines()[:2]
        head, vals = head.split(","), vals.split(",")
        return dict(zip(head[1:], [float(v) for v in vals[-(len(head) - 1):]]))


def write_xml(P, img_path, W, H, polys, path):
    root, region = P.create_page_xml(str(img_path), W, H)
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    for i, poly in enumerate(polys):
        tl = ET.SubElement(region, "TextLine", {"id": f"textline_{i}", "custom": "0"})
        ET.SubElement(tl, "Coords", {"points": " ".join(f"{int(x)},{int(y)}" for x, y in poly)})
        base = P.polygon_to_baseline([tuple(map(int, p)) for p in poly])
        ET.SubElement(tl, "Baseline", {"points": " ".join(f"{x},{y}" for x, y in base)})
        ET.SubElement(ET.SubElement(tl, "TextEquiv"), "Unicode").text = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  "))


def main():
    sub, k = sys.argv[1], int(sys.argv[2])
    P, SC = setup(sub, k)
    import torch
    from transformers import AutoImageProcessor
    out = OUT / f"{sub}_k{k}"
    res_path = out / "result.json"
    if res_path.exists():
        print("done already", res_path); return
    model = P.load_detector(P.MODEL_DIR).to(P.DEVICE)
    proc = AutoImageProcessor.from_pretrained(P.MODEL_DIR)
    thr, width = P.pick_threshold(model, proc)
    seg = P.load_segm_model(P.SEG_WEIGHTS, P.SEG_ENCODER, P.SEG_ARCH, P.DEVICE)
    data = REPO / "00_data/DIVA-HisDB" / sub
    variants = ["std"] + [f"seam_t{t}_k{K}" for t, K in GRID]
    scores = {}
    for split in ("val", "test"):
        img_dir = data / f"img-{sub}/img/{SPLITS[split]}"
        page_dir = data / f"PAGE-gt-{sub}-TASK-2/TASK-2/{SPLITS[split]}"
        run = variants if split == "val" else ["std", best]
        stems = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            stem = img_path.stem
            if not (page_dir / f"{stem}.xml").exists():
                continue
            stems.append(stem)
            image = cv2.imread(str(img_path))
            boxes, W, H = P.detect(model, proc, str(img_path), thr)
            boxes = P._remove_overlapping_rows(P.region_filter(boxes, str(page_dir / f"{stem}.xml"), width))
            polys = {v: [] for v in run}
            for box in boxes:
                x1, y1, x2, y2 = (int(v) for v in box)
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[y1:y2, x1:x2]
                rgb = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (P.RESIZE_W, P.RESIZE_H))
                with torch.no_grad():
                    x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(P.DEVICE) / 255.0
                    prob = cv2.resize(torch.sigmoid(seg(x))[0, 0].cpu().numpy(), (x2 - x1, y2 - y1))
                # the pipeline's mask and its foreground filter decide whether the box is exported at all
                mask = P.segment_crop(crop, seg, P.RESIZE_W, P.RESIZE_H, P.DEVICE, P.BIN_THRESH)
                fg = float((mask > 0).mean())
                if fg > 0.95 or fg < 0.005:
                    continue
                for v in run:
                    if v == "std":
                        poly = P.mask_to_polygon(mask, x1, y1)
                    else:
                        t, K = float(v.split("_t")[1].split("_k")[0]), int(v.split("_k")[1])
                        sp = SC.seam_polygon(prob, t, K)
                        poly = None if sp is None else [(int(px) + x1, int(py) + y1) for px, py in sp]
                    if poly is not None and len(poly) >= 3:
                        polys[v].append(poly)
            for v in run:
                write_xml(P, img_path, W, H, polys[v], out / split / v / f"{stem}.xml")
            print(split, stem, len(boxes), "boxes", flush=True)
        for v in run:
            with ThreadPoolExecutor(6) as ex:
                rows = list(ex.map(lambda st: diva_metrics(sub, split, st, out / split / v / f"{st}.xml"), stems))
            scores.setdefault(split, {})[v] = {m: round(100 * float(np.mean([r[m] for r in rows])), 2)
                                               for m in ("LinesFMeasure", "LinesIU", "PixelIU", "MatchedPixelIU")}
            print(split, v, scores[split][v]["LinesFMeasure"], flush=True)
        if split == "val":
            best = max((v for v in variants if v != "std"), key=lambda v: scores["val"][v]["LinesFMeasure"])
            print("selected on val:", best, flush=True)
    res = {"subset": sub, "k": k, "confidence": thr, "width_fraction": width, "selected": best, "scores": scores}
    res_path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["scores"]["test"]))


if __name__ == "__main__":
    main()
