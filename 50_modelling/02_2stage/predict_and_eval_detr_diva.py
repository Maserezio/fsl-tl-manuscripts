"""PAGE-XML export + DIVA Java evaluator scoring for the RF-DETR / RT-DETR
checkpoints trained in rf_detr_vs_rt_detr_cb55.ipynb.

Those checkpoints only produce boxes (no polygon masks), so each detection is
written as an axis-aligned rectangle TextLine (Coords = the 4 box corners,
Baseline = the bottom edge) into a PAGE XML clipped to the full page (CB55 has
no separate main-text-region GT for this split), then scored with the same
`ch.unifr.LineSegmentationEvaluatorTool` jar used by 50_modelling/02_2stage/evaluate.py.

Run after the notebook has produced 80_models/02_2stage/diva-hisdb/detection/
hf_detr_cb55/{rf_detr,rt_detr}/best_model.
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from dataset import create_page_xml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "80_models/02_2stage/diva-hisdb/detection/hf_detr_cb55"
TEST_IMG_DIR = REPO_ROOT / "00_data/DIVA-HisDB/yolo_dataset_CB55/images/test"
GT_PIXEL_DIR = REPO_ROOT / "00_data/DIVA-HisDB/CB55/pixel-level-gt-CB55/pixel-level-gt/public-test"
GT_PAGE_DIR = REPO_ROOT / "00_data/DIVA-HisDB/CB55/PAGE-gt-CB55-TASK-2/TASK-2/public-test"

JAVA_CP = (
    "/usr/share/openjfx/lib/*:"
    "/home/artur/Thesis/DIVA_Line_Segmentation_Evaluator/out/artifacts/LineSegmentationEvaluator.jar"
)
JAVA_MAIN = "ch.unifr.LineSegmentationEvaluatorTool"

MODELS = {"RF-DETR": "rf_detr", "RT-DETR": "rt_detr"}
CONF_THRESHOLD = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict_page_xmls(run_name: str, checkpoint_dir: Path, out_xml_dir: Path):
    image_processor = AutoImageProcessor.from_pretrained(str(checkpoint_dir))
    model = AutoModelForObjectDetection.from_pretrained(str(checkpoint_dir)).to(DEVICE).eval()

    out_xml_dir.mkdir(parents=True, exist_ok=True)
    img_files = sorted(p for p in TEST_IMG_DIR.iterdir() if p.suffix.lower() in (".jpg", ".png"))

    for img_path in tqdm(img_files, desc=f"{run_name}: predicting"):
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        result = image_processor.post_process_object_detection(
            outputs, threshold=CONF_THRESHOLD, target_sizes=[(H, W)]
        )[0]

        root, region = create_page_xml(str(img_path), W, H)
        for i, box in enumerate(result["boxes"].cpu().tolist()):
            x1, y1, x2, y2 = (max(0, min(v, [W, H, W, H][j])) for j, v in enumerate(box))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            coords = f"{x1:.0f},{y1:.0f} {x2:.0f},{y1:.0f} {x2:.0f},{y2:.0f} {x1:.0f},{y2:.0f}"
            baseline = f"{x1:.0f},{y2:.0f} {x2:.0f},{y2:.0f}"
            tl = ET.SubElement(region, "TextLine", {"id": f"textline_{i}", "custom": "0"})
            ET.SubElement(tl, "Coords", {"points": coords})
            ET.SubElement(tl, "Baseline", {"points": baseline})
            ET.SubElement(ET.SubElement(tl, "TextEquiv"), "Unicode").text = ""

        xml_str = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
        (out_xml_dir / f"{img_path.stem}.xml").write_text(xml_str, encoding="utf-8")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_with_java_evaluator(run_name: str, pred_xml_dir: Path) -> pd.DataFrame:
    # The evaluator does NOT print CSV to stdout under -csv (stdout is just
    # log4j DEBUG/INFO chatter) -- it appends one row per page to a shared
    # <output-dir-of-xp>/results.csv. Remove any stale file before the run
    # since it accumulates across invocations.
    results_csv = pred_xml_dir / "results.csv"
    if results_csv.exists():
        results_csv.unlink()

    for xml_path in tqdm(sorted(pred_xml_dir.glob("*.xml")), desc=f"{run_name}: DIVA evaluator"):
        stem = xml_path.stem
        cmd = [
            "java", "-cp", JAVA_CP, JAVA_MAIN,
            "-igt", str(GT_PIXEL_DIR / f"{stem}.png"),
            "-xgt", str(GT_PAGE_DIR / f"{stem}.xml"),
            "-xp", str(xml_path),
            "-overlap", str(TEST_IMG_DIR / f"{stem}.jpg"),
            "-csv",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"[ERROR] evaluator failed on {stem}\n{result.stderr}")

    if not results_csv.exists():
        raise RuntimeError(f"{run_name}: no evaluator output produced at {results_csv}")
    df = pd.read_csv(results_csv)
    return df


def main():
    if not GT_PIXEL_DIR.is_dir() or not GT_PAGE_DIR.is_dir():
        raise FileNotFoundError(f"missing GT dirs: {GT_PIXEL_DIR} / {GT_PAGE_DIR}")

    summary = []
    for run_name, run_dir_name in MODELS.items():
        checkpoint_dir = MODEL_ROOT / run_dir_name / "best_model"
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"{run_name}: no checkpoint at {checkpoint_dir} (run the notebook first)")

        pred_xml_dir = MODEL_ROOT / run_dir_name / "pred_xml_diva"
        predict_page_xmls(run_name, checkpoint_dir, pred_xml_dir)
        df = score_with_java_evaluator(run_name, pred_xml_dir)
        summary.append({
            "model": run_name,
            "LinesIU": df["LinesIU"].mean(),
            "PixelIU": df["PixelIU"].mean(),
            "n_pages": len(df),
        })
        print(f"{run_name}: LinesIU={df['LinesIU'].mean():.4f}  PixelIU={df['PixelIU'].mean():.4f}")

    summary_df = pd.DataFrame(summary).set_index("model")
    summary_df.to_csv(MODEL_ROOT / "diva_eval_summary.csv")
    print("\n=== DIVA Java evaluator summary (test, public-test split) ===")
    print(summary_df)


if __name__ == "__main__":
    main()
