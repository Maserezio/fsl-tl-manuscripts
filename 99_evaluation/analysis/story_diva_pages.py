#!/usr/bin/env python3
"""Per-page DIVA-HisDB test FM of the four line representations (ConvNeXt-Tiny, full training partition):
U-Net (DINOv3 LVD-1689M), two-stage RT-DETR + crop U-Net (CATMuS), Mask R-CNN (CATMuS), center-line model
(cBAD-distilled encoder, band-limited decoding selected on validation). Used to pick the story figure page.

    .venv/bin/python 99_evaluation/analysis/story_diva_pages.py [render SUB STEM]
render: writes the evaluator visualization overlapped with the page image for each method to
99_evaluation/analysis/story_figures/<STEM>_<method>-overlap.png
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import csv, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "99_evaluation"
DIVA = ROOT / "00_data/DIVA-HisDB"
JAR = Path.home() / "Thesis/DIVA_Line_Segmentation_Evaluator/out/artifacts/LineSegmentationEvaluator.jar"
OUT = ROOT / "99_evaluation/analysis/story_figures"
SUBS = ["CB55", "CS18", "CS863"]


def pred_dir(method, sub):
    if method == "unet":
        return EV / f"semantic_segmentation/unet/diva-hisdb/lines_eval/unet_tu-convnext_tiny.dinov3_lvd1689m_diva_{sub}/pred_xml"
    if method == "twostage":
        base = EV / "instance_segmentation/rtdetr_bbox_unet/diva-hisdb/rtdetr_hf"
        return (base if sub == "CB55" else base / sub) / "rtdetr_convnext_tiny_catmus/pred_xml"
    if method == "maskrcnn":
        # score threshold selected on validation (differs per subset)
        return next((EV / f"instance_segmentation/mask_rcnn/diva-hisdb/{sub}/maskrcnn_convnext_tiny_catmus_704").glob("diva_test_s*"))
    return ROOT / f"99_evaluation/semantic_segmentation/center_line/diva_cbad/pred/test/{sub}_full/band1.0_inside"


METHODS = ["unet", "twostage", "maskrcnn", "centerline"]


def run(sub, stem, xml, out=None):
    with tempfile.TemporaryDirectory() as cwd:
        local = Path(cwd) / f"{stem}.xml"
        shutil.copy(xml, local)
        args = ["java", "-Djava.awt.headless=true", "-cp", f"/usr/share/openjfx/lib/*:{JAR}",
                "ch.unifr.LineSegmentationEvaluatorTool",
                "-igt", str(DIVA / sub / f"pixel-level-gt-{sub}/pixel-level-gt/public-test/{stem}.png"),
                "-xgt", str(DIVA / sub / f"PAGE-gt-{sub}-TASK-2/TASK-2/public-test/{stem}.xml"),
                "-xp", str(local), "-csv"]
        if out:
            args += ["-overlap", str(next((DIVA / sub / f"img-{sub}/img/public-test").glob(stem + ".*")))]
        subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
        lines = (Path(cwd) / "results.csv").read_text().splitlines()
        head, vals = lines[0].split(","), lines[1].split(",")
        res = dict(zip(head[1:], [float(v) for v in vals[-(len(head) - 1):]]))
        if out:
            for f in Path(cwd).glob("*.png"):
                kind = "overlap" if f.name.endswith("-overlap.png") else "visualization"
                shutil.copy(f, out.parent / f"{out.name}-{kind}.png")
        return res


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        sub, stem = sys.argv[2], sys.argv[3]
        OUT.mkdir(parents=True, exist_ok=True)
        for m in METHODS:
            r = run(sub, stem, pred_dir(m, sub) / f"{stem}.xml", out=OUT / f"{stem}_{m}")
            print(m, {k: round(100 * r[k], 2) for k in ("LinesFMeasure", "LinesRecall", "LinesPrecision", "PixelIU")})
        return
    jobs = []
    for sub in SUBS:
        stems = sorted(p.stem for p in (DIVA / sub / f"pixel-level-gt-{sub}/pixel-level-gt/public-test").glob("*.png"))
        for stem in stems:
            for m in METHODS:
                jobs.append((sub, stem, m))
    with ThreadPoolExecutor(int(os.environ.get("DIVA_WORKERS", "6"))) as ex:
        res = list(ex.map(lambda j: run(j[0], j[1], pred_dir(j[2], j[0]) / f"{j[1]}.xml")["LinesFMeasure"], jobs))
    table = {}
    for (sub, stem, m), fm in zip(jobs, res):
        table.setdefault((sub, stem), {})[m] = round(100 * fm, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_page_fm.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subset", "page"] + METHODS)
        for (sub, stem), r in table.items():
            w.writerow([sub, stem] + [r[m] for m in METHODS])
            print(f"{sub:6s} {stem[-14:]:14s} " + " ".join(f"{r[m]:6.1f}" for m in METHODS))
    for sub in SUBS:
        rows = [r for (s, _), r in table.items() if s == sub]
        print(sub, "mean", " ".join(f"{sum(r[m] for r in rows) / len(rows):6.2f}" for m in METHODS))


if __name__ == "__main__":
    main()
