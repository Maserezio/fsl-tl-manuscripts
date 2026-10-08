#!/usr/bin/env python3
"""Pixel metrics of Mask R-CNN + BBox U-Net on the DIVA-HisDB test pages with the largest-contour export (std) and
with the DP seam (validation-selected settings), DIVA evaluator, mean over the 30 pages.
    .venv/bin/python 99_evaluation/analysis/dp_seam_pixel_metrics.py
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_qualitative_v2 as Q  # noqa: E402
import story_diva_pages as S  # noqa: E402

EV = Q.ROOT / "99_evaluation/instance_segmentation/dp_seam/diva-hisdb"
RUNS = {"std": "seam_test_std_s0.9_t*_k0", "seam": "seam_test_seam_s0.9_t0.5_k64"}   # std threshold differs per subset
KEYS = ["LinesFMeasure", "PixelIU", "MatchedPixelIU", "MatchedPixelRecall", "MatchedPixelPrecision"]


def metrics(sub, stem, xml):
    gtx, pix, _ = Q.files(sub, stem)
    with tempfile.TemporaryDirectory() as cwd:
        shutil.copy(xml, Path(cwd) / f"{stem}.xml")
        subprocess.run(["java", "-Djava.awt.headless=true", "-cp", f"/usr/share/openjfx/lib/*:{S.JAR}",
                        "ch.unifr.LineSegmentationEvaluatorTool", "-igt", str(pix), "-xgt", str(gtx),
                        "-xp", str(Path(cwd) / f"{stem}.xml"), "-csv"], cwd=cwd, capture_output=True, text=True, check=True)
        head, vals = (Path(cwd) / "results.csv").read_text().splitlines()[:2]
        head, vals = head.split(","), vals.split(",")
        return dict(zip(head[1:], [float(v) for v in vals[-(len(head) - 1):]]))


jobs = [(name, sub, x) for name, run in RUNS.items() for sub in ("CB55", "CS18", "CS863") for d in (EV / sub).glob(run) for x in sorted(d.glob("*.xml"))]
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(lambda j: (j[0], metrics(j[1], j[2].stem, j[2])), jobs))
keys = list(res[0][1])
print(f"{'metric':28s} {'std':>8s} {'seam':>8s} {'diff':>7s}")
for k in keys:
    v = {name: 100 * float(np.nanmean([m[k] for n, m in res if n == name])) for name in RUNS}
    print(f"{k:28s} {v['std']:8.2f} {v['seam']:8.2f} {v['seam'] - v['std']:+7.2f}")
