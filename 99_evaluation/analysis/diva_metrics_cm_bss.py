#!/usr/bin/env python3
"""Per-page DIVA metrics of the CATMuS center-line model (20 pages) with the bandseamstroke decoder. Replaces the
center-line rows of 99_evaluation/analysis/diva_taxonomy/page_metrics.csv in a copy (page_metrics_cm_bss.csv) and prints the
Spearman correlation of line FM and matched pixel IoU over all pipelines and pages, old and new.
    .venv/bin/python 99_evaluation/analysis/diva_metrics_cm_bss.py
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import shutil, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_qualitative_v2 as Q  # noqa: E402
import story_diva_pages as S  # noqa: E402

SRC = Q.ROOT / "99_evaluation/analysis/diva_taxonomy/page_metrics.csv"
OUT = Q.ROOT / "99_evaluation/analysis/diva_taxonomy/page_metrics_cm_bss.csv"
BSS = Q.ROOT / "99_evaluation/semantic_segmentation/center_line/diva_cbad/pred_cm_cellseam/test/{}_full/bandseamstroke_inside"


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


def rho(df):
    d = df.dropna(subset=["LinesFMeasure", "MatchedPixelIU"])
    return d["LinesFMeasure"].corr(d["MatchedPixelIU"], method="spearman")


old = pd.read_csv(SRC)
rows = []
for _, r in old[old.model == "centerline"].iterrows():
    rows.append({"sub": r["sub"], "page": r["page"], "model": "centerline",
                 **metrics(r["sub"], r["page"], Path(str(BSS).format(r["sub"])) / f"{r['page']}.xml")})
new = pd.concat([old[old.model != "centerline"], pd.DataFrame(rows)], ignore_index=True)
new.to_csv(OUT, index=False)
c = new[new.model == "centerline"]
print("center-line pages", len(c), "FM %.2f mIU %.2f rec %.2f prec %.2f" % tuple(100 * c[k].mean() for k in
      ("LinesFMeasure", "MatchedPixelIU", "MatchedPixelRecall", "MatchedPixelPrecision")))
print("Spearman old %.3f new %.3f" % (rho(old), rho(new)))
