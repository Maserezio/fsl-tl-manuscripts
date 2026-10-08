#!/usr/bin/env python3
"""All DIVA evaluator metrics per test page for the five thesis pipelines (CPU, sequential).
    nice -n 19 .venv/bin/python 99_evaluation/analysis/diva_all_metrics.py  -> 99_evaluation/analysis/diva_taxonomy/page_metrics.csv
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import shutil, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_qualitative_v2 as Q  # noqa: E402
import story_diva_pages as S  # noqa: E402

OUT = Q.ROOT / "99_evaluation/analysis/diva_taxonomy/page_metrics.csv"


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


rows = []
for sub in ("CB55", "CS18", "CS863"):
    pr = Q.diva_preds(sub)
    for f in sorted(pr["maskrcnn"].glob("*.xml")):
        for k, d in pr.items():
            rows.append({"sub": sub, "page": f.stem, "model": k, **metrics(sub, f.stem, d / f"{f.stem}.xml")})
        print("done", sub, f.stem, flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
