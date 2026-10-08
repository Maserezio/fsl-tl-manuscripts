#!/usr/bin/env python3
"""HTR-based evaluation of the RQ3 line polygons on NorHand v3 test pages, against the real transcriptions.

Recogniser: Riksarkivet/trocr-base-handwritten-hist-swe-2 (Swedish historical handwriting; NorHand is Norwegian, so the
absolute CER is high and only differences between segmentations are interpreted).
Page CER in reading order (lines sorted by vertical center, joined with spaces) against
  (a) the NorHand transcription of the GT lines ("true" CER), and
  (b) the recogniser output on the GT polygons (segmentation-induced CER).
Models: GT polygons, zero-shot and protocol-A (trained on ONB Cod. Syr. 1) Mask R-CNN and center-line model.

    nice -n 5 ../../.venv/bin/python norhand_htr_cer.py [n_pages]   (cwd 99_evaluation/analysis/htr_eval)
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import fig_qualitative_v2 as Q  # noqa: E402
import overlap_lines_diva as O  # noqa: E402
import htr_common as H  # noqa: E402

DATA = Q.ROOT / "00_data"
DS = "NorHand_v3"


def gt_lines(stem):
    out = []
    for el in ET.parse(DATA / f"RQ3_sources/NorHand_v3_selected/test/page/{stem}.xml").getroot().iter():
        if el.tag.endswith("TextLine"):
            c = next(ch for ch in el if ch.tag.endswith("Coords"))
            p = np.array([[int(float(v)) for v in xy.split(",")] for xy in c.get("points").split()], np.int32)
            t = [u.text or "" for te in el if te.tag.endswith("TextEquiv") for u in te if u.tag.endswith("Unicode")]
            if len(p) >= 3:
                out.append((p, t[0] if t else ""))
    return out


def order(ps):
    return sorted(range(len(ps)), key=lambda i: (round(ps[i][:, 1].mean() / 60), ps[i][:, 0].mean()))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    torch.set_num_threads(int(__import__("os").environ.get("HTR_THREADS", "12")))
    proc, model = H.load("Riksarkivet/trocr-base-handwritten-hist-swe-2")
    preds = {f"{k}_{m}": d for k in ("zs", "onb") for m, d in Q.rq3_preds(DS, k).items() if d.is_dir()}
    # zero-shot Mask R-CNN predictions kept on 2026-10-07 (thesis zero-shot settings)
    zsm = Q.ROOT / f"99_evaluation/instance_segmentation/mask_rcnn/cross_collection/{DS}_maskrcnn_convnext_tiny_catmus_1152_zeroshot_story_zeroshot/test_pred_xml"
    if zsm.is_dir():
        preds["zs_maskrcnn"] = zsm
    print("models:", list(preds), flush=True)
    stems = sorted(p.stem for p in preds["zs_centerline"].glob("*.xml"))
    stems = stems[:: max(1, len(stems) // n)][:n]
    rows, texts = [], []
    for stem in stems:
        src = stem.replace(f"{DS}__", "")
        gl = gt_lines(src)
        img = cv2.cvtColor(cv2.imread(str(next((DATA / f"RQ3/final_roots/{DS}/images/test").glob(stem + ".*")))), cv2.COLOR_BGR2RGB)
        gp = [p for p, _ in gl]
        true_page = " ".join(gl[j][1] for j in order(gp))
        ref = H.recognise(proc, model, [H.line_image(img, p) for p in gp])
        ref_page = " ".join(ref[j] for j in order(gp))
        rows.append({"page": src, "model": "gt", "cer_true": 100 * H.cer(true_page, ref_page), "cer_seg": 0.0, "n": len(gp)})
        for key, d in preds.items():
            pp = O.polys(d / f"{stem}.xml")
            hyp = H.recognise(proc, model, [H.line_image(img, p) for p in pp]) if pp else []
            hyp_page = " ".join(hyp[i] for i in order(pp))
            rows.append({"page": src, "model": key, "cer_true": 100 * H.cer(true_page, hyp_page),
                         "cer_seg": 100 * H.cer(ref_page, hyp_page), "n": len(pp)})
            texts.append({"page": src, "model": key, "text": hyp_page})
            print(src, key, "CER true %.1f seg %.1f" % (rows[-1]["cer_true"], rows[-1]["cer_seg"]), flush=True)
        texts.append({"page": src, "model": "gt", "text": ref_page}); texts.append({"page": src, "model": "transcription", "text": true_page})
        pd.DataFrame(rows).to_csv(HERE / f"norhand_htr_pages_n{n}.csv", index=False)
        pd.DataFrame(texts).to_csv(HERE / f"norhand_htr_texts_n{n}.csv", index=False)
    print(pd.DataFrame(rows).groupby("model")[["cer_true", "cer_seg"]].mean().round(2))


if __name__ == "__main__":
    main()
