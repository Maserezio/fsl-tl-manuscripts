#!/usr/bin/env python3
"""Qualitative thesis figures, version 2 (2026-10-06): bright page crops in the style of the overlap figure
(ink colored by the polygon that contains it, polygon outlines drawn), so that line extents are visible.

    .venv/bin/python 99_evaluation/analysis/fig_qualitative_v2.py overview DS STEM          downscaled page + GT + grid
    .venv/bin/python 99_evaluation/analysis/fig_qualitative_v2.py diva SUB STEM x0 y0 x1 y1  GT, U-Net, RT-DETR + BBox U-Net,
                                                                              Mask R-CNN, + DP seam, center-line
    .venv/bin/python 99_evaluation/analysis/fig_qualitative_v2.py rq3 DS STEM x0 y0 x1 y1 KEY  GT + Mask R-CNN + center-line
         KEY = zs (zero-shot) or onb (protocol A, models trained on ONB Cod. Syr. 1)
    .venv/bin/python 99_evaluation/analysis/fig_qualitative_v2.py pages DS KEY             per-page FM of both RQ3 models
Writes PNGs to 99_evaluation/analysis/story_figures/q2_* and copies them to the thesis graphics folder.
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import shutil, subprocess, sys, tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overlap_lines_diva as O  # noqa: E402
import story_diva_pages as S  # noqa: E402

ROOT = S.ROOT
OUT = S.OUT
THESIS = Path.home() / "Thesis/fsl-tl-manuscripts-thesis/graphics"
PAL = O.PAL
EV = ROOT / "99_evaluation"
WITH_IU = False


def diva_preds(sub):
    return {"unet": S.pred_dir("unet", sub), "twostage": S.pred_dir("twostage", sub), "maskrcnn": S.pred_dir("maskrcnn", sub),
            "seam": EV / f"instance_segmentation/dp_seam/diva-hisdb/{sub}/seam_test_seam_s0.9_t0.5_k64", "centerline": S.pred_dir("centerline", sub)}


def rq3_preds(ds, key):
    if key == "zs":
        return {"maskrcnn": EV / f"instance_segmentation/mask_rcnn/cross_collection/{ds}_maskrcnn_convnext_tiny_catmus_1152_zeroshot_story_zs/test_pred_xml",
                "centerline": ROOT / f"99_evaluation/semantic_segmentation/center_line/pred_lf/CATMuS_Z_test/row_1.0/{ds}"}
    return {"maskrcnn": EV / f"instance_segmentation/mask_rcnn/cross_collection/{ds}_maskrcnn_convnext_tiny_catmus_1152_k3_story_from_ONB/test_pred_xml",
            "centerline": ROOT / f"99_evaluation/semantic_segmentation/center_line/pred_lf/ONB_A_w50_test/row_1.0/{ds}"}


def files(ds, stem):
    if ds in S.SUBS:
        b = S.DIVA / ds
        return (b / f"PAGE-gt-{ds}-TASK-2/TASK-2/public-test/{stem}.xml", b / f"pixel-level-gt-{ds}/pixel-level-gt/public-test/{stem}.png",
                next((b / f"img-{ds}/img/public-test").glob(stem + ".*")))
    r = O.rq3_root(ds)
    return r / f"page-gt/test/{stem}.xml", next((r / "pixel-gt/test").glob(stem + ".*")), next((r / "images/test").glob(stem + ".*"))


def page_fm(ds, stem, pred_xml):
    gtx, pix, _ = files(ds, stem)
    with tempfile.TemporaryDirectory() as cwd:
        local = Path(cwd) / f"{stem}.xml"; shutil.copy(pred_xml, local)
        subprocess.run(["java", "-Djava.awt.headless=true", "-cp", f"/usr/share/openjfx/lib/*:{S.JAR}",
                        "ch.unifr.LineSegmentationEvaluatorTool", "-igt", str(pix), "-xgt", str(gtx), "-xp", str(local), "-csv"],
                       cwd=cwd, capture_output=True, text=True, check=True)
        lines = (Path(cwd) / "results.csv").read_text().splitlines()
        head, vals = lines[0].split(","), lines[1].split(",")
        r = dict(zip(head[1:], [float(v) for v in vals[-(len(head) - 1):]]))
        return (100 * r["LinesFMeasure"], 100 * r["LinesIU"]) if WITH_IU else 100 * r["LinesFMeasure"]


def panel(img, fg, ps, box, lw):
    x0, y0, x1, y1 = box
    ps = sorted([p for p in ps if O.inter(O.bbox(p), box)], key=lambda p: p[:, 1].mean())
    base = (0.55 * img[y0:y1, x0:x1] + 0.45 * 255).astype(np.float32)
    cnt = np.zeros(base.shape[:2], np.uint8)
    for k, p in enumerate(ps):
        m = O.rast(p, box); cnt += m
        sel = m & fg[y0:y1, x0:x1]
        base[sel] = 0.35 * base[sel] + 0.65 * np.array(PAL[k % len(PAL)])
    base[(cnt >= 2) & fg[y0:y1, x0:x1]] = (20, 20, 20)
    o = base.astype(np.uint8).copy()
    for k, p in enumerate(ps):
        cv2.polylines(o, [p - [x0, y0]], True, PAL[k % len(PAL)], lw)
    return o


def render(ds, stem, box, preds, tag, fm=True):
    gtx, pix, imf = files(ds, stem)
    img = cv2.cvtColor(cv2.imread(str(imf)), cv2.COLOR_BGR2RGB)
    pg = cv2.imread(str(pix))
    fg = (pg[:, :, 0] & 0x08) > 0  # main-text ink bit (DIVA-style pixel GT, also used for the RQ3 collections)
    lw = max(2, int(round((box[2] - box[0]) / 450)))
    for name, ps in [("gt", O.polys(gtx))] + [(m, O.polys(d / f"{stem}.xml")) for m, d in preds.items()]:
        o = panel(img, fg, ps, box, lw)
        f = f"q2_{tag}_{name}.png"
        cv2.imwrite(str(OUT / f), cv2.cvtColor(o, cv2.COLOR_RGB2BGR)); shutil.copy(OUT / f, THESIS / f)
        print(f, o.shape[1], "x", o.shape[0], ("FM %.2f" % page_fm(ds, stem, preds[name] / f"{stem}.xml")) if fm and name != "gt" else "")


def overview(ds, stem):
    gtx, _, imf = files(ds, stem)
    img = cv2.imread(str(imf)); H, W = img.shape[:2]
    for k, p in enumerate(O.polys(gtx)):
        cv2.polylines(img, [p], True, PAL[k % len(PAL)][::-1], max(2, W // 600))
    for x in range(0, W, 200):
        cv2.line(img, (x, 0), (x, H), (0, 0, 0), 1); cv2.putText(img, str(x), (x + 3, 30), 0, 1, (0, 0, 255), 2)
    for y in range(0, H, 200):
        cv2.line(img, (0, y), (W, y), (0, 0, 0), 1); cv2.putText(img, str(y), (3, y - 3), 0, 1, (0, 0, 255), 2)
    s = 1400 / max(H, W)
    cv2.imwrite(str(OUT / f"q2_overview_{ds}_{stem}.jpg"), cv2.resize(img, (int(W * s), int(H * s))))
    print(OUT / f"q2_overview_{ds}_{stem}.jpg", W, H)


def pages(ds, key):
    preds = rq3_preds(ds, key)
    stems = sorted(p.stem for p in preds["maskrcnn"].glob("*.xml"))
    for st in stems:
        print(st, " ".join("%s %.1f" % (m, page_fm(ds, st, d / f"{st}.xml")) for m, d in preds.items()), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "overview":
        overview(a[1], a[2])
    elif a[0] == "diva":
        render(a[1], a[2], tuple(map(int, a[3:7])), diva_preds(a[1]), f"diva_{a[1]}")
    elif a[0] == "rq3":
        render(a[1], a[2], tuple(map(int, a[3:7])), rq3_preds(a[1], a[7]), f"{a[7]}_{a[1]}")
    elif a[0] == "pages":
        pages(a[1], a[2])
