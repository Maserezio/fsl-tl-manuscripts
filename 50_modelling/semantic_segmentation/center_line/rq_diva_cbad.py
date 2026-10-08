#!/usr/bin/env python3
"""Center-line model with the cBAD-distilled encoder on DIVA-HisDB (RQ1: all 20 training pages; RQ2: k pages
of the RQ2 'random' selection), no CATMuS pretraining.

  prepare           roots 00_data/RQ3/diva_lf/<SUB>_<MODE> (MODE: full, k1, k3, k5, k10, k15) with COCO TASK-2
                    train / val / test and symlinked images.
  train SUB MODE... center-line model, encoder from the cBAD distillation (LF_ENC=cbad), DIVA_STEPS (1500) steps
                    -> 80_models/semantic_segmentation/center_line/DIVA<SUB>_<MODE>_cbad.pt
  select            decoder with the best mean val FM over all models -> printed (used for test)
  eval SPLIT [DEC]  every trained model on val or test; decoders: thesis row:1.0, v3 (graph + whole strokes +
                    traced ends), v3 limited to a band of the predicted line height; each with two main-text
                    region rules: clip (lines clipped to the TASK-2 region, as the RQ1 Mask R-CNN export) or
                    inside (a line counts only if >= 50 % of its region lies in the TASK-2 region).
                    DEC restricts the decoders (the one selected on val is used on test).
Results: 99_evaluation/semantic_segmentation/center_line/diva_cbad/<split>.json
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "50_modelling").is_dir())  # repo root
sys.path.insert(0, str(HERE))
SUBS = ["CB55", "CS18", "CS863"]
MODES = ["full", "k1", "k3", "k5", "k10", "k15"]
DIVA = ROOT / "00_data/DIVA-HisDB"
OUT_ROOT = ROOT / "00_data/RQ3/diva_lf"
RES = ROOT / "99_evaluation/semantic_segmentation/center_line/diva_cbad"
JAR = Path.home() / "Thesis/DIVA_Line_Segmentation_Evaluator/out/artifacts/LineSegmentationEvaluator.jar"
SPLIT_DIR = {"train": "training", "val": "validation", "test": "public-test"}
DECODERS = [f"{d}|{r}" for d in ("row", "v3", "band1.0") for r in ("clip", "inside")]
# DIVA_TAG=cbpw50 / cmw50: fine-tuned from the cBAD- / CATMuS-pretrained center-line model, WiSE-FT (0.5)
TAG = os.environ.get("DIVA_TAG", "cbad")
MODES = os.environ.get("DIVA_MODES", " ".join(MODES)).split()
SFX = "" if TAG == "cbad" else f"_{TAG}"


def prepare():
    manifest = json.loads((ROOT / "99_evaluation/summaries/rq2/shot_selection_manifest.json").read_text())
    jobs = {j["job"]: j for j in manifest["jobs"]}
    for sub in SUBS:
        for mode in MODES:
            keep = None
            if mode != "full":
                k = int(mode[1:])
                job = next(c["job"] for c in manifest["cells"] if c["subset"] == sub and c["k"] == k and c["method"] == "random")
                keep = set(jobs[job]["pages"])
            r = OUT_ROOT / f"{sub}_{mode}"
            (r / "coco_instances").mkdir(parents=True, exist_ok=True)
            (r / "images").mkdir(parents=True, exist_ok=True)
            for split in ("train", "val", "test"):
                d = json.loads((DIVA / f"coco_task2_{sub}/{split}.json").read_text())
                if split == "train" and keep is not None:
                    ids = {im["id"] for im in d["images"] if Path(im["file_name"]).stem in keep}
                    d = {**d, "images": [im for im in d["images"] if im["id"] in ids],
                         "annotations": [a for a in d["annotations"] if a["image_id"] in ids]}
                (r / f"coco_instances/{split}.json").write_text(json.dumps(d))
                link = r / "images" / split
                if not link.exists():
                    link.symlink_to(DIVA / sub / f"img-{sub}/img/{SPLIT_DIR[split]}")
            print("prepared", r.name, len(json.loads((r / "coco_instances/train.json").read_text())["images"]), "pages", flush=True)


def train(sub, modes):
    import rq3_linefield as LF
    # cmw50: the CATMuS-pretrained center-line model of RQ3 (CATMuS Mask R-CNN encoder), fine-tuned + WiSE-FT 0.5
    assert os.environ.get("LF_ENC") == ("" if TAG == "cmw50" else "cbad") or (TAG == "cmw50" and not os.environ.get("LF_ENC"))
    if TAG in ("cbpw50", "cmw50"):
        import torch
        pre_path = LF.MODELS / ("cbad_pretrain.pt" if TAG == "cbpw50" else "catmus_pretrain.pt")
        ft_tag = "cbp" if TAG == "cbpw50" else "cm"
        pre = torch.load(pre_path, map_location="cpu", weights_only=False)["model"]
        for mode in modes:
            ft_path, out = LF.MODELS / f"DIVA{sub}_{mode}_{ft_tag}.pt", LF.MODELS / f"DIVA{sub}_{mode}_{TAG}.pt"
            if out.exists():
                print("skip", out, flush=True); continue
            LF.train(f"DIVA{sub}", f"{mode}_{ft_tag}", int(os.environ.get("DIVA_STEPS", "1500")),
                     root=OUT_ROOT / f"{sub}_{mode}", init=pre_path, out=ft_path)
            ft = torch.load(ft_path, map_location="cpu", weights_only=False)
            mix = {k: (0.5 * pre[k].float() + 0.5 * v.float()).to(v.dtype) if v.is_floating_point() else v
                   for k, v in ft["model"].items()}
            torch.save({**ft, "model": mix, "wise_alpha": 0.5, "init": pre_path.name}, out)
            print("saved", out, flush=True)
        return
    for mode in modes:
        if (LF.MODELS / f"DIVA{sub}_{mode}_cbad.pt").exists():      # already trained by another stream
            print("skip", sub, mode, flush=True); continue
        LF.train(f"DIVA{sub}", f"{mode}_cbadenc", int(os.environ.get("DIVA_STEPS", "1500")), root=OUT_ROOT / f"{sub}_{mode}",
                 out=LF.MODELS / f"DIVA{sub}_{mode}_cbad.pt")


def region_mask(sub, split, stem, shape, s):
    import xml.etree.ElementTree as ET
    ns = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}"
    root = ET.parse(DIVA / sub / f"PAGE-gt-{sub}-TASK-2/TASK-2/{SPLIT_DIR[split]}/{stem}.xml").getroot()
    c = root.find(f".//{ns}TextRegion/{ns}Coords")
    if c is None:
        return np.ones(shape, bool)
    m = np.zeros(shape, np.uint8)
    pts = np.array([[float(v) for v in p.split(",")] for p in c.get("points").split()]) * s
    cv2.fillPoly(m, [np.rint(pts).astype(np.int32)], 1)
    return m > 0


def diva_page(sub, split, stem, xml):
    with tempfile.TemporaryDirectory(dir=RES) as cwd:
        local = Path(cwd) / f"{stem}.xml"
        local.write_bytes(Path(xml).read_bytes())
        sd = SPLIT_DIR[split]
        subprocess.run(["java", "-Djava.awt.headless=true", "-cp", f"/usr/share/openjfx/lib/*:{JAR}",
                        "ch.unifr.LineSegmentationEvaluatorTool",
                        "-igt", str(DIVA / sub / f"pixel-level-gt-{sub}/pixel-level-gt/{sd}/{stem}.png"),
                        "-xgt", str(DIVA / sub / f"PAGE-gt-{sub}-TASK-2/TASK-2/{sd}/{stem}.xml"),
                        "-xp", str(local), "-csv"], cwd=cwd, capture_output=True, text=True, check=True)
        lines = (Path(cwd) / "results.csv").read_text().splitlines()
        head, vals = lines[0].split(","), lines[1].split(",")
        return dict(zip(head[1:], [float(v) for v in vals[-(len(head) - 1):]]))


def decode(CL, pg, s, shape, reg):
    """All candidate decoders for one page -> {name: [masks at scale s]}."""
    inst = cv2.resize(CL.page_instances(pg, "row", 1.0), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    base = {"row": [inst == i for i in range(1, int(inst.max()) + 1)]}
    v3, _ = CL.page_instances(pg, "graph", 1.0, stroke="safe", ext="trace")
    base["v3"] = v3
    frg = [f for f in pg["frags"] if len(f["xs"]) > 2]
    band = []
    if frg:
        pp = CL.frag_pitch(frg)
        lines = [l for l in CL.group_graph(frg, pp) if l["xs"][-1] - l["xs"][0] >= 0.5 * pp]
        lines = CL.trace_ends(lines, pp, CL.page_ink(pg, s), s)
        cells = CL.CI.cells([np.stack([l["xs"] * s, l["cy"] * s], 1).astype(np.float32) for l in lines], shape, pp * s)
        for i, l in enumerate(lines):
            hu, hd = CL.line_height(l)
            step = max(1, len(l["xs"]) // 60)
            x, y = l["xs"][::step] * s, l["cy"][::step] * s
            poly = np.concatenate([np.stack([x, y - hu * s], 1), np.stack([x, y + hd * s], 1)[::-1]])
            b = np.zeros(shape, np.uint8); cv2.fillPoly(b, [np.rint(poly).astype(np.int32)], 1)
            band.append((cells == i + 1) & (b > 0))
    base["band1.0"] = band
    out = {}
    for name, ms in base.items():
        ms = [m for m in ms if m.any()]
        out[f"{name}|clip"] = [m & reg for m in ms]
        out[f"{name}|inside"] = [m & reg for m in ms if (m & reg).sum() >= 0.5 * m.sum()]
    return out


def evaluate(split, decoders):
    import torch
    import rq3_linefield as LF
    import rq3_centre_lf as CL
    RES.mkdir(parents=True, exist_ok=True)
    out_path = RES / f"{split}{SFX}.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for sub in SUBS:
        for mode in MODES:
            mfile = LF.MODELS / f"DIVA{sub}_{mode}_{TAG}.pt"
            key = f"{sub}|{mode}"
            if not mfile.exists() or (key in res and all(d in res[key] for d in decoders)):
                continue
            net = LF.build_model()
            net.load_state_dict(torch.load(mfile, map_location="cpu", weights_only=False)["model"])
            net = net.to(LF.DEV).eval()
            root = OUT_ROOT / f"{sub}_full"
            coco = json.loads((root / f"coco_instances/{split}.json").read_text())
            work = {d: RES / f"pred{SFX}" / split / key.replace("|", "_") / d.replace("|", "_") for d in decoders}
            for w in work.values():
                w.mkdir(parents=True, exist_ok=True)
            stems = []
            for im in coco["images"]:
                img = np.asarray(Image.open(root / "images" / split / im["file_name"]).convert("RGB"))
                H, W = img.shape[:2]
                with torch.inference_mode():
                    s1 = 1152 / max(H, W)
                    cen, _, up, dn = LF.run_net(net, cv2.resize(img, (int(W * s1), int(H * s1)), interpolation=cv2.INTER_AREA))
                    sel = cen > 0.5
                    thick = float(np.median(up[sel] + dn[sel])) if sel.sum() > 50 else LF.T0
                    s2 = min(s1 * LF.T0 / max(thick, 1.0), 3200 / max(H, W))
                    cen, end, up, dn = LF.run_net(net, cv2.resize(img, (int(W * s2), int(H * s2)),
                                                                  interpolation=cv2.INTER_AREA if s2 < 1 else cv2.INTER_LINEAR))
                frags = []
                for f in LF._fragments(cen, end, 0.4):
                    ys_, xs_ = f["px"]
                    frags.append({"xs": f["xs"] / s2, "cy": f["cy"] / s2, "n": len(f["xs"]),
                                  "hu": float(np.percentile(up[ys_, xs_], 75)) / s2, "hd": float(np.percentile(dn[ys_, xs_], 75)) / s2})
                pg = {"coll": sub, "im": {"height": H, "width": W, "file_name": im["file_name"]}, "frags": frags,
                      "T0": LF.T0 / s2, "root": str(root), "sp": split}
                stem = Path(im["file_name"]).stem
                stems.append(stem)
                s = min(1.0, CL.CI.WORK_LONG / max(H, W))
                shape = (round(H * s), round(W * s))
                outs = decode(CL, pg, s, shape, region_mask(sub, split, stem, shape, s))
                for d in decoders:
                    ms = [m for m in outs[d] if m.sum() * (1 / s) ** 2 >= 100]      # RQ1: contour area >= 100 px
                    CL.page_xml_multi(work[d] / f"{stem}.xml", im["file_name"], W, H, ms, s)
            for d, w in work.items():
                with ThreadPoolExecutor(int(os.environ.get("DIVA_WORKERS", "6"))) as ex:
                    rows = list(ex.map(lambda st: diva_page(sub, split, st, w / f"{st}.xml"), stems))
                res.setdefault(key, {})[d] = {k: round(100 * float(np.mean([r[k] for r in rows])), 2)
                                              for k in ("PixelIU", "LinesIU", "LinesRecall", "LinesPrecision", "LinesFMeasure",
                                                        "MatchedPixelIU", "MatchedPixelRecall", "MatchedPixelPrecision")}
            out_path.write_text(json.dumps(res, indent=1))
            print(split, key, {d: res[key][d]["LinesFMeasure"] for d in decoders}, flush=True)
            del net; torch.cuda.empty_cache()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        prepare()
    elif cmd == "select":
        val = json.loads((RES / f"val{SFX}.json").read_text())
        print(max(DECODERS, key=lambda d: np.mean([v[d]["LinesFMeasure"] for v in val.values()])))
    elif cmd == "train":
        train(sys.argv[2], sys.argv[3:])
    else:
        evaluate(sys.argv[2], sys.argv[3:] or DECODERS)
