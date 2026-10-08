#!/usr/bin/env python3
"""Test-time scale selection (TTS) for the RQ3 models of the augmentation search (GPU).

  grid   (dev)  every page at every scale in SCALES; per-page label-free statistics and
                official DIVA per-page metrics -> tts/grid/<tag>.csv
  select        choose one rule on the dev grids of the frozen config (Protocol A, all
                sources), same objective as the search -> tts/tts_selection.json
  apply  (test) frozen rule; each page is predicted only at the scale(s) the rule needs
                -> tts/apply/<tag>.json

Rules never read labels of the page they rescale: "pitch" compares the image-based line
pitch of the page (rq3_pitch.estimate_pitch) with the label-based pitch of the source's
own training pages. Only shrinking is possible (the model input size is fixed).
"""
from __future__ import annotations
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "99_evaluation/analysis"))
import rq3_aug_search as S  # noqa: E402

OUT = ROOT / "99_evaluation/analysis/rq3_aug_search/tts"
SCALES = (1.0, 0.75, 0.6, 0.45)
RULES = ["none", "const:0.75", "const:0.6", "pitch", "pitch_margin", "pitch_score", "pitch_score_fb", "pitchc:1.3", "pitchc:1.6", "pitchc:2.0", "step:1.3:0.6", "step:1.3:0.75", "step:1.6:0.6", "step:1.6:0.75", "step:2.0:0.6", "step:2.0:0.75"]


def page_stem(name: str) -> str:
    """File stem that keeps dots inside page names (e.g. Phil.gr.130_0089r)."""
    n = Path(name).name
    for ext in (".xml", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        if n.lower().endswith(ext):
            return n[:-len(ext)]
    return n


def source_ref_pitch(source, protocol):
    from rq3_pitch import label_pitch
    root = S.train_root(source, protocol)
    return label_pitch(json.loads((root / "coco_instances/train.json").read_text()))


def rule_scales(rule, pitch, ref):
    """Candidate scales a rule needs for one page (several only for pitch_score)."""
    if rule == "none":
        return [1.0]
    if rule.startswith("const:"):
        return [float(rule.split(":")[1])]
    if rule.startswith("step:"):   # "step:T:s": shrink to s only when the page pitch exceeds T x source pitch
        _, t, sc = rule.split(":")
        big = bool(pitch) and pitch == pitch and pitch > float(t) * ref
        return [float(sc) if big else 1.0]
    # "pitchc:<c>": shrink only until the page pitch is c times the source pitch (the
    # training scale jitter shows the model lines up to ~1.6x larger than its own).
    c = float(rule.split(":")[1]) if rule.startswith("pitchc:") else 1.0
    f = 1.0 if not pitch or pitch != pitch else min(1.0, c * ref / pitch)
    if rule == "pitch_margin":
        f *= 0.85
    if rule == "pitch_score_fb" and (not pitch or pitch != pitch):
        return list(SCALES)          # no pitch estimate: let the model's confidence decide
    if rule in ("pitch_score", "pitch_score_fb"):
        f = max(f, min(SCALES))      # the neighbourhood of a target below the grid is its low end
        near = [s for s in SCALES if abs(np.log(s) - np.log(f)) <= np.log(1.4)]
        return near or [min(SCALES, key=lambda s: abs(np.log(s) - np.log(f)))]
    return [min(SCALES, key=lambda s: abs(np.log(s) - np.log(f)))]


def choose(rule, per_scale, pitch, ref):
    """per_scale: {scale: {"mean_score", "n", ...}} -> chosen scale."""
    cands = [s for s in rule_scales(rule, pitch, ref) if s in per_scale]
    if len(cands) == 1:
        return cands[0]
    return max(cands, key=lambda s: per_scale[s]["mean_score"])


def load_model(source, protocol, name):
    import torch
    sys.path.insert(0, str(ROOT / "99_evaluation/analysis"))
    import rq3_cheap_exps as ce
    m = ce.m
    tag = S.model_tag(source, protocol, name)
    summary = json.loads((S.EVAL / tag / "summary.json").read_text())
    st, mt = summary["score_threshold"], summary["mask_threshold"]
    model = m.build_model("convnext_tiny_catmus", 1152)
    m.configure_dense_inference(model, 300, replace_anchors=False)
    m.load_init_checkpoint(model, S.ckpt_path(source, protocol, name))
    model.roi_heads.score_thresh = st
    return m, ce, model.to(ce.DEV).eval(), st, mt, torch


def predict_page(m, ce, model, torch, img, meta, f, st, mt, assign):
    h2, w2 = max(1, round(meta["new_h"] * f)), max(1, round(meta["new_w"] * f))
    canvas = img
    if f != 1.0:
        small = torch.nn.functional.interpolate(img[:, :meta["new_h"], :meta["new_w"]][None], size=(h2, w2),
                                                mode="bilinear", antialias=True)[0]
        canvas = torch.ones_like(img)
        canvas[:, :h2, :w2] = small
    with torch.inference_mode(), torch.autocast(ce.DEV.type, dtype=torch.float16, enabled=ce.DEV.type == "cuda"):
        o = model([canvas.to(ce.DEV)])[0]
    o = {k: v.float() if v.is_floating_point() else v for k, v in o.items()}
    meta2 = {**meta, "new_h": h2, "new_w": w2}
    inst = m.prediction_to_instances(o, meta2, st, mt) if assign == "score" else ce.pixel_competition(o, meta2, st, mt)
    keep = o["scores"] >= st
    stats = {"n": int(keep.sum()), "mean_score": float(o["scores"][keep].mean()) if keep.any() else 0.0}
    return inst, stats


def page_pitch(path):
    from PIL import Image
    from rq3_pitch import estimate_pitch
    return estimate_pitch(np.asarray(Image.open(path).convert("RGB")))


def run(source, protocol, name, split, mode, rule, assign):
    import pandas as pd
    tag = S.model_tag(source, protocol, name)
    out = OUT / ("grid" if mode == "grid" else "apply") / (f"{tag}__{split}.csv" if mode == "grid" else f"{tag}__{split}.json")
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    m, ce, model, st, mt, torch = load_model(source, protocol, name)
    ref = source_ref_pitch(source, protocol)
    work = OUT / "work" / f"{tag}__{split}_{mode}"
    rows, result = [], {"tag": tag, "rule": rule, "assign": assign, "ref_pitch": ref, "scales_grid": list(SCALES),
                     "pitch_estimator": "estimate_pitch", "targets": {}}
    for coll, root, sp in S.eval_targets(source, protocol, split):
        dataset, _ = m.make_loader(root, sp, 1152, False)
        chosen = {}
        for i in range(len(dataset)):
            img, _, meta = dataset[i]
            stem = Path(meta["path"]).stem
            pitch = page_pitch(meta["path"])
            scales = SCALES if mode == "grid" else rule_scales(rule, pitch, ref)
            per_scale, insts = {}, {}
            page_assign = assign
            if assign.startswith("adaptive:"):   # per-pixel competition on dense pages only (label-free)
                dense = bool(pitch) and pitch == pitch and pitch < float(assign.split(":")[1]) * ref
                page_assign = "pixel" if dense else "score"
            for f in scales:
                inst, stats = predict_page(m, ce, model, torch, img, meta, f, st, mt, page_assign)
                per_scale[f], insts[f] = stats, inst
            if mode == "grid":
                for f in scales:
                    d = work / coll / f"s{f}"
                    d.mkdir(parents=True, exist_ok=True)
                    m.page_xml(meta, insts[f], d / f"{stem}.xml")
                    rows.append({"target": coll, "page": stem, "scale": f, "pitch": pitch, **per_scale[f]})
            else:
                f = choose(rule, per_scale, pitch, ref)
                chosen[stem] = f
                d = work / coll
                d.mkdir(parents=True, exist_ok=True)
                m.page_xml(meta, insts[f], d / f"{stem}.xml")
        if mode == "grid":
            fm = {}
            for f in SCALES:
                per_page, _ = m.diva_eval(root, sp, work / coll / f"s{f}")
                for r in per_page:
                    fm[(page_stem(r["filename"]), f)] = {k: float(r[k]) if r[k] not in ("", "NaN") else np.nan
                                                        for k in m.METRICS}
            for r in rows:
                if r["target"] == coll:
                    r.update(fm.get((r["page"], r["scale"]), {}))
        else:
            _, metrics = m.diva_eval(root, sp, work / coll)
            result["targets"][coll] = {**metrics, "scales": chosen}
        print(tag, coll, "done", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    if mode == "grid":
        pd.DataFrame(rows).to_csv(out, index=False)
    else:
        out.write_text(json.dumps(result, indent=1))


def select(name, assign):
    """Rule with the best dev objective over Protocol-A sources (mean of 0.5*(cross mean + worst))."""
    import pandas as pd
    scores = {}
    for rule in RULES:
        per_source = []
        for s in S.COLLS:
            g = pd.read_csv(OUT / "grid" / f"{S.model_tag(s, 'A', name)}__dev.csv")
            ref = source_ref_pitch(s, "A")
            fms = {}
            for t, gt in g.groupby("target"):
                vals = []
                for _, gp in gt.groupby("page"):
                    per = {r.scale: {"mean_score": r.mean_score} for r in gp.itertuples()}
                    f = choose(rule, per, gp.pitch.iloc[0], ref)
                    vals.append(float(gp[gp.scale == f].LinesFMeasure.fillna(0).iloc[0]) * 100)
                fms[t] = float(np.mean(vals))
            cross = [v for t, v in fms.items() if t != s]
            per_source.append(0.5 * (np.mean(cross) + min(cross)))
        scores[rule] = float(np.mean(per_source))
    best = max(scores, key=scores.get)
    sel = {"rule": best, "assign": assign, "config": name, "dev_objective": scores, "note": "frozen before test"}
    (OUT / "tts_selection.json").write_text(json.dumps(sel, indent=1))
    print("TTS selection", json.dumps(sel), flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=7, metavar=("SRC", "PROT", "NAME", "SPLIT", "MODE", "RULE", "ASSIGN"))
    a = ap.parse_args()
    if a.one:
        run(*a.one)
        return
    import subprocess
    sel = json.loads((S.OUT / "selection.json").read_text())
    final, assign = sel["config"], sel["assign"]

    def sub(*args):
        subprocess.run([str(S.PY), __file__, "--one", *map(str, args)], cwd=ROOT,
                       env={**os.environ, "DIVA_WORKERS": "14", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})

    for s in S.COLLS:
        sub(s, "A", final, "dev", "grid", "-", assign)
    sel_path = OUT / "tts_selection.json"
    rule = json.loads(sel_path.read_text())["rule"] if sel_path.exists() else select(final, assign)
    print("TTS_RULE", rule, flush=True)
    for h in S.COLLS:
        sub(h, "B", final, "test", "apply", rule, assign)
    for s in S.COLLS:
        sub(s, "A", final, "test", "apply", rule, assign)
    print("TTS_FINAL_DONE", flush=True)
    # Ablation: the frozen rule on the baseline models (TTS without augmentation).
    for h in S.COLLS:
        sub(h, "B", "base", "test", "apply", rule, assign)
    for s in S.COLLS:
        sub(s, "A", "base", "test", "apply", rule, assign)
    print("TTS_DONE", flush=True)


if __name__ == "__main__":
    main()
