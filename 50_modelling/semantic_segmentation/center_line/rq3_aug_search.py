#!/usr/bin/env python3
"""Overnight RQ3 augmentation search (PageForge + degradations), selected on dev pages only.

Stage 1  six augmentation configs x three sources (GRPOLY, ONB, Phil_gr_130), Protocol A.
Stage 2  two derived configs from the stage-1 ranking.
Stage 3  the best config on all seven Protocol-A sources and all seven LOCO-6 (Protocol B)
         holdouts, then a single test evaluation of the frozen config and of the baseline.

Selection data: dev = train+val pages of every *other* collection (00_data/RQ3/dev, built
by rq3_build_dev.py) and the source's own val pages. Test pages are read only in stage 3,
after the configuration has been frozen (recorded in selection.json before any test run).
Objective per config: mean over sources of 0.5 * (cross-collection mean FM + worst FM),
for the better of the two overlap assignments (score order / per-pixel competition).
Everything is cached; rerunning continues where it stopped.

    systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block \
        .venv/bin/python 50_modelling/semantic_segmentation/center_line/rq3_aug_search.py
"""
from __future__ import annotations
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = ROOT / ".venv/bin/python"
OUT = ROOT / "99_evaluation/analysis/rq3_aug_search"
EVALS = OUT / "evals"
EVAL = ROOT / "99_evaluation/instance_segmentation/mask_rcnn/cross_collection"
MODELS = ROOT / "80_models/instance_segmentation/mask_rcnn/cross_collection"
CKPT = MODELS / "final_ab"
STEM = "maskrcnn_convnext_tiny_catmus_1152"
COLLS = ["GRPOLY", "NorHand_v3", "ONB", "Phil_gr_130", "Pinkas", "RASAM", "RASM"]
STAGE_SOURCES = ["GRPOLY", "ONB", "Phil_gr_130"]
ALL_KNOBS = {"pitch": [0.7, 2.2], "scale": [0.5, 1.2], "gamma": [0.6, 2.0], "curve": 0.5,
             "slope": 8, "deg_p": 0.6, "stroke_p": 0.3}
STAGE1 = {  # name: (forge probability, RQ3_FORGE_CFG)
    "all": (0.7, ALL_KNOBS),
    "f05": (0.5, {}),
    "dense": (0.5, {"pitch": [0.7, 2.2]}),
    "small": (0.5, {"scale": [0.5, 1.2]}),
    "deg": (0.5, {"deg_p": 0.6}),
    "degonly": (0.0, {"deg_p": 0.6}),
}


def log(*a):
    print(time.strftime("%H:%M"), *a, flush=True)


def train_root(source, protocol):
    if protocol == "B":
        return ROOT / "00_data/RQ3/loco" / source
    return ROOT / ("00_data/RQ3/final_roots/NorHand_v3" if source == "NorHand_v3" else f"00_data/RQ3/matrix/{source}")


def model_tag(source, protocol, name):
    k = 6 if protocol == "B" else 3
    if name == "base":
        return f"{source}_{STEM}_k{k}_final_ab_j1.6" + ("_d1" if source == "NorHand_v3" and protocol == "A" else "")
    suffix = "final_ab_j1.6_forge05" if name == "f05" else f"final_ab_j1.6_aug_{name}"
    return f"{source}_{STEM}_k{k}_{suffix}"


def ckpt_path(source, protocol, name):
    tag = model_tag(source, protocol, name)
    return (CKPT / f"{tag}.pt") if name == "base" else (MODELS / tag / "best.pt")


def train(source, protocol, name, cfg):
    tag = model_tag(source, protocol, name)
    if name == "base" or ((EVAL / tag / "summary.json").exists() and ckpt_path(source, protocol, name).exists()):
        return
    p, knobs = cfg
    env = {k: v for k, v in os.environ.items() if not k.startswith("RQ3_")}
    env.update(RQ3_REPLACE_ANCHORS="0", RQ3_SCALE_JITTER="1", RQ3_JITTER_UP_MAX="1.6", DIVA_WORKERS="6",
               RQ3_FORGE=str(p), RQ3_FORGE_CFG=json.dumps(knobs),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    suffix = tag.split(f"_k{6 if protocol == 'B' else 3}_", 1)[1]
    log(f"train {tag} cfg={json.dumps(cfg)}")
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    with (OUT / "logs" / f"{tag}.log").open("w") as fh:
        r = subprocess.run([str(PY), "maskrcnn_rq3_loo.py", "--data-root", str(train_root(source, protocol)),
                            "--result-suffix", suffix, "--arm", "convnext_tiny_catmus", "--image-size", "1152",
                            "--epochs", "50", "--fast-grid", "--clean-predictions", "--skip-test", "--seed", "42"],
                           cwd=ROOT / "50_modelling/instance_segmentation/mask_rcnn/cross_collection", env=env, stdout=fh, stderr=subprocess.STDOUT)
    log(f"  exit {r.returncode}")


def eval_targets(source, protocol, split):
    """(collection, data root, split name) pairs a model is scored on."""
    if protocol == "B":
        root = ROOT / ("00_data/RQ3/dev" if split == "dev" else "00_data/RQ3/matrix") / source
        return [(source, root, split)]
    out = []
    for t in COLLS:
        if t == source:
            out.append((t, train_root(source, "A"), "val" if split == "dev" else "test"))
        else:
            out.append((t, ROOT / ("00_data/RQ3/dev" if split == "dev" else "00_data/RQ3/matrix") / t, split))
    return out


def evaluate(source, protocol, name, split):
    """Cached per-target DIVA metrics for both overlap assignments (run in a subprocess)."""
    tag = model_tag(source, protocol, name)
    out = EVALS / f"{tag}__{split}.json"
    if out.exists():
        return json.loads(out.read_text())
    if not ckpt_path(source, protocol, name).exists():
        log(f"  missing checkpoint {tag}")
        return None
    EVALS.mkdir(parents=True, exist_ok=True)
    log(f"eval {tag} on {split}")
    r = subprocess.run([str(PY), __file__, "--eval-one", source, protocol, name, split], cwd=ROOT,
                       env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode or not out.exists():
        log(f"  eval failed: {r.stdout[-1500:]}")
        return None
    return json.loads(out.read_text())


def eval_one(source, protocol, name, split):
    import shutil
    import torch
    sys.path.insert(0, str(ROOT / "99_evaluation/analysis"))
    import rq3_cheap_exps as ce
    m = ce.m
    EVALS.mkdir(parents=True, exist_ok=True)
    tag = model_tag(source, protocol, name)
    summary = json.loads((EVAL / tag / "summary.json").read_text())
    st, mt = summary["score_threshold"], summary["mask_threshold"]
    model = m.build_model("convnext_tiny_catmus", 1152)
    m.configure_dense_inference(model, 300, replace_anchors=False)
    m.load_init_checkpoint(model, ckpt_path(source, protocol, name))
    model.roi_heads.score_thresh = st
    model = model.to(ce.DEV).eval()
    work = OUT / "pred" / f"{tag}__{split}"
    results = {"tag": tag, "score_threshold": st, "mask_threshold": mt, "targets": {}}
    # Test of Protocol A: only the frozen overlap assignment (DIVA per page is the bottleneck).
    assigns = ("score", "pixel")
    if split == "test" and protocol == "A" and (OUT / "selection.json").exists():
        assigns = (json.loads((OUT / "selection.json").read_text())["assign"],)
    for coll, root, sp in eval_targets(source, protocol, split):
        dataset, _ = m.make_loader(root, sp, 1152, False)
        with torch.inference_mode():
            for i in range(len(dataset)):
                img, _, meta = dataset[i]
                with torch.autocast(ce.DEV.type, dtype=torch.float16, enabled=ce.DEV.type == "cuda"):
                    o = model([img.to(ce.DEV)])[0]
                o = {k: v.float() if v.is_floating_point() else v for k, v in o.items()}
                stem = Path(meta["path"]).stem
                for assign in assigns:
                    inst = (m.prediction_to_instances(o, meta, st, mt) if assign == "score"
                            else ce.pixel_competition(o, meta, st, mt))
                    d = work / assign / coll
                    d.mkdir(parents=True, exist_ok=True)
                    m.page_xml(meta, inst, d / f"{stem}.xml")
        for assign in assigns:
            _, metrics = m.diva_eval(root, sp, work / assign / coll)
            results["targets"].setdefault(coll, {})[assign] = metrics
    shutil.rmtree(work, ignore_errors=True)
    (EVALS / f"{tag}__{split}.json").write_text(json.dumps(results, indent=1))


def objective(names, sources, split="dev"):
    """{name: {assign: {"obj", "per_source"}}} from cached evals (Protocol A)."""
    table = {}
    for name in names:
        for assign in ("score", "pixel"):
            per = {}
            for s in sources:
                path = EVALS / f"{model_tag(s, 'A', name)}__{split}.json"
                if not path.exists():
                    break
                t = json.loads(path.read_text())["targets"]
                cross = [t[c][assign]["LinesFMeasure"] * 100 for c in COLLS if c != s]
                per[s] = {"mean": sum(cross) / len(cross), "min": min(cross),
                          "own": t[s][assign]["LinesFMeasure"] * 100}
            if len(per) == len(sources):
                obj = sum(0.5 * (v["mean"] + v["min"]) for v in per.values()) / len(per)
                table.setdefault(name, {})[assign] = {"obj": obj, "per_source": per}
    return table


def best(table, exclude_base=True):
    cands = [(v["obj"], n, a) for n, d in table.items() for a, v in d.items() if not (exclude_base and n == "base")]
    return max(cands) if cands else None


def cfg_key(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configs = dict(STAGE1)
    # Baseline on dev for the stage sources.
    for s in STAGE_SOURCES:
        evaluate(s, "A", "base", "dev")
    # Stage 1.
    for name, cfg in STAGE1.items():
        for s in STAGE_SOURCES:
            train(s, "A", name, cfg)
            evaluate(s, "A", name, "dev")
        (OUT / "stage1.json").write_text(json.dumps(objective(["base", *STAGE1], STAGE_SOURCES), indent=1))
    t1 = objective(list(STAGE1), STAGE_SOURCES)
    ranked = sorted(((max(v["obj"] for v in d.values()), n) for n, d in t1.items()), reverse=True)
    log("stage1 ranking", [(n, round(o, 2)) for o, n in ranked])
    # Stage 2: combination of the two best, and the best with a higher forge probability.
    (_, n1), (_, n2) = ranked[0], ranked[1]
    p1, k1 = configs[n1]
    p2, k2 = configs[n2]
    stage2 = {f"mix_{cfg_key([n1, n2])}": (max(p1, p2, 0.5), {**k2, **k1}),
              f"hi_{cfg_key([n1, 0.9])}": (0.9 if p1 > 0 else 0.0, k1)}
    for name, cfg in stage2.items():
        if any(json.dumps(cfg, sort_keys=True) == json.dumps(c, sort_keys=True) for c in configs.values()):
            continue
        configs[name] = cfg
        for s in STAGE_SOURCES:
            train(s, "A", name, cfg)
            evaluate(s, "A", name, "dev")
    t12 = objective(list(configs), STAGE_SOURCES)
    obj, final, assign = best(t12)
    base_obj = max(v["obj"] for v in objective(["base"], STAGE_SOURCES)["base"].values())
    selection = {"config": final, "forge_p": configs[final][0], "knobs": configs[final][1], "assign": assign,
                 "dev_objective": obj, "base_dev_objective_best_assign": base_obj,
                 "all": {n: {a: round(v["obj"], 2) for a, v in d.items()} for n, d in t12.items()},
                 "frozen_at": time.strftime("%Y-%m-%d %H:%M"), "note": "selected before any test evaluation"}
    if not (OUT / "selection.json").exists():
        (OUT / "selection.json").write_text(json.dumps(selection, indent=1))
    selection = json.loads((OUT / "selection.json").read_text())
    final = selection["config"]
    log("FROZEN", json.dumps(selection)[:400])
    # Stage 3: remaining Protocol-A sources, all LOCO holdouts, dev then test.
    for s in COLLS:
        train(s, "A", final, configs[final])
        evaluate(s, "A", "base", "dev")
        evaluate(s, "A", final, "dev")
    for h in COLLS:
        train(h, "B", final, configs[final])
        evaluate(h, "B", "base", "dev")
        evaluate(h, "B", final, "dev")
    (OUT / "stage3_dev.json").write_text(json.dumps(objective(["base", final], COLLS), indent=1))
    log("STAGE3 dev done; test evaluation of the frozen config")
    for h in COLLS:
        evaluate(h, "B", "base", "test")
        evaluate(h, "B", final, "test")
    for s in COLLS:  # baseline Protocol-A test = official final_ab/A.csv (B re-evaluation matched it)
        evaluate(s, "A", final, "test")
    log("SEARCH_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-one", nargs=4, metavar=("SOURCE", "PROTOCOL", "NAME", "SPLIT"))
    a = ap.parse_args()
    if a.eval_one:
        eval_one(*a.eval_one)
    else:
        main()
