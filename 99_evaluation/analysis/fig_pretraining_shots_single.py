#!/usr/bin/env python3
"""Single-panel version of the RQ2 pretraining x shots figure (left panel of codex pretraining_shots.pdf).
Same data: codex_story_audit_20261003/results_v2/rq2_curves.csv (random selection) and
rq2_selection_sensitivity.csv (range of the six selection-method means). Writes SVG and PDF."""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "99_evaluation/analysis/codex_story_audit_20261003/results_v2"
OUT = Path.home() / "Thesis/fsl-tl-manuscripts-thesis/graphics"
INITS = ["imagenet", "dinov3", "cbad", "catmus"]
NAMES = {"imagenet": "ImageNet", "dinov3": "DINOv3 (LVD-1689M)", "cbad": "DINOv3 (cBAD)", "catmus": "CATMuS"}
COLORS = dict(zip(INITS, ["#64748b", "#9b59b6", "#e69522", "#167e75"]))
KS = [1, 3, 5, 10, 15, 20]

curves = pd.read_csv(SRC / "rq2_curves.csv")
sens = pd.read_csv(SRC / "rq2_selection_sensitivity.csv")
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
for init in INITS:
    g = curves[(curves.method == "random") & (curves.init == init)].sort_values("k")
    s = sens[sens.init == init].sort_values("k")
    ax.fill_between(s.k, s.min_method_fm, s.max_method_fm, alpha=.12, color=COLORS[init], lw=0)
    ax.plot(g.k, g.fm, "o-", color=COLORS[init], label=NAMES[init], lw=2, ms=6)
ax.set(xlabel="Training pages per manuscript", ylabel="Mean line FM (%)", ylim=(-2, 102), xticks=KS)
ax.grid(alpha=.2)
ax.legend(fontsize=8, loc="lower right")
for ext in ("svg", "pdf"):
    fig.savefig(OUT / f"rq2_pretraining_shots.{ext}")
print("written", OUT / "rq2_pretraining_shots.{svg,pdf}")
