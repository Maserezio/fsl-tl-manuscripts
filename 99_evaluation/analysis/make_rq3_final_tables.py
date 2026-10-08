#!/usr/bin/env python3
"""LaTeX tables, radar figure, and text numbers for the RQ3 robust-configuration results.

Reads the per-run summary.json files (seed-42 robustness ablation, scale sweep) and the
final_ab/{A,B}.csv of run_rq3_final_ab.py, and writes into the thesis repository:
graphics/rq3_radar.pdf and the LaTeX fragments in OUT_TEX/*.tex; the numbers quoted in
the text are printed and stored in OUT_TEX/numbers.json.

    .venv/bin/python 99_evaluation/analysis/make_rq3_final_tables.py
"""
from __future__ import annotations
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
THESIS = REPO.parent / "fsl-tl-manuscripts-thesis"
EVAL = REPO / "99_evaluation/instance_segmentation/mask_rcnn/cross_collection"
OUT_TEX = REPO / "99_evaluation/analysis/rq3_final_tex"
STEM = "maskrcnn_convnext_tiny_catmus_1152_k3"
D = ["Pinkas", "ONB", "RASAM", "RASM", "Phil_gr_130", "GRPOLY", "NorHand_v3"]
N = {"Pinkas": "Pinkas", "ONB": "ONB", "RASAM": "RASAM", "RASM": "RASM",
     "Phil_gr_130": "Phil.", "GRPOLY": "GRPOLY", "NorHand_v3": "NorHand"}
# Categorical slots 1-3 of the dataviz reference palette (validated all-pairs, light mode).
COLORS = {"base": "#2a78d6", "ab": "#eb6834", "ab+post": "#1baf7a"}
STYLES = {"base": ("--", "o"), "ab": ("-", "s"), "ab+post": (":", "^")}
f2 = lambda x: f"{100 * x:.2f}"


def summary(tag):
    p = EVAL / tag / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------- seed-42 ablation
PAIRS = [("ONB", "GRPOLY"), ("RASAM", "GRPOLY"), ("Phil_gr_130", "GRPOLY"), ("RASAM", "NorHand_v3"),
         ("ONB", "Phil_gr_130"), ("RASAM", "ONB"), ("RASM", "ONB"), ("ONB", "ONB"), ("GRPOLY", "GRPOLY")]
SOURCES = ["ONB", "RASAM", "Phil_gr_130", "RASM", "GRPOLY"]
CONDS = [("base", "base"), ("base_post", "base + post"), ("a", "a"), ("ab", "ab"),
         ("ab_post", "ab + post"), ("abc", "abc"), ("abcd", "abcd"), ("full", "abcde"),
         ("full_post", "abcde + post"), ("ab_j2", r"ab, $s_{\max}=2.0$")]


def ablation():
    rows, vals = [], {}
    for cond, label in CONDS:
        cells = []
        for s, t in PAIRS:
            tag = f"{s}_{STEM}_abl_{cond}_s42" + ("" if s == t else f"_from_{s}")
            tag = f"{t}_{STEM}_abl_{cond}_s42" + ("" if s == t else f"_from_{s}")
            sm = summary(tag)
            cells.append(sm["test"]["LinesFMeasure"] if sm and sm.get("test") else np.nan)
        val = [summary(f"{s}_{STEM}_abl_{cond}_s42")["validation"]["LinesFMeasure"] for s in SOURCES]
        vals[cond] = dict(zip(SOURCES, val))
        rows.append((label, cells, float(np.mean(val))))
    head = (" & ".join(N[s] for s, _ in PAIRS) + r" & \\" + "\n        Config. & "
            + " & ".join(f"$\\to${N[t]}" for _, t in PAIRS))
    body = "\n".join(f"        {lab} & " + " & ".join("--" if np.isnan(c) else f2(c) for c in cells)
                     + f" & {f2(v)} \\\\" for lab, cells, v in rows)
    tex = rf"""\begin{{table}}[htbp]
    \centering
    \caption[RQ3 robustness ablation.]{{Robustness ablation (\gls{{fm}}, \%, seed 42): source$\to$target pairs as columns, configurations as rows. (a) keeps the CATMuS-pretrained proposal network, (b) adds train-only scale jitter with $s_{{\max}}=1.6$, (c) enlarges the mask ROI to 28, (d) adds box NMS to the validation grid, (e) freezes the stem and stages 0--1 and stops early on source validation; \emph{{post}} applies the post-processing rules. \emph{{Val.}} is the mean source-validation \gls{{fm}} over the five source models, the only quantity used to choose a configuration. Higher is better.}}
    \label{{tab:rq3-ablation}}
    \footnotesize
    \setlength{{\tabcolsep}}{{2.5pt}}
    \resizebox{{\textwidth}}{{!}}{{%
    \begin{{tabular}}{{@{{}}l*{{9}}{{r}}r@{{}}}}
        \toprule
        Source & {head} & Val. \\
        \midrule
{body}
        \bottomrule
    \end{{tabular}}}}
\end{{table}}
"""
    return tex, {lab: dict(zip([f"{s}->{t}" for s, t in PAIRS], cells)) | {"val": v} for lab, cells, v in rows}, vals


# --------------------------------------------------------------- final matrices
def final():
    A = pd.read_csv(EVAL / "final_ab/A.csv")
    B = pd.read_csv(EVAL / "final_ab/B.csv")
    base = pd.read_csv(EVAL / "rq3_matrix/A_LinesFMeasure.csv", index_col=0)[D]
    mat = lambda m, p: A[(A.model == m) & (A.post == p)].pivot(index="source", columns="target", values="FM")
    base_d1 = base.copy()
    base_d1.loc["NorHand_v3", D] = mat("base", False).loc["NorHand_v3", D]   # same page draw as ab
    ab, abp = mat("ab", False).reindex(index=D, columns=D), mat("ab", True).reindex(index=D, columns=D)
    return base, base_d1, ab, abp, A, B


def matrix_tex(m, label, caption, short):
    lines = []
    for r in D:
        cells = [(r"\cellcolor{gray!20}" if r == c else "") + f2(m.loc[r, c]) for c in D]
        tr = np.mean([m.loc[r, c] for c in D if c != r])
        lines.append(f"        {N[r]} & " + " & ".join(cells) + f" & {f2(m.loc[r, D].mean())} & {f2(tr)} \\\\")
    col = " & ".join(f2(np.mean([m.loc[s, t] for s in D if s != t])) for t in D)
    return rf"""\begin{{table}}[htbp]
    \centering
    \caption[{short}]{{{caption}}}
    \label{{{label}}}
    \footnotesize
    \setlength{{\tabcolsep}}{{3pt}}
    \begin{{tabular}}{{@{{}}l*{{7}}{{r}}rr@{{}}}}
        \toprule
        & \multicolumn{{7}}{{c}}{{Test collection}} & \multicolumn{{2}}{{c}}{{Mean}} \\
        \cmidrule(lr){{2-8}}\cmidrule(l){{9-10}}
        Train & {" & ".join(N[d] for d in D)} & All & Transfer \\
        \midrule
{chr(10).join(lines)}
        \midrule
        Transfer into & {col} & & \\
        \bottomrule
    \end{{tabular}}
\end{{table}}
"""


def loco_tex(B, bB):
    rows, acc = [], {k: [] for k in ("base", "ab", "abp", "val")}
    for d in D:
        b = bB.loc[d, d]
        a = B[(B.source == d) & (~B.post)].iloc[0]
        p = B[(B.source == d) & (B.post)].iloc[0]
        for k, v in zip(acc, (b, a.FM, p.FM, a.source_val_FM)):
            acc[k].append(v)
        rows.append(f"        {N[d]} & {f2(b)} & {f2(a.FM)} & {f2(p.FM)} & {f2(a.recall)} & {f2(a.precision)} \\\\")
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    tex = rf"""\begin{{table}}[htbp]
    \centering
    \caption[RQ3 LOCO-6 with the robust configuration.]{{\gls{{loco}}-6 results on each held-out collection (\%): the baseline configuration of Table~\ref{{tab:rq3-loco-summary}}, the robust configuration \emph{{ab}}, and \emph{{ab}} with post-processing. Recall and precision refer to \emph{{ab}} without post-processing. Training pages, validation pages, and thresholds are selected exactly as for the baseline. Higher is better; one run with seed 42.}}
    \label{{tab:rq3-loco-ab}}
    \small
    \begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
        \toprule
        & \multicolumn{{3}}{{c}}{{\gls{{fm}}}} & \multicolumn{{2}}{{c}}{{\emph{{ab}}}} \\
        \cmidrule(lr){{2-4}}\cmidrule(l){{5-6}}
        Held out & Baseline & \emph{{ab}} & \emph{{ab}} + post & \gls{{dr}} & \gls{{ra}} \\
        \midrule
{chr(10).join(rows)}
        \midrule
        Mean & {f2(m['base'])} & {f2(m['ab'])} & {f2(m['abp'])} & & \\
        \bottomrule
    \end{{tabular}}
\end{{table}}
"""
    return tex, m, acc


def sweep_tex():
    d = pd.read_csv(REPO / "99_evaluation/analysis/scale_sweep_out/scale_sweep.csv")
    scales = sorted(d.scale.unique())
    order = [("ONB", "GRPOLY"), ("RASAM", "GRPOLY"), ("RASM", "GRPOLY"), ("Phil_gr_130", "GRPOLY"),
             ("Pinkas", "GRPOLY"), ("NorHand_v3", "GRPOLY"), ("GRPOLY", "GRPOLY"), ("RASAM", "ONB"), ("RASM", "ONB")]
    rows = []
    for s, t in order:
        g = d[(d.source == s) & (d.target == t)].set_index("scale").LinesFMeasure
        best = g.max()
        cells = [(r"\textbf{" + f2(g[r]) + "}") if g[r] == best else f2(g[r]) for r in scales]
        rows.append(f"        {N[s]}$\\to${N[t]} & " + " & ".join(cells) + " \\\\")
    return rf"""\begin{{table}}[htbp]
    \centering
    \caption[RQ3 inference-time scale sweep.]{{Inference-time rescaling of the baseline protocol-A models (\gls{{fm}}, \%). A factor $r$ resizes each test page to $r$ times its usual size on the model input without retraining; $r=1$ is the normal evaluation. The best factor of each row is set in bold. The factor is chosen with knowledge of the test results and serves as a diagnostic, not as a method. Retrained models; at most 40 test pages per target.}}
    \label{{tab:rq3-scale-sweep}}
    \small
    \begin{{tabular}}{{@{{}}l{"r" * len(scales)}@{{}}}}
        \toprule
        Source$\to$target & {" & ".join(f"$r={r:g}$" for r in scales)} \\
        \midrule
{chr(10).join(rows)}
        \bottomrule
    \end{{tabular}}
\end{{table}}
""", d


# --------------------------------------------------------------- radar figure
def radar(panels, path):
    angles = np.linspace(0, 2 * np.pi, len(D), endpoint=False)
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 3.0), subplot_kw={"projection": "polar"})
    for ax, (title, series) in zip(axes, panels):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100"], fontsize=5, color="#6b6b6b")
        ax.set_rlabel_position(180)  # between RASM and Phil., away from most data
        ax.set_xticks(angles)
        ax.set_xticklabels([N[d] for d in D], fontsize=6.5, color="#2b2b2b")
        ax.tick_params(axis="x", pad=2)
        ax.grid(color="#d9d9d6", linewidth=0.45)
        ax.spines["polar"].set_visible(False)
        # heptagonal grid instead of circles
        ax.yaxis.grid(False)
        for r in (25, 50, 75, 100):
            ax.plot(np.append(angles, angles[0]), [r] * (len(D) + 1), color="#d9d9d6", linewidth=0.45, zorder=0)
        for name, values in series.items():
            v = np.append(values, values[0])
            a = np.append(angles, angles[0])
            ls, mk = STYLES[name]
            ax.plot(a, v, linestyle=ls, marker=mk, markersize=2.8, linewidth=1.3, color=COLORS[name],
                    label=name, zorder=3, markeredgecolor="white", markeredgewidth=0.5)
            ax.fill(a, v, color=COLORS[name], alpha=0.07, zorder=2)
        ax.set_title(title, fontsize=7.5, pad=10, color="#2b2b2b")
    handles, labels = axes[0].get_legend_handles_labels()
    names = {"base": "Baseline", "ab": "Robust (ab)", "ab+post": "Robust + post-processing"}
    fig.legend(handles, [names[l] for l in labels], loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0.0, 0.07, 1, 1), w_pad=1.2)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_TEX.mkdir(exist_ok=True)
    abl_tex, abl, vals = ablation()
    base, base_d1, ab, abp, A, B = final()
    bB = pd.read_csv(EVAL / "rq3_matrix/B_LinesFMeasure.csv", index_col=0)
    (OUT_TEX / "ablation.tex").write_text(abl_tex)
    (OUT_TEX / "matrix_ab.tex").write_text(matrix_tex(
        ab, "tab:rq3-matrix-ab-fm",
        r"Protocol~A transfer matrix with the robust configuration \emph{ab} (\gls{fm}, \%), laid out as Table~\ref{tab:rq3-matrix-a-fm}. NorHand v3 is trained on the page draw d1 (Section~\ref{sec:results-rq3-failures}). \emph{Transfer into} averages the six off-diagonal entries of each column. Higher is better; one run with seed 42.",
        "RQ3 transfer matrix, robust configuration."))
    (OUT_TEX / "matrix_abpost.tex").write_text(matrix_tex(
        abp, "tab:rq3-matrix-abpost-fm",
        r"Protocol~A transfer matrix with the robust configuration \emph{ab} and post-processing (\gls{fm}, \%), laid out as Table~\ref{tab:rq3-matrix-ab-fm}. Thresholds are re-selected on the source validation pages with post-processing enabled. Higher is better; one run with seed 42.",
        "RQ3 transfer matrix, robust configuration with post-processing."))
    lt, lm, lacc = loco_tex(B, bB)
    (OUT_TEX / "loco_ab.tex").write_text(lt)
    st, sweep = sweep_tex()
    (OUT_TEX / "scale_sweep.tex").write_text(st)

    diag = lambda m: np.array([m.loc[d, d] for d in D])
    into = lambda m: np.array([np.mean([m.loc[s, t] for s in D if s != t]) for t in D])
    panels = [("In-domain, three pages", {"base": 100 * diag(base_d1), "ab": 100 * diag(ab), "ab+post": 100 * diag(abp)}),
              ("Mean transfer into collection", {"base": 100 * into(base_d1), "ab": 100 * into(ab), "ab+post": 100 * into(abp)}),
              (r"LOCO-6 on held-out collection", {"base": 100 * np.array(lacc["base"]),
                                                     "ab": 100 * np.array(lacc["ab"]), "ab+post": 100 * np.array(lacc["abp"])})]
    radar(panels, THESIS / "graphics/rq3_radar.pdf")

    off = lambda m: float(np.mean([m.loc[s, t] for s in D for t in D if s != t]))
    numbers = {
        "diag": {k: float(diag(m).mean()) for k, m in (("base", base), ("base_d1", base_d1), ("ab", ab), ("abp", abp))},
        "transfer": {k: off(m) for k, m in (("base", base), ("base_d1", base_d1), ("ab", ab), ("abp", abp))},
        "into": {k: dict(zip(D, into(m).tolist())) for k, m in (("base_d1", base_d1), ("ab", ab), ("abp", abp))},
        "diag_each": {k: dict(zip(D, diag(m).tolist())) for k, m in (("base_d1", base_d1), ("ab", ab), ("abp", abp))},
        "loco": lm, "ablation": abl, "val": vals,
        "ab_cells": {f"{s}->{t}": float(ab.loc[s, t]) for s in D for t in D},
        "abp_cells": {f"{s}->{t}": float(abp.loc[s, t]) for s in D for t in D},
        "base_d1_row": {t: float(base_d1.loc["NorHand_v3", t]) for t in D},
        "decision": json.loads((EVAL / "final_ab/decision.json").read_text()),
    }
    (OUT_TEX / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps({k: numbers[k] for k in ("diag", "transfer", "loco")}, indent=1))


if __name__ == "__main__":
    main()
