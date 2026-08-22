"""Render the U-DIADS size-axis matrix as markdown, one table per size bracket.

Reads the CSV written by eval_udiads_matrix.py and emits
99_evaluation/01_simple_segmentation/u-diads-tl/size_axis_results.md

Rows are grouped subset-first, then model; the best value per (subset, metric)
inside each bracket is bolded. Bolding compares the ROUNDED values, so visually
tied cells are all bolded rather than one of them silently winning on a digit
the table does not show.
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "99_evaluation/01_simple_segmentation/u-diads-tl/size_axis_zottin.csv"
OUT = REPO / "99_evaluation/01_simple_segmentation/u-diads-tl/size_axis_results.md"

DECIMALS = 2
SUBSETS = ["Latin14396", "Latin2", "Syr341"]
METRICS = ["Pixel_IU", "Line_IU", "DR", "RA", "FM"]
INIT_LABEL = {"imagenet": "IN", "random": "rand", "dinov3": "DINOv3"}
# Order arms consistently: pretrained, random, then SSL where it exists.
INIT_ORDER = ["imagenet", "random", "dinov3"]

SHORT = {
    "tu-convnext_femto": "convnext_femto", "tu-convnext_pico": "convnext_pico",
    "tu-convnext_tiny": "convnext_tiny", "tu-pvt_v2_b0": "pvt_v2_b0",
    "tu-pvt_v2_b1": "pvt_v2_b1", "tu-pvt_v2_b2": "pvt_v2_b2",
    "vit_tiny_patch16_224.augreg_in21k": "vit_tiny",
    "vit_small_patch16_224.augreg_in21k": "vit_small",
}
PARAMS = {
    "tu-convnext_femto": 4.83, "tu-convnext_pico": 8.53, "tu-convnext_tiny": 27.82,
    "tu-pvt_v2_b0": 3.41, "tu-pvt_v2_b1": 13.50, "tu-pvt_v2_b2": 24.85,
    "vit_tiny_patch16_224.augreg_in21k": 5.5,
    "vit_small_patch16_224.augreg_in21k": 21.7,
}
BRACKETS = {
    "XS": ["tu-convnext_femto", "tu-pvt_v2_b0", "vit_tiny_patch16_224.augreg_in21k"],
    "S": ["tu-convnext_pico", "tu-pvt_v2_b1"],
    "M": ["tu-convnext_tiny", "tu-pvt_v2_b2", "vit_small_patch16_224.augreg_in21k"],
}

HEADER = """# U-DIADS-TL — size axis, U-Net semantic segmentation

Test split, Zottin metric. 54 cells: 8 encoders x 3 subsets x 2 initialisations,
plus a DINO-SSL arm for the two bracket-M architectures that have SSL weights.
One recipe throughout: 100 epochs, k_shot=3, crop 448, batch 4, lr 1e-3,
lr_backbone 1e-4, weight_decay 1e-4, lambda_boundary 0.

Source: `99_evaluation/01_simple_segmentation/u-diads-tl/size_axis_zottin.csv`
Training: `50_modelling/01_simple_segmentation/run_train_udiads_size_axis{,_random}.sh`
Scoring: `50_modelling/01_simple_segmentation/eval_udiads_matrix.py`

`IN` = ImageNet-pretrained encoder, `rand` = random init, `DINOv3` = self-supervised
DINOv3 weights. Values rounded to {dec} decimals; bold marks the best value per
subset within each table.

SSL exists only at bracket M: the smallest published DINOv3 ConvNeXt is tiny
(27.8M) and the smallest DINOv3/DINOv2 ViT is small (~21M); nothing below that was
released, and there are no SSL weights for hierarchical ViTs (PvtV2) at any size.
"""

FOOTER = """
## Reading these numbers

**Size barely moves anything.** Best FM per bracket on Latin14396 is 0.95 (XS),
0.96 (S), 0.96 (M); on Syr341 it is 0.77, 0.78, 0.78. Going from 3.4M to 27.8M
parameters buys about one point on the hard subset. The spread between encoder
families inside one bracket is larger than the spread between brackets.

**The two architectures with an SSL arm disagree about what pretraining is for.**
Mean FM over the three subsets:

| architecture | random | ImageNet | DINOv3 |
|---|---:|---:|---:|
| convnext_tiny | 0.7989 | **0.8738** | 0.8581 |
| vit_small/16 | 0.8297 | 0.8404 | **0.8638** |

For the CNN, supervised ImageNet wins and DINOv3 lands between it and random --
self-supervision is *worse* than supervised pretraining here. For the flat ViT the
order reverses: DINOv3 is best on all three subsets individually, and it is the only
thing that moves that architecture at all (ImageNet buys +0.011 over random, DINOv3
+0.034). The best flat-ViT cell in the whole matrix is `vit_small (DINOv3)`.

**The random-init penalty lands on RA, not on pixels.** On Syr341, convnext_femto
scores a *higher* Pixel_IU at random init than with ImageNet (0.71 vs 0.69) and an
almost identical Line_IU, yet RA collapses from 0.77 to 0.55. The model still finds
and fills the lines; it emits extra components. That is a line-separation failure,
not a feature-quality one, which is why FM drops while the pixel metrics do not.

**Line_IU hardly separates encoders at all.** Whole-table range is 0.93-0.98 on
Latin14396 and 0.88-0.92 on Syr341, all three arms included. Any ranking by FM is
effectively a ranking by DR/RA.

## Caveats

- **`lr_backbone` is 1e-4 in every arm.** Identical recipe, so the arms isolate the
  weights -- but a randomly initialised encoder trains 10x slower than its own
  decoder. Part of the CNN random gap (-0.075 FM) is plausibly this, not the
  initialisation. A control run at `lr_backbone=1e-3` is needed before treating that
  gap as measured. It does not affect the ImageNet-vs-DINOv3 comparison, where both
  arms start from pretrained weights.
- **Flat-ViT rows here train the backbone**; the DIVA matrix elsewhere in this repo
  keeps it frozen. These rows are therefore not comparable with those.
- **DINOv3 ViT loads through timm, not Meta's torch.hub**, which is gated. Same
  `lvd1689m` checkpoint, weights verified identical; see `dinov3_vits16_timm` in
  `71_misc/hier_encoder/backbones.py`.
- **Not comparable with `udiads_comparison/zottin_metrics.csv`**, which mixes
  HPO-tuned runs (60 epochs, per-encoder lr/wd/lambda) with fixed-recipe ones.
"""


def main():
    df = pd.read_csv(CSV)
    # .replace, not .format: HEADER contains a shell brace expansion
    # ("run_train_udiads_size_axis{,_random}.sh") that str.format would try to
    # read as a field name.
    lines = [HEADER.replace("{dec}", str(DECIMALS))]

    for bracket, encoders in BRACKETS.items():
        spec = " · ".join(f"{SHORT[e]} {PARAMS[e]:.1f}M" for e in encoders)
        lines.append(f"\n## {bracket} ({spec})\n")
        lines.append("| subset | model | " + " | ".join(METRICS) + " |")
        lines.append("|---|---|" + "---:|" * len(METRICS))

        for subset in SUBSETS:
            block = df[(df.subset == subset) & (df.encoder.isin(encoders))]
            if block.empty:
                continue
            best = {m: block[m].round(DECIMALS).max() for m in METRICS}
            for enc in encoders:
                for init in INIT_ORDER:
                    row = block[(block.encoder == enc) & (block.init == init)]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    cells = []
                    for m in METRICS:
                        v = round(float(row[m]), DECIMALS)
                        s = f"{v:.{DECIMALS}f}"
                        cells.append(f"**{s}**" if v == best[m] else s)
                    name = f"{SHORT[enc]} ({INIT_LABEL[init]})"
                    lines.append(f"| {subset} | {name} | " + " | ".join(cells) + " |")

    lines.append(FOOTER)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
