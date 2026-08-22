# Few-shot text line segmentation for historical manuscripts — RQ1 work plan

RQ1: backbone comparison. Two axes per track, run on both datasets.

**Datasets** — DIVA-HisDB (`CB55`, `CS18`, `CS863`), U-DIADS-TL (`Latin14396`, `Latin2`, `Syr341`).
**Metrics** — Pixel IU, Line IU, DR, RA, FM.

Status: **done** = trained and scored · **partial** = some subsets only · **todo** = not started.
A cell counts as done only when every subset of that dataset is scored under one shared recipe.

DINO SSL exists only in bracket M — the smallest published DINOv3 ConvNeXt is tiny
(27.8M), the smallest DINOv3/DINOv2 ViT is small (~21M), and no SSL weights exist for
hierarchical ViTs at any size. Those cells are marked `n/a`, not `todo`.

---

## 01. Simple segmentation (semantic)

SMP U-Net. Baseline outside the matrix: `resnet34` (21.3M).

### Pretrain axis (bracket M)

| family | backbone | Random | ImageNet | DINO SSL |
|---|---|---|---|---|
| CNN | `tu-convnext_tiny` 27.8M | IN: **done** · DIVA: todo | IN: **done** · DIVA: todo | IN: **done** · DIVA: todo |
| ViT hier. | `tu-pvt_v2_b2` 24.9M | IN: **done** · DIVA: todo | IN: **done** · DIVA: todo | `n/a` |
| ViT flat (SFP) | `vit_small_patch16` 21.7M | IN: **done** · DIVA: todo | IN: **done** · DIVA: todo | IN: **done** · DIVA: todo |

SSL checkpoints: `tu-convnext_tiny.dinov3_lvd1689m`, `vit_small_patch16_dinov3`.

### Size axis (Random / ImageNet)

| bracket | CNN | ViT hier. | ViT flat (SFP) |
|---|---|---|---|
| XS | `tu-convnext_femto` 4.8M | `tu-pvt_v2_b0` 3.4M | `vit_tiny_patch16` 5.5M |
| S | `tu-convnext_pico` 8.5M | `tu-pvt_v2_b1` 13.5M | — |
| M | `tu-convnext_tiny` 27.8M | `tu-pvt_v2_b2` 24.9M | `vit_small_patch16` 21.7M |

| dataset | status |
|---|---|
| U-DIADS-TL | **done** — 8 encoders x 2 arms x 3 subsets = 48 cells |
| DIVA-HisDB | **todo** — no size-axis runs exist |

### Additional (outside both axes)

| backbone | U-DIADS-TL | DIVA-HisDB |
|---|---|---|
| `dinov2` (ViT-S/14, SFP) | todo | **done** (CB55, CS18) |
| `dinov2_reg` (ViT-S/14 +reg, SFP) | todo | **done** (CB55, CS18) |
| `am-radio` (RADIO v2.5-B, SFP) | partial (Latin2 only) | todo |

### Loss study — BCE / Tversky / SuperVoxel

| dataset | status |
|---|---|
| U-DIADS-TL | **todo** — whole matrix trained on BCE (`supervoxel.enabled: false`, `lambda_boundary: 0`) |
| DIVA-HisDB | **todo** |

### Legacy runs, not part of any axis

`resnet34`, `resnet50`, `efficientnet-b4`, `mit_b2`, plain `tu-convnext_tiny`, `tu-pvt_v2_b2`
on U-DIADS-TL are HPO-tuned (60 or 200 epochs, per-encoder lr / weight decay /
lambda_boundary). Not comparable with the fixed-recipe matrix; kept for reference only.

---

## 02. Two-stage (bbox detector + crop segmentor)

### RT-DETR — pretrain axis (bracket M)

| family | backbone | Random | ImageNet | DINO SSL |
|---|---|---|---|---|
| CNN | `ConvNextConfig` tiny 27.8M | CB55: **done** | todo | todo (`dinov3-convnext-tiny`) |
| ViT hier. | `PvtV2Config` b2 24.9M | CB55: **done** | CB55: **done** | `n/a` |
| ViT flat (SFP) | `ViTConfig` small 21.7M | todo | todo | todo (`dinov3-vits16`) |

Baseline outside the matrix: stock R50-vd COCO 23.5M — CB55 **done**.

### RT-DETR — size axis (Random / ImageNet)

| bracket | CNN | ViT hier. | ViT flat (SFP) |
|---|---|---|---|
| XS | `ConvNextConfig` femto 5.2M | `PvtV2Config` b0 3.4M | `ViTConfig` tiny 5.7M |
| S | `ConvNextConfig` pico 9.0M | `PvtV2Config` b1 13.1M | — |
| M | `ConvNextConfig` tiny 27.8M | `PvtV2Config` b2 24.9M | `ViTConfig` small 21.7M |

| dataset | status |
|---|---|
| DIVA-HisDB | **partial** — CB55 only, and only b0 / b2 / ConvNeXt-tiny / stock. CS18, CS863 todo |
| U-DIADS-TL | **todo** |

### YOLO — reference track, not an RQ1 axis

| arm | status |
|---|---|
| detector size (YOLOv8 n/s/m) | **done** — metrics in training logs |
| pretrain (DINO distill / cBAD / COCO / random) | **done** — CB55 |
| swappable backbones (10 encoders x 3 DIVA subsets) | **done** — `02_2stage/detection_eval.csv` |
| zero-shot / cross-collection (bbox only) | **partial** — cBAD to CB55 |

### Second-stage segmentor

| item | status |
|---|---|
| U-Net loss study (BCE / Tversky / SuperVoxel) | **done** — CB55 |
| architecture sweep (6 variants) | **done** — CB55 |

### Additional detectors

| model | status |
|---|---|
| RF-DETR | **done** — CB55, archived under `80_models/.../detection/_archive/` |
| LT-DETR, DEIM, TAO-DETR | todo |
| Catmus pretraining | todo |

---

## 03. Instance segmentation

Mask2Former (HF), swappable backbones. Baseline outside the matrix: Swin-T COCO 28M — todo.

### Pretrain axis (bracket M)

| family | backbone | Random | ImageNet | DINO SSL |
|---|---|---|---|---|
| CNN | `ConvNextConfig` tiny 27.8M | todo | todo | todo |
| ViT hier. | `PvtV2Config` b2 24.9M | todo | todo | `n/a` |
| ViT flat (SFP) | `ViTConfig` small 21.7M | todo | todo | todo |

### Size axis (Random / ImageNet)

| bracket | CNN | ViT hier. | ViT flat (SFP) |
|---|---|---|---|
| XS | `ConvNextConfig` femto 5.2M | `PvtV2Config` b0 3.4M | `ViTConfig` tiny 5.7M |
| S | `ConvNextConfig` pico 9.0M | `PvtV2Config` b1 13.1M | — |
| M | `ConvNextConfig` tiny 27.8M | `PvtV2Config` b2 24.9M | `ViTConfig` small 21.7M |

| dataset | status |
|---|---|
| DIVA-HisDB | **todo** — two R50 runs exist (CS18, CS863) but score LineIU 0.02-0.04 |
| U-DIADS-TL | **todo** — 4 checkpoints on Latin14396 (resnet-34/50, swin-t), never scored |

### Additional (not matrix cells)

| model | status |
|---|---|
| YOLOv8m-seg | **done** — Mask mAP50 0.955 |
| RF-DETR-Seg | todo |
| EoMT | todo |

---

## Where things live

| what | path |
|---|---|
| training scripts | `50_modelling/{01_simple_segmentation,02_2stage}/` |
| checkpoints | `80_models/` |
| metrics, PAGE-XML | `99_evaluation/` |
| U-DIADS size-axis results | `99_evaluation/01_simple_segmentation/u-diads-tl/size_axis_results.md` |
| RT-DETR backbone results | `99_evaluation/02_2stage/diva-hisdb/rtdetr_hf/` |
