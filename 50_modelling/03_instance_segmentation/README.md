# Mask2Former on U-DIADS-TL

Hugging Face `facebook/mask2former-swin-tiny-coco-instance` fine-tuning and
instance-level evaluation for `Latin14396`, `Latin2`, and `Syr341`.

The script reads the existing
`00_data/U-DIADS-TL/coco_dataset_<subset>/{train,val,test}.json` files. Each
connected text-line component is one instance. Predictions are written as
16-bit instance-label PNGs and scored with the repository's U-DIADS/Zottin
definitions (`Pixel_IU`, `Line_IU`, `DR`, `RA`, `FM`).

## Environment

From the repository root:

```bash
python3 -m venv --system-site-packages ~/.venvs/fsl-mask2former
~/.venvs/fsl-mask2former/bin/python -m pip install \
  -r 50_modelling/03_instance_segmentation/requirements.txt
```

`--system-site-packages` reuses the repository machine's CUDA-enabled PyTorch
and Transformers installation. A clean environment can instead install the
root `requirements.txt` first.

## Colab tiled 1152 run

Open
[`colab_03_mask2former_udiads.ipynb`](colab_03_mask2former_udiads.ipynb) in a
GPU Colab runtime. It keeps a 1152 px maximum image side, batch 1, seed 42,
Drive checkpoint sync, and a real forward/backward VRAM guard. Latin14396 uses
full pages; Latin2 and Syr341 use two and three vertical tiles. All stages use
100 queries and full-resolution instance targets.

The dense stages both initialize directly from the Latin14396 checkpoint and
use gradient accumulation 1. Their epoch counts are calculated for about 1000
optimizer updates (167 epochs for Latin2 and 112 for Syr341 with the current
three training pages). Checkpoints maximize validation FM every ten epochs;
score/mask/NMS thresholds are then selected on validation before one test run.
The notebook supports BF16 Colab GPUs and T4 FP16 through GradScaler.

## Commands

One-batch smoke test:

```bash
~/.venvs/fsl-mask2former/bin/python \
  50_modelling/03_instance_segmentation/mask2former_udiads.py run \
  --subsets Latin14396 --epochs 1 --max-train-images 1 \
  --max-eval-images 1 --run-name smoke
```

Train and test all three subsets:

```bash
~/.venvs/fsl-mask2former/bin/python \
  50_modelling/03_instance_segmentation/mask2former_udiads.py run \
  --subsets all --epochs 200
```

For the best local results, train the subsets sequentially. Latin14396 learns
the text-line task first; the two dense multi-column subsets reuse that
checkpoint and train one physical page column per sample:

```bash
PY=~/.venvs/fsl-mask2former/bin/python
SCRIPT=50_modelling/03_instance_segmentation/mask2former_udiads.py

$PY $SCRIPT run --subsets Latin14396 --epochs 150 --num-queries 100 \
  --shortest-edge 704 --longest-edge 1056 --mask-label-stride 1 \
  --train-num-points 12544 --eval-every 15 \
  --score-threshold 0.6 --mask-threshold 0.4 \
  --run-name m2f_swin_t_704_q100_fullmask_150ep

LATIN_CKPT=80_models/03_instance_segmentation/u-diads-tl/Latin14396/m2f_swin_t_704_q100_fullmask_150ep

$PY $SCRIPT run --subsets Latin2 --epochs 100 --num-queries 100 \
  --shortest-edge 576 --longest-edge 1440 --mask-label-stride 1 \
  --train-num-points 12544 --train-vertical-tiles 2 --vertical-tiles 2 \
  --init-checkpoint "$LATIN_CKPT" --eval-every 20 \
  --score-threshold 0.8 --mask-threshold 0.5 --nms-threshold 0.3 \
  --run-name m2f_swin_t_column2_q100_fullmask_transfer_100ep

$PY $SCRIPT run --subsets Syr341 --epochs 100 --num-queries 100 \
  --shortest-edge 576 --longest-edge 1728 --mask-label-stride 1 \
  --train-num-points 12544 --train-vertical-tiles 3 --vertical-tiles 3 \
  --init-checkpoint "$LATIN_CKPT" --eval-every 20 \
  --score-threshold 0.8 --mask-threshold 0.5 --nms-threshold 0.3 \
  --run-name m2f_swin_t_column3_q100_fullmask_transfer_100ep
```

The thresholds above were selected on `--eval-split val`; see `RESULTS.md` for
the full test metrics. For a new dataset, tune them on validation rather than
copying them blindly.

Evaluate an existing Hugging Face checkpoint on all three subsets:

```bash
~/.venvs/fsl-mask2former/bin/python \
  50_modelling/03_instance_segmentation/mask2former_udiads.py evaluate \
  --subsets all \
  --checkpoint 80_models/03_Mask2Former/u-diads-tl/Latin14396/m2f_latin14396_swin-t \
  --run-name legacy_latin14396_cross_subset
```

New checkpoints go to
`80_models/03_instance_segmentation/u-diads-tl/<subset>/<run-name>`; predictions
and CSV/JSON metrics go to
`99_evaluation/03_instance_segmentation/u-diads-tl/<subset>/<run-name>`.

The default 200 queries cover full pages containing up to 190 annotated lines.
Column tiling reduces the per-sample count below 100, allowing all query slots
to inherit useful COCO/task weights. The default 512/768 resize and
quarter-resolution targets are conservative 8 GB settings; the best runs use
full-resolution targets and subset-specific input sizes. `--mask-label-bf16`
is available for borderline memory cases.

The loader also repairs the exact extra `encoder.backbone` nesting found in
the old `80_models/03_Mask2Former` checkpoints. Loading those folders directly
with `from_pretrained` reports hundreds of missing/unexpected Swin tensors and
silently evaluates a randomly initialized backbone.
