# Local Mask2Former experiments — 2026-08-18

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB. Software: PyTorch 2.9.1
CUDA 12.8, Transformers 5.14.1, SciPy 1.18.0. All models start from
`facebook/mask2former-swin-tiny-coco-instance`; model selection and
post-processing thresholds use validation data only.

## Best test results

| subset | layout / input | queries | score / mask | Pixel IU | Line IU | DR | RA | FM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Latin14396 | full page, 704/1056 | 100 | 0.60 / 0.40 | 0.6071 | 0.7361 | 0.6371 | 0.7286 | **0.6795** |
| Latin2 | 2 columns, 576/1440 | 100 | 0.80 / 0.50 | 0.6155 | 0.7281 | 0.4707 | 0.4598 | **0.4641** |
| Syr341 | 3 columns, 576/1728 | 100 | 0.80 / 0.50 | 0.5811 | 0.5203 | 0.3341 | 0.3034 | **0.3179** |

Every row is the mean over all 15 test pages. The mean over the three subsets
is Pixel IU 0.6012, Line IU 0.6615, and FM 0.4872.

Checkpoints:

- `Latin14396/m2f_swin_t_704_q100_fullmask_150ep`
- `Latin2/m2f_swin_t_column2_q100_fullmask_transfer_100ep`
- `Syr341/m2f_swin_t_column3_q100_fullmask_transfer_100ep`

They are under `80_models/03_instance_segmentation/u-diads-tl`. Tuned instance
maps and per-page CSVs use the same names with `_tuned` under
`99_evaluation/03_instance_segmentation/u-diads-tl/<subset>`.

## What changed the result

The first end-to-end feasibility run used 512/768 inputs, quarter-resolution
target masks, 200 queries, and 20 epochs. Its FM scores were 0.0107, 0, and 0
for Latin14396, Latin2, and Syr341. The useful improvements were:

1. Keep full-resolution target masks (`--mask-label-stride 1`). On
   Latin14396 this alone raised FM from 0.192 for half-resolution masks to
   0.679 for full masks.
2. Initialize the other scripts from the fine-tuned Latin14396 checkpoint.
   This halves the initial validation loss compared with restarting from COCO.
3. Train and infer page columns independently. Latin2 has two columns and
   Syr341 has three; tiling preserves thin glyph strokes and lets the model use
   the 100 already-trained queries rather than learning 60–100 new query slots
   from only three pages.
4. Use the aspect-preserving custom text-line postprocessor and select its
   score/mask thresholds on the ten validation pages only.

Column training raised Latin2 test FM from 0.1758 (transferred full page) to
0.4641, and Syr341 from 0.0143 (transferred full page) to 0.3179. The two
column models use mask-IoU NMS 0.3, also selected on validation.

## Legacy checkpoint finding

The old `80_models/03_Mask2Former` Swin-T checkpoint stores its backbone under
an extra `encoder.backbone` state-dict level. Vanilla `from_pretrained`
silently leaves 227 backbone tensors random. The loader repairs that wrapper
exactly, but the correctly converted old Latin14396 checkpoint still scored
Line IU 0, so none of the best runs use it.
