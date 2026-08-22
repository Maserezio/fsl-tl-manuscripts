# RT-DETR (HuggingFace) inside 02_2stage

A second stage-1 detector alongside the Ultralytics YOLO path. Stage 2 is unchanged:
the same crop segmenter turns each detected box into a polygon, so LineIU/PixelIU are
comparable with every other detector in this stage.

## Files

| file | role |
|---|---|
| `train_rtdetr_hf.py` | trains one arm; constants at the top, arm selected by env vars |
| `predict_and_eval_rtdetr_diva.py` | detections → PAGE-XML → official DIVA Java evaluator |
| `run_rtdetr_backbone_matrix.sh` | trains the three local backbone arms in sequence |
| `colab_rtdetr_pvtv2_b2.ipynb` | same recipe on Colab for PvtV2-b2 (does not fit on 8 GB) |
| `experiments/diva_cb55_rtdetr_hf.yaml` | the run documented in this stage's experiment schema |

## Where things live

```
80_models/02_2stage/diva-hisdb/detection/rtdetr_hf/<arm>/   checkpoints + metrics_summary.json
99_evaluation/02_2stage/diva-hisdb/rtdetr_hf/<arm>/pred_xml/  PAGE-XML + per-page diva_results.csv
99_evaluation/02_2stage/diva-hisdb/rtdetr_hf/results.csv      one row per arm
99_evaluation/02_2stage/diva-hisdb/rtdetr_hf/backbone_matrix.csv  the comparison table
80_models/third_party/                                        DEIMv2, RT-DETR-main clones
80_models/02_2stage/diva-hisdb/detection/_archive/            RF-DETR + DEIMv2 runs, kept but retired
```

## Usage

```bash
# one arm (BACKBONE: "" = stock R50-vd | convnext | pvt_v2)
BACKBONE=pvt_v2 RUN_NAME=rtdetr_bb_pvt_v2 IMAGE_SIZE=1152 python train_rtdetr_hf.py

# all three local arms
bash run_rtdetr_backbone_matrix.sh

# score a trained arm (writes PAGE-XML, runs the Java evaluator, updates results.csv)
RUN_NAME=rtdetr_bb_pvt_v2 python predict_and_eval_rtdetr_diva.py

# SMOKE=1 gives a 2-epoch end-to-end check into <arm>_smoke/
```

The DIVA evaluator jar lives outside the repo; override its location with
`DIVA_EVALUATOR_JAR=/path/to/LineSegmentationEvaluator.jar`.

## Two things that will bite you

**Filter boxes to the GT `TextRegion` before scoring.** TASK-2 ground truth only
annotates lines inside that region, while `coco_dataset_CB55` also labels the marginal
gloss lines outside it. Verified on all 10 test pages: COCO boxes whose centre falls in
the region match the TASK-2 line count exactly. Without the filter every correct gloss
detection scores as a false positive and LineIU collapses (measured: 0.181 vs 0.968).

**Post-process against the square canvas, not `(H, W)`.** The HF processor resizes the
longest side then pads to a square, so boxes are normalised against `max(H, W)`. Calling
`post_process_object_detection(target_sizes=(H, W))` compresses x by `W/H` — 0.75 on
CB55 — and drives LineIU to 0.0000 on every page. This is what was wrong with the
earlier `hf_detr_cb55` export.

## Results (CB55 test, imgsz 1152, identical recipe)

| backbone | params | mAP50 | mAP50-95 | mAR@100 | LineIU | PixelIU | LineF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock R50-vd (COCO) | 23,474,016 | 0.8820 | 0.7733 | 0.8208 | 0.9589 | 0.8819 | 0.9788 |
| PvtV2-b0 (random) | 3,409,760 | 0.8739 | 0.6528 | 0.7183 | 0.9563 | 0.8739 | 0.9772 |
| PvtV2-b2 (random) | 24,849,856 | 0.8956 | 0.6826 | 0.7457 | 0.9442 | 0.8668 | 0.9709 |
| PvtV2-b2 (ImageNet) | 24,849,856 | 0.8782 | 0.7282 | 0.7882 | 0.9337 | 0.8543 | 0.9650 |
| ConvNeXt-tiny (random) | 27,821,280 | 0.8177 | 0.6132 | 0.7043 | 0.9135 | 0.8345 | 0.9534 |

Backbone pretraining barely moves the end metric once encoder/decoder are COCO-pretrained:
ImageNet-initialised b2 is *worse* on LineIU than the random one (0.9337 vs 0.9442), and a
random 3.4M b0 lands within 0.3 % of the COCO-pretrained 23.5M R50-vd. Scaling inside one
family does not help either — b0 beats b2 on LineIU at 7x fewer parameters.
