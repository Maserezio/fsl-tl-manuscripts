# Few-shot text line segmentation for historical manuscripts — experiments

Code, models, and results of the master's thesis. Every experiment is started through `./run.sh` (see below).

## Layout

```
00_data/                        datasets (DIVA-HisDB, U-DIADS-TL, CATMuS, cBAD, RQ3 collections and their splits)
  scripts/                      dataset converters (PAGE XML / masks -> COCO, YOLO, previews, audits)
50_modelling/
  common/                       shared code: few-shot page selection, hier_encoder backbones, evaluation helpers
  semantic_segmentation/
    unet/                       U-Net with projection cuts (training, inference, postprocessing, Colab generators)
    center_line/                center-line model (LineField U-Net, decoders, CATMuS pretraining, WiSE-FT,
                                DIVA-HisDB and RQ3 runs)
  instance_segmentation/
    rtdetr_bbox_unet/           RT-DETR detector + BBox U-Net (crop segmenter), CATMuS pretraining notebook
    mask_rcnn/                  Mask R-CNN (DIVA-HisDB, U-DIADS-TL, CATMuS pretraining, cBAD distillation)
      cross_collection/         RQ3 Mask R-CNN runs (zero-shot, single collection, LOCO, WiSE-FT)
    dp_seam/                    dynamic-programming seam for the masks of detected lines
  legacy_runners/               run_rq3_wise_test.sh only (WiSE-FT of Mask R-CNN in RQ3, not ported to run.sh)
80_models/                      checkpoints, same semantic_segmentation/... instance_segmentation/... layout
99_evaluation/                  predictions and metrics, same layout
  summaries/                    aggregated tables (RQ2 manifest, curves, ...)
  scripts/                      table fillers for the thesis
  analysis/                     analyses and figures of chapters 3-6 (error types, glosses, word gaps, HTR, ...)
  logs/                         run logs
run.sh                          single entry point
```

The analysis scripts and the center-line modules import each other; `_paths.py` in both folders puts
`50_modelling/semantic_segmentation/center_line`, `99_evaluation/analysis` and `50_modelling/common` on `sys.path`.

## Thesis pipelines

| Thesis name | Code | Models | Results |
|---|---|---|---|
| U-Net with projection cuts | `50_modelling/semantic_segmentation/unet` | `80_models/semantic_segmentation/unet` | `99_evaluation/semantic_segmentation/unet` |
| Center-line model | `50_modelling/semantic_segmentation/center_line` | `80_models/semantic_segmentation/center_line` | `99_evaluation/semantic_segmentation/center_line` |
| RT-DETR + BBox U-Net | `50_modelling/instance_segmentation/rtdetr_bbox_unet` | `80_models/instance_segmentation/rtdetr_bbox_unet` | `99_evaluation/instance_segmentation/rtdetr_bbox_unet` |
| Mask R-CNN (+ BBox U-Net on U-DIADS-TL) | `50_modelling/instance_segmentation/mask_rcnn` | `80_models/instance_segmentation/mask_rcnn` | `99_evaluation/instance_segmentation/mask_rcnn` |
| DP seam | `50_modelling/instance_segmentation/dp_seam` | — | `99_evaluation/instance_segmentation/dp_seam` |

## Running

```
./run.sh help
./run.sh rq1-rtdetr "catmus" "convnext_tiny"      # RQ1, one initialization and encoder
./run.sh rq1-maskrcnn convnext_tiny_catmus
./run.sh pretrain rtdetr convnext_tiny             # CATMuS pretraining (1152 px, 30 epochs)
./run.sh rq2-maskrcnn                              # all 77 page sets of the RQ2 manifest
./run.sh rq3-centerline test                       # cache + score the RQ3 center-line models on the test pages
./run.sh figures                                   # regenerate the thesis figures
```

All commands skip finished runs. Long runs: `systemd-inhibit --what=sleep:idle ./run.sh ...`.
U-Net runs and parts of RQ2 were trained on Colab; `./run.sh rq1-unet` and `./run.sh pretrain colab` write those notebooks.

Environment: `.venv` (Python 3.12, torch 2.9, transformers 5.14.1); DIVA evaluator jar in `~/Thesis/DIVA_Line_Segmentation_Evaluator`.
