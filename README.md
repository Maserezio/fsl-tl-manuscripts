# Few-Shot Text Line Segmentation for Historical Manuscripts

This repository contains the code and resources for my master's thesis on few-shot text line segmentation for historical manuscripts. It includes dataset preparation utilities, multiple modelling pipelines, trained model artifacts, and evaluation outputs.

## Repository structure

The repository is organized by workflow stage rather than by package type.

```text
fsl-tl-manuscripts/
├── 00_data/                Raw and converted datasets used by the experiments
├── 10_img_preprocessing/   Dataset conversion and preprocessing scripts
├── 50_modelling/           Main training and inference pipelines
├── 71_misc/                Experimental outputs, grid searches, and auxiliary runs
├── 80_models/              Saved model artifacts and checkpoints
├── 99_evaluation/          Predictions and evaluation outputs
├── runs/                   Additional run directories
└── README.md
```

### Top-level folders

- `00_data/`: source datasets and converted training sets for DIVA-HisDB and U-DIADS-TL, including COCO- and YOLO-style exports.
- `10_img_preprocessing/`: scripts for format conversion, tiling, polygon and bounding-box extraction, and manuscript-specific preprocessing utilities.
- `50_modelling/`: the active modelling code used for segmentation and detection experiments.
- `71_misc/`: ad hoc experiment outputs, grid-search artifacts, and intermediate run folders.
- `80_models/`: stored trained weights and exported models.
- `99_evaluation/`: prediction dumps and post-training evaluation outputs.
- `runs/`: additional run logs and outputs outside the main modelling folders.

## Modelling overview

The main experiment code lives in [50_modelling](/home/artur/Thesis/fsl-tl-manuscripts/50_modelling).

### Current model folders

- [simple_segmentation](/home/artur/Thesis/fsl-tl-manuscripts/50_modelling/simple_segmentation): unified semi-supervised text-line segmentation pipeline.
	Uses either ResNet34 or DINOv2 backbones and supports two segmentation regimes through configuration:
	`simple` for the supervoxel-regularized baseline and `allspark` for the pseudo-label based variant.

- [2stage](/home/artur/Thesis/fsl-tl-manuscripts/50_modelling/2stage): two-stage experiments combining detection and segmentation baselines.
	This folder contains YOLO-based bounding-box detection experiments and segmentation baselines built with `segmentation_models_pytorch` for text-line mask prediction.

- [mask2former](/home/artur/Thesis/fsl-tl-manuscripts/50_modelling/mask2former): unified Detectron2-based instance segmentation pipeline for text-line extraction.
	It uses a DINOv2 ViT-B/14 backbone with a Mask2Former-style head and supports both tile training and full-page training through the config flag `INPUT.FULL_PAGE`.
	The default is `false`, which enables tile-based training.

- [detectron2](/home/artur/Thesis/fsl-tl-manuscripts/50_modelling/detectron2): vendored Detectron2 code used by the Detectron2-based pipelines.

- [few_shot_sampler.py](/home/artur/Thesis/fsl-tl-manuscripts/50_modelling/few_shot_sampler.py): shared few-shot page selection logic reused across experiments so k-shot splits stay consistent.

### Model families in more detail

#### 1. Simple segmentation

The [simple_segmentation](/home/artur/Thesis/fsl-tl-manuscripts/50_modelling/simple_segmentation) pipeline is designed for binary text-line segmentation with limited supervision.

- Backbone options: ResNet34 and DINOv2.
- Decoder: lightweight segmentation decoder over shared backbone features.
- Training variants:
	- `simple`: supervised segmentation with supervoxel consistency regularization.
	- `allspark`: pseudo-label driven semi-supervised training.
- Configs: `dinov2_simple.yaml`, `dinov2_allspark.yaml`, `resnet_simple.yaml`, `resnet_allspark.yaml`.

This is the most direct segmentation pipeline in the repository and is useful for controlled few-shot comparisons.

#### 2. Two-stage baselines

The [2stage](/home/artur/Thesis/fsl-tl-manuscripts/50_modelling/2stage) folder groups baselines that separate localization and segmentation concerns.

- YOLO experiments for bounding-box detection.
- Segmentation baselines based on `segmentation_models_pytorch`.
- Shared dataset handling for polygons, XML parsing, and bounding-box conversion.

This folder is primarily useful for comparing simpler or more modular baselines against the unified segmentation and Mask2Former pipelines.

#### 3. Mask2Former pipeline

The [mask2former](/home/artur/Thesis/fsl-tl-manuscripts/50_modelling/mask2former) folder is the main instance-segmentation pipeline.

- Framework: Detectron2.
- Backbone: DINOv2 ViT-B/14 adapted as a multi-scale backbone.
- Head: custom Mask2Former-style text-line instance segmentation head.
- Outputs: instance masks, polygon extraction, baseline extraction, and PAGE XML export.
- Training modes:
	- `INPUT.FULL_PAGE: false`: tile-based training with random 768x768 crops.
	- `INPUT.FULL_PAGE: true`: full-page training with the full-page finetuning settings.

This pipeline is intended for structured instance segmentation rather than plain binary mask prediction.

## Typical workflow

The repository roughly follows this sequence:

1. Prepare or convert datasets in `10_img_preprocessing/`.
2. Train models from `50_modelling/`.
3. Store resulting checkpoints in `80_models/` or run-specific output folders.
4. Export predictions and evaluation results to `99_evaluation/` or `71_misc/`.

## Notes

- Most modelling code assumes the repository-local virtual environment in `.venv/`.
- The Detectron2-based pipelines depend on the vendored [detectron2](/home/artur/Thesis/fsl-tl-manuscripts/50_modelling/detectron2) folder.
- Several folders under `71_misc/`, `80_models/`, and `99_evaluation/` contain experiment artifacts rather than reusable library code.

