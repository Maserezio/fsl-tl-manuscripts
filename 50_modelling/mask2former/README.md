# mask2former

Detectron2 training and inference pipeline for historical document text-line instance segmentation and PAGE XML export.

## What is included

- DINOv2 ViT-B/14 backbone adapted to Detectron2.
- Mask2Former-style query head for 1-class instance segmentation.
- COCO dataset registration and a mode-aware mapper for tile or full-page training.
- Full-image inference with no downsampling.
- Mask to polygon conversion with OpenCV contours.
- Mask to baseline extraction with skeletonization, graph construction, and longest-path decoding.
- PAGE XML generation with `Coords` and `Baseline` for each `TextLine`.

## Training modes

The unified config uses tile training by default:

- `INPUT.FULL_PAGE: false` enables random 768x768 training crops.
- `INPUT.FULL_PAGE: true` disables cropping and switches to the previous full-page finetuning defaults.

## Training

```bash
/home/artur/Thesis/fsl-tl-manuscripts/.venv/bin/python 50_modelling/mask2former/scripts/train.py \
  --config-file 50_modelling/mask2former/configs/mask2former_dinov2.yaml \
  --train-json /path/to/train.json \
  --train-images /path/to/train_images \
  --val-json /path/to/val.json \
  --val-images /path/to/val_images \
  --output-dir 50_modelling/mask2former/output/run_01
```

To train on full pages instead of tiles:

```bash
/home/artur/Thesis/fsl-tl-manuscripts/.venv/bin/python 50_modelling/mask2former/scripts/train.py \
  --config-file 50_modelling/mask2former/configs/mask2former_dinov2.yaml \
  --train-json /path/to/train.json \
  --train-images /path/to/train_images \
  --val-json /path/to/val.json \
  --val-images /path/to/val_images \
  --output-dir 50_modelling/mask2former/output/full_page_run \
  INPUT.FULL_PAGE True
```

## Evaluation

```bash
/home/artur/Thesis/fsl-tl-manuscripts/.venv/bin/python 50_modelling/mask2former/scripts/evaluate.py \
  --config-file 50_modelling/mask2former/configs/mask2former_dinov2.yaml \
  --dataset-json /path/to/val.json \
  --dataset-images /path/to/val_images \
  --weights 50_modelling/mask2former/output/run_01/model_final.pth \
  --output-dir 50_modelling/mask2former/output/eval_01
```

## Inference and PAGE XML export

```bash
/home/artur/Thesis/fsl-tl-manuscripts/.venv/bin/python 50_modelling/mask2former/scripts/inference.py \
  --config-file 50_modelling/mask2former/configs/mask2former_dinov2.yaml \
  --weights 50_modelling/mask2former/output/run_01/model_final.pth \
  --input /path/to/page_images \
  --output-dir 50_modelling/mask2former/output/pagexml
```

The first DINOv2 run uses `torch.hub` and may download weights if `MODEL.DINO.WEIGHTS` is empty.