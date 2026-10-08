"""Generate the autonomous Colab notebook for the RQ2 two-stage shot study with the
DINOv3 (cBAD) detector initialization (the local run_shot_selection_init.sh cbad queue).

    python 50_modelling/instance_segmentation/rtdetr_bbox_unet/make_colab_rq2_twostage_cbad.py <out.ipynb>

Inputs are read from MyDrive/thesis_rq2_twostage_cbad/repo (code, COCO annotations,
page-selection files, cBAD encoder, 77 crop segmenters; same layout as the repository).
"""
import json
import sys


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


cells = [md(r"""
# RQ2 — two-stage shot-count study on DIVA-HisDB, DINOv3 (cBAD) detector (autonomous Colab run)

The local queue `run_shot_selection_init.sh cbad`, unchanged: for each of the **77 page sets** of the RQ2 shot-selection manifest, an RT-DETR detector with a ConvNeXt-Tiny backbone initialized from the cBAD-distilled encoder is trained (image size 1152, evaluation every 5 epochs) and evaluated with the **existing** ResNet-34 crop segmenter of the same page set (Tversky arm, shared with the ImageNet, DINOv3 and CATMuS rows) by the official DIVA evaluator on the public test split.

Inputs: `MyDrive/thesis_rq2_twostage_cbad/repo/` (uploaded from the experiment repository), DIVA-HisDB from Zenodo, the RT-DETR COCO checkpoint from Hugging Face.

**Runtime → Change runtime type → A100 GPU**, then *Run all*. Detectors train in parallel; evaluations run one at a time (they update a shared per-subset results file). Resumable: finished page sets are skipped, trained detectors are kept on Drive and only re-evaluated.
""")]

cells.append(md("## 1. Configuration"))
cells.append(code(r'''
from pathlib import Path
import os, sys, subprocess, shutil, json, time

DRIVE_OUT  = Path("/content/drive/MyDrive/thesis_rq2_twostage_cbad")
REPO_DRIVE = DRIVE_OUT / "repo"
REPO       = Path("/content/repo")
DATA       = REPO / "00_data" / "DIVA-HisDB"
JAR        = Path("/content/LineSegmentationEvaluator.jar")
ZENODO     = "https://zenodo.org/api/records/19127869/files/DIVAHisDB.tar.gz/content"
JAR_URL    = "https://github.com/DIVA-DIA/DIVA_Line_Segmentation_Evaluator/raw/master/out/artifacts/LineSegmentationEvaluator.jar"

SUBSETS = ["CB55", "CS18", "CS863"]
INIT, BACKBONE, IMAGE_SIZE, ARM, EVAL_EVERY = "cbad", "convnext_tiny", 1152, "tversky", 5   # as run_shot_selection_init.sh
SEG_ROOT = "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256"
GB_PER_RUN, RAM_PER_RUN, PARALLEL = 9, 12, None   # PARALLEL None = derived from GPU memory and RAM
# Inputs uploaded to REPO_DRIVE: relative path -> size in bytes (the notebook waits until all are complete).
EXPECTED = json.loads(r"""{"00_data/DIVA-HisDB/coco_dataset_CB55/test.json": 6354275, "00_data/DIVA-HisDB/coco_dataset_CB55/train.json": 11943897, "00_data/DIVA-HisDB/coco_dataset_CB55/val.json": 6501177, "00_data/DIVA-HisDB/coco_dataset_CS18/test.json": 3787076, "00_data/DIVA-HisDB/coco_dataset_CS18/train.json": 8622516, "00_data/DIVA-HisDB/coco_dataset_CS18/val.json": 3947521, "00_data/DIVA-HisDB/coco_dataset_CS863/test.json": 4067817, "00_data/DIVA-HisDB/coco_dataset_CS863/train.json": 7938979, "00_data/DIVA-HisDB/coco_dataset_CS863/val.json": 3878138, "00_data/DIVA-HisDB/shot_selection/diva_cb55_10_diverse_images.txt": 2473, "00_data/DIVA-HisDB/shot_selection/diva_cb55_15_diverse_images.txt": 3673, "00_data/DIVA-HisDB/shot_selection/diva_cb55_1_diverse_images.txt": 358, "00_data/DIVA-HisDB/shot_selection/diva_cb55_3_diverse_images.txt": 828, "00_data/DIVA-HisDB/shot_selection/diva_cb55_5_diverse_images.txt": 1298, "00_data/DIVA-HisDB/shot_selection/diva_cs18_10_diverse_images.txt": 2223, "00_data/DIVA-HisDB/shot_selection/diva_cs18_15_diverse_images.txt": 3298, "00_data/DIVA-HisDB/shot_selection/diva_cs18_1_diverse_images.txt": 333, "00_data/DIVA-HisDB/shot_selection/diva_cs18_3_diverse_images.txt": 753, "00_data/DIVA-HisDB/shot_selection/diva_cs18_5_diverse_images.txt": 1173, "00_data/DIVA-HisDB/shot_selection/diva_cs863_10_diverse_images.txt": 2223, "00_data/DIVA-HisDB/shot_selection/diva_cs863_15_diverse_images.txt": 3298, "00_data/DIVA-HisDB/shot_selection/diva_cs863_1_diverse_images.txt": 333, "00_data/DIVA-HisDB/shot_selection/diva_cs863_3_diverse_images.txt": 753, "00_data/DIVA-HisDB/shot_selection/diva_cs863_5_diverse_images.txt": 1173, "50_modelling/semantic_segmentation/unet/configs/unet_resnet_diva.yaml": 752, "50_modelling/semantic_segmentation/unet/configs/unet_resnet_u_diads.yaml": 547, "50_modelling/semantic_segmentation/unet/configs/unet_sizeaxis_random_u_diads.yaml": 941, "50_modelling/semantic_segmentation/unet/configs/unet_sizeaxis_u_diads.yaml": 1010, "50_modelling/semantic_segmentation/unet/data/diva_dataset.py": 19847, "50_modelling/semantic_segmentation/unet/data/few_shot_sampler.py": 318, "50_modelling/semantic_segmentation/unet/data/__init__.py": 154, "50_modelling/semantic_segmentation/unet/eval_small_components.py": 7579, "50_modelling/semantic_segmentation/unet/evaluate_lines.py": 9659, "50_modelling/semantic_segmentation/unet/evaluate.py": 18912, "50_modelling/semantic_segmentation/unet/eval_udiads_matrix.py": 8242, "50_modelling/semantic_segmentation/unet/make_colab_rq11_unet.py": 17105, "50_modelling/semantic_segmentation/unet/make_colab_rq2_unet_inits.py": 17808, "50_modelling/semantic_segmentation/unet/make_colab_rq2_unet.py": 16922, "50_modelling/semantic_segmentation/unet/models/arunet.py": 5751, "50_modelling/semantic_segmentation/unet/models/__init__.py": 309, "50_modelling/semantic_segmentation/unet/models/smp_unet.py": 5803, "50_modelling/semantic_segmentation/unet/models/vit_hier.py": 4629, "50_modelling/semantic_segmentation/unet/postproc.py": 5349, "50_modelling/semantic_segmentation/unet/predict_diva_seamcarve.py": 23929, "50_modelling/semantic_segmentation/unet/predict.py": 1073, "50_modelling/semantic_segmentation/unet/predict_u_diads.py": 4810, "50_modelling/semantic_segmentation/unet/report_udiads_size_axis.py": 7192, "50_modelling/semantic_segmentation/unet/train.py": 22571, "50_modelling/instance_segmentation/rtdetr_bbox_unet/cbad_backbone_maps.py": 2169, "50_modelling/instance_segmentation/rtdetr_bbox_unet/dataset.py": 9593, "50_modelling/instance_segmentation/rtdetr_bbox_unet/evaluate_loss_ablation_detr_udiads.py": 26098, "50_modelling/instance_segmentation/rtdetr_bbox_unet/evaluate.py": 25714, "50_modelling/instance_segmentation/rtdetr_bbox_unet/predict_and_eval_rtdetr_diva.py": 23531, "50_modelling/instance_segmentation/rtdetr_bbox_unet/rtdetr_fullpage_reconstruct_udiads.py": 14402, "50_modelling/instance_segmentation/rtdetr_bbox_unet/rtdetr_load.py": 3319, "50_modelling/instance_segmentation/rtdetr_bbox_unet/syr341_fm_target.py": 10644, "50_modelling/instance_segmentation/rtdetr_bbox_unet/train_crops_loss_ablation_2stage.py": 31984, "50_modelling/instance_segmentation/rtdetr_bbox_unet/train_rtdetr_hf.py": 46100, "50_modelling/common/few_shot_sampler.py": 5484, "50_modelling/common/evaluate_util.py": 4969, "50_modelling/common/hier_encoder/adapters/__init__.py": 567, "50_modelling/common/hier_encoder/adapters/unet_adapter.py": 3393, "50_modelling/common/hier_encoder/adapters/yolo_adapter.py": 11777, "50_modelling/common/hier_encoder/backbones.py": 17399, "50_modelling/common/hier_encoder/build.py": 8269, "50_modelling/common/hier_encoder/config.py": 4499, "50_modelling/common/hier_encoder/__init__.py": 655, "50_modelling/common/hier_encoder/lora.py": 3916, "50_modelling/common/hier_encoder/neck_convstem.py": 2409, "50_modelling/common/hier_encoder/neck_multiblock.py": 2582, "50_modelling/common/hier_encoder/neck_sfp.py": 3831, "50_modelling/common/hier_encoder/smoke_test.py": 3408, "50_modelling/common/hier_encoder/windowed_attn.py": 2134, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k10_5a5195ef/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k10_69341241/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k10_69b45a69/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k10_ad7bd714/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k10_d688d043/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k1_2c752f4a/k1/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k15_5c363272/k15/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k15_634568b4/k15/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k15_668f4fda/k15/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k15_7e3c513c/k15/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k1_7a1608db/k1/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k1_8c26820d/k1/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k1_ddbc80ee/k1/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k3_2017fce1/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k3_458ebb5d/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k3_9c4b2a61/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k3_ce938512/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k3_e5547c3a/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k3_f2700177/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k5_45043204/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k5_4d2a742b/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k5_82d0bdca/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k5_86a6d5af/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k5_87c37879/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CB55/CB55_k5_9072af77/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k10_1e188476/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k10_3049391c/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k10_5d8b6b68/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k10_9dc97bff/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k10_a753e501/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k10_d1a49d99/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k15_60eb6abb/k15/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k15_7e55208c/k15/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k15_8a9edb1c/k15/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k15_9d9d1f6a/k15/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k15_c0630f5c/k15/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k1_789b4135/k1/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k1_eec0092d/k1/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k3_4666bf9f/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k3_48a31c0d/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k3_a4e24e97/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k3_e5d541e7/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k3_f09412e0/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k5_45a250d0/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k5_4889c850/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k5_4aa3851f/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k5_815dbac0/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS18/CS18_k5_f9cecfe9/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k10_6949b64f/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k10_73d6bbc6/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k10_b045f2ec/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k10_ba6c5cfd/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k10_efc33dfa/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k10_f95fa471/k10/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k1_0fd5a0da/k1/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k1_23a82488/k1/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k15_02e6fe88/k15/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k15_351ed81e/k15/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k15_8895bb47/k15/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k15_a02442ed/k15/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k15_eb326e37/k15/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k15_f84a921c/k15/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k1_c19c7a4c/k1/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k1_c2389398/k1/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k1_f4f0a4c0/k1/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k3_002db31f/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k3_063a713a/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k3_37b23fac/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k3_59db31e9/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k3_a4e975f7/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k3_f6fbf2ad/k3/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k5_5feb910b/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k5_6565430e/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k5_7cb780c5/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k5_a902634d/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k5_d03a5cb8/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_kshot_1024x256/CS863/CS863_k5_f2677637/k5/tversky/best.pth": 97904127, "80_models/instance_segmentation/mask_rcnn/cbad_distilled_backbones/convnext_tiny_cbad_dinov3_epoch1100.pt": 111351219, "99_evaluation/summaries/rq2/shot_selection_manifest.json": 35406}""")

def sh(cmd, check=True, **kw):
    print("$", cmd)
    return subprocess.run(cmd, shell=True, check=check, **kw)
'''))

cells.append(md("## 2. Drive and GPU"))
cells.append(code(r'''
from google.colab import drive
drive.mount("/content/drive")
for d in ("detectors", "results", "logs"):
    (DRIVE_OUT / d).mkdir(parents=True, exist_ok=True)

import torch, psutil
assert torch.cuda.is_available(), "Select a GPU runtime"
vram = torch.cuda.get_device_properties(0).total_memory / 2**30
ram, cpus = psutil.virtual_memory().total / 2**30, os.cpu_count()
if PARALLEL is None:
    PARALLEL = max(1, min(int(0.9 * vram // GB_PER_RUN), int(0.85 * ram // RAM_PER_RUN), 8))
THREADS = max(1, cpus // PARALLEL)
print(f"{torch.cuda.get_device_name(0)}: {vram:.0f} GiB VRAM, {ram:.0f} GiB RAM, {cpus} cores -> {PARALLEL} parallel trainings")
'''))

cells.append(md("## 3. Dependencies\n\nPinned to the local runs (`transformers==5.14.1`, so detector checkpoints reload exactly as for the other initializations)."))
cells.append(code(r'''
sh('pip -q install "transformers==5.14.1" "timm==1.0.24" "datasets==5.0.1" "accelerate==1.14.0" '
   '"torchmetrics==1.9.0" "faster-coco-eval==1.7.2" "segmentation-models-pytorch==0.5.0" "albumentations==2.0.8" '
   'pycocotools safetensors einops pyyaml opencv-python-headless scikit-image shapely')
sh("apt-get -qq update > /dev/null && apt-get -qq install -y default-jre-headless openjfx > /dev/null")
# The evaluator build used for all local runs (the GitHub jar lacks LineSegmentationEvaluatorTool).
JAR_DRIVE = REPO_DRIVE / "tools/LineSegmentationEvaluator.jar"
assert JAR_DRIVE.exists(), f"missing {JAR_DRIVE}"
shutil.copy2(JAR_DRIVE, JAR)
listing = subprocess.run(["unzip", "-l", str(JAR)], capture_output=True, text=True).stdout
assert "ch/unifr/LineSegmentationEvaluatorTool.class" in listing, "wrong DIVA evaluator build"
import transformers, timm
print("torch", torch.__version__, "transformers", transformers.__version__, "timm", timm.__version__)
'''))

cells.append(md("## 4. DIVA-HisDB from Zenodo\n\nExtracts images, pixel-level GT and Task-2 PAGE XML, and links the pages into `yolo_dataset_<subset>/images/{train,val,test}` (identical to the training / validation / public-test splits)."))
cells.append(code(r'''
import tarfile, zipfile
ARCHIVE = Path("/content/DIVAHisDB.tar.gz")
PREFIX = "hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/"
SPLITS = (("training", "train", 20), ("validation", "val", 10), ("public-test", "test", 10))

def ready():
    try:
        return all(len(list((DATA / s / f"img-{s}/img" / sp).glob("*.jpg"))) == n for s in SUBSETS for sp, _, n in SPLITS)
    except FileNotFoundError:
        return False

if not ready():
    if not ARCHIVE.exists():
        sh(f"curl -sL -o {ARCHIVE} {ZENODO}")
    wanted = {}
    for s in SUBSETS:
        task2 = "CSG863" if s == "CS863" else s
        wanted[f"img-{s}.zip"] = DATA / s / f"img-{s}"
        wanted[f"pixel-level-gt-{s}.zip"] = DATA / s / f"pixel-level-gt-{s}"
        wanted[f"PAGE-gt-{task2}-TASK-2.zip"] = DATA / s / f"PAGE-gt-{s}-TASK-2"
    tmp = Path("/content/diva_zips"); tmp.mkdir(exist_ok=True)
    with tarfile.open(ARCHIVE) as tar:
        for name in wanted:
            with tar.extractfile(tar.getmember(PREFIX + name)) as src, open(tmp / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    for name, target in wanted.items():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmp / name) as z:
            z.extractall(target)
    shutil.rmtree(tmp)
assert ready(), "DIVA-HisDB layout check failed"

for s in SUBSETS:
    for src_split, dst_split, n in SPLITS:
        dst = DATA / f"yolo_dataset_{s}" / "images" / dst_split
        dst.mkdir(parents=True, exist_ok=True)
        for img in (DATA / s / f"img-{s}/img" / src_split).glob("*.jpg"):
            if not (dst / img.name).exists():
                (dst / img.name).symlink_to(img)
        assert len(list(dst.glob("*.jpg"))) == n
    print(s, "images linked")
'''))

cells.append(md("## 5. Inputs from Drive\n\nWaits until every uploaded input file is complete on Drive (checked by size; the upload may still be running when the notebook starts), then copies them (≈7 GB, mostly the crop segmenters) to the local disk."))
cells.append(code(r'''
def pending():
    out = []
    for rel, size in EXPECTED.items():
        p = REPO_DRIVE / rel
        try:
            if p.stat().st_size != size:
                out.append(rel)
        except OSError:
            out.append(rel)
    return out

t0 = time.time()
while (left := pending()):
    print(time.strftime("%H:%M:%S"), f"waiting for the upload: {len(left)}/{len(EXPECTED)} files not complete yet"
          f" (e.g. {left[0]})", flush=True)
    time.sleep(120)
print(f"all {len(EXPECTED)} input files complete on Drive")
for rel in EXPECTED:
    dst = REPO / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != EXPECTED[rel]:
        shutil.copy2(REPO_DRIVE / rel, dst)
MANIFEST = json.loads((REPO / "99_evaluation/summaries/rq2/shot_selection_manifest.json").read_text())
first = {}
for c in MANIFEST["cells"]:
    first.setdefault(c["job"], c)
JOBS = [dict(job=j["job"], subset=j["subset"], k=j["k"], method=first[j["job"]]["method"]) for j in MANIFEST["jobs"]]
missing = [j["job"] for j in JOBS if not (REPO / SEG_ROOT / j["subset"] / j["job"] / f"k{j['k']}" / ARM / "best.pth").exists()]
assert not missing, f"missing crop segmenters: {missing[:5]}"
cbad = REPO / "80_models/instance_segmentation/mask_rcnn/cbad_distilled_backbones/convnext_tiny_cbad_dinov3_epoch1100.pt"
assert cbad.exists(), cbad
for s in SUBSETS:
    coco = json.loads((DATA / f"coco_dataset_{s}" / "train.json").read_text())
    have = {p.name for p in (DATA / f"yolo_dataset_{s}" / "images" / "train").iterdir()}
    assert {Path(i["file_name"]).name for i in coco["images"]} <= have, f"{s}: COCO train images not found"
print(f"{len(JOBS)} page sets, all segmenters present, cBAD encoder ok, COCO images linked, {(time.time() - t0) / 60:.1f} min")
'''))

cells.append(md("## 6. Train and evaluate\n\nPer page set: detector training → `best_model` copied to Drive → evaluation with the existing crop segmenter → `results/<job>.csv` on Drive."))
cells.append(code(r'''
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

STAGE = REPO / "50_modelling"
DET_ROOT = REPO / "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/detection/rtdetr_hf"
eval_lock, print_lock = threading.Lock(), threading.Lock()

def log(msg):
    with print_lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)

def sel_file(j):
    return "" if j["method"] == "random" else \
        str(REPO / f"00_data/DIVA-HisDB/shot_selection/diva_{j['subset'].lower()}_{j['k']}_diverse_images.txt")

def run_job(j):
    job, sub, k = j["job"], j["subset"], j["k"]
    run = f"sel_{INIT}_{job}"
    res = DRIVE_OUT / "results" / f"{job}.csv"
    if res.exists():
        return job, "skip (done)"
    logf = DRIVE_OUT / "logs" / f"{job}.log"
    model_dir = DET_ROOT / ("" if sub == "CB55" else sub) / run      # layout of train_rtdetr_hf.py
    drive_model = DRIVE_OUT / "detectors" / run / "best_model"
    env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS), "DIVA_EVALUATOR_JAR": str(JAR)}
    t0 = time.time()
    if drive_model.exists() and any(drive_model.iterdir()):
        shutil.copytree(drive_model, model_dir / "best_model", dirs_exist_ok=True)
    else:
        shutil.rmtree(model_dir, ignore_errors=True)
        log(f"train {job} ({sub} k={k}, {j['method']})")
        denv = {**env, "DINOV3_SOURCE": "timm", "DATASET": sub, "BACKBONE": BACKBONE, "INIT": INIT,
                "IMAGE_SIZE": str(IMAGE_SIZE), "K_SHOT": str(k), "K_SHOT_METHOD": j["method"],
                "K_SHOT_PRECOMPUTED": sel_file(j), "EVAL_EVERY_EPOCHS": str(EVAL_EVERY), "RUN_NAME": run}
        with open(logf, "a") as fh:
            r = subprocess.run([sys.executable, "instance_segmentation/rtdetr_bbox_unet/train_rtdetr_hf.py"], cwd=STAGE, env=denv,
                               stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode != 0 or not (model_dir / "best_model").exists():
            return job, f"FAILED training (see {logf})"
        for ck in model_dir.glob("checkpoint-*"):          # evaluation reads best_model/ only
            shutil.rmtree(ck, ignore_errors=True)
        shutil.copytree(model_dir / "best_model", drive_model, dirs_exist_ok=True)
    curve = f"99_evaluation/summaries/rq2/colab_cbad/{job}.csv"
    with eval_lock:
        log(f"evaluate {job}")
        eenv = {**env, "SUBSET": sub, "RUN_NAME": run, "K_SHOT": str(k),
                "SEG_KSHOT_ROOT": f"{SEG_ROOT}/{sub}/{job}/k{k}/{ARM}",
                "CURVE_CSV": curve, "CURVE_APPROACH": f"two_stage_{INIT}", "CURVE_METHOD": job}
        with open(logf, "a") as fh:
            r = subprocess.run([sys.executable, "instance_segmentation/rtdetr_bbox_unet/predict_and_eval_rtdetr_diva.py"], cwd=STAGE,
                               env=eenv, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not (REPO / curve).exists():
        return job, f"FAILED evaluation (see {logf})"
    shutil.copy2(REPO / curve, res)
    return job, f"ok FM={100 * pd.read_csv(res).iloc[0]['FM']:.2f} ({(time.time() - t0) / 60:.0f} min)"

todo = [j for j in JOBS if not (DRIVE_OUT / "results" / f"{j['job']}.csv").exists()]
print(f"{len(JOBS) - len(todo)} finished, {len(todo)} to run, {PARALLEL} trainings in parallel")
pool = ThreadPoolExecutor(max_workers=PARALLEL)
try:
    for fut in as_completed([pool.submit(run_job, j) for j in todo]):
        job, status = fut.result()
        log(f"[{status}] {job}")
    pool.shutdown()
except KeyboardInterrupt:
    pool.shutdown(wait=False, cancel_futures=True)
    print("Interrupted: re-run this cell to continue (finished page sets are kept).")
'''))

cells.append(md("## 7. Results\n\nExpands the 77 page-set results to the 90 (subset, method, k) cells: `MyDrive/thesis_rq2_twostage_cbad/shot_selection_diva_by_method_cbad.csv` (format of the local runs, metrics as fractions)."))
cells.append(code(r'''
by_job = {p.stem: pd.read_csv(p).iloc[0] for p in (DRIVE_OUT / "results").glob("*.csv")}
metrics = ["Pixel_IU", "Line_IU", "DR", "RA", "FM"]
rows = [{"subset": c["subset"], "method": c["method"], "k": c["k"], "pages": c["k"], "job": c["job"],
         **{m: round(float(by_job[c["job"]][m]), 4) for m in metrics}}
        for c in MANIFEST["cells"] if c["job"] in by_job]
df = pd.DataFrame(rows)
out = DRIVE_OUT / "shot_selection_diva_by_method_cbad.csv"
df.to_csv(out, index=False)
pd.DataFrame([r.to_dict() for r in by_job.values()]).to_csv(DRIVE_OUT / "shot_selection_diva_cbad.csv", index=False)
print(f"{len(by_job)}/{len(JOBS)} page sets, {len(df)}/{len(MANIFEST['cells'])} cells -> {out}")
if len(df):
    g = df.groupby(["method", "k"])
    display((100 * g["FM"].mean()).where(g["subset"].nunique() == len(SUBSETS)).round(2).unstack("k"))
'''))

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("written", sys.argv[1])
