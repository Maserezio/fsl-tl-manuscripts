"""Generate the Colab notebook that pretrains RT-DETR and the BBox U-Net on CATMuS Medieval with the same budget as
the CATMuS Mask R-CNN arm (image size 1152, 30 epochs, random initialization, best epoch on the CATMuS validation
pages).

    python 50_modelling/instance_segmentation/rtdetr_bbox_unet/make_colab_catmus_pretrain.py <out.ipynb>

Inputs are read from MyDrive/thesis_catmus_pretrain/repo (code and the CATMuS COCO annotations, same layout as the
repository); the page images are downloaded from Hugging Face (CATMuS/medieval-segmentation).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FILES = (["50_modelling/common/few_shot_sampler.py", "50_modelling/common/evaluate_util.py",
          "00_data/CATMuS/medieval-segmentation/coco_instances/train.json",
          "00_data/CATMuS/medieval-segmentation/coco_instances/val.json"]
         + [f"50_modelling/instance_segmentation/rtdetr_bbox_unet/{n}" for n in ("train_rtdetr_hf.py", "rtdetr_load.py", "dataset.py",
                                                    "train_crops_loss_ablation_2stage.py", "prepare_catmus_crops.py")]
         + sorted(str(p.relative_to(REPO)) for p in (REPO / "50_modelling/common/hier_encoder").rglob("*.py")))
EXPECTED = {f: (REPO / f).stat().st_size for f in FILES}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


cells = [md(r"""
# CATMuS pretraining of RT-DETR and the BBox U-Net (autonomous Colab run)

Aligns the two detection-based pipelines. The CATMuS Mask R-CNN arm pretrains the complete detector, including its
mask head, on the CATMuS Medieval line polygons. Here the RT-DETR detector and the BBox U-Net that segments its boxes
are pretrained on the same pages:

| Stage | Model | Data | Budget |
|---|---|---|---|
| A | RT-DETR (one run per backbone, random initialization) | CATMuS train 1335 pages, best epoch on CATMuS val 191 pages | 1152 px, 30 epochs |
| B | BBox U-Net (U-Net, ResNet-34), Tversky arm with BCE warm-up | line crops of the same pages, resized to 1024x256 | `CROP_EPOCHS` epochs |

Outputs on Drive (`MyDrive/thesis_catmus_pretrain/`):
- `rtdetr_hf/rtdetr_<backbone>_random_1152e30/best_model` (load with `INIT=catmus CATMUS_PRETRAIN_DIR=<that dir>`),
- `bbox_unet/best.pth` (pass to `train_crops_loss_ablation_2stage.py --init-checkpoint`),
- `logs/`, `summary.json`.

**Runtime → Change runtime type → A100 GPU**, then *Run all*. Resumable: RT-DETR checkpoints are written to Drive
every epoch and training continues from the newest one after a disconnect; finished stages are skipped.
""")]

cells.append(md("## 1. Configuration"))
cells.append(code(r'''
from pathlib import Path
import os, sys, subprocess, shutil, json, time

DRIVE_OUT  = Path("/content/drive/MyDrive/thesis_catmus_pretrain")
REPO_DRIVE = DRIVE_OUT / "repo"
REPO       = Path("/content/repo")
CATMUS     = REPO / "00_data/CATMuS/medieval-segmentation"
HF_DATASET = "CATMuS/medieval-segmentation"          # v1.5.0 was used for the local COCO export

BACKBONES  = ["convnext_tiny"]                       # all RQ1 encoders: ["convnext_tiny", "pvt_v2_b2", "vit_small"]
IMAGE_SIZE, EPOCHS = 1152, 30                        # = CATMuS Mask R-CNN budget
CROP_EPOCHS, CROP_WARMUP, CROP_BATCH = 10, 2, 8      # BBox U-Net: Tversky arm after a BCE warm-up
CROP_LONG_SIDE = 2400                                # pages are resized once; crops are 1024x256 anyway

EXPECTED = json.loads(r"""__EXPECTED__""")

def sh(cmd, check=True, **kw):
    print("$", cmd)
    return subprocess.run(cmd, shell=True, check=check, **kw)
'''.replace("__EXPECTED__", json.dumps(EXPECTED))))

cells.append(md("## 2. Drive and GPU"))
cells.append(code(r'''
from google.colab import drive
drive.mount("/content/drive")
for d in ("rtdetr_hf", "bbox_unet", "logs"):
    (DRIVE_OUT / d).mkdir(parents=True, exist_ok=True)
import torch, psutil
assert torch.cuda.is_available(), "Select a GPU runtime"
print(torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB VRAM,",
      f"{psutil.virtual_memory().total / 2**30:.0f} GiB RAM, {os.cpu_count()} cores")
'''))

cells.append(md("## 3. Dependencies\n\nPinned to the local runs (`transformers==5.14.1`, so the checkpoints reload exactly as for the other initializations)."))
cells.append(code(r'''
sh('pip -q install "transformers==5.14.1" "timm==1.0.24" "datasets==5.0.1" "accelerate==1.14.0" '
   '"torchmetrics==1.9.0" "faster-coco-eval==1.7.2" "segmentation-models-pytorch==0.5.0" "albumentations==2.0.8" '
   'pycocotools safetensors einops pyyaml opencv-python-headless scikit-image shapely huggingface_hub '
   '"git+https://github.com/AllenNeuralDynamics/supervoxel-loss@fbf309abc6042d2585d330b62e6cbfe8f1a9b640"')
import transformers, timm
print("torch", torch.__version__, "transformers", transformers.__version__, "timm", timm.__version__)
'''))

cells.append(md("## 4. Inputs from Drive\n\nWaits until every uploaded file is complete on Drive (checked by size), then copies the code and the COCO annotations."))
cells.append(code(r'''
def pending():
    out = []
    for rel, size in EXPECTED.items():
        try:
            if (REPO_DRIVE / rel).stat().st_size != size:
                out.append(rel)
        except OSError:
            out.append(rel)
    return out

while (left := pending()):
    print(time.strftime("%H:%M:%S"), f"waiting for the upload: {len(left)}/{len(EXPECTED)} files missing (e.g. {left[0]})", flush=True)
    time.sleep(60)
for rel, size in EXPECTED.items():
    dst = REPO / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != size:
        shutil.copy2(REPO_DRIVE / rel, dst)
print(f"{len(EXPECTED)} input files copied")
'''))

cells.append(md("## 5. CATMuS page images from Hugging Face\n\n`data/train` and `data/dev` of the dataset are linked as `yolo_seg_dataset/images/{train,val}`, the layout of the COCO export (val = dev)."))
cells.append(code(r'''
from huggingface_hub import snapshot_download
src = Path(snapshot_download(HF_DATASET, repo_type="dataset", local_dir="/content/catmus_src",
                             allow_patterns=["data/train/**", "data/dev/**"], max_workers=16))
img_root = CATMUS / "yolo_seg_dataset/images"
img_root.mkdir(parents=True, exist_ok=True)
for split, hf in (("train", "train"), ("val", "dev")):
    link = img_root / split
    if not link.exists():
        link.symlink_to(src / "data" / hf, target_is_directory=True)
    coco = json.loads((CATMUS / f"coco_instances/{split}.json").read_text())
    missing = [im["file_name"] for im in coco["images"] if not (link / im["file_name"]).exists()]
    assert not missing, f"{split}: {len(missing)} COCO pages not in the Hugging Face snapshot, e.g. {missing[:3]}"
    print(split, len(coco["images"]), "pages,", len(coco["annotations"]), "lines: all images present")
'''))

cells.append(md("## 6. Stage A: RT-DETR\n\nOne run per backbone (`train_rtdetr_hf.py`, `DATASET=CATMuS`, `INIT=random`). The output directory lives on Drive, so the per-epoch checkpoints survive a disconnect (`RESUME=1`)."))
cells.append(code(r'''
STAGE = REPO / "50_modelling"
local_root = REPO / "80_models/instance_segmentation/rtdetr_bbox_unet/catmus_pretrain/rtdetr_hf"
local_root.parent.mkdir(parents=True, exist_ok=True)
if not local_root.exists():
    local_root.symlink_to(DRIVE_OUT / "rtdetr_hf", target_is_directory=True)
summary = json.loads((DRIVE_OUT / "summary.json").read_text()) if (DRIVE_OUT / "summary.json").exists() else {}

for bb in BACKBONES:
    run = f"rtdetr_{bb}_random_{IMAGE_SIZE}e{EPOCHS}"
    out = local_root / run
    if (out / "best_model/model.safetensors").exists() and (out / "metrics_summary.json").exists():
        print("[skip]", run); continue
    env = {**os.environ, "DATASET": "CATMuS", "BACKBONE": bb, "INIT": "random", "RUN_NAME": run,
           "IMAGE_SIZE": str(IMAGE_SIZE), "NUM_EPOCHS": str(EPOCHS), "RESUME": "1",
           "BACKBONE_GRAD_CKPT": "1" if bb == "pvt_v2_b2" else "0"}
    t0 = time.time()
    print(time.strftime("%H:%M:%S"), "train", run, flush=True)
    with open(DRIVE_OUT / "logs" / f"{run}.log", "a") as fh:
        r = subprocess.run([sys.executable, "instance_segmentation/rtdetr_bbox_unet/train_rtdetr_hf.py"], cwd=STAGE, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not (out / "best_model").exists():
        raise RuntimeError(f"{run} failed, see {DRIVE_OUT / 'logs' / (run + '.log')}")
    for ck in out.glob("checkpoint-*"):                 # best_model/ is all that later stages read
        shutil.rmtree(ck, ignore_errors=True)
    summary[run] = {"minutes": round((time.time() - t0) / 60), **json.loads((out / "metrics_summary.json").read_text())}
    (DRIVE_OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(time.strftime("%H:%M:%S"), run, "done in", summary[run]["minutes"], "min", flush=True)
'''))

cells.append(md("## 7. Stage B: BBox U-Net\n\nPages are resized once (`prepare_catmus_crops.py`), then the crop segmenter is trained on all CATMuS train lines and selected on the val lines (`train_crops_loss_ablation_2stage.py --family catmus`)."))
cells.append(code(r'''
target = DRIVE_OUT / "bbox_unet" / "best.pth"
if target.exists():
    print("[skip] BBox U-Net")
else:
    sh(f"cd {REPO} && {sys.executable} 50_modelling/instance_segmentation/rtdetr_bbox_unet/prepare_catmus_crops.py --long-side {CROP_LONG_SIDE} --workers {os.cpu_count()}")
    t0 = time.time()
    cmd = (f"cd {STAGE / 'instance_segmentation/rtdetr_bbox_unet'} && {sys.executable} train_crops_loss_ablation_2stage.py --family catmus --subset CATMuS "
           f"--arms tversky --epochs {CROP_EPOCHS} --warmup-epochs {CROP_WARMUP} --batch-size {CROP_BATCH} "
           f"--workers {max(2, os.cpu_count() - 2)} > {DRIVE_OUT / 'logs' / 'bbox_unet.log'} 2>&1")
    sh(cmd)
    out = REPO / "80_models/instance_segmentation/rtdetr_bbox_unet/catmus_pretrain/crop_seg_loss_ablation_components_1024x256/CATMuS"
    shutil.copy2(out / "tversky" / "best.pth", target)
    for f in ("summary.csv", "history_all_arms.csv", "curves.png"):
        if (out / f).exists():
            shutil.copy2(out / f, DRIVE_OUT / "bbox_unet" / f)
    summary = json.loads((DRIVE_OUT / "summary.json").read_text()) if (DRIVE_OUT / "summary.json").exists() else {}
    summary["bbox_unet"] = {"minutes": round((time.time() - t0) / 60), "epochs": CROP_EPOCHS, "warmup": CROP_WARMUP}
    (DRIVE_OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(open(out / "summary.csv").read())
'''))

cells.append(md("## 8. Summary"))
cells.append(code(r'''
print(json.dumps(json.loads((DRIVE_OUT / "summary.json").read_text()), indent=1)[:3000])
for p in sorted(DRIVE_OUT.rglob("best_model")) + [DRIVE_OUT / "bbox_unet" / "best.pth"]:
    print("ok" if p.exists() else "MISSING", p)
'''))

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("written", sys.argv[1], "with", len(EXPECTED), "input files")
