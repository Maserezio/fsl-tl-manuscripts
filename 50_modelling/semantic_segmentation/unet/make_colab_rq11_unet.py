import base64, json, sys
bundle = base64.b64encode(open(sys.argv[1], 'rb').read()).decode()
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}
cells = []
cells.append(md(r"""
# RQ1.1 — U-Net on DIVA-HisDB (autonomous Colab run)

Trains and evaluates the missing U-Net configurations of RQ1.1 on DIVA-HisDB:

| Encoder | Initializations |
|---|---|
| ConvNeXt-Tiny | Random |
| PVTv2-B2 | Random, ImageNet |
| ViT-Small (SFP neck) | Random, ImageNet, DINOv3 (LVD-1689M) |

6 configurations × 3 manuscripts (CB55, CS18, CS863) = 18 runs, minus ViT-Small/ImageNet on CB55 and CS18, which are already in the thesis tables = **16 runs**.

Everything is fetched by the notebook itself: the project code (embedded below), DIVA-HisDB (Zenodo, CC BY 4.0), the encoder weights (timm / Hugging Face) and the official DIVA Line Segmentation Evaluator (GitHub). Recipe is identical to the existing thesis rows: 100 epochs, patience 30, lr 1e-3 (backbone 1e-4), batch 4, crop 448, weight decay 1e-4, AMP, seed 42, all 20 training pages; seam-carving postprocessing and the Java DIVA evaluator on the public test split.

**Runtime → Change runtime type → A100 GPU**, then *Run all*. The number of simultaneous runs is derived from GPU memory (about 12 GB per run) and capped by host RAM. Validation runs every epoch as for the existing rows, but with batched sliding windows and cached pages (same numbers, much faster). Everything that matters is written to Drive (`MyDrive/thesis_rq11_unet_diva/`), and the notebook is resumable: re-running skips finished runs, re-uses checkpoints already on Drive, and waits for runs that are still alive from an interrupted execution instead of restarting them.
"""))
cells.append(md("## 1. Configuration"))
cells.append(code(r'''
from pathlib import Path
import os, sys, subprocess, shutil, json

DRIVE_OUT = Path("/content/drive/MyDrive/thesis_rq11_unet_diva")   # persistent results
REPO      = Path("/content/repo")                                   # code + data (local VM)
DATA      = REPO / "00_data" / "DIVA-HisDB"
JAR       = Path("/content/LineSegmentationEvaluator.jar")
ZENODO    = "https://zenodo.org/api/records/19127869/files/DIVAHisDB.tar.gz/content"
JAR_URL   = "https://github.com/DIVA-DIA/DIVA_Line_Segmentation_Evaluator/raw/master/out/artifacts/LineSegmentationEvaluator.jar"

SUBSETS = ["CB55", "CS18", "CS863"]
# (short tag, encoder name as used by the project, initialization)
CONFIGS = [
    ("convnext_tiny", "tu-convnext_tiny",                   "random"),
    ("pvt_v2_b2",     "tu-pvt_v2_b2",                       "random"),
    ("pvt_v2_b2",     "tu-pvt_v2_b2",                       "imagenet"),
    ("vit_small",     "vit_small_patch16_224.augreg_in21k", "random"),
    ("vit_small",     "vit_small_patch16_224.augreg_in21k", "imagenet"),
    ("vit_small",     "vit_small_patch16_dinov3",           "dinov3"),
]
# Recipe pinned to the existing thesis rows -- do not change.
EPOCHS, PATIENCE, LR, LR_BACKBONE = 100, 30, 1e-3, 1e-4
BATCH_SIZE, WEIGHT_DECAY, SEED, K_SHOT = 4, 1e-4, 42, 20
# Validation: batched sliding windows and cached pages give the same numbers as before, only
# faster. VAL_EVERY must stay 1 to match the existing RQ1 rows (every-epoch validation).
VAL_EVERY, VAL_WINDOW_BATCH = 1, 16
GB_PER_RUN  = 12    # peak GPU memory of one run incl. batched fp32 validation
RAM_PER_RUN = 10    # host RAM of one run incl. cached training and validation pages
PARALLEL    = None  # None = as many runs as GPU memory and RAM allow

# Rows that already exist in the thesis tables -> not recomputed.
SKIP = {("vit_small", "imagenet", "CB55"), ("vit_small", "imagenet", "CS18")}

JOBS = [dict(tag=t, encoder=e, init=i, subset=s, name=f"unet_{t}_{i}_diva_{s}")
        for s in SUBSETS for (t, e, i) in CONFIGS if (t, i, s) not in SKIP]
print(len(JOBS), "runs")

def sh(cmd, check=True, **kw):
    print("$", cmd)
    return subprocess.run(cmd, shell=True, check=check, **kw)
'''))
cells.append(md("## 2. Drive and GPU"))
cells.append(code(r'''
from google.colab import drive
drive.mount("/content/drive")
DRIVE_OUT.mkdir(parents=True, exist_ok=True)
(DRIVE_OUT / "runs").mkdir(exist_ok=True)
(DRIVE_OUT / "results").mkdir(exist_ok=True)
(DRIVE_OUT / "logs").mkdir(exist_ok=True)

import torch
assert torch.cuda.is_available(), "Select a GPU runtime"
vram = torch.cuda.get_device_properties(0).total_memory / 2**30
import psutil
ram, cpus = psutil.virtual_memory().total / 2**30, os.cpu_count()
if PARALLEL is None:
    PARALLEL = max(1, min(int(0.9 * vram // GB_PER_RUN), int(0.85 * ram // RAM_PER_RUN), len(JOBS)))
THREADS = max(1, cpus // PARALLEL)   # CPU threads per run (pages are cached, no loader workers)
print(f"{torch.cuda.get_device_name(0)}: {vram:.0f} GiB VRAM, {ram:.0f} GiB RAM, {cpus} CPU cores"
      f" -> {PARALLEL} parallel runs")
'''))
cells.append(md("## 3. Dependencies (Python packages and Java for the DIVA evaluator)"))
cells.append(code(r'''
sh('pip -q install "timm==1.0.24" "segmentation-models-pytorch==0.5.0" einops pyyaml '
   'opencv-python-headless scikit-image scikit-learn shapely')
sh("apt-get -qq update > /dev/null && apt-get -qq install -y default-jre-headless openjfx > /dev/null")
sh("java -version")
if not JAR.exists():
    sh(f"curl -sL -o {JAR} {JAR_URL}")
assert JAR.stat().st_size > 1_000_000, "DIVA evaluator jar download failed"
JAVA_CP = f"/usr/share/openjfx/lib/*:{JAR}"
print("evaluator:", JAR, JAR.stat().st_size, "bytes")
'''))
cells.append(md("## 4. Project code (embedded snapshot of the thesis repository)"))
cells.append(code("BUNDLE_B64 = (\n" + "\n".join(f'    "{bundle[i:i+120]}"' for i in range(0, len(bundle), 120)) + "\n)\n" + r'''
import base64, io, tarfile
REPO.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE_B64)), mode="r:gz") as tar:
    tar.extractall(REPO)
STAGE = REPO / "50_modelling/semantic_segmentation/unet"

import yaml
base = yaml.safe_load(open(STAGE / "configs/unet_resnet_diva.yaml"))
for init in ("random", "pretrained"):
    c = json.loads(json.dumps(base))
    c["model"].update(encoder_weights=None if init == "random" else "imagenet",
                      pretrained=(init != "random"), freeze_backbone=False)
    yaml.safe_dump(c, open(STAGE / f"configs/unet_diva_{init}.yaml", "w"), sort_keys=False)
print("code ready:", sorted(p.name for p in STAGE.iterdir()))
'''))
cells.append(md("## 5. DIVA-HisDB from Zenodo\n\nDownloads the 1.3 GB archive once, extracts only the images, pixel-level ground truth and Task-2 PAGE XML of the three manuscripts, and checks the page counts (20/10/10 per manuscript)."))
cells.append(code(r'''
import tarfile, zipfile
ARCHIVE = Path("/content/DIVAHisDB.tar.gz")
PREFIX = "hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/"

def ready():
    try:
        return all(len(list((DATA / s / f"img-{s}/img" / sp).glob("*.jpg"))) == n
                   for s in SUBSETS for sp, n in (("training", 20), ("validation", 10), ("public-test", 10)))
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
            member = tar.getmember(PREFIX + name)
            with tar.extractfile(member) as src, open(tmp / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    for name, target in wanted.items():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmp / name) as z:
            z.extractall(target)
    shutil.rmtree(tmp)

assert ready(), "DIVA-HisDB layout check failed"
for s in SUBSETS:
    n_gt = len(list((DATA / s / f"pixel-level-gt-{s}/pixel-level-gt").rglob("*.png")))
    n_xml = len(list((DATA / s / f"PAGE-gt-{s}-TASK-2/TASK-2").rglob("*.xml")))
    print(s, "images ok, pixel GT:", n_gt, "PAGE XML:", n_xml)
'''))
cells.append(md("## 6. Encoder check\n\nBuilds each configuration once (this also downloads the pretrained weights) and verifies that the pretrained initializations really load their checkpoint, while the random ones do not."))
cells.append(code(r'''
import copy, torch, timm
sys.path.insert(0, str(STAGE))
os.chdir(STAGE)
from models import build_model

REF = {("vit_small_patch16_224.augreg_in21k", "imagenet"): "vit_small_patch16_224.augreg_in21k",
       ("vit_small_patch16_dinov3", "dinov3"): "vit_small_patch16_dinov3.lvd1689m",
       ("tu-pvt_v2_b2", "imagenet"): "pvt_v2_b2.in1k"}

def cfg_for(encoder, init):
    c = yaml.safe_load(open(STAGE / f"configs/unet_diva_{'random' if init == 'random' else 'pretrained'}.yaml"))
    c["model"]["encoder_name"] = encoder
    return c

for tag, enc, init in CONFIGS:
    net = build_model(cfg_for(enc, init)).eval()
    sd = net.state_dict()
    n = sum(p.numel() for p in net.parameters()) / 1e6
    status = "random"
    if init != "random":
        ref = timm.create_model(REF[(enc, init)], pretrained=True, num_classes=0).state_dict()
        pe = lambda d: [v for k, v in d.items() if "patch_embed" in k and k.endswith("proj.weight")]
        ok = any(a.shape == b.shape and torch.allclose(a.float(), b.float()) for a in pe(sd) for b in pe(ref))
        assert ok, f"{enc}/{init}: pretrained weights NOT loaded"
        status = "pretrained weights verified"
    print(f"{tag:14s} {init:9s} {n:6.1f}M params  {status}")
    del net
torch.cuda.empty_cache()
'''))
cells.append(md("## 7. Train and evaluate\n\nEach run: train (log to Drive) → copy `best.pth` to Drive → seam-carving postprocessing + DIVA evaluator on the public test split → `results/<run>.json` on Drive. Finished runs are skipped; a checkpoint already on Drive is re-used without retraining."))
cells.append(code(r'''
import csv, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

LOCAL_RUNS = Path("/content/runs")
LOCAL_RUNS.mkdir(exist_ok=True)
lock = threading.Lock()

def log(msg):
    with lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)

def pids(*parts):
    """PIDs of live processes whose command line contains all parts."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            state = (d / "stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            continue
        if state != "Z" and "python" in cmd.split(" ", 1)[0] and all(p in cmd for p in parts):
            out.append(int(d.name))
    return out

def job_pids(name):
    return pids("train.py", f"--run-name {name} ") + pids("predict_diva_seamcarve.py", f"/{name}/best.pth")

def wait_for(ps):
    while any(p in pids() for p in ps):
        time.sleep(30)

# Runs removed from the queue that may still be alive from an earlier execution.
for (t, i, s) in SKIP:
    for p in job_pids(f"unet_{t}_{i}_diva_{s}"):
        os.kill(p, 15); print("stopped skipped run", t, i, s, "pid", p)

def run_job(j):
    name = j["name"]
    res = DRIVE_OUT / "results" / f"{name}.json"
    if res.exists():
        return name, "skip (done)"
    logf = DRIVE_OUT / "logs" / f"{name}.log"
    ck_local = LOCAL_RUNS / name / "best.pth"
    ck_drive = DRIVE_OUT / "runs" / name / "best.pth"
    env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS)}
    t0 = time.time()
    alive = job_pids(name)
    if alive:   # still running from an interrupted execution -> wait instead of restarting
        log(f"waiting for {name} (pid {alive}) from previous execution")
        wait_for(alive)
    last_attempt = logf.read_text(errors="ignore").split("Seed: ")[-1] if logf.exists() else ""
    if ck_drive.exists():   # training finished earlier -> evaluate only
        if not ck_local.exists():
            ck_local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ck_drive, ck_local)
    elif ck_local.exists() and "Training complete" in last_attempt:
        ck_drive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ck_local, ck_drive)
    else:                   # (re)train from scratch; drop any interrupted local run
        shutil.rmtree(LOCAL_RUNS / name, ignore_errors=True)
        log(f"train {name}")
        cfg = f"configs/unet_diva_{'random' if j['init'] == 'random' else 'pretrained'}.yaml"
        cmd = (f"{sys.executable} train.py --config {cfg} --encoder {j['encoder']} --manuscript {j['subset']}"
               f" --k-shot {K_SHOT} --epochs {EPOCHS} --patience {PATIENCE} --lr {LR} --lr-backbone {LR_BACKBONE}"
               f" --batch-size {BATCH_SIZE} --weight-decay {WEIGHT_DECAY} --seed {SEED}"
               f" --val-every {VAL_EVERY} --val-window-batch {VAL_WINDOW_BATCH} --cache-pages"
               f" --run-name {name} --out-dir {LOCAL_RUNS}")
        with open(logf, "a") as fh:
            r = subprocess.run(cmd, shell=True, cwd=STAGE, stdout=fh, stderr=subprocess.STDOUT, env=env,
                               start_new_session=True)   # survives a cell interrupt
        if r.returncode != 0 or not ck_local.exists():
            return name, f"FAILED training (see {logf})"
        ck_drive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ck_local, ck_drive)
    out = LOCAL_RUNS / name / "eval"
    summary = out / "diva_summary.csv"
    if not summary.exists() or summary.stat().st_mtime < ck_local.stat().st_mtime:
        log(f"evaluate {name}")
        cmd = (f"{sys.executable} predict_diva_seamcarve.py --checkpoint {ck_local} --manuscript {j['subset']}"
               f" --split test --output-dir {out} --approx-ratio 0.001 --java-cp '{JAVA_CP}'")
        with open(logf, "a") as fh:
            r = subprocess.run(cmd, shell=True, cwd=STAGE, stdout=fh, stderr=subprocess.STDOUT, env=env,
                               start_new_session=True)   # survives a cell interrupt
        if r.returncode != 0 or not summary.exists():
            return name, f"FAILED evaluation (see {logf})"
    row = next(csv.DictReader(open(summary)))
    keys = {"PixelIU": "Pixel_IU", "LinesIU": "Line_IU", "LinesRecall": "DR", "LinesPrecision": "RA", "LinesFMeasure": "FM"}
    metrics = {v: float(row[k]) for k, v in keys.items()}
    shutil.copytree(out, DRIVE_OUT / "runs" / name / "eval", dirs_exist_ok=True)
    res.write_text(json.dumps({**{k: j[k] for k in ("tag", "encoder", "init", "subset", "name")}, **metrics,
                               "minutes": round((time.time() - t0) / 60, 1)}, indent=2))
    return name, "ok FM=%.2f" % (100 * metrics["FM"])

todo = [j for j in JOBS if not (DRIVE_OUT / "results" / f"{j['name']}.json").exists()]
print(f"{len(JOBS) - len(todo)} finished, {len(todo)} to run, {PARALLEL} in parallel")
todo.sort(key=lambda j: not job_pids(j["name"]))   # runs still alive first
pool = ThreadPoolExecutor(max_workers=PARALLEL)
try:
    for fut in as_completed([pool.submit(run_job, j) for j in todo]):
        name, status = fut.result()
        log(f"[{status}] {name}")
    pool.shutdown()
except KeyboardInterrupt:
    # Queued runs are cancelled; runs already started keep going in the background and are
    # picked up (waited for, then evaluated) by the next execution of this cell.
    pool.shutdown(wait=False, cancel_futures=True)
    print("Interrupted: queue cancelled, running trainings continue:",
          [j["name"] for j in todo if job_pids(j["name"])])
'''))
cells.append(md("## 8. Results\n\nCollects all finished runs into `MyDrive/thesis_rq11_unet_diva/rq11_unet_diva_results.csv` (metrics as fractions, like the evaluator)."))
cells.append(code(r'''
import pandas as pd
rows = [json.loads(p.read_text()) for p in sorted((DRIVE_OUT / "results").glob("*.json"))]
df = pd.DataFrame(rows)
df.to_csv(DRIVE_OUT / "rq11_unet_diva_results.csv", index=False)
print(f"{len(df)}/{len(JOBS)} runs finished -> {DRIVE_OUT / 'rq11_unet_diva_results.csv'}")
if len(df):
    view = df.assign(**{m: (100 * df[m]).round(2) for m in ("Pixel_IU", "Line_IU", "DR", "RA", "FM")})
    display(view.pivot_table(index=["tag", "init"], columns="subset", values="FM"))
    display(view[["tag", "init", "subset", "Pixel_IU", "Line_IU", "DR", "RA", "FM", "minutes"]])
'''))
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[2], "w"), indent=1)
print("written", sys.argv[2])
