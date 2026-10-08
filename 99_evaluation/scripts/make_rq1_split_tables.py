"""Rebuild the RQ1.1 / RQ1.2 tables from the raw evaluation outputs.

Encoder metadata (parameter count, CNN/ViT, size bracket) is imported from
make_rq1_tables.py rather than copied, so the numbers cannot drift apart from
the tables already in RQ1_ALL_RESULTS.md.
"""
import csv, glob, os, re, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_rq1_tables import BRACKET, FAMILY, PARAMS

M = ["Pixel_IU", "Line_IU", "DR", "RA", "FM"]
BUCKET = {enc: b for b, encs in BRACKET.items() for enc in encs}


def meta(encoder):
    key = "stock r50vd" if encoder.startswith("stock") else encoder
    return PARAMS.get(key, "—"), BUCKET.get(key, "—"), FAMILY.get(key, "—")


def table(frame, cols):
    head = cols + ["bucket", "params", "type"] + M
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * (len(cols) + 3) + ["---:"] * len(M)) + "|"]
    order = {"XS": 0, "S": 1, "M": 2, "—": 3}
    frame = frame.assign(_b=frame.bucket.map(order)).sort_values(["_b"] + cols)
    for _, r in frame.iterrows():
        vals = " | ".join("—" if pd.isna(r.get(c)) else f"{r[c]:.3f}" for c in M)
        out.append("| " + " | ".join(str(r[c]) for c in cols)
                   + f" | {r.bucket} | {r['params']} | {r['type']} | " + vals + " |")
    return out


def add_meta(frame, col):
    frame[["params", "bucket", "type"]] = frame[col].apply(lambda e: pd.Series(meta(e)))
    return frame


# ---------------------------------------------------------------- RQ1.1
u = pd.read_csv("99_evaluation/semantic_segmentation/unet/u-diads-tl/size_axis_zottin.csv")
u["encoder"] = (u.encoder.str.replace("^tu-", "", regex=True)
                .str.replace("_patch16_224.augreg_in21k", "", regex=False))

dv = pd.read_csv("99_evaluation/summaries/diagnostics/simple_segmentation/diva_backbone_matrix.csv")


def split(name):                      # DIVA has no init column: it lives in the weight tag
    if "dinov3" in name:                   return "convnext_tiny", "dinov3"
    if name in ("dinov2", "dinov2_reg"):   return name, "ssl"
    if "in12k_ft_in1k" in name:            return "convnext_tiny", "imagenet"
    if "augreg_in21k" in name:             return "vit_small", "imagenet"
    return name, "imagenet"


dv[["encoder", "init"]] = dv.encoder.apply(lambda n: pd.Series(split(n.replace("tu-", ""))))
u, dv = add_meta(u, "encoder"), add_meta(dv, "encoder")

L = ["# RQ1.1 — сравнение бэкбонов, одностадийная схема (U-Net)\n",
     "Полностраничная U-Net, предсказание -> связные компоненты -> PAGE XML -> "
     "официальный оценщик DIVA.\n",
     "Рецепт обучения одинаков для обоих датасетов: 200 эпох, lr 1e-3, "
     "lr бэкбона 1e-4, батч 4.\n",
     "`bucket` — группа размера (XS < 9M, S 9-20M, M > 20M), `params` — параметры "
     "энкодера, не всей U-Net. Обе величины импортированы из "
     "`99_evaluation/scripts/make_rq1_tables.py`.\n",
     f"\n## U-DIADS-TL ({len(u)} ячеек)\n",
     "Полная сетка: 8 энкодеров x 3 инициализации x 3 подмножества.\n"]
L += table(u, ["encoder", "init", "subset"])
L += [f"\n## DIVA-HisDB ({len(dv)} ячеек)\n",
      "**Инициализация `random` не обучалась ни для одного энкодера.**",
      "Набор энкодеров другой: тут `resnet34/50`, `dinov2`, `dinov2_reg`; "
      "общие с U-DIADS-TL только `convnext_tiny` и `vit_small`.\n"]
L += table(dv, ["encoder", "init", "subset"])
open("99_evaluation/summaries/rq1/RQ1.1_backbones_unet.md", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- RQ1.2
ud = []
for f in glob.glob("99_evaluation/instance_segmentation/rtdetr_bbox_unet/u-diads-tl/*_1024x256/*/per_page_zottin.csv"):
    sub = os.path.basename(os.path.dirname(os.path.dirname(f))).split("_")[0]
    m = re.match(r"rtdetr_(.+)_(random|imagenet|dinov3)_(bce|tversky|supervoxel)$",
                 os.path.basename(os.path.dirname(f)))
    if not m:
        continue
    d = pd.read_csv(f)
    ud.append(dict(subset={"latin14396": "Latin14396", "latin2": "Latin2",
                           "syr341": "Syr341"}[sub],
                   backbone=m.group(1), init=m.group(2), loss=m.group(3),
                   **{c: d[c].mean() for c in M if c in d}))
ud = add_meta(pd.DataFrame(ud), "backbone")

dv2 = []
for sub, f in (("CB55", "99_evaluation/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/rtdetr_hf/results.csv"),
               ("CS18", "99_evaluation/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/rtdetr_hf/CS18/results.csv"),
               ("CS863", "99_evaluation/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/rtdetr_hf/CS863/results.csv")):
    rows = list(csv.reader(open(f)))
    ix = {k: i for i, k in enumerate(rows[0])}
    for r in rows[1:]:
        name = r[ix["platform"]]
        if not name.startswith("rtdetr_") or name.startswith("rtdetr_bb_"):
            continue
        g = re.search(r"LineIU=([\d.]+) PixelIU=([\d.]+)", r[ix["notes"]])
        if not g:
            continue
        init = ("dinov3" if "dinov3" in name else
                "imagenet" if "imagenet" in name else "random")
        dv2.append(dict(subset=sub, backbone=name.replace("rtdetr_", "").replace("_" + init, ""),
                        init=init, Line_IU=float(g.group(1)), Pixel_IU=float(g.group(2))))
dv2 = add_meta(pd.DataFrame(dv2), "backbone")

L = ["# RQ1.2 — сравнение бэкбонов, двухстадийная схема (RT-DETR -> U-Net по кропам)\n",
     "Детектор RT-DETR даёт боксы строк, U-Net сегментирует каждый кроп, результат "
     "экспортируется в PAGE XML и считается официальным оценщиком DIVA.\n",
     "**Рецепты обучения двух датасетов различаются.** Общее: 100 эпох, lr 1e-4, батч 1, "
     "seed 42, bf16, разрешение 1152. Различия: накопление градиента 2 (DIVA) против "
     "16 (U-DIADS-TL) — эффективный батч отличается в 8 раз — и планировщик lr "
     "constant против cosine. Внутри каждого датасета рецепт единый, между датасетами "
     "абсолютные значения несопоставимы.\n",
     "`bucket` — группа размера (XS < 9M, S 9-20M, M > 20M), `params` — параметры "
     "бэкбона детектора, не всей связки.\n",
     f"\n## U-DIADS-TL ({len(ud)} ячеек)\n",
     "Ось функции потерь (bce/tversky/supervoxel) есть только здесь.\n"]
L += table(ud, ["backbone", "init", "loss", "subset"])
L += [f"\n## DIVA-HisDB ({len(dv2)} ячеек)\n",
      "**CS863 отсутствует полностью** — прогон был начат и остановлен, метрик нет.",
      "**DR/RA/FM не сохранялись**: в `results.csv` пишутся только LineIU и PixelIU. "
      "Полные пять метрик лежат по-странично в `<прогон>/pred_xml/diva_results.csv`.",
      "Каждая ячейка считалась одной функцией потерь, оси loss нет.\n"]
L += table(dv2, ["backbone", "init", "subset"])
open("99_evaluation/summaries/rq1/RQ1.2_backbones_2stage.md", "w").write("\n".join(L) + "\n")

# machine-readable twins of the two tables
cols1 = ["dataset", "encoder", "init", "subset", "bucket", "params", "type"] + M
pd.concat([u.assign(dataset="U-DIADS-TL"), dv.assign(dataset="DIVA-HisDB")],
          ignore_index=True)[cols1].to_csv(
              "99_evaluation/summaries/rq1/RQ1.1_backbones_unet.csv", index=False)
cols2 = ["dataset", "backbone", "init", "loss", "subset", "bucket", "params", "type"] + M
both = pd.concat([ud.assign(dataset="U-DIADS-TL"),
                  dv2.assign(dataset="DIVA-HisDB", loss="—")], ignore_index=True)
both[cols2].to_csv("99_evaluation/summaries/rq1/RQ1.2_backbones_2stage.csv", index=False)

print(f"RQ1.1: U-DIADS {len(u)} + DIVA {len(dv)} -> md + csv")
print(f"RQ1.2: U-DIADS {len(ud)} + DIVA {len(dv2)} -> md + csv")
