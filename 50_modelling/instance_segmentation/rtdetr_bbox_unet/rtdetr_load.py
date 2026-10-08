"""Load a trained RT-DETR run directory, including the flat-ViT + SFP arms.

`AutoModelForObjectDetection.from_pretrained(best_model)` is enough for every
HF-native backbone (stock R50-vd, ConvNeXt, PVTv2): the backbone is described by
the saved config. The vit_small arms are not -- their backbone is hier_encoder's
SFP module bolted on by train_rtdetr_hf._build_sfp_model, which no config can
express, so from_pretrained rebuilds a ResNet and dies on the size mismatch.
train_rtdetr_hf writes best_model/sfp_backbone.json for those runs; here the same
builder recreates the graph and the saved weights are loaded into it.

The saved safetensors use the legacy key names the Trainer writes
(encoder.encoder.N / out_proj / fc1); the live modules are named
encoder.aifi.N / o_proj / mlp.fc1. from_pretrained applies that rename
internally; for the manual path it is applied here, and the key sets are
asserted equal so any drift fails loudly instead of silently zero-initialising.
"""

import json
import os
import re
import sys
from pathlib import Path

from safetensors.torch import load_file
from transformers import AutoModelForObjectDetection

SFP_META = "sfp_backbone.json"

# Mirrors transformers.conversion_mapping["rt_detr"].
_LEGACY_RENAMES = (
    (r"out_proj", "o_proj"),
    (r"layers\.(\d+)\.fc1", r"layers.\1.mlp.fc1"),
    (r"layers\.(\d+)\.fc2", r"layers.\1.mlp.fc2"),
    (r"encoder\.encoder\.(\d+)\.layers", r"encoder.aifi.\1.layers"),
)


def _rename(key):
    for pattern, repl in _LEGACY_RENAMES:
        key = re.sub(pattern, repl, key)
    return key


def load_detector(model_dir):
    """Return the eval-mode model for a best_model/ directory (on CPU)."""
    model_dir = Path(model_dir)
    meta_path = model_dir / SFP_META
    if not meta_path.exists():
        return AutoModelForObjectDetection.from_pretrained(str(model_dir)).eval()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # hier_encoder sizes its position embeddings from IMAGE_SIZE at build time and
    # train_rtdetr_hf reads that from the environment on import.
    os.environ["IMAGE_SIZE"] = str(meta["image_size"])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_rtdetr_hf as trainer_module  # noqa: E402  (import-time env parsing only)

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    label2id = {v: k for k, v in id2label.items()}
    model = trainer_module._build_sfp_model(meta["backbone"], meta["init"], id2label, label2id)

    state = {_rename(k): v for k, v in load_file(str(model_dir / "model.safetensors")).items()}
    expected = set(model.state_dict())
    # BatchNorm's num_batches_tracked is a step counter, not a weight; the saved
    # checkpoints omit it and it plays no part in eval-mode inference.
    missing = {k for k in expected - set(state) if not k.endswith("num_batches_tracked")}
    unexpected = set(state) - expected
    if missing or unexpected:
        raise RuntimeError(
            f"{model_dir}: key mismatch after rename -- {len(missing)} missing "
            f"(e.g. {sorted(missing)[:3]}), {len(unexpected)} unexpected "
            f"(e.g. {sorted(unexpected)[:3]})")
    model.load_state_dict(state, strict=False)
    return model.eval()
