"""Map timm state dicts of the cBAD-distilled encoders onto the HF backbones of RT-DETR.

The cBAD exports are plain timm state dicts (see maskrcnn_diva.CBAD_BACKBONE_ROOT).
ConvNeXt reuses train_rtdetr_hf._timm_convnext_to_hf; PVTv2 needs its own rename,
including the split of timm's fused kv projection into HF's key and value.
"""
from __future__ import annotations

import re

import torch


def timm_pvt_v2_to_hf(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if k.startswith("patch_embed."):
            out["encoder.layers.0.patch_embedding." + k[len("patch_embed."):].replace("norm.", "layer_norm.")] = v
            continue
        m = re.match(r"stages\.(\d+)\.downsample\.(proj|norm)\.(weight|bias)", k)
        if m:
            s, part, t = m.groups()
            out[f"encoder.layers.{s}.patch_embedding.{'proj' if part == 'proj' else 'layer_norm'}.{t}"] = v
            continue
        m = re.match(r"stages\.(\d+)\.norm\.(weight|bias)", k)
        if m:
            out[f"encoder.layers.{m.group(1)}.layer_norm.{m.group(2)}"] = v
            continue
        m = re.match(r"stages\.(\d+)\.blocks\.(\d+)\.(.+)", k)
        if not m:
            raise KeyError(k)
        s, b, tail = m.groups()
        p = f"encoder.layers.{s}.blocks.{b}."
        if tail.startswith("attn.kv."):
            t = tail.split(".")[-1]
            half = v.shape[0] // 2
            out[p + f"attention.key.{t}"] = v[:half].clone()
            out[p + f"attention.value.{t}"] = v[half:].clone()
            continue
        tail = (tail.replace("attn.q.", "attention.query.")
                    .replace("attn.proj.", "attention.proj.")
                    .replace("attn.sr.", "attention.spatial_reduction.")
                    .replace("attn.norm.", "attention.layer_norm.")
                    .replace("mlp.fc1.", "mlp.dense1.")
                    .replace("mlp.fc2.", "mlp.dense2.")
                    .replace("mlp.dwconv.", "mlp.dwconv.dwconv.")
                    .replace("norm1.", "layer_norm_1.")
                    .replace("norm2.", "layer_norm_2."))
        out[p + tail] = v
    return out
