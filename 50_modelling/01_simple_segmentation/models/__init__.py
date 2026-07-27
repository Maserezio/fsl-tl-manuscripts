import torch.nn as nn

from .smp_unet import SMPUNet


def build_model(cfg: dict) -> nn.Module:
    model_type = cfg.get("model", {}).get("type", "smp_unet")
    if model_type == "smp_unet":
        return SMPUNet(cfg)
    raise ValueError(f"unknown model type {model_type!r} (only 'smp_unet' is supported)")
