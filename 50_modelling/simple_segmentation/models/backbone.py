"""
Backbone wrappers: DINOv2 (ViT-B/14) and ResNet34.

Unified interface:
    features = backbone(x)   # [B, out_channels, H_f, W_f]

DINOv2
------
  - Loads dinov2_vitb14 from torch.hub ("facebookresearch/dinov2").
  - Input H, W are padded to the next multiple of 14 automatically.
  - Output stride = 14  →  H_f = ceil(H / 14).
  - Backbone weights are frozen by default; only the 1×1 projection is trained.

ResNet34
--------
  - Loads ResNet34_Weights.IMAGENET1K_V1 from torchvision.
  - use_layer3=True  → stride 16, 256-ch features  (recommended)
  - use_layer3=False → stride 32, 512-ch features
  - Backbone weights are frozen by default; only the 1×1 projection is trained.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Backbone(nn.Module):
    PATCH_SIZE = 14
    EMBED_DIM  = 768   # ViT-B hidden size

    def __init__(self, out_channels: int = 256, freeze: bool = True):
        super().__init__()
        self.dino = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=True,
        )
        if freeze:
            for p in self.dino.parameters():
                p.requires_grad_(False)

        self.proj = nn.Sequential(
            nn.Conv2d(self.EMBED_DIM, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.out_channels = out_channels

    def _pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        _, _, H, W = x.shape
        pH = (-H) % self.PATCH_SIZE
        pW = (-W) % self.PATCH_SIZE
        if pH or pW:
            x = F.pad(x, (0, pW, 0, pH))
        return x, (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad, _ = self._pad(x)
        _, _, H_pad, W_pad = x_pad.shape
        H_f = H_pad // self.PATCH_SIZE
        W_f = W_pad // self.PATCH_SIZE

        feats        = self.dino.forward_features(x_pad)
        patch_tokens = feats["x_norm_patchtokens"]   # [B, H_f*W_f, D]
        B, _, D      = patch_tokens.shape
        feat_map     = patch_tokens.permute(0, 2, 1).reshape(B, D, H_f, W_f)
        return self.proj(feat_map)                    # [B, out_channels, H_f, W_f]


class ResNetBackbone(nn.Module):
    """ResNet34 feature extractor with a learnable 1×1 projection head."""

    def __init__(
        self,
        out_channels: int = 256,
        freeze: bool = True,
        use_layer3: bool = True,
    ):
        super().__init__()
        import torchvision.models as tvm

        resnet = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1)
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.use_layer3 = use_layer3
        in_ch = 256 if use_layer3 else 512

        if freeze:
            for m in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
                for p in m.parameters():
                    p.requires_grad_(False)

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if not self.use_layer3:
            x = self.layer4(x)
        return self.proj(x)


def get_backbone(cfg: dict) -> nn.Module:
    btype = cfg["type"]
    if btype == "dinov2":
        return DINOv2Backbone(
            out_channels=cfg.get("out_channels", 256),
            freeze=cfg.get("freeze", True),
        )
    elif btype == "resnet34":
        return ResNetBackbone(
            out_channels=cfg.get("out_channels", 256),
            freeze=cfg.get("freeze", True),
            use_layer3=cfg.get("use_layer3", True),
        )
    else:
        raise ValueError(f"Unknown backbone type: {btype!r}")
