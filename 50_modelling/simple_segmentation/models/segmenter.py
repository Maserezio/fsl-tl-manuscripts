from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .allspark import AllSparkModule
from .backbone import get_backbone
from .decoder import SegDecoder


class UnifiedSegmenter(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        bcfg = cfg["backbone"]
        dcfg = cfg["decoder"]
        acfg = cfg.get("allspark", {})

        self.use_allspark = acfg.get("enabled", False)
        self.creator_name = "AllSpark" if self.use_allspark else "SimpleSegmenter"

        self.backbone = get_backbone(bcfg)
        channels = bcfg["out_channels"]
        self.decoder = SegDecoder(
            in_channels=channels,
            num_classes=dcfg.get("num_classes", 1),
            hidden=dcfg.get("hidden", 128),
        )

        if self.use_allspark:
            self.allspark = AllSparkModule(
                in_channels=channels,
                ec=acfg["embedding_channels"],
                num_heads=acfg["num_heads"],
                num_class=acfg["num_class"],
                patch_num=acfg["patch_num"],
            )
        else:
            self.allspark = None

        self.pseudo_prob: Optional[torch.Tensor] = None
        self.using_smem = False

    def _forward_identity_logits(self, img: torch.Tensor) -> torch.Tensor:
        _, _, height, width = img.shape
        features = self.backbone(img)
        if self.use_allspark:
            features_2x = torch.cat([features, features], dim=0)
            features = self.allspark(
                features_2x,
                pseudo_prob=None,
                using_smem=self.using_smem,
            )[: img.size(0)]
        return self.decoder(features, (height, width))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self._forward_identity_logits(img)

    @torch.no_grad()
    def update_pseudo_labels(self, img_u: torch.Tensor) -> None:
        if not self.use_allspark:
            return

        was_training = self.training
        self.eval()

        batch_size, _, height, width = img_u.shape
        feat_u = self.backbone(img_u)
        feat_2x = torch.cat([feat_u, feat_u], dim=0)
        feat_as = self.allspark(feat_2x, pseudo_prob=None, using_smem=self.using_smem)
        logits_u = self.decoder(feat_as[:batch_size], (height, width))
        prob_u = torch.sigmoid(logits_u)

        _, _, feat_h, feat_w = feat_u.shape
        prob_u_ds = F.interpolate(
            prob_u,
            size=(feat_h, feat_w),
            mode="bilinear",
            align_corners=False,
        )
        background = 1.0 - prob_u_ds
        self.pseudo_prob = torch.cat([background, prob_u_ds], dim=1).detach()

        if was_training:
            self.train()

    def forward_train(
        self,
        img_l: torch.Tensor,
        img_u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_allspark:
            return self.forward(img_l), self.forward(img_u)

        batch_size, _, height, width = img_l.shape
        img_cat = torch.cat([img_l, img_u], dim=0)
        feat_cat = self.backbone(img_cat)
        feat_as = self.allspark(feat_cat, self.pseudo_prob, self.using_smem)
        logits = self.decoder(feat_as, (height, width))
        return logits[:batch_size], logits[batch_size:]

    @torch.no_grad()
    def predict(self, img: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(img))

    def enable_smem(self) -> None:
        if self.use_allspark:
            self.using_smem = True
