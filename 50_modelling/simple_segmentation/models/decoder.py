"""
Segmentation decoder: [B, C, H_f, W_f] → [B, num_classes, H_out, W_out].

Lightweight conv head followed by bilinear upsampling to the requested
output resolution. The upsampling factor is determined at runtime from the
target output_size, so it works for any backbone stride.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegDecoder(nn.Module):
    """
    Parameters
    ----------
    in_channels  : feature channels from backbone projection (default 256)
    num_classes  : 1 for binary text-line segmentation
    hidden       : intermediate channel width in the conv head
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 1,
        hidden: int = 128,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, num_classes, 1),
        )

    def forward(
        self,
        features: torch.Tensor,    # [B, in_channels, H_f, W_f]
        output_size: tuple,        # (H_out, W_out)
    ) -> torch.Tensor:
        logits = self.head(features)
        return F.interpolate(logits, size=output_size, mode="bilinear",
                             align_corners=False)
