import torch.nn as nn
import torch.nn.functional as F


class SimpleFPNPixelDecoder(nn.Module):
    def __init__(self, input_shapes, conv_dim, mask_dim):
        super().__init__()
        self.in_features = ["res2", "res3", "res4", "res5"]

        self.lateral_convs = nn.ModuleDict()
        self.output_convs = nn.ModuleDict()
        for name in self.in_features:
            in_channels = input_shapes[name].channels
            self.lateral_convs[name] = nn.Sequential(
                nn.Conv2d(in_channels, conv_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, conv_dim),
            )
            self.output_convs[name] = nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, conv_dim),
                nn.GELU(),
            )

        self.mask_projection = nn.Sequential(
            nn.Conv2d(conv_dim, mask_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, mask_dim),
            nn.GELU(),
        )

    def forward_features(self, features):
        fused = {}
        top_down = None
        for name in reversed(self.in_features):
            current = self.lateral_convs[name](features[name])
            if top_down is not None:
                top_down = F.interpolate(top_down, size=current.shape[-2:], mode="bilinear", align_corners=False)
                current = current + top_down
            top_down = self.output_convs[name](current)
            fused[name] = top_down

        mask_features = self.mask_projection(fused["res2"])
        multi_scale_features = [fused["res5"], fused["res4"], fused["res3"]]
        return mask_features, multi_scale_features