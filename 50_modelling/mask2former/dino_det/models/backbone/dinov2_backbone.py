import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone


class DinoV2FPNBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        del input_shape
        self.patch_size = cfg.MODEL.DINO.PATCH_SIZE
        self.embed_dim = cfg.MODEL.DINO.EMBED_DIM
        self.out_channels = cfg.MODEL.DINO.OUT_CHANNELS
        self.finetune_last_blocks = cfg.MODEL.DINO.FINETUNE_LAST_BLOCKS

        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            name: self.out_channels for name in self._out_features
        }
        self._size_divisibility = self.patch_size

        self.dino = self._load_dinov2(cfg)
        self.base_projection = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, self.out_channels),
            nn.GELU(),
        )
        self.scale_refinement = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(32, self.out_channels),
                    nn.GELU(),
                )
                for name in self._out_features
            }
        )

    @staticmethod
    def _extract_state_dict(state_dict):
        if "model" in state_dict:
            return state_dict["model"]
        if "state_dict" in state_dict:
            return state_dict["state_dict"]
        return state_dict

    def _load_dinov2(self, cfg):
        weights_path = cfg.MODEL.DINO.WEIGHTS
        pretrained = not bool(weights_path)
        try:
            model = torch.hub.load(
                "facebookresearch/dinov2",
                cfg.MODEL.DINO.MODEL_NAME,
                pretrained=pretrained,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DINOv2 via torch.hub. Either allow the initial download or set MODEL.DINO.WEIGHTS."
            ) from exc

        if weights_path:
            state_dict = torch.load(weights_path, map_location="cpu")
            state_dict = self._extract_state_dict(state_dict)
            model.load_state_dict(state_dict, strict=False)

        for parameter in model.parameters():
            parameter.requires_grad_(False)

        blocks = getattr(model, "blocks", None)
        if blocks is None or self.finetune_last_blocks >= len(blocks):
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        else:
            for block in blocks[-self.finetune_last_blocks :]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for attribute in ["norm", "fc_norm"]:
                module = getattr(model, attribute, None)
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
        return model

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def _pad_to_patch(self, x):
        height, width = x.shape[-2:]
        pad_h = (-height) % self.patch_size
        pad_w = (-width) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, height, width

    def forward(self, x):
        x, height, width = self._pad_to_patch(x)
        padded_height, padded_width = x.shape[-2:]

        features = self.dino.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]
        patch_height = padded_height // self.patch_size
        patch_width = padded_width // self.patch_size
        token_map = patch_tokens.permute(0, 2, 1).reshape(
            patch_tokens.shape[0], self.embed_dim, patch_height, patch_width
        )
        base_feature = self.base_projection(token_map)

        outputs = {}
        for name, stride in self._out_feature_strides.items():
            size = (math.ceil(height / stride), math.ceil(width / stride))
            scaled = F.interpolate(base_feature, size=size, mode="bilinear", align_corners=False)
            outputs[name] = self.scale_refinement[name](scaled)
        return outputs


@BACKBONE_REGISTRY.register()
def build_dinov2_fpn_backbone(cfg, input_shape):
    return DinoV2FPNBackbone(cfg, input_shape)