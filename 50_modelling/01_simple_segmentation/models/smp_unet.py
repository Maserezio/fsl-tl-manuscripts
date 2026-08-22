import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from .vit_hier import HIER_VIT_BACKBONES, ensure_hier_encoder_registered

# All backbones go through smp.Unet uniformly -- including the two plain,
# non-hierarchical ViT-S/16 backbones. A flat ViT (timm features_only=True gives
# 3 stages all at the SAME reduction=16, not a real pyramid) can't feed smp.Unet's
# decoder directly ("Unsupported model downsampling pattern"); instead we wrap it
# with hier_encoder's SFP ("Simple Feature Pyramid", ViTDet) neck, which
# synthesizes a genuine [4,8,16,32]-stride pyramid from the flat tokens, then
# register that as a real smp encoder via hier_encoder.adapters.register_smp_encoder.
# See models/vit_hier.py for the encoder_name -> hier_encoder backbone mapping.
_ARCH = {
    "unet": smp.Unet,
    "segformer": smp.Segformer,
}

# --- skip-connection ablation -------------------------------------------------
# smp's encoders return [x, s2, s4, s8, s16, s32]; UnetDecoder drops the first
# (full-resolution input), takes s32 as the head and feeds the remaining four as
# skips, deepest first: block0<-s16, block1<-s8, block2<-s4, block3<-s2 (block4
# never gets one). Ablating a skip must NOT change the decoder: dropping a skip
# tensor would shrink that block's conv in_channels and turn this into a
# capacity ablation. Instead we keep every tensor and zero it, so parameter count
# and every layer shape stay bit-for-bit identical to the baseline runs -- only
# the information carried by the skip is removed.
#
# Ordered deepest (coarsest) skip first, so n_skips=N keeps the N deepest and
# zeroes the higher-resolution ones -- i.e. the high-frequency detail is what
# goes away first, which is exactly the signal this ablation is about.
_SKIP_FEATURE_INDICES = (4, 3, 2, 1)   # encoder-output index of s16, s8, s4, s2
MAX_SKIPS = len(_SKIP_FEATURE_INDICES)


class SMPUNet(nn.Module):
    """SMP segmentation model with a swappable architecture + encoder.

    cfg.model.arch         : 'unet' | 'segformer'   (default 'unet')
    cfg.model.encoder_name : any smp/timm encoder, or one of HIER_VIT_BACKBONES
                             (default 'resnet34')
    cfg.model.n_skips      : 0..4 U-Net skip connections kept, deepest first
                             (default 4 = untouched baseline; unet only)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg.get("model", {})
        arch = mcfg.get("arch", "unet").lower()
        if arch not in _ARCH:
            raise ValueError(f"unknown arch {arch!r} (choose from {list(_ARCH)})")
        encoder_name = mcfg.get("encoder_name", "resnet34")

        is_hier_vit = encoder_name in HIER_VIT_BACKBONES
        if is_hier_vit:
            # Flat ViTs take their weights through hier_encoder, not smp's
            # encoder_weights, so they need their own two keys. Defaults keep the
            # original behaviour (ImageNet, backbone frozen, only the SFP neck
            # trains); set freeze_backbone: false to train the ViT end to end like
            # every CNN/hierarchical encoder in the same matrix.
            ensure_hier_encoder_registered(
                encoder_name,
                pretrained=mcfg.get("pretrained", True),
                freeze_backbone=mcfg.get("freeze_backbone", True),
            )
        # hier_encoder-registered encoders load their own pretrained weights inside
        # HierSMPEncoder construction; smp's own encoder_weights mechanism looks up
        # pretrained_settings[encoder_weights] (registered empty), so it must be
        # None here or smp.Unet raises a KeyError.
        encoder_weights = None if is_hier_vit else mcfg.get("encoder_weights", "imagenet")

        self.net = _ARCH[arch](
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
        self.backbone = self.net.encoder          # for the optimizer's backbone LR group
        self.arch = arch
        self.encoder_name = encoder_name
        self.creator_name = f"SMP-{arch}-{encoder_name}"

        self.n_skips = int(mcfg.get("n_skips", MAX_SKIPS))
        if not 0 <= self.n_skips <= MAX_SKIPS:
            raise ValueError(f"model.n_skips must be in 0..{MAX_SKIPS}, got {self.n_skips}")
        if self.n_skips < MAX_SKIPS:
            if arch != "unet":
                raise ValueError(f"model.n_skips is only meaningful for arch 'unet', not {arch!r}")
            self._install_skip_ablation()
            self.creator_name += f"-skips{self.n_skips}"

    def _install_skip_ablation(self) -> None:
        """Zero the ablated skip tensors on their way out of the encoder.

        A forward hook on the encoder is enough: SegmentationModel.forward feeds
        the encoder's output list straight into the decoder, so replacing entries
        here removes exactly those skips and nothing else. Zeroed (not dropped)
        keeps every decoder conv's in_channels -- and therefore the parameter
        count -- identical to the 4-skip baseline. The bottleneck path (s32) is
        untouched, so the backbone still trains through the deep route.
        """
        dropped = tuple(_SKIP_FEATURE_INDICES[self.n_skips:])

        def _zero_skips(_module, _args, output):
            features = list(output)
            for index in dropped:
                features[index] = torch.zeros_like(features[index])
            return features

        self._skip_hook = self.net.encoder.register_forward_hook(_zero_skips)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.net(img)

    @torch.no_grad()
    def predict(self, img: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(img))
