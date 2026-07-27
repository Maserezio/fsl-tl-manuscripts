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


class SMPUNet(nn.Module):
    """SMP segmentation model with a swappable architecture + encoder.

    cfg.model.arch         : 'unet' | 'segformer'   (default 'unet')
    cfg.model.encoder_name : any smp/timm encoder, or one of HIER_VIT_BACKBONES
                             (default 'resnet34')
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
            ensure_hier_encoder_registered(encoder_name)
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

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.net(img)

    @torch.no_grad()
    def predict(self, img: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(img))
