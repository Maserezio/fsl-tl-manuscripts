"""Registers the two plain ViT-S/16 backbones as real smp encoders via
hier_encoder's SFP neck, so they go through smp.Unet like every other backbone.

encoder_name (as used in configs/CLI throughout this project) -> hier_encoder
backbone name (hier_encoder/backbones.py's registry). Deliberately NOT prefixed
with "tu-": smp.encoders.get_encoder() special-cases any "tu-"-prefixed name to
TimmUniversalEncoder unconditionally, before ever checking the custom registry --
our registration would silently never take effect under that prefix.
    "vit_small_patch16_224.augreg_in21k"  (supervised ImageNet)
    "dinov3_vits16"                        (DINOv3 SSL; gated by Meta)

hier_encoder.backbones.load_backbone silently falls back to a randomly-initialized
ViT stub (with only a warning) when the real weights can't be loaded -- fine for
hier_encoder's own offline smoke tests, not acceptable here. ensure_hier_encoder_registered
builds the encoder once immediately and raises if that fallback was used, instead of
training silently on random weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

# hier_encoder is the shared SFP package under 71_misc/; ensure it's importable from any
# cwd / Jupyter kernel / Colab even without the editable install active.
_HIER_PARENT = Path(__file__).resolve().parents[3] / "71_misc"
if str(_HIER_PARENT) not in sys.path:
    sys.path.insert(0, str(_HIER_PARENT))

HIER_VIT_BACKBONES = {
    "vit_small_patch16_224.augreg_in21k": "vit_small_patch16_224.augreg_in21k",
    "vit_tiny_patch16_224.augreg_in21k": "vit_tiny_patch16_224.augreg_in21k",
    # Repointed from the "dinov3_vits16" torch.hub spec to the timm one: Meta gates
    # the hub weights, so that path raised "silently fell back to a random-init
    # backbone" and the encoder was unusable. timm serves the same lvd1689m
    # checkpoint ungated -- verified identical weights, 21.59M, real SFP pyramid.
    "vit_small_patch16_dinov3": "dinov3_vits16_timm",
    # Foundation-backbone comparison aliases (small variants unless the family
    # only exposes a base model, as for RADIO v2.5).
    "dinov2": "dinov2_vits14",
    "dinov2_reg": "dinov2_vits14_reg",
    "dinov3": "dinov3_vits16",
    "am-radio": "radio_v2.5-b",
}

# encoder_name -> the (pretrained, freeze_backbone) it is currently registered
# under. smp's encoder registry is global and keyed by name alone, so a second
# request with different settings has to re-register rather than reuse.
_registered: dict[str, tuple[bool, bool]] = {}


def ensure_hier_encoder_registered(
    encoder_name: str,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> None:
    """Register a flat ViT as an smp encoder behind hier_encoder's SFP neck.

    Defaults reproduce the original behaviour (ImageNet weights, backbone frozen,
    only the SFP neck trains) so existing runs stay reproducible. Both are worth
    setting explicitly:

    freeze_backbone=True leaves ~30M of ViT-S frozen while CNN and hierarchical-ViT
    encoders in the same comparison train end to end -- fine for a foundation-feature
    probe, a confound in a size or pretrain matrix.

    pretrained=False is required for a random-init arm; the fallback check below is
    skipped in that case, since a random backbone is then the intent rather than a
    silent failure.
    """
    if _registered.get(encoder_name) == (pretrained, freeze_backbone):
        return
    if encoder_name not in HIER_VIT_BACKBONES:
        raise ValueError(f"{encoder_name!r} is not a known hier_encoder ViT backbone")

    import segmentation_models_pytorch as smp
    from hier_encoder import EncoderConfig
    from hier_encoder.adapters import register_smp_encoder
    from hier_encoder.backbones import FallbackViTBackbone

    hier_name = HIER_VIT_BACKBONES[encoder_name]
    cfg = EncoderConfig(
        backbone=hier_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        feature_strategy="sfp",
    )
    register_smp_encoder(encoder_name, cfg)

    entry = smp.encoders.encoders[encoder_name]
    probe = entry["encoder"](**entry["params"])
    try:
        if pretrained and isinstance(probe.encoder.backbone, FallbackViTBackbone):
            raise RuntimeError(
                f"{hier_name!r} silently fell back to a random-init backbone (see the "
                f"warning printed above for the underlying loading error -- likely "
                f"gated/unreachable pretrained weights). Refusing to train on it."
            )
    finally:
        del probe

    _registered[encoder_name] = (pretrained, freeze_backbone)
