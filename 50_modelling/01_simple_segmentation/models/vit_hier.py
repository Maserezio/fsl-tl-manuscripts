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

HIER_VIT_BACKBONES = {
    "vit_small_patch16_224.augreg_in21k": "vit_small_patch16_224.augreg_in21k",
    "vit_small_patch16_dinov3": "dinov3_vits16",
    # Foundation-backbone comparison aliases (small variants unless the family
    # only exposes a base model, as for RADIO v2.5).
    "dinov2": "dinov2_vits14",
    "dinov2_reg": "dinov2_vits14_reg",
    "dinov3": "dinov3_vits16",
    "am-radio": "radio_v2.5-b",
}

_registered: set[str] = set()


def ensure_hier_encoder_registered(encoder_name: str) -> None:
    if encoder_name in _registered:
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
        pretrained=True,
        freeze_backbone=True,
        feature_strategy="sfp",
    )
    register_smp_encoder(encoder_name, cfg)

    entry = smp.encoders.encoders[encoder_name]
    probe = entry["encoder"](**entry["params"])
    try:
        if isinstance(probe.encoder.backbone, FallbackViTBackbone):
            raise RuntimeError(
                f"{hier_name!r} silently fell back to a random-init backbone (see the "
                f"warning printed above for the underlying loading error -- likely "
                f"gated/unreachable pretrained weights). Refusing to train on it."
            )
    finally:
        del probe

    _registered.add(encoder_name)
