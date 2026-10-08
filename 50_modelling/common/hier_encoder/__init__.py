"""hier_encoder -- turn vision foundation models into hierarchical encoders.

Public API::

    from hier_encoder import build_hierarchical_encoder, EncoderConfig
    cfg = EncoderConfig(backbone="dinov3_vitb16", feature_strategy="sfp")
    encoder = build_hierarchical_encoder(cfg)
    feats = encoder(x)   # OrderedDict keyed by stride
"""
from .backbones import list_backbones
from .build import HierarchicalEncoder, Stride2Stem, build_hierarchical_encoder
from .config import EncoderConfig

__all__ = [
    "EncoderConfig",
    "HierarchicalEncoder",
    "Stride2Stem",
    "build_hierarchical_encoder",
    "list_backbones",
]

__version__ = "0.1.0"
