"""Runnable sanity check (no pytest). ``python -m hier_encoder.smoke_test``.

Builds several encoder configs on CPU, forwards a non-square tensor, and prints
per-stride shapes / stats and parameter counts. Does not hard-fail when
pretrained weights / internet are unreachable -- loaders fall back to a
random-init ViT stub of matching dimensions.
"""
from __future__ import annotations

import torch

from .build import build_hierarchical_encoder
from .config import EncoderConfig


def _params(m):
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, train


def _check(enc, x, tol_note=""):
    with torch.no_grad():
        feats = enc(x)
    ph, pw = enc.last_padding
    H, W = x.shape[-2] + ph, x.shape[-1] + pw
    ps = enc.patch_size
    # patch=14 backbones produce H/14 (not H/16) at the nominal "16" level, so
    # the SFP ladder is anchored on the patch grid: level-s size = Hp * 16/s.
    # That introduces up to ±1 rounding vs an ideal H/s; patch=16 is exact.
    tol = 1 if ps != 16 else 0
    Hp, Wp = H // ps, W // ps
    print(f"  padded input: {H}x{W} (pad {ph},{pw}), patch={ps} {tol_note}")
    for s, t in feats.items():
        if s == 2:  # true stride-2 stem, not a nominal SFP level
            exp_h, exp_w = H // 2, W // 2
        else:
            exp_h, exp_w = round(Hp * 16 / s), round(Wp * 16 / s)
        ok = abs(t.shape[-2] - exp_h) <= tol and abs(t.shape[-1] - exp_w) <= tol
        print(
            f"  stride {s:>2}: {tuple(t.shape)}  mean={t.mean():+.3f} std={t.std():.3f}"
            f"  expected~({exp_h},{exp_w}) [tol=±{tol}] {'OK' if ok else 'MISMATCH'}"
        )
        assert ok, f"stride {s}: got {tuple(t.shape[-2:])} expected ~({exp_h},{exp_w})"
    tot, tr = _params(enc)
    print(f"  params: total={tot / 1e6:.2f}M  trainable={tr / 1e6:.2f}M")
    print(f"  channels per stride: {dict(zip(enc.strides, enc.out_channels))}")
    print(f"  strides={enc.strides}  num_features={enc.num_features}\n")


def main():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 518, 728)  # non-square, not a patch multiple

    print("=" * 70)
    print("[1] default config (dinov3_vitb16, sfp, frozen, windowed)")
    enc = build_hierarchical_encoder(EncoderConfig())
    _check(enc, x, tol_note="(patch=16 -> exact)")

    print("[2] feature_strategy=multiblock")
    enc = build_hierarchical_encoder(EncoderConfig(feature_strategy="multiblock"))
    _check(enc, x)

    print("[3] include_stride2=True (5-level output for SMP Unet)")
    enc = build_hierarchical_encoder(EncoderConfig(include_stride2=True))
    assert enc.num_features == 5, enc.num_features
    _check(enc, x)

    print("[4] LoRA rank=8 on a frozen backbone (trainable should grow)")
    enc = build_hierarchical_encoder(EncoderConfig(lora_rank=8))
    _check(enc, x)

    print("[5] patch-14 backbone (dinov2_vitb14_reg, ±1 tolerance)")
    enc = build_hierarchical_encoder(EncoderConfig(backbone="dinov2_vitb14_reg"))
    _check(enc, x, tol_note="(patch=14 -> ±1 tolerance)")

    print("[6] fake RADIO-like backbone stub (offline-safe: pretrained=False)")
    enc = build_hierarchical_encoder(
        EncoderConfig(backbone="radio_v2.5-b", feature_strategy="sfp", pretrained=False)
    )
    _check(enc, x)

    print("=" * 70)
    print("smoke test PASSED")


if __name__ == "__main__":
    main()
