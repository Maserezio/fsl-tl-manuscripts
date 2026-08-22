"""Generic hierarchical timm backbone for Ultralytics YOLO.

Any hierarchical backbone (native strides 8/16/32) drops into YOLO's neck via `features_only`.
Handles the two ViT-hierarchical quirks:
  * NHWC output (swin)                       -> permuted to NCHW
  * fixed input size (swin, hiera)           -> built at `imgsz`; off-size calls (Ultralytics'
                                                256x256 stride-probe) return correctly-shaped
                                                zeros so strides still resolve to 8/16/32.
Fixed-size backbones therefore require: square `imgsz`, `rect=False`, no `multi_scale`.

Import + register() in any notebook that builds or loads one of these models (checkpoints
reference `timm_backbone.TimmBackbone`, so they load anywhere on sys.path -- like ResNet).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _transplant_pretrained(feat_model, name, imgsz):
    """Load pretrained weights into a fixed-size backbone built at `imgsz`, interpolating any
    pos_embed from its native grid (needed for hiera, whose pos_embed is locked to 224)."""
    src = timm.create_model(name, pretrained=True)
    native = src.pretrained_cfg['input_size'][-1]
    sd = src.state_dict()
    for k in list(sd):
        if 'pos_embed' in k and sd[k].dim() == 3:               # (1, N, C) grid pos-embed
            pe = sd[k]; N, C = pe.shape[1], pe.shape[2]; g = int(round(N ** 0.5))
            tg = imgsz * g // native
            pe = pe.reshape(1, g, g, C).permute(0, 3, 1, 2)
            pe = F.interpolate(pe, size=(tg, tg), mode='bicubic', align_corners=False)
            sd[k] = pe.permute(0, 2, 3, 1).reshape(1, tg * tg, C)
    feat_model.load_state_dict({f'model.{k}': v for k, v in sd.items()}, strict=False)

BACKBONES = {
    'resnet34':   dict(id='resnet34',                        dynamic=True,  nhwc=False),
    'resnet50':   dict(id='resnet50',                        dynamic=True,  nhwc=False),
    'convnext':   dict(id='convnext_tiny.in12k_ft_in1k',     dynamic=True,  nhwc=False),
    'convnextv2': dict(id='convnextv2_tiny',                 dynamic=True,  nhwc=False),
    'convnext_dinov3': dict(id='convnext_tiny.dinov3_lvd1689m', dynamic=True, nhwc=False),
    'swin':       dict(id='swin_tiny_patch4_window7_224',    dynamic=False, nhwc=True),
    'hiera':      dict(id='hiera_tiny_224.mae_in1k_ft_in1k', dynamic=False, nhwc=False),
    # Hierarchical transformer with a genuinely dynamic input size -- unlike swin/hiera it
    # needs no img_size pinning or stride-probe stub, so rect/multi_scale stay usable.
    # ImageNet-only weights: no DINOv2/v3 exists for any hierarchical transformer, so this
    # fills the "supervised hierarchical ViT" slot and cannot stand in for the SFP branch.
    'pvt_v2':     dict(id='pvt_v2_b2.in1k',                  dynamic=True,  nhwc=False),
}
STRIDES = (8, 16, 32)


def pyramid_channels(key, imgsz=1024):
    """Channels at strides 8/16/32 for the chosen backbone (random init, no download)."""
    spec = BACKBONES[key]
    kw = {} if spec['dynamic'] else {'img_size': imgsz}
    fe = timm.create_model(spec['id'], pretrained=False, features_only=True, **kw)
    red, ch = list(fe.feature_info.reduction()), list(fe.feature_info.channels())
    idx = [i for i, r in enumerate(red) if r in STRIDES]
    return [ch[i] for i in idx]


class TimmBackbone(nn.Module):
    """timm hierarchical backbone -> [P3, P4, P5] feature list for a YOLO neck."""
    def __init__(self, key, imgsz=1024, freeze=False):
        super().__init__()
        spec = BACKBONES[key]
        self.nhwc = spec['nhwc']
        self.built = None if spec['dynamic'] else int(imgsz)   # fixed-size backbones remember their size
        kw = {} if spec['dynamic'] else {'img_size': int(imgsz)}
        probe = timm.create_model(spec['id'], pretrained=False, features_only=True, **kw)
        idx = [i for i, r in enumerate(probe.feature_info.reduction()) if r in STRIDES]
        try:
            self.m = timm.create_model(spec['id'], pretrained=True, features_only=True,
                                       out_indices=tuple(idx), **kw)
        except Exception:                       # fixed-size backbone w/ non-interpolatable pos_embed
            self.m = timm.create_model(spec['id'], pretrained=False, features_only=True,
                                       out_indices=tuple(idx), **kw)
            _transplant_pretrained(self.m, spec['id'], int(imgsz))
        self.chs = list(self.m.feature_info.channels())
        self.frozen = freeze
        if freeze:
            for p in self.m.parameters():
                p.requires_grad = False
        cfg = self.m.pretrained_cfg
        self.register_buffer('mean', torch.tensor(cfg['mean']).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(cfg['std']).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.m.eval()
        return self

    def forward(self, x):
        H, W = x.shape[-2:]
        if self.built is not None and (H != self.built or W != self.built):
            # off-size (e.g. Ultralytics' 256x256 stride probe): shapes-only, so strides resolve
            return [x.new_zeros(x.shape[0], c, H // s, W // s) for c, s in zip(self.chs, STRIDES)]
        x = (x - self.mean) / self.std
        run = torch.no_grad() if self.frozen else torch.enable_grad()
        with run:
            feats = self.m(x)
        if self.nhwc:
            feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        return list(feats)


def register():
    import ultralytics.nn.tasks as tasks
    tasks.TimmBackbone = TimmBackbone


def build_yaml(key, imgsz=1024, freeze=False, nc=1):
    """Full YOLOv8 detection YAML dict: TimmBackbone -> Index taps -> scale-n neck -> Detect."""
    c3, c4, c5 = pyramid_channels(key, imgsz)
    return {
        'nc': nc,
        'scales': {'n': [0.33, 0.25, 1024]},
        'backbone': [[-1, 1, 'TimmBackbone', [key, imgsz, freeze]],
                     [0, 1, 'Index', [c3, 0]], [0, 1, 'Index', [c4, 1]], [0, 1, 'Index', [c5, 2]]],
        'head': [
            [-1, 1, 'Conv', [512, 1, 1]], [-1, 1, 'nn.Upsample', [None, 2, 'nearest']], [[-1, 2], 1, 'Concat', [1]], [-1, 3, 'C2f', [512]],
            [-1, 1, 'Conv', [256, 1, 1]], [-1, 1, 'nn.Upsample', [None, 2, 'nearest']], [[-1, 1], 1, 'Concat', [1]], [-1, 3, 'C2f', [256]],
            [-1, 1, 'Conv', [256, 3, 2]], [[-1, 8], 1, 'Concat', [1]], [-1, 3, 'C2f', [512]],
            [-1, 1, 'Conv', [512, 3, 2]], [[-1, 4], 1, 'Concat', [1]], [-1, 3, 'C2f', [1024]],
            [[11, 14, 17], 1, 'Detect', ['nc']],
        ]}
