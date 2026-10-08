#!/usr/bin/env python3
"""WiSE-FT for LineField: theta = (1 - a) * theta_CATMuS + a * theta_finetuned (Wortsman et al., 2022).

    .venv/bin/python 50_modelling/semantic_segmentation/center_line/rq3_lf_wise.py 0.5 SRC...          -> <SRC>_A_w50.pt (tag "_w50")
    .venv/bin/python 50_modelling/semantic_segmentation/center_line/rq3_lf_wise.py 0.5 --protocol L3 H  -> <H>_L3_w50.pt (LOCO-k3)
The fine-tuned models are <SRC>_<P>_cm.pt (rq3_lf_pretrain.py finetune / loco); nothing is trained here.
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq3_linefield as LF  # noqa: E402


def main():
    args = sys.argv[1:]
    prot = "A"
    if "--protocol" in args:
        i = args.index("--protocol")
        prot = args[i + 1]
        args = args[:i] + args[i + 2:]
    a = float(args[0])
    pre = torch.load(LF.MODELS / "catmus_pretrain.pt", map_location="cpu", weights_only=False)["model"]
    for src in args[1:]:
        out = LF.MODELS / f"{src}_{prot}_w{round(100 * a)}.pt"
        ft_path = LF.MODELS / f"{src}_{prot}_cm.pt"
        if out.exists() or not ft_path.exists():
            continue
        ft = torch.load(ft_path, map_location="cpu", weights_only=False)
        mix = {k: ((1 - a) * pre[k].float() + a * v.float()).to(v.dtype) if v.is_floating_point() else v
               for k, v in ft["model"].items()}
        torch.save({**ft, "model": mix, "wise_alpha": a, "init": "catmus_pretrain.pt"}, out)
        print("saved", out, flush=True)


if __name__ == "__main__":
    main()
