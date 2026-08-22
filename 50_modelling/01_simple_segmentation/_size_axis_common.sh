# Shared body for run_train_udiads_size_axis{,_random}.sh -- sourced, not run.
# Expects: PY CFG OUT SUBSETS EPOCHS ENCODERS RUN_SUFFIX, and optionally
# LR / LR_BACKBONE (default 0.001 / 0.0001).
LR="${LR:-0.001}"
LR_BACKBONE="${LR_BACKBONE:-0.0001}"

# Single-instance lock. Two copies writing the same run dirs corrupt each other's
# checkpoints, and the loser silently skips runs it never trained.
_LOCK="/tmp/udiads_size_axis.lock"
exec 9>"$_LOCK"
if ! flock -n 9; then
  echo "ERROR: another size-axis run holds $_LOCK -- refusing to start a second one." >&2
  exit 1
fi

# --- preflight -------------------------------------------------------------
# Goes through build_model, not smp.Unet directly: the flat ViTs only become smp
# encoders after ensure_hier_encoder_registered() runs, so a bare smp.Unet call
# would report them as unknown. This also verifies the two things most likely to
# be silently wrong -- that the config's init actually took, and that a flat ViT
# did not fall back to hier_encoder's random stub.
echo "### preflight: $CFG ###"
"$PY" - "$CFG" "${ENCODERS[@]}" <<'PYEOF'
import sys, yaml, torch
sys.path.insert(0, ".")
from models import build_model
from models.vit_hier import HIER_VIT_BACKBONES
from hier_encoder.backbones import FallbackViTBackbone

cfg_path, encoders = sys.argv[1], sys.argv[2:]
base = yaml.safe_load(open(cfg_path))
mcfg = dict(base["model"])
want_pre = mcfg.get("pretrained", True) and mcfg.get("encoder_weights") is not None
print(f"  init: encoder_weights={mcfg.get('encoder_weights')!r} "
      f"pretrained={mcfg.get('pretrained', True)} "
      f"freeze_backbone={mcfg.get('freeze_backbone', True)}")

bad = []
for enc in encoders:
    try:
        m = build_model({"model": {**mcfg, "encoder_name": enc}})
        with torch.no_grad():
            m(torch.randn(1, 3, 224, 224))
        tot = sum(p.numel() for p in m.backbone.parameters())
        tr = sum(p.numel() for p in m.backbone.parameters() if p.requires_grad)
        note = ""
        if enc in HIER_VIT_BACKBONES:
            inner = m.backbone.encoder.backbone
            if isinstance(inner, FallbackViTBackbone):
                raise RuntimeError("fell back to hier_encoder's random ViT stub")
            note = "  [SFP]"
        if tr != tot:
            note += f"  WARNING: {(tot-tr)/1e6:.1f}M frozen"
        print(f"  OK   {enc:36} enc={tot/1e6:6.2f}M trainable={tr/1e6:6.2f}M{note}")
    except Exception as e:
        bad.append(enc)
        print(f"  FAIL {enc:36} {type(e).__name__}: {str(e)[:110]}")
sys.exit(1 if bad else 0)
PYEOF
echo

train_one() {
  local subset="$1" enc="$2"
  local run="unet_${enc}${RUN_SUFFIX}_udiads_${subset}"
  local dir="../../$OUT/$run"
  # Completion marker, not best.pth. train.py writes best.pth from the first epoch
  # that improves, so an interrupted run leaves one behind and a best.pth-based
  # guard would skip it forever as "already trained" -- which is exactly what
  # happened when two copies of this script ran at once: the second saw the first's
  # partial checkpoints and skipped straight past them.
  if [[ -f "$dir/.done" ]]; then
    echo "[skip] $run already trained"
    return
  fi
  if [[ -f "$dir/best.pth" ]]; then
    echo "[redo] $run has best.pth but no .done marker -> incomplete, retraining"
  fi
  echo "######## TRAIN  ${run}  (${EPOCHS} epochs) ########"
  local t0=$SECONDS
  "$PY" train.py --config "$CFG" \
    --arch unet --encoder "$enc" --manuscript "$subset" \
    --batch-size 4 --epochs "$EPOCHS" \
    --lr "$LR" --lr-backbone "$LR_BACKBONE" --weight-decay 0.0001 \
    --run-name "$run" --out-dir "$OUT"
  touch "$dir/.done"
  echo "[ok] $run in $(( (SECONDS - t0) / 60 )) min"
}

# Subset-major: finishes one manuscript across every size before moving on, so an
# interrupted run still leaves a complete sweep for at least one subset.
for ms in "${SUBSETS[@]}"; do
  for enc in "${ENCODERS[@]}"; do
    train_one "$ms" "$enc"
  done
done

echo "DONE: ${#ENCODERS[@]} encoders x ${#SUBSETS[@]} subsets (suffix='${RUN_SUFFIX}', lr_backbone=$LR_BACKBONE)"
