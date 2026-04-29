#!/bin/bash
#SBATCH --job-name=m2f
#SBATCH --account=p71186
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

module purge
module load python/3.11.3-gcc-12.2.0-hn7p65z


# ---- virtual env ----
PROJECT_ROOT=/gpfs/data/fs71186/ohanian/fsl-tl-manuscripts
VENV_DIR=/gpfs/data/fs71186/ohanian/venv
M2F_DIR="$PROJECT_ROOT/50_modelling/Mask2Former"

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: venv activation failed"
    exit 1
fi

# ---- redirect all caches to gpfs (avoid home dir quota) ----
export CACHE_DIR=/gpfs/data/fs71186/ohanian/.cache
export TORCH_HOME=$CACHE_DIR/torch
export HF_HOME=$CACHE_DIR/huggingface
export XDG_CACHE_HOME=$CACHE_DIR
mkdir -p "$TORCH_HOME" "$HF_HOME"

# ---- run ----
cd "$PROJECT_ROOT"
srun python "$M2F_DIR/train_cs18.py" \
    --config-file "$M2F_DIR/configs/diva/maskformer2_R50_diva.yaml" \
    --num-gpus 1 \
    DIVA.SUBSET cs18 \
    OUTPUT_DIR "$PROJECT_ROOT/80_models/segmentation/mask2former_cs18_R50" \
    SOLVER.CHECKPOINT_PERIOD 200