#!/usr/bin/env bash
# Run ConvNeXtV2 base 5-seed ensemble (wrapper)
# Usage: bash scripts/run_convnextv2_base_ensemble.sh [--epochs N] [--patch_size P]

set -euo pipefail

EPOCHS=1500
PATCH_SIZE=256
SEEDS=(42 7 13 99 1234)
EXTRA_ARGS=""

# parse simple args
while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs) EPOCHS="$2"; shift 2;;
    --patch_size) PATCH_SIZE="$2"; shift 2;;
    --extra_args) EXTRA_ARGS="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

BACKBONE="convnextv2_base"

# 1) Train the seeds sequentially (uses existing Python helper)
# Force-disable enhanced point-query and use the new hold-cosine LR schedule
python scripts/run_backbone_seeds.py --backbone ${BACKBONE} --seeds ${SEEDS[@]} --extra_args "--epochs ${EPOCHS} --patch_size ${PATCH_SIZE} --lr_scheduler warmup_hold_cosine --warmup_epochs 10 --hold_epochs 100 --min_lr 1e-7 ${EXTRA_ARGS} --no-enhanced_point_query"

# 2) Evaluate ensemble from produced checkpoints
python scripts/ensemble_evaluate.py --backbone ${BACKBONE} --checkpoints "outputs/SHA/${BACKBONE}_seed_*/best_checkpoint.pth" --threshold_sweep

echo "Done: trained seeds ${SEEDS[*]} and evaluated ensemble for backbone ${BACKBONE}."
