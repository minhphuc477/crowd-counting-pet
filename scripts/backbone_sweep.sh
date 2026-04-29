#!/bin/bash
# Bash script to run backbone sweep on Linux/Mac
# 
# Usage:
#   bash scripts/backbone_sweep.sh
# 
# This script runs multiple backbones with 1 seed each for quick comparison

set -e

BACKBONES=(
    "swinv2_base_window8_256"
    "convnextv2_base"
    "maxvit_rmlp_tiny_rw_256"
)

SEED=42
EPOCHS=1500
PATCH_SIZE=256

echo "========================================"
echo "PET Backbone Sweep (Quick)"
echo "========================================"
echo ""
echo "Backbones to test: ${BACKBONES[@]}"
echo "Seed: $SEED"
echo "Epochs: $EPOCHS"
echo ""

declare -A RESULTS

for backbone in "${BACKBONES[@]}"; do
    echo "========================================"
    echo "Testing: $backbone"
    echo "========================================"
    
    output_dir="backbone_${backbone}_seed_${SEED}"
    cmd="python main.py --backbone $backbone --seed $SEED --epochs $EPOCHS --output_dir $output_dir --patch_size $PATCH_SIZE"
    
    echo "Command: $cmd"
    echo ""
    
    if eval "$cmd"; then
        # Try to extract best MAE from log
        log_file="outputs/SHA/${output_dir}/run_log.txt"
        if [ -f "$log_file" ]; then
            best_mae=$(grep "best mae:" "$log_file" | tail -1 | grep -oP 'best mae:\s+\K[0-9.]+' || echo "N/A")
            RESULTS[$backbone]="SUCCESS : MAE=$best_mae"
            echo "✓ $backbone - Best MAE: $best_mae"
        fi
    else
        RESULTS[$backbone]="FAILED"
        echo "✗ $backbone - Training failed"
    fi
    
    echo ""
done

echo "========================================"
echo "Backbone Sweep Summary"
echo "========================================"

for backbone in "${BACKBONES[@]}"; do
    echo "$backbone : ${RESULTS[$backbone]}"
done

echo ""
echo "Next steps:"
echo "1. Pick the backbone with lowest MAE"
echo "2. Run ensemble with multiple seeds:"
echo "   python scripts/run_backbone_seeds.py --backbone <best> --seeds 42 7 13 99 1234"
echo "3. Evaluate ensemble:"
echo "   python scripts/ensemble_evaluate.py --backbone <best> --checkpoints outputs/SHA/<best>_seed_*/best_checkpoint.pth"
echo ""
