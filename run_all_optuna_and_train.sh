#!/usr/bin/env sh
set -e

# Optuna search + training for all backbones (Ubuntu/Linux)
# This script is designed to be resumable - it will continue from where it left off
# It handles failed trials and missing results gracefully
backbones="
convnext_tiny
convnext_base
convnextv2_tiny
convnextv2_base
swinv2_tiny
swinv2_base
maxvit_tiny
maxvit_small
maxvit_rmlp_tiny
fastvit_tiny
fastvit_small
efficientvit_tiny
efficientvit_small
efficientnetv2_tiny
efficientnetv2_small
mobilenetv4_small
mobilenetv4_hybrid
hgnetv2_tiny
hgnetv2_small
pvtv2_b0
pvtv2_b1
edgenext_tiny
edgenext_small
repvit_tiny
repvit_small
"

for backbone in $backbones; do
  echo "========================================"
  echo "Starting Optuna search for ${backbone}..."
  echo "========================================"
  
  # Run optuna search with error handling - continue even if it fails
  if ! python3 scripts/optuna_search.py \
    --backbone "${backbone}" \
    --trials 10 \
    --seeds 7 \
    --output_dir results; then
    echo "Warning: Optuna search for ${backbone} had issues. Check logs for details."
    echo "Attempting to continue with training using best params if available..."
  fi

  echo "========================================"
  echo "Checking if training should proceed for ${backbone}..."
  echo "========================================"
  
  best_params_file="results/${backbone}/optuna_best.json"
  if [ ! -f "$best_params_file" ]; then
    echo "Error: Best params file not found at $best_params_file. Skipping training for ${backbone}."
    continue
  fi

  # Check if best_params is None (no completed trials) and extract parameters
  best_args=$(python3 - "$backbone" <<'PY'
import json
import shlex
import sys
from pathlib import Path

path = Path("results") / sys.argv[1] / "optuna_best.json"
if not path.exists():
    print("ERROR: No best params file", file=sys.stderr)
    sys.exit(1)

try:
    data = json.loads(path.read_text(encoding="utf-8"))
    trials_completed = data.get("trials_completed", 0)
    
    if trials_completed == 0:
        print("ERROR: No completed trials yet", file=sys.stderr)
        sys.exit(1)
    
    p = data.get("best_params")
    if p is None:
        print("ERROR: No best params found", file=sys.stderr)
        sys.exit(1)
    
    args = [
        "--lr", str(p["lr"]),
        "--lr_backbone", str(p["lr_backbone"]),
        "--batch_size", str(p["batch_size"]),
        "--warmup_epochs", str(p["warmup"]),
        "--score_threshold", str(p["score_threshold"]),
    ]
    print(" ".join(shlex.quote(x) for x in args))
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PY
) || {
    echo "Skipping training for ${backbone} - no successful Optuna trials yet."
    echo "Run the script again to resume and complete the trials."
    continue
  }

  echo "========================================"
  echo "Starting training for ${backbone}..."
  echo "========================================"
  
  if python3 main.py \
    --backbone "${backbone}" \
    --epochs 1500 \
    --patch_size 256 \
    --seed 7 \
    --output_dir "results/${backbone}/final_train" \
    ${best_args}; then
    echo "Completed training for ${backbone}"
  else
    echo "Warning: Training for ${backbone} had issues. Check logs for details."
  fi

  echo

done

echo "All backbones processing completed - check individual logs!"
