#!/usr/bin/env sh
set -eu

# Optuna search + training for all backbones (Ubuntu/Linux)
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
  python3 scripts/optuna_search.py \
    --backbone "${backbone}" \
    --trials 10 \
    --seeds 7 \
    --output_dir results

  echo "========================================"
  echo "Starting training for ${backbone}..."
  echo "========================================"
  best_args=$(python3 - "$backbone" <<'PY'
import json
import shlex
import sys
from pathlib import Path

path = Path("results") / sys.argv[1] / "optuna_best.json"
data = json.loads(path.read_text(encoding="utf-8"))
p = data["best_params"]
args = [
    "--lr", str(p["lr"]),
    "--lr_backbone", str(p["lr_backbone"]),
    "--batch_size", str(p["batch_size"]),
    "--warmup_epochs", str(p["warmup"]),
    "--score_threshold", str(p["score_threshold"]),
]
print(" ".join(shlex.quote(x) for x in args))
PY
)
  python3 main.py \
    --backbone "${backbone}" \
    --epochs 1500 \
    --patch_size 256 \
    --seed 7 \
    --output_dir "results/${backbone}/final_train" \
    ${best_args}

  echo "Completed ${backbone}"
  echo

done

echo "All backbones completed!"
