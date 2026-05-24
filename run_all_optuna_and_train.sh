#!/usr/bin/env sh
set -e

# Optuna search + training + evaluation for all backbones (Ubuntu/Linux)
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
efficientnet_b0
efficientnet_b1
resnet18
resnet50
resnet101
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
  echo "Starting processing for ${backbone}..."
  echo "========================================"

  dataset_file="SHA"
  train_output_rel="results/${backbone}/final_train"
  final_train_dir="outputs/${dataset_file}/${train_output_rel}"
  legacy_final_train_dir="${train_output_rel}"
  if [ ! -d "$final_train_dir" ] && [ -d "$legacy_final_train_dir" ]; then
    # Backward compatibility in case checkpoints were saved outside outputs/<dataset>/.
    final_train_dir="$legacy_final_train_dir"
  fi

  eval_results_file="results/${backbone}/eval_results.json"

  # ── Phase 1: Check if everything is already done ──
  if [ -f "$eval_results_file" ]; then
    echo "✓ Eval results already exist for ${backbone}. Skipping entirely."
    echo
    continue
  fi

  # ── Phase 2: Optuna search ──
  # Check if training is already complete (skip optuna if so)
  need_optuna=true
  best_params_file="results/${backbone}/optuna_best.json"
  if [ -f "$best_params_file" ]; then
    echo "✓ Optuna best params already exist for ${backbone}. Skipping search."
    need_optuna=false
  fi

  if [ "$need_optuna" = true ]; then
    echo "Starting Optuna search for ${backbone}..."
    echo "========================================"
    
    if ! python3 scripts/optuna_search.py \
      --backbone "${backbone}" \
      --trials 5 \
      --seeds 7 \
      --output_dir results; then
      echo "Warning: Optuna search for ${backbone} had issues. Check logs for details."
      echo "Attempting to continue with training using best params if available..."
    fi
  fi

  # ── Phase 3: Extract best params ──
  echo "========================================"
  echo "Checking if training should proceed for ${backbone}..."
  echo "========================================"
  
  if [ ! -f "$best_params_file" ]; then
    echo "Error: Best params file not found at $best_params_file. Skipping ${backbone}."
    continue
  fi

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
    
    p = data.get("best_params")
    if p is None:
        print("ERROR: No best params found", file=sys.stderr)
        sys.exit(1)
    
    best_value = data.get("best_value")
    if best_value is None:
        print("ERROR: No best value found", file=sys.stderr)
        sys.exit(1)
    
    args = [
        "--lr_scheduler", "warmup_hold_cosine",
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
    echo "Skipping ${backbone} - no valid best params found."
    echo "Run Optuna again or check optuna_best.json"
    continue
  }

  # ── Phase 4: Training ──
  # Check if training already completed
  training_done=false
  if [ -d "$final_train_dir" ]; then
    if [ -f "$final_train_dir/best_checkpoint.pth" ]; then
      echo "✓ Training already completed for ${backbone} (best_checkpoint.pth exists). Skipping training."
      training_done=true
    elif [ -f "$final_train_dir/checkpoint.pth" ]; then
      echo "⚠ Training output exists but no best_checkpoint. Will resume training."
    fi
  fi

  if [ "$training_done" = false ]; then
    echo "========================================"
    echo "Starting training for ${backbone}..."
    echo "========================================"
    
    resume_flag=""
    checkpoint_path="${final_train_dir}/checkpoint.pth"
    if [ -f "$checkpoint_path" ]; then
      echo "Found existing checkpoint. Will resume training from $checkpoint_path"
      resume_flag="--resume $checkpoint_path"
    fi
    
    if python3 main.py \
      --backbone "${backbone}" \
      --epochs 1500 \
      --patch_size 256 \
      --seed 7 \
      --output_dir "${train_output_rel}" \
      ${resume_flag} \
      ${best_args}; then
      echo "✓ Completed training for ${backbone}"
    else
      EXIT_CODE=$?
      echo "✗ Training for ${backbone} failed with exit code ${EXIT_CODE}"
      echo "  Check ${final_train_dir}/ for error details"
      echo "  This backbone will retry on next script run"
      echo
      continue
    fi
  fi

  # ── Phase 5: Evaluation on best checkpoint ──
  best_ckpt="${final_train_dir}/best_checkpoint.pth"
  if [ ! -f "$best_ckpt" ]; then
    # Fall back to regular checkpoint if best doesn't exist
    best_ckpt="${final_train_dir}/checkpoint.pth"
  fi

  if [ ! -f "$best_ckpt" ]; then
    echo "✗ No checkpoint found for ${backbone}. Cannot evaluate."
    echo
    continue
  fi

  echo "========================================"
  echo "Running evaluation for ${backbone}..."
  echo "  Checkpoint: ${best_ckpt}"
  echo "========================================"

  eval_log="results/${backbone}/eval_log.txt"
  mkdir -p "results/${backbone}"

  if python3 eval.py \
    --backbone "${backbone}" \
    --resume "${best_ckpt}" \
    --seed 7 2>&1 | tee "${eval_log}"; then
    echo "✓ Completed evaluation for ${backbone}"
  else
    echo "✗ Evaluation for ${backbone} failed"
    echo
    continue
  fi

  # ── Phase 6: Save eval results to JSON ──
  python3 - "${backbone}" "${best_ckpt}" "${eval_log}" <<'PY'
import json
import re
import sys
import torch
from pathlib import Path

backbone = sys.argv[1]
ckpt_path = sys.argv[2]
eval_log_path = sys.argv[3]
out_dir = Path("results") / backbone

# Extract best metrics from the checkpoint
try:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_mae = ckpt.get("best_mae", None)
    best_mse = ckpt.get("best_mse", None)
    best_epoch = ckpt.get("best_epoch", None)
    epoch = ckpt.get("epoch", None)
except Exception as e:
    print(f"Warning: could not read checkpoint: {e}", file=sys.stderr)
    best_mae, best_mse, best_epoch, epoch = None, None, None, None

# Parse actual eval-time MAE/MSE from eval log
eval_mae, eval_mse = None, None
eval_log = Path(eval_log_path)
if eval_log.exists():
    text = eval_log.read_text(encoding="utf-8", errors="replace")
    # eval.py prints: "epoch: X, mae: Y, mse: Z"
    match = re.search(r'mae:\s*([0-9.]+).*?mse:\s*([0-9.]+)', text)
    if match:
        eval_mae = float(match.group(1))
        eval_mse = float(match.group(2))

# Read optuna best value
optuna_file = out_dir / "optuna_best.json"
optuna_mae = None
if optuna_file.exists():
    try:
        optuna_data = json.loads(optuna_file.read_text(encoding="utf-8"))
        optuna_mae = optuna_data.get("best_value")
    except Exception:
        pass

results = {
    "backbone": backbone,
    "checkpoint": ckpt_path,
    "eval_mae": eval_mae,
    "eval_mse": eval_mse,
    "train_best_mae": float(best_mae) if best_mae is not None else None,
    "train_best_mse": float(best_mse) if best_mse is not None else None,
    "best_epoch": int(best_epoch) if best_epoch is not None else None,
    "total_epochs": int(epoch) + 1 if epoch is not None else None,
    "optuna_search_mae": float(optuna_mae) if optuna_mae is not None else None,
}

eval_out = out_dir / "eval_results.json"
eval_out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
mae_str = f"{eval_mae:.2f}" if eval_mae is not None else "N/A"
mse_str = f"{eval_mse:.2f}" if eval_mse is not None else "N/A"
print(f"  Saved eval results to {eval_out}")
print(f"  Eval MAE: {mae_str}, Eval MSE: {mse_str}, Best Epoch: {results['best_epoch']}")
PY

  echo
done

# ── Final Summary: Collect all results into one table ──
echo "========================================"
echo "FINAL ABLATION SUMMARY"
echo "========================================"

python3 - <<'SUMMARY'
import json
from pathlib import Path

results_dir = Path("results")
rows = []
for eval_file in sorted(results_dir.glob("*/eval_results.json")):
    try:
        data = json.loads(eval_file.read_text(encoding="utf-8"))
        rows.append(data)
    except Exception:
        pass

if not rows:
    print("No eval results found yet.")
else:
    # Sort by eval MAE (best first), fall back to train_best_mae
    rows.sort(key=lambda r: r.get("eval_mae") or r.get("train_best_mae") or 9999)

    # Print table
    header = f"{'Backbone':<25} {'Eval MAE':>9} {'Eval MSE':>9} {'Train MAE':>10} {'Epoch':>6} {'Optuna':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        eval_mae = f"{r['eval_mae']:.2f}" if r.get('eval_mae') is not None else "N/A"
        eval_mse = f"{r['eval_mse']:.2f}" if r.get('eval_mse') is not None else "N/A"
        train_mae = f"{r['train_best_mae']:.2f}" if r.get('train_best_mae') is not None else "N/A"
        epoch = str(r.get('best_epoch', 'N/A'))
        optuna = f"{r['optuna_search_mae']:.2f}" if r.get('optuna_search_mae') is not None else "N/A"
        print(f"{r['backbone']:<25} {eval_mae:>9} {eval_mse:>9} {train_mae:>10} {epoch:>6} {optuna:>7}")

    # Save combined results
    combined_out = results_dir / "ablation_summary.json"
    combined_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved combined results to {combined_out}")
SUMMARY

echo
echo "All backbones processing completed!"
