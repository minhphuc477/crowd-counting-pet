#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

args=(
  --data_root "${NWPU_DATA:-./data/NWPU-Crowd}"
  --work_dir "${NWPU_WORK_DIR:-./data/downloads}"
  --eval_split "${NWPU_EVAL_SPLIT:-val}"
)

if [[ -n "${NWPU_URL:-}" ]]; then
  args+=(--url "$NWPU_URL")
fi
if [[ -n "${NWPU_ARCHIVE:-}" ]]; then
  args+=(--archive "$NWPU_ARCHIVE")
fi
if [[ -n "${CHECKPOINT:-}" ]]; then
  args+=(--checkpoint "$CHECKPOINT")
fi

python scripts/setup_nwpu_crowd.py "${args[@]}" "$@"
