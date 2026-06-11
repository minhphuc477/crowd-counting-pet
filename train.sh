#!/bin/bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA="${DATA:-./data/ShanghaiTech/part_A}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python main.py \
  --backbone vgg16_bn \
  --timm_adapter lite_fpn \
  --dataset_file SHA \
  --data_path "$DATA" \
  --output_dir outputs/SHA/vgg16_bn_drop700_apg_lc_seed42 \
  --device cuda \
  --num_workers 2 \
  --batch_size 8 \
  --epochs 1500 \
  --eval_freq 5 \
  --lr_scheduler step \
  --lr_drop 700 \
  --lr_gamma 0.1 \
  --lr 0.0001 \
  --lr_backbone 0.00001 \
  --weight_decay 0.0001 \
  --clip_max_norm 0.1 \
  --patch_size 256 \
  --crop_attempts 1 \
  --min_crop_points 0 \
  --pet_loss_variant paper \
  --apg_loss_coef 1.0 \
  --apg_pos_k 1 \
  --apg_point_coef 5.0 \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --eval_nms_radius 0 \
  --eval_branch_gate none \
  --seed 42
