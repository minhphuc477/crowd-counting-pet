#!/bin/bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA="${DATA:-./data/ShanghaiTech/part_A}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python main.py \
  --backbone vgg16_bn \
  --timm_adapter lite_fpn \
  --dataset_file SHA \
  --data_path "$DATA" \
  --output_dir outputs/SHA/vgg16_bn_apglc_density_counthead_seed42 \
  --device cuda \
  --num_workers 2 \
  --batch_size 8 \
  --epochs 120 \
  --eval_freq 2 \
  --eval_start_epoch 0 \
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
  --count_head_loss_coef 1.0 \
  --count_head_loss_type log_l1 \
  --count_head_start_epoch 0 \
  --count_head_end_epoch -1 \
  --count_head_init_count 40 \
  --count_head_init_cells 1024 \
  --count_head_feature_grad_scale 1.0 \
  --allow_count_head_fresh_train \
  --allow_count_head_from_start \
  --density_map_loss_coef 0 \
  --eval_count_mode threshold \
  --score_threshold 0.575 \
  --split_threshold 0.47 \
  --eval_nms_radius 0 \
  --eval_branch_gate none \
  --eval_soft_split_gate none \
  --bad_count_start_epoch 20 \
  --seed 42
