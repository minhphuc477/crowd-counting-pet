#!/bin/bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA="${DATA:-./data/ShanghaiTech/part_A}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python main.py \
  --backbone convnextv2_base \
  --timm_adapter rcc_fpn \
  --timm_output_norm gn \
  --dataset_file SHA \
  --data_path "$DATA" \
  --output_dir outputs/SHA/convnextv2_base_rccfpn_ce_safe_apg_seed42 \
  --device cuda \
  --num_workers 2 \
  --batch_size 1 \
  --accum_iter 8 \
  --amp \
  --epochs 1500 \
  --eval_freq 5 \
  --eval_start_epoch 100 \
  --lr_scheduler warmup_hold_cosine \
  --warmup_epochs 20 \
  --hold_epochs 300 \
  --min_lr 1e-7 \
  --lr 0.00001 \
  --lr_backbone 0.000001 \
  --lr_backbone_adapter 0.00002 \
  --weight_decay 0.0001 \
  --clip_max_norm 0.1 \
  --ema_decay 0 \
  --eval_model raw \
  --patch_size 256 \
  --patch_size_choices 192,256 \
  --crop_attempts 12 \
  --min_crop_points 1 \
  --pet_loss_variant paper \
  --split_loss_variant gt \
  --quadtree_loss_coef 0.25 \
  --split_count_threshold 5 \
  --split_pos_weight 2.0 \
  --class_loss_type ce \
  --class_prior_prob 0.0023 \
  --eos_coef 0.5 \
  --apg_loss_coef 0.1 \
  --apg_pos_k 1 \
  --apg_point_coef 5.0 \
  --apg_bg_coef 1.0 \
  --apg_bg_k 32 \
  --apg_bg_min_dist 16 \
  --apg_start_epoch 120 \
  --apg_warmup_epochs 240 \
  --bayesian_sigma 8.0 \
  --bayesian_bg_coef 0.02 \
  --bayesian_loss_gate detach \
  --bayesian_start_epoch 300 \
  --count_head_loss_coef 0 \
  --density_map_loss_coef 0 \
  --eval_count_mode threshold \
  --score_threshold 0.55 \
  --split_threshold 0.5 \
  --split_threshold_quantile 0.5 \
  --eval_nms_radius 8 \
  --eval_branch_gate pred \
  --eval_soft_split_gate none \
  --bad_count_start_epoch 100 \
  --bad_count_direction over \
  --eval_debug_counting \
  --seed 42
