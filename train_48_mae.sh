#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=./data/ShanghaiTech/part_A
export BATCH=4
export ACCUM=10

CUDA_VISIBLE_DEVICES=0 python main.py \
  --backbone convnextv2_base \
  --timm_adapter lite_fpn \
  --dataset_file SHA \
  --data_path "$DATA" \
  --output_dir outputs/SHA/convnextv2_base_SOTA_48 \
  --resume outputs/SHA/convnextv2_base_gt_split_fixed/checkpoint.pth \
  --resume_model_only \
  --device cuda \
  --num_workers 2 \
  --batch_size "$BATCH" \
  --accum_iter "$ACCUM" \
  --amp \
  --epochs 100 \
  --eval_freq 2 \
  --lr_scheduler warmup_hold_cosine \
  --warmup_epochs 0 \
  --hold_epochs 0 \
  --min_lr 1e-7 \
  --lr 0.00001 \
  --lr_backbone 0.000001 \
  --lr_backbone_adapter 0.00002 \
  --weight_decay 0.0001 \
  --clip_max_norm 0.1 \
  --ema_decay 0.999 \
  --patch_size 384 \
  --patch_size_choices 256,384 \
  --crop_attempts 12 \
  --min_crop_points 1 \
  --split_loss_variant gt \
  --quadtree_loss_coef 0.1 \
  --split_count_threshold 1 \
  --split_pos_weight 2.0 \
  --qd_apg_loss_coef 0.3 \
  --qd_apg_point_coef 5.0 \
  --qd_apg_suppress_coef 0.5 \
  --qd_apg_route_source gt_count \
  --ifi_loss_coef 0.1 \
  --ifi_neg_radius 12.0 \
  --apg_contrastive_coef 0.05 \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --eval_nms_radius 0 \
  --eval_branch_gate query \
  --eval_soft_split_gate none \
  --seed 42
