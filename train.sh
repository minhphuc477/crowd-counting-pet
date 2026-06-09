#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=./data/ShanghaiTech/part_A
export BATCH=10
export ACCUM=4

CUDA_VISIBLE_DEVICES=0 python main.py \
  --backbone convnextv2_base \
  --timm_adapter lite_fpn \
  --dataset_file SHA \
  --data_path "$DATA" \
  --output_dir outputs/SHA/convnextv2_base_gt_split_fixed \
  --device cuda \
  --num_workers 2 \
  --batch_size "$BATCH" \
  --accum_iter "$ACCUM" \
  --amp \
  --epochs 1500 \
  --eval_freq 5 \
  --lr_scheduler warmup_hold_cosine \
  --warmup_epochs 20 \
  --hold_epochs 200 \
  --min_lr 1e-7 \
  --lr 0.00005 \
  --lr_backbone 0.000005 \
  --lr_backbone_adapter 0.0001 \
  --weight_decay 0.0001 \
  --clip_max_norm 0.1 \
  --ema_decay 0.999 \
  --patch_size 256 \
  --patch_size_choices 192,256 \
  --crop_attempts 12 \
  --min_crop_points 1 \
  --split_loss_variant gt \
  --quadtree_loss_coef 0.1 \
  --split_count_threshold 1 \
  --split_pos_weight 2.0 \
  --apg_loss_coef 0.0 \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --eval_nms_radius 0 \
  --eval_branch_gate query \
  --eval_soft_split_gate none \
  --seed 42
