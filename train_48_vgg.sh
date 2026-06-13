export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=./data/ShanghaiTech/part_A
export BATCH=4
export ACCUM=10

CUDA_VISIBLE_DEVICES=0 python main.py \
  --backbone vgg16_bn \
  --dataset_file SHA \
  --data_path "$DATA" \
  --output_dir outputs/SHA/vgg16_bayesian_48 \
  --resume outputs/SHA/vgg16_bn_drop700_apg_lc_seed42/best_checkpoint.pth \
  --resume_model_only \
  --resume_allow_arch_change \
  --device cuda \
  --num_workers 2 \
  --batch_size "$BATCH" \
  --accum_iter "$ACCUM" \
  --amp \
  --epochs 100 \
  --eval_freq 2 \
  --lr_scheduler warmup_hold_cosine \
  --warmup_epochs 10 \
  --hold_epochs 0 \
  --min_lr 1e-7 \
  --lr 0.00001 \
  --lr_backbone 0.000001 \
  --weight_decay 0.0001 \
  --ema_decay 0.999 \
  --patch_size 256 \
  --patch_size_choices 192,256 \
  --crop_attempts 12 \
  --min_crop_points 1 \
  --split_loss_variant gt \
  --quadtree_loss_coef 0.1 \
  --split_count_threshold 1 \
  --split_pos_weight 2.0 \
  --bayesian_loss_coef 0.05 \
  --bayesian_sigma 8.0 \
  --bayesian_bg_coef 0.02 \
  --bayesian_loss_gate detach \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --eval_nms_radius 4 \
  --eval_branch_gate query \
  --eval_soft_split_gate pred \
  --seed 42
