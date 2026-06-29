**bdd13944dc442535901c6d0fcca6a9fb4c2825c8 commit result:**

Best: mae=71.3516 mse=103.1025 score_threshold=0.5 split_threshold=0.45 eval_nms_radius=0.0 eval_branch_gate=none eval_soft_split_gate=none eval_score_calibration=count_head_bias
Results saved to: eval_results/SHA/vgg16_bn_scratch_apglc_countcal_sweep

**e8203857468720a3e6502042d224e94eb5f410d7 commit result:**

```
vgg16_bn_scratch_apglc_countcal_densecurr_apgcal_seed42 
```

epoch: 885 mae: 64.08241758241758 mse: 95.60969765517662 pred_cnt: 392.4120879120879 gt_cnt: 433.3076923076923

best mae: 59.895604395604394 best epoch: 755

**f161ca132e4a04bdc228bf79eaa096c42f1c6e76 commit result:**

```
vgg16_bn_apglc_lowloss_count_feedback_seed42 
```

Best: mae=68.0879 mse=104.8842 score_threshold=0.52 split_threshold=0.47 eval_nms_radius=0.0 eval_branch_gate=none eval_soft_split_gate=pred eval_score_calibration=none
Results saved to: eval_results/SHA/vgg16_bn_apglc_lowloss_count_feedback_seed42_sweep

**60c9065d5c3b641b7611179f0c373ab067bc658a commit result**

```
vgg16_bn_apglc_apgccneg_late_countreg_seed42 
```

Best: mae=50.9121 mse=82.1399 score_threshold=0.61 split_threshold=0.5 eval_nms_radius=0.0 eval_branch_gate=none eval_soft_split_gate=none eval_score_calibration=none

cd ~/crowd-counting-pet
conda activate crowd_counting


conda activate crowd_counting
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


CUDA_VISIBLE_DEVICES=0 python main.py \
  --resume outputs/NWPU/vgg16_bn_apglc_unified_ifi_stage1_seed42/checkpoint.pth \
  --output_dir outputs/NWPU/vgg16_bn_apglc_unified_ifi_stage1_seed42 \
  --device cuda --num_workers 2 --batch_size 16 --amp




RUN="outputs/SHB/vgg_pet_branch_ifi_seed42_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=0 python main.py \
  --model_recipe vgg_pet_branch_ifi \
  --dataset_file SHB \
  --data_path data/ShanghaiTech/part_B \
  --output_dir "$RUN" \
  --device cuda \
  --num_workers 2 \
  --batch_size 16 \
  --epochs 1500 \
  --lr 0.0001 \
  --lr_backbone 0.00001 \
  --lr_scheduler step \
  --lr_drop -1 \
  --patch_size 256 \
  --pet_loss_variant paper \
  --split_loss_variant paper \
  --query_feature_interpolation implicit \
  --query_ifi_sharing independent \
  --ifi_interpolation implicit \
  --ifi_feature_source branch \
  --ifi_loss_coef 0.02 \
  --ifi_head_source routed \
  --ifi_point_coef 0.2 \
  --ifi_pos_k 2 \
  --ifi_pos_radius 2 \
  --ifi_random_sampling \
  --ifi_neg_k 2 \
  --ifi_neg_radius 8 \
  --ifi_neg_min_dist 2 \
  --ifi_start_epoch 0 \
  --ifi_end_epoch -1 \
  --apg_loss_coef 0 \
  --count_head_loss_coef 0 \
  --density_map_loss_coef 0 \
  --branch_target_routing none \
  --score_threshold 0.5 \
  --split_threshold 0.5 \
  --query_prune_threshold 0.5 \
  --eval_freq 5 \
  --eval_start_epoch 0 \
  --seed 42