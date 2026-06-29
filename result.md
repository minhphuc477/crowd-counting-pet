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


python scripts/sweep_eval_thresholds.py \
  --resume outputs/SHB/vgg16_bn_apglc_branch_ifi_stage1_seed42/best_checkpoint.pth \
  --dataset_file SHB \
  --data_path ./data/ShanghaiTech/part_B \
  --backbone vgg16_bn \
  --device cuda \
  --num_workers 2 \
  --output_dir eval_results/SHB/vgg16_bn_apglc_branch_ifi_stage1_seed42_sweep \
  --tta_scales 1.0 \
  --localization_protocol fixed \
  --localization_large_threshold 8 \
  --localization_small_threshold 4 \
  --eval_nms_radii 0 2 4 \
  --eval_branch_gates none \
  --eval_soft_split_gates none \
  --eval_count_modes threshold \
  --eval_count_sources pet \
  --eval_count_blend_alphas 0.5 \
  --score_thresholds 0.42 0.45 0.48 0.50 0.52 0.54 0.56 0.58 0.60 \
  --split_thresholds 0.45 0.47 0.50 \
  --query_prune_thresholds 0.5

  python scripts/sweep_eval_thresholds.py \
  --resume outputs/SHB/vgg16_bn_apglc_branch_ifi_counthead_stage2_seed42/best_checkpoint.pth \
  --dataset_file SHB \
  --data_path ./data/ShanghaiTech/part_B \
  --backbone vgg16_bn \
  --eval_image_set train_holdout \
  --device cuda \
  --num_workers 2 \
  --output_dir eval_results/SHB/stage2_holdout_sweep `
  --eval_nms_radii 0 `
  --eval_branch_gates none `
  --eval_soft_split_gates none `
  --eval_count_modes threshold `
  --eval_count_sources pet `
  --score_thresholds 0.48 0.50 0.52 0.54 0.56 `
  --split_thresholds 0.45 0.47 0.50 `
  --query_prune_thresholds 0.5