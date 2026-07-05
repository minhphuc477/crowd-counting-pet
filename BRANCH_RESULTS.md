# best-model

This branch keeps only the runtime code paths needed for the best result set below.

## Included Model Recipes

- `vgg_apglc`
- `vgg_apglc_density_counthead_ft_legacy`
- `vgg_apglc_branch_ifi`
- `vgg_apglc_branch_ifi_counthead_stage2`
- `vgg_apglc_branch_ifi_nwpu`

## Included Datasets

- `SHA`
- `SHB`
- `NWPU`

## Recorded Results

### SHA: `outputs/SHA/vgg16_bn_apglc_density_counthead_seed42/final_results.json`

- epoch: 32
- test_mae: 48.83516483516483
- test_mse: 76.81532012947291
- best_epoch: 32
- best_test_mae: 48.83516483516483
- best_test_mse: 76.81532012947291
- eval_model: raw

### SHB: `outputs/SHB/vgg16_bn_apglc_branch_ifi_counthead_stage2_seed42/final_results.json`

- epoch: 10
- test_mae: 5.775
- test_mse: 8.501470461043784
- best_epoch: 10
- best_test_mae: 5.775
- best_test_mse: 8.501470461043784
- lowest_mse: 8.035857141587325
- best_mse_epoch: 2
- best_loc_f1_large: 0.7922920140818973
- best_loc_f1_small: 0.5918102649620159
- validation_protocol: train_holdout
- eval_model: auto

### NWPU: `outputs/NWPU/vgg16_bn_apglc_branch_ifi_stage1_b16_seed42/best_eval_results.json`

- epoch: 620
- test_mae: 64.206
- test_mse: 566.4011564253732
- pred_cnt: 361.45
- gt_cnt: 392.476
- best_epoch: 620
- best_test_mae: 64.206
- best_test_mse: 566.4011564253732
- lowest_mse: 565.660406958097
- best_mse_epoch: 340
- best_loc_f1_large: 0.5493003763740999
- best_loc_f1_small: 0.48421312601457644
- validation_protocol: benchmark_test
- eval_model: raw
