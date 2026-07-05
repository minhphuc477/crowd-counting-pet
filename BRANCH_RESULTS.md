# test-model

This branch keeps only the runtime code paths needed for the test/result set below.

## Included Model Recipes

- `vgg_pet_branch_ifi`
- `vgg_pet_paper`
- `vgg_apglc`

## Included Datasets

- `SHA`
- `SHB`
- `QNRF`

## Recorded Results

### SHB: `outputs/SHB/vgg_pet_branch_ifi_seed42_20260629_041929/final_results.json`

- epoch: 1095
- test_mae: 7.094936708860759
- test_mse: 11.517185837705892
- best_epoch: 1095
- best_test_mae: 7.094936708860759
- best_test_mse: 11.517185837705892
- lowest_mse: 11.015093212543082
- best_mse_epoch: 1305
- best_loc_f1_large: 0.8167742777016892
- best_loc_f1_small: 0.6171003717472119
- eval_model: auto

### QNRF: `outputs/QNRF/vgg16_bn_qnrf_paper/best_eval_results.json`

- epoch: 545
- test_mae: 94.56586826347305
- test_mse: 155.47290231192514
- pred_cnt: 693.2425149700599
- gt_cnt: 718.7724550898204
- best_epoch: 545
- best_test_mae: 94.56586826347305
- best_test_mse: 155.47290231192514
- eval_model: raw

### SHA: `outputs/SHA/vgg16_bn_drop700_apg_lc_seed42/best_eval_results.json`

- epoch: 585
- test_mae: 50.934065934065934
- test_mse: 79.4391051596846
- pred_cnt: 434.2637362637363
- gt_cnt: 433.3076923076923
- best_epoch: 585
- best_test_mae: 50.934065934065934
- best_test_mse: 79.4391051596846
- eval_model: raw

### SHA: `vgg16_bn_drop700_apg_seed42`

- test_mae: 50.92
