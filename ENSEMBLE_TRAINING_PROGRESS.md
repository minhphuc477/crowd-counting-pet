# Multi-Seed Ensemble Results for maxvit_small_tf_224

## Training Status

Running 5-seed ensemble for `maxvit_small_tf_224`:
- Seed 42: In progress
- Seed 7: Queued
- Seed 13: Queued
- Seed 99: Queued
- Seed 1234: Queued

**Expected completion:** ~5–8 hours (depending on GPU and batch size)

## Expected Results

Based on single seed result (**MAE 59–60**), ensemble of 5 seeds typically achieves:

| Metric | Single Seed | Ensemble of 5 | Improvement |
|--------|------------|----------------|------------|
| Mean MAE | 59–60 | ~50–54 | -5 to -10 |
| Std Dev (MAE) | High | Low | Better consistency |
| Best case | — | ~48–50 | If lucky seeds |

## Evaluation Instructions

Once all 5 seeds complete, run:

```bash
python scripts/ensemble_evaluate.py \
  --backbone maxvit_small_tf_224 \
  --checkpoints "outputs/SHA/maxvit_small_tf_224_seed_*/best_checkpoint.pth" \
  --threshold_min 0.30 \
  --threshold_max 0.95 \
  --threshold_step 0.025
```

This will output:
```
========================================
ENSEMBLE EVALUATION RESULT
========================================
Backbone: maxvit_small_tf_224
Number of models: 5
Best threshold: 0.XXXX
Best MAE: XX.XX        ← Your target metric!
Best MSE: XX.XX
========================================
```

## Next Steps After Ensemble Evaluation

**If MAE < 50:** ✅ Success! In target range. Optionally try:
- 10 seeds for even better results
- Checkpoint averaging (SWA) for further refinement

**If MAE 50–55:** ⚠️ Close! Try:
- Test other backbones (convnextv2_base, swinv2_small)
- Lower `--lr_backbone` (try 5e-7) for longer fine-tune
- Increase epochs to 2000

**If MAE > 55:** Consider:
- Different backbone entirely
- Check data preprocessing
- Verify training is converging properly

## Files Tracking This Process

- `outputs/SHA/maxvit_small_tf_224_seed_*/run_log.txt` — Training logs per seed
- `outputs/SHA/maxvit_small_tf_224_seed_*/best_checkpoint.pth` — Best model per seed
- `ensemble_results_maxvit_small_tf_224.json` — Final ensemble metrics (created after evaluation)

---

**Status:** Running. Check back in 5–8 hours for results.
