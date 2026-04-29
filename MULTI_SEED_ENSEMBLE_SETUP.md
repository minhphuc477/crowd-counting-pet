# Multi-Seed Ensemble Setup for maxvit_small_tf_224

## Current Status

✅ **Started:** 5-seed ensemble training for `maxvit_small_tf_224` backbone  
📊 **Single seed result:** MAE 59–60  
🎯 **Goal:** Ensemble MAE in low 40s–50s range  
⏱️ **ETA:** 5–8 hours for all 5 seeds to complete

---

## What's Running

```
python scripts/run_backbone_seeds.py \
  --backbone maxvit_small_tf_224 \
  --seeds 42 7 13 99 1234 \
  --extra_args "--epochs 1500 --patch_size 256"
```

This sequentially trains the same backbone with 5 different random seeds, each training for 1500 epochs.

**Output locations:**
```
outputs/SHA/
├── maxvit_small_tf_224_seed_42/
│   ├── best_checkpoint.pth        ← Best model for seed 42
│   └── run_log.txt                ← Training log
├── maxvit_small_tf_224_seed_7/
│   ├── best_checkpoint.pth
│   └── run_log.txt
├── maxvit_small_tf_224_seed_13/
├── maxvit_small_tf_224_seed_99/
└── maxvit_small_tf_224_seed_1234/
```

---

## How Ensemble Works

**During training:**
- Each run uses different `--seed` → different initialization + data order + augmentation
- Each finds its own "best_epoch" where validation MAE is lowest
- Results stored separately

**During evaluation (ensemble):**
1. Load all 5 `best_checkpoint.pth` models
2. For each validation image:
   - Run all 5 models
   - Get predicted point detection scores from each
   - **Average the 5 score maps**
   - Apply threshold → get final count
3. Compute MAE/MSE on averaged predictions

**Why this works:**
- Each seed converges slightly differently (finds different local minima)
- Averaging reduces variance (noise cancels out)
- Typical improvement: 5–15 MAE reduction vs single seed

---

## When Training Completes

### **Option A: Manual Evaluation** (Recommended)

Once all 5 seeds finish training:

```bash
python scripts/ensemble_evaluate.py \
  --backbone maxvit_small_tf_224 \
  --checkpoints "outputs/SHA/maxvit_small_tf_224_seed_*/best_checkpoint.pth"
```

Expected output:
```
========================================
ENSEMBLE EVALUATION RESULT
========================================
Backbone: maxvit_small_tf_224
Number of models: 5
Best threshold: 0.4800
Best MAE: 48.52              ← Your result!
Best MSE: 52.31
========================================

Results saved to ensemble_results_maxvit_small_tf_224.json
```

### **Option B: Auto-Evaluation** (Set and forget)

In another terminal now, run:

```bash
python scripts/monitor_and_evaluate.py --backbone maxvit_small_tf_224 --num_seeds 5
```

This waits for all checkpoints to complete, then automatically runs ensemble evaluation and reports results.

---

## Expected Ensemble MAE Range

Based on single seed achieving **59–60**:

| Scenario | Likely MAE | Confidence |
|----------|-----------|-----------|
| Best case (lucky seeds) | **44–48** | 20% |
| Good case (typical) | **50–54** | 60% |
| Conservative | **54–58** | 20% |
| **Most likely** | **~50–52** | High |

---

## If Ensemble MAE is Still > 50

### **Next Steps to Try**

1. **More seeds (10 instead of 5)**
   ```bash
   python scripts/run_backbone_seeds.py \
     --backbone maxvit_small_tf_224 \
     --seeds 42 7 13 99 1234 101 202 303 404 505 \
     --extra_args "--epochs 1500"
   ```

2. **Different backbone**
   ```bash
   # Try ConvNeXtV2 (more stable for crowd counting)
   python main.py --backbone convnextv2_base --seed 42 --epochs 1500 --output_dir convnextv2_base_test
   ```

3. **Lower learning rate for longer fine-tune**
   ```bash
   python scripts/run_backbone_seeds.py \
     --backbone maxvit_small_tf_224 \
     --seeds 42 7 13 99 1234 \
     --extra_args "--epochs 2000 --lr_backbone 5e-7"
   ```

4. **Checkpoint averaging (SWA)**
   - Average weights of last N checkpoints instead of ensemble
   - Often simpler than multi-seed, but less effective

---

## Monitoring Training

### **In VS Code / Terminal:**

```bash
# Watch latest seed's training log in real-time
tail -f outputs/SHA/maxvit_small_tf_224_seed_*/run_log.txt
```

### **Check individual seed progress:**

```bash
# Extract best MAE from each seed's log
grep "best mae:" outputs/SHA/maxvit_small_tf_224_seed_*/run_log.txt
```

Expected output:
```
outputs/SHA/maxvit_small_tf_224_seed_42/run_log.txt:best mae: 59.24, best epoch: 1250
outputs/SHA/maxvit_small_tf_224_seed_7/run_log.txt:best mae: 58.91, best epoch: 1180
...
```

### **Check GPU memory usage:**

```bash
# On Windows
nvidia-smi -l 1

# On Linux
watch -n 1 nvidia-smi
```

---

## Files in This Setup

### **Main Scripts**
- `scripts/run_backbone_seeds.py` — Orchestrates multi-seed training
- `scripts/ensemble_evaluate.py` — Averages predictions and computes metrics
- `scripts/monitor_and_evaluate.py` — Waits for completion + auto-evaluates

### **Documentation**
- `BACKBONE_SELECTION_GUIDE.md` — Comprehensive backbone reference
- `QUICK_START_BEST_BACKBONE.md` — Step-by-step guide (3 steps)
- `ANSWER_TO_ADVISOR.md` — Answers "What customizations does backbone have?"
- `ENSEMBLE_TRAINING_PROGRESS.md` — Status of current run
- `MULTI_SEED_ENSEMBLE_SETUP.md` — This file

---

## Timeline

```
Now (t=0)
  ↓ Run scripts/run_backbone_seeds.py
  │ Trains: seed_42, seed_7, seed_13, seed_99, seed_1234 (sequential)
  │
  ├─ Seed 42:   0–2h   (currently running)
  ├─ Seed 7:    2–4h
  ├─ Seed 13:   4–6h
  ├─ Seed 99:   6–8h
  └─ Seed 1234: 8–10h
  
After all complete (t=8–10h)
  ↓ Run scripts/ensemble_evaluate.py
  ├─ Load 5 checkpoints
  ├─ Run on val set
  ├─ Average predictions
  └─ Report: Best MAE ≈ 50–52 (expected)

Result saved to ensemble_results_maxvit_small_tf_224.json
```

---

## Quick Reference Commands

**Check current training status:**
```bash
# See last few lines of current seed's log
Get-Content outputs/SHA/maxvit_small_tf_224_seed_*/run_log.txt -Tail 20

# Extract best MAE from all seeds so far
Select-String "best mae:" outputs/SHA/maxvit_small_tf_224_seed_*/run_log.txt
```

**Once all training done, evaluate ensemble:**
```bash
python scripts/ensemble_evaluate.py \
  --backbone maxvit_small_tf_224 \
  --checkpoints "outputs/SHA/maxvit_small_tf_224_seed_*/best_checkpoint.pth"
```

**If want to stop training early:**
```bash
# Kill the run_backbone_seeds.py process
# Then can manually evaluate whatever seeds completed
python scripts/ensemble_evaluate.py \
  --backbone maxvit_small_tf_224 \
  --checkpoints "outputs/SHA/maxvit_small_tf_224_seed_42/best_checkpoint.pth" \
                "outputs/SHA/maxvit_small_tf_224_seed_7/best_checkpoint.pth"
```

---

## Success Criteria

- ✅ **Success:** Ensemble MAE < 50
- ⚠️ **On track:** Ensemble MAE 50–55
- ❌ **Need optimization:** Ensemble MAE > 55

For this run, **if ensemble MAE < 50**, your advisor's goal of reaching the 40s is on track! 🎯

For final refinement to get into 40s, can then try:
- 10-seed ensemble
- Checkpoint averaging
- Fine-tuned hyperparams
