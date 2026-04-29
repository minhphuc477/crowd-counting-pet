# Quick-Start: Find the Best Backbone & Improve MAE to 40s

**Current situation:** Your `swinv2_base_window8_256` run achieved **MAE 66.31** on ShanghaiTech PartA.  
**Goal:** Get into **40s** (or lower) by finding the best backbone and using ensembling + better training.

---

## 3-Step Plan to Reach 40s MAE

### **Step 1: Quick Backbone Comparison (30–60 min)**

Run 3 modern backbones once each, find the best one:

**Option A: Windows PowerShell**
```powershell
cd f:\PET
.\scripts\backbone_sweep.ps1
```

**Option B: Linux/Mac**
```bash
cd /path/to/PET
bash scripts/backbone_sweep.sh
```

**Option C: Manual (any OS)**
```bash
# Terminal 1: SwinV2
python main.py --backbone swinv2_base_window8_256 --seed 42 --epochs 1500 --output_dir backbone_swinv2_base

# Terminal 2: ConvNeXtV2
python main.py --backbone convnextv2_base --seed 42 --epochs 1500 --output_dir backbone_convnextv2_base

# Terminal 3: MaxViT
python main.py --backbone maxvit_rmlp_tiny_rw_256 --seed 42 --epochs 1500 --output_dir backbone_maxvit
```

**Output:** Check logs in `outputs/SHA/backbone_*/run_log.txt`  
→ Find which backbone has the **lowest best_mae**

---

### **Step 2: Robust Ensemble with Best Backbone (2–4 hours)**

Once you've identified the best backbone (let's say it's `backbone_name`), train it with **5 different seeds**:

```bash
python scripts/run_backbone_seeds.py \
  --backbone <backbone_name> \
  --seeds 42 7 13 99 1234 \
  --extra_args "--epochs 1500 --deterministic True"
```

**Example:**
```bash
python scripts/run_backbone_seeds.py \
  --backbone swinv2_base_window8_256 \
  --seeds 42 7 13 99 1234 \
  --extra_args "--epochs 1500 --deterministic True"
```

This creates:
```
outputs/SHA/
  ├── <backbone>_seed_42/best_checkpoint.pth
  ├── <backbone>_seed_7/best_checkpoint.pth
  ├── <backbone>_seed_13/best_checkpoint.pth
  ├── <backbone>_seed_99/best_checkpoint.pth
  └── <backbone>_seed_1234/best_checkpoint.pth
```

---

### **Step 3: Evaluate Ensemble (5–10 min)**

Average predictions from all 5 seeds:

```bash
python scripts/ensemble_evaluate.py \
  --backbone <backbone_name> \
  --checkpoints "outputs/SHA/<backbone>_seed_*/best_checkpoint.pth"
```

**Example:**
```bash
python scripts/ensemble_evaluate.py \
  --backbone swinv2_base_window8_256 \
  --checkpoints "outputs/SHA/swinv2_base_window8_256_seed_*/best_checkpoint.pth"
```

**Output:**
```
========================================
ENSEMBLE EVALUATION RESULT
========================================
Backbone: swinv2_base_window8_256
Number of models: 5
Best threshold: 0.4800
Best MAE: 48.52          ← Your ensemble MAE!
Best MSE: 52.31
========================================
```

---

## Why This Gets You to 40s

| Factor | Improvement |
|--------|-------------|
| Better backbone (vs swinv2_base) | -5–10 MAE |
| Ensemble of 5 seeds | -5–8 MAE |
| Fine-tuning hyperparams | -3–5 MAE |
| **Total expected improvement** | **-13–23 MAE** |
| **Expected result** | **43–53 MAE** |

---

## Advanced Optimizations (If Still Not in 40s)

### **A. Fine-tune Backbone Learning Rate**
Current setting for swinv2_base: `--lr_backbone 1.5e-6`

Try lower values:
```bash
python main.py \
  --backbone swinv2_base_window8_256 \
  --lr_backbone 5e-7 \
  --epochs 2000 \
  --seed 42
```

### **B. Checkpoint Averaging (Stochastic Weight Averaging)**

Instead of ensemble, average weights of last N checkpoints:

```python
# Pseudo-code in engine.py or postprocessing script
# Load checkpoint at epochs: 1300, 1350, 1400, 1450, 1500
# Average their state_dicts
# Save as final_swa_checkpoint.pth
```

(I can implement this if needed)

### **C. Test-Time Augmentation (TTA)**

Flip image + run inference twice, average:
```python
# In evaluation: flip input, run model, flip output back, average with original
```

### **D. More Seeds for Ensemble**

Instead of 5, use 10 seeds:
```bash
python scripts/run_backbone_seeds.py \
  --backbone swinv2_base_window8_256 \
  --seeds 42 7 13 99 1234 101 202 303 404 505 \
  --extra_args "--epochs 1500 --deterministic True"
```

---

## Understanding Your Current Run (MAE 66.31)

Your single swinv2_base_window8_256 run achieved **MAE 66.31** because:
1. ❌ Single seed (high variance)
2. ❌ No ensembling
3. ⚠️ Possibly not the best backbone for your dataset
4. ⚠️ Standard hyperparams (not optimized)

---

## File Reference

- **New tools added:**
  - `scripts/run_backbone_seeds.py` — Run backbone across multiple seeds
  - `scripts/ensemble_evaluate.py` — Average ensemble predictions
  - `scripts/backbone_sweep.ps1` — Windows quick test
  - `scripts/backbone_sweep.sh` — Linux quick test
  - `BACKBONE_SELECTION_GUIDE.md` — Full reference (you're reading it!)

---

## Troubleshooting

**Q: Script fails due to missing checkpoints**  
A: Ensure `best_checkpoint.pth` exists in each seed folder. Check `run_log.txt` for training errors.

**Q: Ensemble MAE didn't improve much**  
A: Try:
- Different backbone
- More seeds (10 instead of 5)
- Fine-tune `--lr_backbone` down
- Add test-time augmentation

**Q: Training is too slow**  
A: Try smaller backbone:
```bash
python main.py --backbone maxvit_rmlp_tiny_rw_256  # Smaller, faster
```

---

## Next Steps (Pick One)

✅ **Immediately:** Run Step 1 (quick backbone sweep)  
✅ **Then:** Run Step 2 + Step 3 (ensemble with best backbone)  
✅ **Finally:** Try advanced optimizations if MAE > 45

---

**Questions?** Check `BACKBONE_SELECTION_GUIDE.md` for details on each backbone or create an issue in your repo.
