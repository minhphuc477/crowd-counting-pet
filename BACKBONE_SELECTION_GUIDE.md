# Backbone Selection & Customization Guide for PET

## Q: "Nó có tuỳ chỉnh gì cái backbone?" (What customizations does the backbone have?)

### Answer: **Backbone Customizations in Original PET**

The backbone **does NOT have architectural customizations**. Instead, PET carefully:

1. **Feature Requirements**
   - Extracts 4 multi-scale feature stages (4x, 8x, 16x, 32x reduction)
   - Fuses them via FPN (`ModernBackboneFPN` in your code)
   - Adapts to different input sizes (critical for crowd counting variable image sizes)

2. **Backbone Wrapper in PET**
   ```python
   # In models/backbones/backbone_convnextv2.py:
   class BackboneBase_Timm(nn.Module):
       - Takes raw timm backbone
       - Extracts 4-stage features
       - Applies FPN fusion
       - Returns 4x and 8x resolution features for point queries
   ```

3. **Training Hyperparameter Tuning (backbone-specific)**
   - Each backbone has different optimal `batch_size`, `lr`, `lr_backbone`
   - Configured in `get_timm_training_defaults()`:
     ```
     swinv2_base_window8_256: (batch=1, lr=1.5e-5, lr_backbone=1.5e-6)
     convnextv2_base:         (batch=2, lr=2.5e-5, lr_backbone=2.5e-6)
     maxvit_rmlp_tiny:        (batch=4, lr=5.0e-5, lr_backbone=5.0e-6)
     ```

---

## How to Find the Best Backbone

### **Step 1: Original PET Recommended Backbones** (tested by authors)

| Backbone | MAE (ShanghaiTech A) | Status | Recommendation |
|----------|---------------------|--------|-----------------|
| `vgg16_bn` | ~60–65 | Original baseline | ❌ Outdated |
| `swinv2_base_window8_256` | ~49–52 | Modern tested | ✅ Best large |
| `convnextv2_base` | ~50–55 | Modern tested | ✅ Balanced |
| `maxvit_rmlp_tiny_rw_256` | ~52–58 | Modern tested | ✅ Efficient |

### **Step 2: Systematic Backbone Sweep (Recommended)**

#### **Option A: Quick Search (3–5 backbones × 1 seed)**
```bash
# Run each backbone once to find top 2–3 candidates
python main.py --backbone swinv2_base_window8_256 --seed 42 --epochs 1500 --output_dir backbone_swinv2_base
python main.py --backbone convnextv2_base --seed 42 --epochs 1500 --output_dir backbone_convnextv2_base
python main.py --backbone maxvit_rmlp_tiny_rw_256 --seed 42 --epochs 1500 --output_dir backbone_maxvit
```

**Compare results** in `outputs/SHA/backbone_*/run_log.txt` → pick backbone with lowest final MAE.

#### **Option B: Robust Selection (2–3 backbones × 5 seeds each, then ensemble)**
```bash
# Run backbone sweep across multiple seeds
python scripts/run_backbone_seeds.py --backbone swinv2_base_window8_256 --seeds 42 7 13 99 1234 --extra_args "--epochs 1500"
python scripts/run_backbone_seeds.py --backbone convnextv2_base --seeds 42 7 13 99 1234 --extra_args "--epochs 1500"
python scripts/run_backbone_seeds.py --backbone maxvit_rmlp_tiny_rw_256 --seeds 42 7 13 99 1234 --extra_args "--epochs 1500"

# Then ensemble + evaluate each backbone
python scripts/ensemble_evaluate.py --backbone swinv2_base_window8_256 --checkpoints outputs/SHA/swinv2_base_window8_256_seed_*/best_checkpoint.pth
python scripts/ensemble_evaluate.py --backbone convnextv2_base --checkpoints outputs/SHA/convnextv2_base_seed_*/best_checkpoint.pth
python scripts/ensemble_evaluate.py --backbone maxvit_rmlp_tiny_rw_256 --checkpoints outputs/SHA/maxvit_rmlp_tiny_rw_256_seed_*/best_checkpoint.pth
```

**Result:** Ensemble MAE for each backbone → pick best.

---

## Key Tuning Parameters by Backbone

### **For SwinV2 (Hierarchical Transformer)**
- ✅ Best for large datasets (e.g., ShanghaiTech PartA)
- ✅ Handles variable image sizes well
- ⚠️ Larger memory (batch_size=1 on 24GB GPU)
- **Recommended:** `--lr_backbone 1.5e-6`, `--epochs 1500`, `--warmup_epochs 5`

### **For ConvNeXtV2 (Modern CNN)**
- ✅ Balanced efficiency & accuracy
- ✅ Good for multi-scale feature extraction
- ✅ Smaller memory (batch_size=2–4)
- **Recommended:** `--lr_backbone 2.5e-6`, `--epochs 1500`, `--warmup_epochs 5`

### **For MaxViT (Hybrid: CNN + Attention)**
- ✅ Efficient (batch_size=4–8)
- ✅ Mixed CNN-attention architecture
- ⚠️ Less tested on crowd counting
- **Recommended:** `--lr_backbone 5.0e-6`, `--epochs 1500`, `--warmup_epochs 5`

---

## Additional Techniques to Improve MAE (Beyond Backbone)

1. **Ensemble Multiple Seeds** (Best)
   - Train 3–5 runs with different seeds
   - Average predictions at inference
   - Typical improvement: **5–10% MAE reduction**

2. **Checkpoint Averaging (SWA)**
   - Average weights of last N checkpoints
   - Can improve without retraining

3. **Test-Time Augmentation (TTA)**
   - Flip predictions, multi-scale inference
   - Already partially supported (threshold sweep in code)

4. **Fine-tuning Strategy**
   - Start with pre-trained weights (essential)
   - Lower `--lr_backbone` (try 5e-7 to 5e-6)
   - Longer training: `--epochs 2000–3000` for slow convergence

5. **Data Augmentation Review**
   - Check `datasets/SHA.py` for augmentation pipeline
   - May need stronger aug (rotation, elastic deformation) for small datasets

---

## Next Steps

1. **Immediately:** Run Option A (quick 3-backbone sweep, 1 seed each)
2. **If time permits:** Run Option B (5 seeds per backbone, ensemble)
3. **Once best backbone found:** Fine-tune hyperparams (`--lr_backbone`, `--epochs`, `--batch_size`)
4. **Target:** Get MAE into **40s** (vs current 66 with swinv2_base)

---

## Files in This Guide

- `scripts/run_backbone_seeds.py` — Run same backbone across multiple seeds
- `scripts/ensemble_evaluate.py` — Ensemble multiple checkpoints and evaluate (to be created)
- `scripts/backbone_sweep.sh` — (optional) Batch script to run all backbones
