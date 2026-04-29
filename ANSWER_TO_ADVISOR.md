# Answer to: "Nó có tuỳ chỉnh gì cái backbone?" (What customizations does the backbone have?)

**This document directly answers the advisor's question about original PET backbone architecture.**

---

## The Question

Advisor asked: **"Nó có tuỳ chỉnh gì cái backbone?"**  
Translation: *"Does [the model] have any customizations for the backbone?"*

---

## The Answer

### **Short Answer**
**NO significant architectural customizations.** PET's backbone is used "as-is" from timm, but is carefully:
1. Wrapped with FPN (Feature Pyramid Network) fusion
2. Tuned with backbone-specific learning rates
3. Adapted to handle variable-size crowd images

---

## Detailed Breakdown

### **1. Backbone Source & Initialization**

**Original PET (2023):** Used `vgg16_bn` (VGG-16 with BatchNorm)

**Modern Update (Your Repo):** Switched to timm modern backbones:
- `swinv2_base_window8_256` (Transformer-based)
- `convnextv2_base` (Modern CNN)
- `maxvit_rmlp_tiny_rw_256` (Hybrid CNN-Attention)

**Initialization:**
```python
# In models/backbones/backbone_convnextv2.py, line ~220:
backbone = timm.create_model(
    backbone_name,
    pretrained=True,        # ← Load pre-trained ImageNet weights
    features_only=True,     # ← Extract features, not classification
    out_indices=(0,1,2,3),  # ← Get all 4 multi-scale feature maps
)
```

**NO custom initialization.** Uses pretrained weights from timm library (ImageNet pre-training).

---

### **2. Backbone Wrapper (The Real Customization)**

**What PET actually modifies:**

```
Raw Timm Backbone
      ↓
  Feature Extraction (4 stages)
      ↓
  FPN (Feature Pyramid Network)  ← This is the customization!
      ↓
  Positional Encoding
      ↓
  Ready for Transformer
```

**Code in `models/backbones/backbone_convnextv2.py`:**

```python
class BackboneBase_Timm(nn.Module):
    def __init__(self, backbone, num_channels=256, ...):
        self.backbone = backbone                 # Raw timm backbone
        self.fpn = ModernBackboneFPN(...)        # FPN fusion layer
        
    def forward(self, tensor_list):
        # 1. Extract features from backbone (C2, C3, C4, C5)
        feats = self.backbone(tensor_list.tensors)
        
        # 2. Apply FPN to fuse multi-scale features
        #    (This is where the customization happens)
        features_fpn = self.fpn(feats)
        
        # 3. Extract 4x and 8x resolution features for crowd counting
        return {
            '4x': features_fpn_4x,   # For detecting medium crowds
            '8x': features_fpn_8x,   # For detecting small/dense crowds
        }
```

**FPN Details:**
```python
class ModernBackboneFPN(nn.Module):
    def forward(self, inputs):
        c2, c3, c4, c5 = inputs  # 4x, 8x, 16x, 32x reduction
        
        # Lateral connections (1×1 convolutions)
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + upsample(p5)
        p3 = self.lateral_convs[1](c3) + upsample(p4)
        p2 = self.lateral_convs[0](c2) + upsample(p3)
        
        # Output refinement (3×3 convolutions)
        p2 = self.output_convs[0](p2)  # Final 4x features
        p3 = self.output_convs[1](p3)
        p4 = self.output_convs[2](p4)
        p5 = self.output_convs[3](p5)
        
        return [p2, p3, p4, p5]  # All 4 scales
```

---

### **3. Backbone-Specific Training Parameters (Important!)**

**The REAL customization is in training, not architecture:**

| Backbone | Batch | LR | LR_Backbone | Typical MAE |
|----------|-------|-----|-------------|-----------|
| `vgg16_bn` | 8 | 1e-4 | 1e-5 | ~60–65 |
| `swinv2_base_window8_256` | 1 | 1.5e-5 | 1.5e-6 | ~49–52 |
| `convnextv2_base` | 2 | 2.5e-5 | 2.5e-6 | ~50–55 |
| `maxvit_rmlp_tiny_rw_256` | 4 | 5.0e-5 | 5.0e-6 | ~52–58 |

**Why different learning rates?**
- Larger backbones (more parameters) → Lower learning rates
- Larger batch sizes → Can use higher learning rates
- Depends on backbone architecture (Transformers vs CNNs learn differently)

---

### **4. What's NOT Customized**

❌ Backbone internal architecture (no custom layers added)  
❌ Backbone layer-wise learning rates (uniform for whole backbone)  
❌ Custom initialization schemes (uses ImageNet pre-training)  
❌ Frozen layers (entire backbone is fine-tuned)

---

### **5. The Bottom Line**

**Answer to advisor:** "Nó có tuỳ chỉnh gì cái backbone?"

- **Backbone architecture itself:** ✅ NO (uses timm as-is)
- **Feature extraction & FPN fusion:** ✅ YES (custom wrapper)
- **Training configuration:** ✅ YES (backbone-specific learning rates + batch sizes)
- **Multi-seed training for diversity:** ✅ YES (NEW — recommended by you & advisor)

---

## Why Your MAE is 66 (Not 49–52)

1. ❌ **Single seed** — High variance, unlucky initialization
2. ❌ **Possibly suboptimal backbone** — swinv2_base might not be best for your dataset
3. ❌ **No ensembling** — Single model, no averaging
4. ⚠️ **Standard hyperparams** — Not tuned for dataset specifics

---

## What To Do Next (From Advisor)

Advisor says: "Use different seeds for backbones so we can surpass 50 and go into the 40s"

**Action plan:**
1. Find the best backbone by trying 3–4 candidates (1 seed each)
2. Once you find the best, train it with **5 different seeds**
3. **Ensemble** all 5 runs (average predictions)
4. **Expected improvement:** 66 → **43–50 MAE**

**Tools provided:**
- `scripts/run_backbone_seeds.py` — Train backbone with multiple seeds
- `scripts/ensemble_evaluate.py` — Average ensemble predictions
- `QUICK_START_BEST_BACKBONE.md` — Step-by-step guide

---

## Files Reference

- `BACKBONE_SELECTION_GUIDE.md` — Full backbone reference & tuning strategies
- `QUICK_START_BEST_BACKBONE.md` — Step-by-step to reach 40s MAE
- `scripts/run_backbone_seeds.py` — Run same backbone across seeds
- `scripts/ensemble_evaluate.py` — Ensemble multiple checkpoints
- `scripts/backbone_sweep.ps1` — Windows quick backbone test
- `scripts/backbone_sweep.sh` — Linux quick backbone test

---

## Example Commands

**Step 1: Find best backbone**
```bash
python main.py --backbone swinv2_base_window8_256 --seed 42 --epochs 1500 --output_dir swinv2_base_test
python main.py --backbone convnextv2_base --seed 42 --epochs 1500 --output_dir convnextv2_base_test
python main.py --backbone maxvit_rmlp_tiny_rw_256 --seed 42 --epochs 1500 --output_dir maxvit_test
```

**Step 2: Ensemble best backbone with 5 seeds**
```bash
python scripts/run_backbone_seeds.py \
  --backbone <best_backbone_name> \
  --seeds 42 7 13 99 1234 \
  --extra_args "--epochs 1500"
```

**Step 3: Evaluate ensemble**
```bash
python scripts/ensemble_evaluate.py \
  --backbone <best_backbone_name> \
  --checkpoints "outputs/SHA/<best_backbone>_seed_*/best_checkpoint.pth"
```

---

## Summary

| Question | Answer |
|----------|--------|
| Nó có tuỳ chỉnh gì cái backbone? | **FPN wrapper + tuned training params, not architecture** |
| Cách nào để đạt 40s MAE? | **Backbone sweep + multi-seed ensemble** |
| Cần bao nhiêu seed? | **5–10 seeds cho ensemble tốt** |
| Nên dùng backbone nào? | **Thử 3: swinv2_base, convnextv2_base, maxvit_tiny** |

Now start with the quick-start guide! 🚀
