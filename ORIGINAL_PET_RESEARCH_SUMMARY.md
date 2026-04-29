# Original PET Research Summary

## Official Repository & Paper

**Repository URL:** https://github.com/cxliu0/PET  
**Repository Owner:** [cxliu0](https://github.com/cxliu0) (Chengxin Liu)

**Paper Title:** Point-Query Quadtree for Crowd Counting, Localization, and More  
**Conference:** ICCV 2023  
**ArXiv:** https://arxiv.org/abs/2308.13814  
**CVF OpenAccess:** https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Point-Query_Quadtree_for_Crowd_Counting_Localization_and_More_ICCV_2023_paper.html  
**Supplementary Materials:** https://drive.google.com/file/d/1WxdtOaEEccYrXuNQTn1k29lFDAetBm63/view?usp=sharing

## Authors

1. **Chengxin Liu** - Huazhong University of Science and Technology (HUST), China
   - Homepage: https://cxliu0.github.io/
   - Email: cx_liu@hust.edu.cn

2. **Hao Lu** - Huazhong University of Science and Technology, China
   - Homepage: https://sites.google.com/site/poppinace/

3. **Zhiguo Cao** - Huazhong University of Science and Technology, China
   - Homepage: http://english.aia.hust.edu.cn/info/1085/1528.htm

4. **Tongliang Liu** - The University of Sydney, Australia
   - Homepage: https://tongliang-liu.github.io/

## Paper Key Concepts

PET formulates crowd counting as a **decomposable point querying process** using a **point-query quadtree** structure where:
- Sparse input points can split into four new points when necessary
- Input and output are both interpretable and steerable
- Applicable to multiple crowd-related tasks by adjusting input format
- Achieves state-of-the-art crowd counting and localization results

## Backbone Architecture

### Original PET Default Backbone

**Primary Backbone:** `vgg16_bn` (VGG16 with Batch Normalization)

- ImageNet pretrained weights: https://download.pytorch.org/models/vgg16_bn-6c64b313.pth
- Configuration: Standard VGG16 architecture with batch normalization applied
- Feature extraction: Multi-scale hierarchical features from convolutional blocks

### VGG Backbone Details

The original implementation supports VGG variants:

| Model | Config | Layers |
|-------|--------|--------|
| vgg11 | A | 8 conv + 3 FC |
| vgg13 | B | 10 conv + 3 FC |
| vgg16 | D | 13 conv + 3 FC |
| vgg19 | E | 16 conv + 3 FC |
| vgg16_bn | D + BatchNorm | 13 conv (batch normalized) + 3 FC |
| vgg19_bn | E + BatchNorm | 16 conv (batch normalized) + 3 FC |

## Modern Backbone Support (Fork Extensions)

The current fork extends PET with modern backbone options via timm library:

### ConvNeXt V2 Variants
- convnextv2_atto
- convnextv2_femto
- convnextv2_pico
- convnextv2_nano
- convnextv2_tiny
- **convnextv2_small**
- **convnextv2_base** ← Currently the best known baseline in fork
- convnextv2_large

**Reference:** https://arxiv.org/abs/2301.00808

### Swin Transformer Variants
- swin_tiny_patch4_window7_224
- swin_small_patch4_window7_224
- swinv2_tiny_window8_256
- swinv2_small_window8_256
- **swinv2_base_window8_256** ← Recommended as second priority
- swinv2_cr_small_ns_256

**Reference:** https://arxiv.org/abs/2111.09883

### MaxViT Variants
- maxvit_nano_rw_256
- **maxvit_rmlp_tiny_rw_256** ← **Strongest next upgrade candidate** (auto_maxvit)
- maxvit_tiny_pm_256
- maxvit_tiny_tf_224
- maxvit_small_tf_224

**Reference:** https://arxiv.org/abs/2204.01697

## Recommended Backbone Tuning Strategies

### Original Authors' Recommendations (Implicit)
1. **Use ImageNet pretrained weights** - Transfer learning from ImageNet is critical
2. **VGG16-BN is reliable** - Provides stable baseline performance
3. **Multi-scale feature extraction** - PET requires hierarchical features at multiple scales (4x and 8x stride)
4. **Feature fusion** - Backbone features must be compatible with FPN-style feature pyramid adapter

### Fork Research Findings

#### Backbone Selection Criteria for Crowd Counting

The key question is **NOT "best classifier backbone on paper"** but rather:
- **Best backbone that works with DYNAMIC image sizes**
- **Hierarchical multi-scale feature extraction**
- **Compatibility with FPN-style feature fusion**
- **Robust non-square input handling**

#### Tested & Verified Locally (timm 1.0.26)

| Backbone | Dynamic Non-Square | Fixed 256x256 | Status | Recommendation |
|----------|-------------------|---------------|--------|-----------------|
| convnextv2_base | ✅ | ✅ | Baseline | Current standard |
| maxvit_rmlp_tiny_rw_256 | ✅ | ✅ | **Best upgrade** | **Try first** |
| swinv2_base_window8_256 | ✅ | ✅ | Compatible | **Try second** |
| maxvit_small_tf_224 | ❌* | ✅ | Needs geometry pass | Requires input-shape work |
| hiera_small_abswin_256 | ❌ | ✅ | Incompatible | Not recommended |

*maxvit_small_tf_224 requires window divisibility guarantees not in current batching path

#### Why MaxViT is the Strongest Next Candidate

1. **Architecture Design** - Mixes blocked local attention + dilated global attention
2. **Hierarchical Features** - Multi-scale hierarchical design matches PET better than plain ViT
3. **Global Context** - Provides global receptive field without modifying PET encoder
4. **Empirical Validation** - Passed local dynamic-size feature extraction checks
5. **Pretrained Weights** - maxvit_rmlp_tiny_rw_256 has current pretrained weights available

#### Why NOT Other Options (Currently)

- **Shifted-Window Encoder**: Ablations show hard regression; cross-window shifting not suitable
- **Hiera**: Strong paper but timm features_only path not robust for PET's dynamic non-square inference
- **ConvNeXtV2-only**: Good baseline but not proof no better exists; MaxViT is cleaner next step

### Training Configuration Recommendations

#### Scheduler Choices
1. **Primary (Baseline):** `warmup_cosine`
   - Standard for pretrained hierarchical backbones with AdamW
   - Works well with ConvNeXt V2 and Swin training recipes
   
2. **Alternative:** `warmup_poly`
   - Better for dense prediction problems
   - Polynomial decay more gradual late in training
   - Keeps learning rate more useful when pushed close to zero

#### Learning Rates
- **Backbone LR:** Default `1e-5` (from models/backbones)
- **Model LR:** Dataset-specific (auto-tuned when using backbone auto-selector)
- **Backbone learning rate warmup:** Recommended for transfer learning scenarios

#### Batch Size & Accumulation
- Auto-tuned based on selected backbone via `get_timm_training_defaults()`
- Recommended: Use `--backbone="auto"` for automatic hyperparameter tuning
- Accumulation steps scale with batch size for gradient accumulation

## Experimental Results on Benchmark Datasets

### Official Pretrained Models & Performance

| Dataset | Model Weights | Training Log | MAE (Lower = Better) |
|---------|--------------|--------------|----------------------|
| **ShanghaiTech PartA** | [SHA_model.pth](https://drive.google.com/file/d/1QwV8hrEDs1LQ4h1TH4KSL8tB51AImNMT/view?usp=drive_link) | [SHA_log.txt](https://drive.google.com/file/d/1UpY61L0KWRA9c29CM9FMX34bHyprnPUY/view?usp=sharing) | **49.08** |
| **ShanghaiTech PartB** | [SHB_model.pth](https://drive.google.com/file/d/10HK42xC6fmOK-5lQfu-pTn6oAHYeRUhv/view?usp=sharing) | [SHB_log.txt](https://drive.google.com/file/d/1M74PI0XuJtQraPOUiCQJSCUjrWoJUq3n/view?usp=sharing) | **6.18** |
| UCF-QNRF | [UCF_QNRF.pth](https://drive.google.com/file/d/129l__gW51UtTQnPKM-90lTZo508-Eh7I/view?usp=sharing) | - | - |
| JHU-Crowd++ | [JHU_Crowd.pth](https://drive.google.com/file/d/1D4vtoYhQuvj_5onJaXJRtWrlwrl2ckbE/view?usp=sharing) | - | - |
| NWPU-Crowd | [NWPU_Crowd.pth](https://drive.google.com/file/d/1MX7tQAexyc9slrt7TaNSK7j8RtSvnI2H/view?usp=sharing) | - | - |

### Development Environment

- Python: 3.8
- PyTorch: 1.12.1
- TorchVision: 0.13.1

## Reference Papers for Modern Backbones

1. **ConvNeXt V2:** https://arxiv.org/abs/2301.00808  
   *A ConvNet for the 2020s*

2. **Swin Transformer V2:** https://arxiv.org/abs/2111.09883  
   *Swin Transformer V2: Scaling Up Capacity and Resolution*

3. **MaxViT:** https://arxiv.org/abs/2204.01697  
   *Maxvit: Multi-Axis Vision Transformers*

4. **Hiera:** https://arxiv.org/abs/2306.00989  
   *Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles*

## Related Work Acknowledged

- **DETR:** https://github.com/facebookresearch/detr - Transformer-based detection
- **P2PNet:** https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet - Point-based crowd counting

## Key Hyperparameter Query Enhancements

The fork provides optional enhanced point query features via `--enhanced_point_query`:

- **Local Context:** Depthwise + pointwise context block around each query point
- **Coordinate Priors:** Sine-coordinate priors derived from query locations
- **Feature Fusion:** Combines query content, backbone features, local context, and coordinate priors
- **Branch-Specific Biases:** Separate learned biases for sparse and dense branches
- **Query Normalization:** Normalizes fused query tensors before decoding

Note: Ablations show this path is not the source of regression issues, making it worth testing for potential improvements.

## Important Geometry Notes for Dynamic Inputs

- PET does **not run on fixed-size square crops** at inference
- Different backbones handle dynamic/non-square inputs differently
- Feature extraction path must support arbitrary image dimensions
- Window-based attention mechanisms (Swin, MaxViT) must handle divisibility gracefully
- **Padding/geometry handling is critical** for backbone selection success

## Citation

If using PET, please cite the original work:

```bibtex
@InProceedings{liu2023pet,
  title={Point-Query Quadtree for Crowd Counting, Localization, and More},
  author={Liu, Chengxin and Lu, Hao and Cao, Zhiguo and Liu, Tongliang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

## Contact

For questions about the original implementation:  
**Chengxin Liu** (cx_liu@hust.edu.cn)
