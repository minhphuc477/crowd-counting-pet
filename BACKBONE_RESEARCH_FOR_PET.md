# Backbone Research For PET

This note evaluates backbone candidates for this PET fork using two filters:

1. Expected benefit for crowd counting and localization
2. Integration cost into PET's current backbone contract (`4x` and `8x` feature maps, PyTorch-first training loop, minimal custom CUDA burden)

Where integration difficulty is discussed, that is an engineering inference from the official repos plus this repo's current FPN-style backbone adapter pattern.

## Short answer

- Best near-term backbone already in this repo: `ConvNeXt V2`
- Best additional backbone to add next: `Hiera`
- Best speed-focused additions: `FastViT` or `EfficientViT`
- Best research-only candidates, not first integration targets: `InternImage`, `VMamba`

## Candidate table

| Model | Architecture | Why it is interesting for PET | Expected impact vs VGG16 | PET integration risk | Recommendation | Paper | GitHub |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ConvNeXt V2 | Pure CNN | FCMAE pretraining and GRN give stronger semantic features without leaving the CNN regime. PET already has a working adapter for it. | Strong upgrade in representation quality and stability over VGG16. | Low | Keep and continue tuning. This is still the best default non-VGG path here. | [arXiv:2301.00808](https://arxiv.org/abs/2301.00808) | [facebookresearch/ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2) |
| Hiera | Hierarchical ViT | Hiera is explicitly hierarchical, MAE-pretrained, and designed to be simpler and faster than older hierarchical ViTs. That matches PET's multi-scale feature needs well. | Likely stronger global context than VGG16 and a realistic upgrade candidate for dense scenes. | Medium | Best new backbone to add after ConvNeXt V2. | [arXiv:2306.00989](https://arxiv.org/abs/2306.00989) | [facebookresearch/hiera](https://github.com/facebookresearch/hiera) |
| FastViT | Hybrid CNN + ViT | Very strong latency/accuracy trade-off. The official repo also keeps unfused checkpoints for downstream detection/segmentation training, which matters for PET fine-tuning. | Better deployment efficiency than VGG16 and probably better features too. | Medium | Add if inference latency is a top priority. | [arXiv:2303.14189](https://arxiv.org/abs/2303.14189) | [apple/ml-fastvit](https://github.com/apple/ml-fastvit) |
| EfficientViT | Multi-scale linear-attention vision model | Built for high-resolution dense prediction and emphasizes global receptive field plus multi-scale learning with lightweight operations. That is conceptually a strong match for PET. | Clear speed and scale-handling upside over VGG16. | Medium | Good extra candidate to add after Hiera if speed matters. | [arXiv:2205.14756](https://arxiv.org/abs/2205.14756) | [mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit) |
| InternImage | CNN with DCNv3 | Dynamic receptive fields and adaptive spatial aggregation are attractive for scale-aware crowd scenes. | Potentially very strong accuracy gain over VGG16. | High | Do not add first. Revisit only if you accept custom ops and a heavier dependency stack. | [arXiv:2211.05778](https://arxiv.org/abs/2211.05778) | [OpenGVLab/InternImage](https://github.com/OpenGVLab/InternImage) |
| VMamba | Visual state-space model | Linear-time global context and strong scaling behavior are appealing for large dense images. | Research upside is real, but engineering friction is high. | High | Not a good fit for this repo today. | [arXiv:2401.10166](https://arxiv.org/abs/2401.10166) | [MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba) |

## Why the recommendations differ

### 1. ConvNeXt V2 stays the safest strong choice

Official sources state that ConvNeXt V2 combines a fully convolutional masked autoencoder framework with GRN and improves pure ConvNet performance across classification, detection, and segmentation. In this repo, it already works through a simple `features_only` plus FPN adapter, so it has the best accuracy-to-risk ratio.

### 2. Hiera is the cleanest new addition

Hiera is a hierarchical vision transformer with MAE-pretrained checkpoints and straightforward loading through Torch Hub. For PET, that matters because PET already consumes hierarchical feature maps and benefits from pretraining that preserves spatial structure. Among the backbones not yet in this repo, Hiera looks like the cleanest next integration target.

### 3. FastViT and EfficientViT are the best speed-oriented additions

FastViT's official paper emphasizes latency-accuracy trade-offs and reports it as faster than ConvNeXt on mobile hardware at similar accuracy. EfficientViT is even more directly dense-prediction-oriented, with official claims around multi-scale linear attention and major latency reductions on high-resolution workloads. If deployment speed on VPS or edge hardware matters more than absolute research novelty, these are stronger first additions than VMamba or InternImage.

### 4. InternImage is powerful but expensive to integrate

InternImage is attractive because DCNv3 gives adaptive receptive fields that should help scale-aware counting. The problem is engineering cost: the official repo depends on DCNv3 custom operators and a larger downstream stack. That is a much bigger lift than the current ConvNeXt adapter pattern.

### 5. VMamba is not a good first add for this repo

The official VMamba repo recommends PyTorch `>= 2.0`, CUDA `>= 11.8`, and installation of the selective-scan kernel. That clashes with the original PET environment that this fork still tries to remain compatible with. VMamba is interesting research, but it is a bad first integration target for this specific codebase.

## Extra models worth adding to the short list

These were not in the original user list, but they are better PET candidates than some harder-to-integrate options:

- `Hiera`
  - Reason: hierarchical, MAE-pretrained, simple loading path, good match to PET's multi-scale interface.
- `EfficientViT`
  - Reason: designed for high-resolution dense prediction and speed-sensitive deployment.

## Recommended add order

1. Keep `ConvNeXt V2` as the main experimental path and tune it against the restored pre-`d6b9e71` auto recipe.
2. Add `Hiera` next if the goal is a stronger general-purpose backbone.
3. Add `FastViT` or `EfficientViT` if deployment latency matters more than maximum research headroom.
4. Delay `InternImage` and `VMamba` until the repo is willing to depend on custom CUDA operators and newer runtime requirements.

## Source notes

Primary-source facts above come from the official papers and official repos:

- ConvNeXt V2 paper and repo
- Hiera paper and repo
- FastViT paper, Apple research page, and repo
- EfficientViT paper and repo
- InternImage paper and repo
- VMamba paper and repo
