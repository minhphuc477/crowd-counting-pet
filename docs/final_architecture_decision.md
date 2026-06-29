# Final Architecture Decision: RMI-PET

## Scope

Select one PET-derived research candidate that:

- trains from scratch with an ImageNet-pretrained backbone;
- uses the same architecture on SHA, SHB, UCF-QNRF, JHU-Crowd++, and
  NWPU-Crowd;
- predicts points directly for counting and localization;
- does not require a density teacher, count-head inference, test-set threshold
  selection, or a hidden fine-tuning stage.

No published point model, including APGCC, improves every PET MAE/MSE pair.
No architecture can therefore be claimed to improve every dataset before the
fixed cross-dataset experiment is complete. RMI-PET is the final candidate,
not a pre-declared result. `vgg_pet_paper` remains the production reference.

## Decision

Use `--model_recipe vgg_pet_rmi --allow_experimental_model_recipe` only for
the declared cross-dataset experiment.

RMI-PET keeps PET's native:

- VGG16-BN backbone and 4x/8x feature projections;
- context encoder;
- sparse and dense quadtree branches;
- Hungarian point matching;
- classification and point-offset heads;
- threshold counting and localization output.

It adds one shared multi-scale Implicit Feature Interpolation module:

1. Sample continuous representations from projected 4x and 8x features.
2. Fuse both scales into a branch-sized correction.
3. Add the correction to the native PET feature through a learned scalar
   initialized to `1e-3`.
4. Train arbitrary local positive and negative points through the same routed
   sparse/dense heads used at inference.

The residual start preserves PET's initial query representation. IFI and
Auxiliary Point Guidance are active from epoch zero and remain part of one
scratch-training run. No component is enabled by a hidden epoch switch.

## Why This Design

- PET is the strongest verified counting baseline in this repository, and its
  quadtree is efficient on highly variable crowd density.
- APGCC is the publication with the clearest same-backbone evidence of
  simultaneous counting and localization gains. Its key transferable pieces
  are multi-level IFI and auxiliary positive/negative point supervision.
- APGCC improves PET on SHA, SHB, and JHU-Crowd++, but its UCF-QNRF MAE is
  slightly worse than PET. Residual initialization is therefore a risk-control
  hypothesis, not evidence of a universal gain.
- Replacing PET query features directly with IFI improved localization but
  regressed SHB counting in this repository. Residual IFI isolates that risk.
- A scalar count loss gives no spatial information and previously destabilized
  query confidence. It is excluded.
- Density teachers and density-regression heads change the task and evaluation
  contract. They are excluded from the final point model.
- RCCFormer-style wholesale backbone/attention replacement did not show a
  consistent gain over PET on all target datasets. It is not the universal
  default.

Primary references:

- [PET, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Point-Query_Quadtree_for_Crowd_Counting_Localization_and_More_ICCV_2023_paper.html)
- [APGCC, ECCV 2024](https://arxiv.org/abs/2405.10589)
- [Official APGCC implementation](https://github.com/AaronCIH/APGCC)
- [Generalized Loss / unbalanced OT, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Wan_A_Generalized_Loss_Function_for_Crowd_Counting_and_Localization_CVPR_2021_paper.html)
- [LayerScale, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.html)
- [JHU-Crowd++, 2020](https://arxiv.org/abs/2004.03597)
- [Partial annotations in an image, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_Crowd_Counting_With_Partial_Annotations_in_an_Image_ICCV_2021_paper.html)
- [Point-to-Region loss, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_Point-to-Region_Loss_for_Semi-Supervised_Point-Based_Crowd_Counting_CVPR_2025_paper.html)

## Dataset And Supervision Contract

- UCF-QNRF uses the published 1536-pixel long-side cap.
- JHU-Crowd++ and NWPU-Crowd use the published 2048-pixel cap unless
  explicitly overridden.
- JHU-Crowd++ zero-person distractors are valid samples, not annotation errors.
- JHU head files are parsed as `x y width height occlusion blur`; approximate
  boxes are retained only for localization thresholds unless a declared
  scale-loss ablation is enabled.
- Shanghai partial annotation uses one deterministic rectangle per training
  image. At 10%, the default rectangle is `0.5H x 0.2W`, matching the published
  PAL setup.
- Queries, splitter cells, APG negatives, and IFI samples outside the annotated
  rectangle receive no target. They are not background.
- Global count and density auxiliaries are rejected during partial-region
  training because their full-image targets are unknown.

PET's published 10% result uses a declared two-step protocol: train on partial
regions, infer points around those regions, then retrain on fused real and
pseudo annotations. This repository now implements the correct first-stage
supervision contract. It must not claim reproduction of PET's Table 3 until
the pseudo-label generation and fusion stage is separately implemented and
reported.

## Fixed Evaluation Contract

Use the dataset validation/test split exactly once per checkpoint with:

- `score_threshold=0.5`;
- `split_threshold=0.5`;
- `query_prune_threshold=0.5`;
- no NMS;
- no branch gate;
- no soft split gate;
- PET point count as the count source;
- no score calibration.

Save three independent checkpoints:

- lowest validation MAE;
- lowest validation MSE;
- highest localization score, ranked by large-threshold F1 then
  small-threshold F1.

Report all three checkpoints. Do not select a threshold or checkpoint on the
test split.

## Acceptance Rule

Run SHB first because it is the cheapest strict falsification test. Continue
to SHA, UCF-QNRF, JHU-Crowd++, and NWPU only if RMI-PET is not worse than the
verified PET baseline on both MAE and MSE and improves at least one
localization F1.

The final architecture is accepted only when the same recipe, fixed evaluator,
and predeclared checkpoint rule produce a Pareto improvement on all five
datasets and do not regress the 10% partial-region stage. Otherwise retain
`vgg_pet_paper` as the production architecture and report RMI-PET as a rejected
ablation.
