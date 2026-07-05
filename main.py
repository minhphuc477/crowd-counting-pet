import argparse
import copy
import datetime
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets import build_dataset
from engine import evaluate, evaluate_crowd_no_overlap, format_localization_metrics, train_one_epoch
from models import build_model
from models.backbones import get_supported_timm_backbones, is_timm_backbone


BASE_TRAINING_DEFAULTS = {
    'lr': 1e-4,
    'lr_backbone': 1e-5,
    'lr_backbone_adapter': -1.0,
    'batch_size': 8,
    'warmup_epochs': 5,
    'freeze_backbone_epochs': 0,
}

BACKBONE_RECIPES = {
    'heavy': {
        'batch_size': 4,
        'lr': 5e-5,
        'lr_backbone': 5e-6,
        'lr_backbone_adapter': 1e-4,
        'warmup_epochs': 10,
        'freeze_backbone_epochs': 5,
    },
    'mobile': {
        'batch_size': 8,
        'lr': 7.5e-5,
        'lr_backbone': 7.5e-6,
        'lr_backbone_adapter': 1e-4,
        'warmup_epochs': 8,
        'freeze_backbone_epochs': 3,
    },
    'vgg': {
        'batch_size': 8,
        'lr': 1e-4,
        'lr_backbone': 1e-5,
        'lr_backbone_adapter': 1e-4,
        'warmup_epochs': 5,
        'freeze_backbone_epochs': 0,
    },
}

MODEL_RECIPES = {
    # Exact fully supervised PET optimization/evaluation contract used by the
    # official ICCV 2023 implementation. Keep this recipe free of every added
    # auxiliary so dataset and evaluator parity can be established before an
    # architectural ablation is accepted.
    'vgg_pet_paper': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'epochs': 1500,
        'lr': 1e-4,
        'lr_backbone': 1e-5,
        'lr_scheduler': 'step',
        'lr_drop': -1,
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'paper',
        'apg_loss_coef': 0.0,
        'ifi_loss_coef': 0.0,
        'count_head_loss_coef': 0.0,
        'density_map_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.0,
        'foreground_loss_coef': 0.0,
        'branch_target_routing': 'none',
        'query_feature_interpolation': 'nearest',
        'eval_foreground_gate': 'none',
        'eval_count_mode': 'threshold',
        'eval_count_source': 'pet',
        'eval_score_calibration': 'none',
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'score_threshold': 0.5,
        'split_threshold': 0.5,
        'split_threshold_quantile': 0.5,
        'query_prune_threshold': 0.5,
        'bad_count_start_epoch': 100,
        'bad_count_direction': 'all',
    },
    # Known stable scratch path from this repo/session:
    # PET + lite FPN + low-weight APG. The saved best checkpoint
    # vgg16_bn_drop700_apg_lc_seed42 records apg_loss_coef=0.02,
    # apg_end_epoch=350, and apg_contrastive_coef=0.15. Keeping APG active
    # for the full 1500 epochs reproduced worse 52-53 MAE runs, so this recipe
    # intentionally matches the verified checkpoint schedule.
    # No scalar count head, no density map, no routed targets, and no
    # foreground gate.
    'vgg_apglc': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'auto',
        'apg_loss_coef': 0.02,
        'apg_start_epoch': 0,
        'apg_warmup_epochs': 0,
        'apg_end_epoch': 350,
        'apg_sparse_coef': 1.0,
        'apg_dense_coef': 1.0,
        'apg_dense_start_epoch': 0,
        'apg_dense_warmup_epochs': 0,
        'apg_pos_k': 1,
        'apg_point_coef': 5.0,
        'apg_bg_coef': 0.0,
        'apg_contrastive_coef': 0.15,
        'apg_neg_k': 4,
        'apg_margin': 1.0,
        'apg_count_calibration': 'none',
        'count_head_loss_coef': 0.0,
        'density_map_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.0,
        'foreground_loss_coef': 0.0,
        'eval_foreground_gate': 'none',
        'eval_count_mode': 'threshold',
        'eval_score_calibration': 'none',
        'eval_dense_start_epoch': 0,
        'eval_dense_residual_mode': 'none',
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'score_threshold': 0.55,
        'split_threshold': 0.45,
        'split_threshold_quantile': 0.5,
        # Decoder window pruning is an architectural PET constant. Keep it
        # independent from the sweepable branch-routing threshold above.
        'query_prune_threshold': 0.5,
        'bad_count_start_epoch': 100,
        'bad_count_direction': 'all',
    },
    # Bidirectional Scale-Fusion PET (BSF-PET).
    #
    # This is a single-variable architecture ablation over the verified
    # vgg_apglc scratch recipe. A zero-initialized residual block exchanges
    # 4x localization detail and 8x context before PET's encoder/decoders. It
    # uses the original PET counting path: no count head, density loss, score
    # calibration, foreground gate, NMS, or branch gate.
    'vgg_apglc_bsf': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'scale_fusion': 'bidirectional',
        'scale_fusion_activation': 'gelu',
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'auto',
        'apg_loss_coef': 0.02,
        'apg_start_epoch': 0,
        'apg_warmup_epochs': 0,
        'apg_end_epoch': 350,
        'apg_sparse_coef': 1.0,
        'apg_dense_coef': 1.0,
        'apg_dense_start_epoch': 0,
        'apg_dense_warmup_epochs': 0,
        'apg_pos_k': 1,
        'apg_point_coef': 5.0,
        'apg_bg_coef': 0.0,
        'apg_contrastive_coef': 0.15,
        'apg_neg_k': 4,
        'apg_margin': 1.0,
        'apg_count_calibration': 'none',
        'count_head_loss_coef': 0.0,
        'density_map_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.0,
        'foreground_loss_coef': 0.0,
        'eval_foreground_gate': 'none',
        'eval_count_mode': 'threshold',
        'eval_score_calibration': 'none',
        'eval_dense_start_epoch': 0,
        'eval_dense_residual_mode': 'none',
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'score_threshold': 0.55,
        'split_threshold': 0.45,
        'split_threshold_quantile': 0.5,
        'query_prune_threshold': 0.5,
        'bad_count_start_epoch': 100,
        'bad_count_direction': 'all',
    },
    # Scratch version of the observed 50 -> 48 path.
    #
    # APG+LC is trained first. A small scalar density-sum count regularizer is
    # introduced only after the original step-drop point, when PET scores should
    # already be calibrated. This keeps the count head out of early query
    # formation, but lets it provide the count-aware feature nudge that produced
    # the best fine-tune result. It never uses top-k or count-bias evaluation.
    'vgg_apglc_late_countreg': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'paper',
        'apg_loss_coef': 0.02,
        'apg_start_epoch': 0,
        'apg_warmup_epochs': 0,
        'apg_end_epoch': 350,
        'apg_sparse_coef': 1.0,
        'apg_dense_coef': 1.0,
        'apg_dense_start_epoch': 0,
        'apg_dense_warmup_epochs': 0,
        'apg_pos_k': 1,
        'apg_point_coef': 5.0,
        'apg_bg_coef': 0.0,
        'apg_contrastive_coef': 0.15,
        'apg_neg_k': 4,
        'apg_margin': 1.0,
        'apg_count_calibration': 'none',
        'count_head_loss_coef': 0.10,
        'count_head_loss_type': 'log_l1',
        'count_head_start_epoch': 700,
        'count_head_end_epoch': -1,
        'count_head_warmup_epochs': 200,
        'count_head_feature_grad_scale': 0.05,
        'count_head_feature_grad_start_epoch': 700,
        'count_head_feature_grad_warmup_epochs': 200,
        'allow_count_head_fresh_train': True,
        'density_map_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.0,
        'foreground_loss_coef': 0.0,
        'eval_foreground_gate': 'none',
        'eval_count_mode': 'threshold',
        'eval_score_calibration': 'none',
        'eval_dense_start_epoch': 0,
        'eval_dense_residual_mode': 'none',
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'score_threshold': 0.55,
        'split_threshold': 0.45,
        'split_threshold_quantile': 0.5,
        'bad_count_start_epoch': 100,
        'bad_count_direction': 'all',
    },
    # Historical recovery path for the 48.8 MAE result.
    #
    # This is intentionally a checkpoint fine-tune recipe, not a fresh scratch
    # recipe. The successful run was APG+LC first, then a short scalar
    # density-sum count-head auxiliary fine-tune while inference stayed normal
    # threshold PET. Later "safe" count-head recipes detach/delay this signal
    # and reproduced only 52-55 MAE, so this recipe preserves the legacy
    # behavior explicitly instead of hiding it in ad-hoc command flags. APG is
    # disabled in this fine-tune because the archived 48-MAE train logs showed
    # loss_count_head but no loss_apg_* terms; the checkpoint already carries
    # the APG+LC representation.
    'vgg_apglc_density_counthead_ft_legacy': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'epochs': 80,
        'eval_freq': 2,
        'lr': 1e-4,
        'lr_backbone': 0.0,
        'lr_scheduler': 'step',
        'lr_drop': 50,
        'lr_gamma': 0.1,
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'auto',
        'apg_loss_coef': 0.0,
        'apg_start_epoch': 0,
        'apg_warmup_epochs': 0,
        'apg_end_epoch': 350,
        'apg_sparse_coef': 1.0,
        'apg_dense_coef': 1.0,
        'apg_dense_start_epoch': 0,
        'apg_dense_warmup_epochs': 0,
        'apg_pos_k': 1,
        'apg_point_coef': 5.0,
        'apg_bg_coef': 0.0,
        'apg_contrastive_coef': 0.15,
        'apg_neg_k': 4,
        'apg_margin': 1.0,
        'apg_count_calibration': 'none',
        'count_head_loss_coef': 1.0,
        'count_head_loss_type': 'log_l1',
        'count_head_start_epoch': 0,
        'count_head_end_epoch': -1,
        'count_head_warmup_epochs': 0,
        'count_head_feature_grad_scale': 1.0,
        'count_head_feature_grad_start_epoch': 0,
        'count_head_feature_grad_warmup_epochs': 0,
        'train_count_head_only': True,
        'freeze_bn': False,
        'no_auto_freeze_bn_on_count_head_resume': True,
        'density_map_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.0,
        'foreground_loss_coef': 0.0,
        'eval_foreground_gate': 'none',
        'eval_count_mode': 'threshold',
        'eval_score_calibration': 'none',
        'eval_dense_start_epoch': 0,
        'eval_dense_residual_mode': 'none',
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'eval_count_head_min_score': 0.0,
        'score_threshold': 0.59,
        'split_threshold': 0.45,
        'split_threshold_quantile': 0.55,
        'bad_count_start_epoch': 8,
        'bad_count_direction': 'over',
    },
    # Stage-2 adaptation after a trained APG+LC checkpoint. Unlike the legacy
    # recovery recipe, this updates the PET counting/localization heads with a
    # small scalar count-head auxiliary, while keeping the backbone frozen. It
    # is intended for NWPU high-density adaptation where eval still uses PET
    # threshold counting, so training only the count head cannot improve MAE.
    'vgg_apglc_counthead_stage2_adapt': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'epochs': 120,
        'eval_freq': 2,
        'lr': 2e-5,
        'lr_backbone': 0.0,
        'lr_scheduler': 'step',
        'lr_drop': 80,
        'lr_gamma': 0.1,
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'auto',
        'apg_loss_coef': 0.0,
        'count_head_loss_coef': 0.2,
        'count_head_loss_type': 'log_l1',
        'count_head_start_epoch': 0,
        'count_head_end_epoch': -1,
        'count_head_warmup_epochs': 20,
        'count_head_feature_grad_scale': 0.05,
        'count_head_feature_grad_start_epoch': 0,
        'count_head_feature_grad_warmup_epochs': 20,
        'density_map_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.0,
        'foreground_loss_coef': 0.0,
        'eval_foreground_gate': 'none',
        'eval_count_mode': 'threshold',
        'eval_count_source': 'pet',
        'eval_score_calibration': 'none',
        'eval_dense_start_epoch': 0,
        'eval_dense_residual_mode': 'none',
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'score_threshold': 0.50,
        'split_threshold': 0.45,
        'split_threshold_quantile': 0.55,
        'bad_count_start_epoch': 20,
        'bad_count_direction': 'over',
    },
    # Same end-to-end architecture as APG+LC + late scalar count regularizer,
    # but with an explicit APG warmup so scratch training starts near PET's loss
    # scale. Unlike the failed count-feedback variants, this does not suppress
    # PET probabilities early; it only avoids applying full auxiliary APG
    # pressure before the classifier has calibrated.
    'vgg_apglc_warmapg_late_countreg': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'paper',
        'apg_loss_coef': 1.0,
        'apg_start_epoch': 0,
        'apg_warmup_epochs': 100,
        'apg_sparse_coef': 1.0,
        'apg_dense_coef': 1.0,
        'apg_dense_start_epoch': 0,
        'apg_dense_warmup_epochs': 100,
        'apg_pos_k': 1,
        'apg_point_coef': 5.0,
        'apg_bg_coef': 0.0,
        'apg_count_calibration': 'none',
        'count_loss_coef': 0.0,
        'count_head_loss_coef': 0.10,
        'count_head_loss_type': 'log_l1',
        'count_head_start_epoch': 700,
        'count_head_end_epoch': -1,
        'count_head_warmup_epochs': 200,
        'count_head_feature_grad_scale': 0.05,
        'count_head_feature_grad_start_epoch': 700,
        'count_head_feature_grad_warmup_epochs': 200,
        'allow_count_head_fresh_train': True,
        'density_map_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.0,
        'foreground_loss_coef': 0.0,
        'eval_foreground_gate': 'none',
        'eval_count_mode': 'threshold',
        'eval_score_calibration': 'none',
        'eval_dense_start_epoch': 0,
        'eval_dense_residual_mode': 'none',
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'score_threshold': 0.5,
        'split_threshold': 0.5,
        'split_threshold_quantile': 0.5,
        'bad_count_start_epoch': 100,
        'bad_count_direction': 'all',
    },
    # PET-specific novelty: Cross-Branch Mutual Exclusion (CBME).
    #
    # The repeated scratch failures show duplicate sparse+dense confidence as
    # APG ramps. CBME penalizes overlapping sparse/dense person probabilities
    # after projecting sparse scores to the dense grid. It does not impose a
    # global count target and does not replace threshold inference; it only says
    # that one image location should not be counted confidently by both PET
    # branches.
    'vgg_apglc_cbme_late_countreg': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'paper',
        'apg_loss_coef': 1.0,
        'apg_start_epoch': 0,
        'apg_warmup_epochs': 100,
        'apg_sparse_coef': 1.0,
        'apg_dense_coef': 1.0,
        'apg_dense_start_epoch': 0,
        'apg_dense_warmup_epochs': 100,
        'apg_pos_k': 1,
        'apg_point_coef': 5.0,
        'apg_bg_coef': 0.0,
        'apg_count_calibration': 'none',
        'count_loss_coef': 0.0,
        'branch_exclusion_loss_coef': 0.05,
        'branch_exclusion_start_epoch': 0,
        'branch_exclusion_end_epoch': -1,
        'count_head_loss_coef': 0.10,
        'count_head_loss_type': 'log_l1',
        'count_head_start_epoch': 700,
        'count_head_end_epoch': -1,
        'count_head_warmup_epochs': 200,
        'count_head_feature_grad_scale': 0.05,
        'count_head_feature_grad_start_epoch': 700,
        'count_head_feature_grad_warmup_epochs': 200,
        'allow_count_head_fresh_train': True,
        'density_map_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.0,
        'foreground_loss_coef': 0.0,
        'eval_foreground_gate': 'none',
        'eval_count_mode': 'threshold',
        'eval_score_calibration': 'none',
        'eval_dense_start_epoch': 0,
        'eval_dense_residual_mode': 'none',
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'score_threshold': 0.5,
        'split_threshold': 0.5,
        'split_threshold_quantile': 0.5,
        'bad_count_start_epoch': 100,
        'bad_count_direction': 'all',
    },
    # Scratch version of the 50 -> 48 path with the missing half of APGCC
    # restored. APGCC uses both auxiliary positives and negatives; positive-only
    # APG repeatedly produced early SHA score explosions in this repo. This
    # recipe keeps PET/APG+LC intact, adds local APG negatives from epoch 0, and
    # delays the scalar density-sum count regularizer until after the original
    # PET step drop so it cannot form the query scores from scratch.
    'vgg_apglc_balanced_late_countreg': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'paper',
        'apg_loss_coef': 1.0,
        'apg_start_epoch': 0,
        'apg_warmup_epochs': 0,
        'apg_end_epoch': 350,
        'apg_sparse_coef': 1.0,
        'apg_dense_coef': 1.0,
        'apg_dense_start_epoch': 0,
        'apg_dense_warmup_epochs': 0,
        'apg_pos_k': 1,
        'apg_point_coef': 5.0,
        'apg_bg_coef': 0.5,
        'apg_bg_k': 8,
        'apg_bg_min_dist': 12.0,
        'apg_bg_offset_coef': 1.0,
        'apg_count_calibration': 'none',
        'count_head_loss_coef': 0.10,
        'count_head_loss_type': 'log_l1',
        'count_head_start_epoch': 700,
        'count_head_end_epoch': -1,
        'count_head_warmup_epochs': 200,
        'count_head_feature_grad_scale': 0.05,
        'count_head_feature_grad_start_epoch': 700,
        'count_head_feature_grad_warmup_epochs': 200,
        'allow_count_head_fresh_train': True,
        'density_map_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.0,
        'foreground_loss_coef': 0.0,
        'eval_foreground_gate': 'none',
        'eval_count_mode': 'threshold',
        'eval_score_calibration': 'none',
        'eval_dense_start_epoch': 0,
        'eval_dense_residual_mode': 'none',
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'score_threshold': 0.59,
        'split_threshold': 0.45,
        'split_threshold_quantile': 0.5,
        'bad_count_start_epoch': 100,
        'bad_count_direction': 'all',
    },
    # PET-specific scratch novelty: query-budget APG+LC.
    #
    # This is not APGCC reuse. It keeps PET threshold inference and APG+LC, but
    # adds an overcount-only soft budget on the actual sparse+dense PET person
    # probabilities. The loss does nothing when PET is under-counting and only
    # suppresses query-score explosions, which is the failure shown by the
    # epoch-5/10 logs. The scalar count head is still delayed until after the
    # original step-drop point.
    'vgg_apglc_budget_late_countreg': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'paper',
        'apg_loss_coef': 1.0,
        'apg_start_epoch': 0,
        'apg_warmup_epochs': 0,
        'apg_sparse_coef': 1.0,
        'apg_dense_coef': 1.0,
        'apg_dense_start_epoch': 0,
        'apg_dense_warmup_epochs': 0,
        'apg_pos_k': 1,
        'apg_point_coef': 5.0,
        'apg_bg_coef': 0.0,
        'apg_contrastive_coef': 0.15,
        'apg_neg_k': 4,
        'apg_margin': 1.0,
        'apg_count_calibration': 'none',
        'count_loss_coef': 0.10,
        'count_loss_gate': 'none',
        'count_loss_type': 'over_log_l1',
        'count_loss_budget_margin': 1.10,
        'count_loss_start_epoch': 0,
        'count_head_loss_coef': 0.10,
        'count_head_loss_type': 'log_l1',
        'count_head_start_epoch': 700,
        'count_head_end_epoch': -1,
        'count_head_warmup_epochs': 200,
        'count_head_feature_grad_scale': 0.05,
        'count_head_feature_grad_start_epoch': 700,
        'count_head_feature_grad_warmup_epochs': 200,
        'allow_count_head_fresh_train': True,
        'density_map_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.0,
        'foreground_loss_coef': 0.0,
        'eval_foreground_gate': 'none',
        'eval_count_mode': 'threshold',
        'eval_score_calibration': 'none',
        'eval_dense_start_epoch': 0,
        'eval_dense_residual_mode': 'none',
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'score_threshold': 0.59,
        'split_threshold': 0.45,
        'split_threshold_quantile': 0.5,
        'bad_count_start_epoch': 100,
        'bad_count_direction': 'all',
    },
    # Query-budget APG+LC with closed-loop APG pressure.
    #
    # The plain budget loss reduced the early explosion but was too weak while
    # positive APG stayed at full strength. This variant uses PET's own
    # thresholded count estimate to scale positive APG down during over-count,
    # while keeping APG local/background behavior and threshold inference intact.
    'vgg_apglc_count_feedback_late_countreg': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'paper',
        'apg_loss_coef': 1.0,
        'apg_start_epoch': 0,
        'apg_warmup_epochs': 0,
        'apg_sparse_coef': 1.0,
        'apg_dense_coef': 1.0,
        'apg_dense_start_epoch': 0,
        'apg_dense_warmup_epochs': 0,
        'apg_pos_k': 1,
        'apg_point_coef': 5.0,
        'apg_bg_coef': 0.0,
        'apg_count_calibration': 'threshold',
        'apg_count_calibration_gate': 'none',
        'apg_count_calibration_min': 0.20,
        'apg_count_calibration_max': 1.0,
        'apg_count_calibration_eps': 1.0,
        'count_loss_coef': 0.25,
        'count_loss_gate': 'none',
        'count_loss_type': 'over_log_l1',
        'count_loss_budget_margin': 1.05,
        'count_loss_start_epoch': 0,
        'count_head_loss_coef': 0.10,
        'count_head_loss_type': 'log_l1',
        'count_head_start_epoch': 700,
        'count_head_end_epoch': -1,
        'count_head_warmup_epochs': 200,
        'count_head_feature_grad_scale': 0.05,
        'count_head_feature_grad_start_epoch': 700,
        'count_head_feature_grad_warmup_epochs': 200,
        'allow_count_head_fresh_train': True,
        'density_map_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.0,
        'foreground_loss_coef': 0.0,
        'eval_foreground_gate': 'none',
        'eval_count_mode': 'threshold',
        'eval_score_calibration': 'none',
        'eval_dense_start_epoch': 0,
        'eval_dense_residual_mode': 'none',
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'score_threshold': 0.59,
        'split_threshold': 0.45,
        'split_threshold_quantile': 0.5,
        'bad_count_start_epoch': 100,
        'bad_count_direction': 'all',
    },
    # Low-loss scratch curriculum for the same architecture.
    #
    # Full-strength APG from epoch 0 makes the summed objective start around 3
    # and repeatedly over-count. This recipe keeps the same PET+LC+APG+count
    # feedback components, but ramps APG and the overcount budget so the first
    # epochs stay close to PET's calibrated loss scale.
    'vgg_apglc_lowloss_count_feedback': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'paper',
        'apg_loss_coef': 0.75,
        'apg_start_epoch': 0,
        'apg_warmup_epochs': 150,
        'apg_sparse_coef': 1.0,
        'apg_dense_coef': 1.0,
        'apg_dense_start_epoch': 0,
        'apg_dense_warmup_epochs': 150,
        'apg_pos_k': 1,
        'apg_point_coef': 5.0,
        'apg_bg_coef': 0.0,
        'apg_count_calibration': 'threshold',
        'apg_count_calibration_gate': 'none',
        'apg_count_calibration_min': 0.20,
        'apg_count_calibration_max': 1.0,
        'apg_count_calibration_eps': 1.0,
        'count_loss_coef': 0.10,
        'count_loss_gate': 'none',
        'count_loss_type': 'over_log_l1',
        'count_loss_budget_margin': 1.10,
        'count_loss_start_epoch': 0,
        'count_head_loss_coef': 0.10,
        'count_head_loss_type': 'log_l1',
        'count_head_start_epoch': 700,
        'count_head_end_epoch': -1,
        'count_head_warmup_epochs': 200,
        'count_head_feature_grad_scale': 0.05,
        'count_head_feature_grad_start_epoch': 700,
        'count_head_feature_grad_warmup_epochs': 200,
        'allow_count_head_fresh_train': True,
        'density_map_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.0,
        'foreground_loss_coef': 0.0,
        'eval_foreground_gate': 'none',
        'eval_count_mode': 'threshold',
        'eval_score_calibration': 'none',
        'eval_dense_start_epoch': 0,
        'eval_dense_residual_mode': 'none',
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'score_threshold': 0.59,
        'split_threshold': 0.45,
        'split_threshold_quantile': 0.5,
        'bad_count_start_epoch': 100,
        'bad_count_direction': 'all',
    },
    # Publication-inspired scratch path:
    # stable APG+LC + PANet/RCCFormer-style scale-aware local context + late
    # scalar count regularization. The DRF residual starts as identity, so
    # early training remains the known stable APG+LC behavior. Keep APG at the
    # verified 0.02 strength; full-strength APG from scratch repeatedly caused
    # SHA query-score explosions before the architecture could help.
    'vgg_apglc_drf_late_countreg': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'paper',
        'perspective_mixer': 'drf',
        'perspective_mixer_dilations': '1,2,3',
        'perspective_mixer_mid_dim': 64,
        'perspective_mixer_activation': 'gelu',
        'apg_loss_coef': 0.02,
        'apg_start_epoch': 0,
        'apg_warmup_epochs': 0,
        'apg_end_epoch': 350,
        'apg_sparse_coef': 1.0,
        'apg_dense_coef': 1.0,
        'apg_dense_start_epoch': 0,
        'apg_dense_warmup_epochs': 0,
        'apg_pos_k': 1,
        'apg_point_coef': 5.0,
        'apg_bg_coef': 0.0,
        'apg_contrastive_coef': 0.15,
        'apg_neg_k': 4,
        'apg_margin': 1.0,
        'apg_count_calibration': 'none',
        'count_head_loss_coef': 0.10,
        'count_head_loss_type': 'log_l1',
        'count_head_start_epoch': 700,
        'count_head_end_epoch': -1,
        'count_head_warmup_epochs': 200,
        'count_head_feature_grad_scale': 0.05,
        'count_head_feature_grad_start_epoch': 700,
        'count_head_feature_grad_warmup_epochs': 200,
        'allow_count_head_fresh_train': True,
        'density_map_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.0,
        'foreground_loss_coef': 0.0,
        'eval_foreground_gate': 'none',
        'eval_count_mode': 'threshold',
        'eval_score_calibration': 'none',
        'eval_dense_start_epoch': 0,
        'eval_dense_residual_mode': 'none',
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'score_threshold': 0.55,
        'split_threshold': 0.45,
        'split_threshold_quantile': 0.5,
        'bad_count_start_epoch': 100,
        'bad_count_direction': 'all',
    },
    # PET + lite FPN + APG with a spatial foreground-confidence branch.
    #
    # Experimental only. It keeps guidance local and uses suppress-only eval,
    # but it is not the recommended recipe until it beats vgg_apglc from scratch.
    'vgg_apglc_foreground': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'paper',
        'apg_loss_coef': 1.0,
        'apg_start_epoch': 0,
        'apg_warmup_epochs': 0,
        'apg_sparse_coef': 1.0,
        'apg_dense_coef': 1.0,
        'apg_dense_start_epoch': 0,
        'apg_dense_warmup_epochs': 0,
        'apg_pos_k': 1,
        'apg_point_coef': 5.0,
        'apg_bg_coef': 0.0,
        'apg_count_calibration': 'none',
        'count_head_loss_coef': 0.0,
        'density_map_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.0,
        'foreground_loss_coef': 0.10,
        'foreground_sigma': 8.0,
        'foreground_neg_shrink': 16.0,
        'foreground_init_prior': 0.5,
        'eval_foreground_gate': 'query',
        'eval_foreground_gate_mode': 'suppress',
        'eval_foreground_gate_strength': 0.75,
        'score_threshold': 0.5,
        'split_threshold': 0.5,
        'split_threshold_quantile': 0.5,
        'eval_count_mode': 'threshold',
        'eval_score_calibration': 'none',
        'eval_dense_start_epoch': 0,
        'eval_dense_residual_mode': 'none',
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'bad_count_start_epoch': 100,
        'bad_count_direction': 'all',
    },
    # PET-compatible APG plus a scalar density-sum calibration head.
    #
    # This recipe is intentionally conservative. The count head is trained from
    # detached encoder features, so it learns global count calibration without
    # pushing PET's query logits into the severe over-counting failure mode seen
    # with density-map and top-k variants. APG is kept on PET's real inference
    # heads, but its positive pressure is damped when the current query field
    # already over-counts.
    'vgg_apglc_countcal': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'warmup_epochs': 0,
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'paper',
        'apg_loss_coef': 1.0,
        'apg_start_epoch': 0,
        'apg_warmup_epochs': 100,
        'apg_sparse_coef': 1.0,
        'apg_dense_coef': 0.25,
        'apg_dense_start_epoch': 150,
        'apg_dense_warmup_epochs': 550,
        'pq_sparse_coef': 1.0,
        'pq_dense_coef': 1.0,
        'pq_dense_start_epoch': 150,
        'pq_dense_warmup_epochs': 550,
        'apg_pos_k': 1,
        'apg_point_coef': 5.0,
        'apg_bg_coef': 0.05,
        'apg_bg_k': 1,
        'apg_bg_min_dist': 12.0,
        'apg_count_calibration': 'threshold',
        'apg_count_calibration_gate': 'none',
        'apg_count_calibration_min': 0.25,
        'apg_count_calibration_max': 1.0,
        'count_head_loss_coef': 0.2,
        'count_head_loss_type': 'log_l1',
        'count_head_start_epoch': 0,
        'count_head_end_epoch': -1,
        'count_head_warmup_epochs': 100,
        'count_head_feature_grad_scale': 0.0,
        'count_head_feature_grad_start_epoch': 0,
        'count_head_feature_grad_warmup_epochs': 0,
        'allow_count_head_fresh_train': True,
        'density_map_loss_coef': 0.0,
        'eval_count_mode': 'threshold',
        'eval_score_calibration': 'count_head_bias',
        'eval_score_calibration_strength': 0.50,
        'eval_score_calibration_start_epoch': 100,
        'eval_score_calibration_min_bias': -1.0,
        'eval_score_calibration_max_bias': 1.0,
        'eval_score_calibration_count_blend': 0.50,
        'eval_score_calibration_count_ratio_min': 0.85,
        'eval_score_calibration_count_ratio_max': 1.15,
        'eval_dense_start_epoch': 700,
        'eval_dense_residual_mode': 'count_head',
        'eval_dense_residual_start_epoch': 150,
        'eval_dense_residual_min_score': 0.30,
        'score_threshold': 0.5,
        'split_threshold': 0.5,
        'split_threshold_quantile': 0.5,
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'bad_count_start_epoch': 100,
        'bad_count_direction': 'over',
    },
    # Count-Calibrated PET (CC-PET).
    #
    # This is the solid scratch architecture path: APG+LC remains the detector,
    # a detached scalar density-sum count branch learns global crowd mass, and
    # inference uses a bounded logit-bias solver to nudge PET's expected count
    # within a small ratio band. The count branch cannot top-k the result and
    # cannot force a large score shift, so it avoids the over-count failures
    # caused by direct count losses and unbounded count-head calibration.
    'vgg_apglc_ccpet': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'auto',
        'apg_loss_coef': 0.02,
        'apg_start_epoch': 0,
        'apg_warmup_epochs': 0,
        'apg_end_epoch': 350,
        'apg_sparse_coef': 1.0,
        'apg_dense_coef': 1.0,
        'apg_dense_start_epoch': 0,
        'apg_dense_warmup_epochs': 0,
        'apg_pos_k': 1,
        'apg_point_coef': 5.0,
        'apg_bg_coef': 0.0,
        'apg_contrastive_coef': 0.15,
        'apg_neg_k': 4,
        'apg_margin': 1.0,
        'apg_count_calibration': 'none',
        'count_loss_coef': 0.0,
        'count_head_loss_coef': 0.10,
        'count_head_loss_type': 'log_l1',
        'count_head_start_epoch': 0,
        'count_head_end_epoch': -1,
        'count_head_warmup_epochs': 0,
        'count_head_feature_grad_scale': 0.0,
        'count_head_feature_grad_start_epoch': 0,
        'count_head_feature_grad_warmup_epochs': 0,
        'allow_count_head_fresh_train': True,
        'density_map_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.0,
        'foreground_loss_coef': 0.0,
        'eval_foreground_gate': 'none',
        'eval_count_mode': 'threshold',
        'eval_count_head_min_score': 0.0,
        'eval_score_calibration': 'count_head_bias',
        'eval_score_calibration_strength': 0.50,
        'eval_score_calibration_start_epoch': 0,
        'eval_score_calibration_min_bias': -1.0,
        'eval_score_calibration_max_bias': 1.0,
        'eval_score_calibration_count_blend': 0.50,
        'eval_score_calibration_count_ratio_min': 0.85,
        'eval_score_calibration_count_ratio_max': 1.15,
        'eval_dense_start_epoch': 0,
        'eval_dense_residual_mode': 'none',
        'eval_nms_radius': 0.0,
        'eval_branch_gate': 'none',
        'eval_soft_split_gate': 'none',
        'score_threshold': 0.55,
        'split_threshold': 0.45,
        'split_threshold_quantile': 0.55,
        'bad_count_start_epoch': 100,
        'bad_count_direction': 'all',
    },
    # PET-compatible routed APG plus detached scalar count calibration.
    #
    # This is the conservative scratch-training version. It keeps PET's base
    # Hungarian point-query loss intact, because hard-routing those targets made
    # sparse nearly background-only and caused dense over-counting. Branch
    # ownership is applied through routed APG and GT split supervision instead.
    # The count head remains detached and is not used to bias validation logits
    # by default; the epoch-135 under-count run showed that online count-head
    # score calibration can suppress valid PET predictions. All active losses
    # in this recipe start at epoch 0; disabled components are disabled by mode,
    # not hidden behind unreachable start epochs.
    'vgg_routed_apglc_countcal': {
        'backbone': 'vgg16_bn',
        'timm_adapter': 'lite_fpn',
        'warmup_epochs': 0,
        'pet_loss_variant': 'paper',
        'split_loss_variant': 'paper_gt',
        'branch_target_routing': 'none',
        'pq_sparse_coef': 1.0,
        'pq_dense_coef': 1.0,
        'pq_dense_start_epoch': 0,
        'pq_dense_warmup_epochs': 0,
        'apg_loss_coef': 0.0,
        'routed_apg_loss_coef': 0.25,
        'routed_apg_point_coef': 5.0,
        'routed_apg_pos_k': 1,
        'routed_apg_bg_coef': 0.05,
        'routed_apg_bg_k': 1,
        'routed_apg_bg_min_dist': 12.0,
        'routed_apg_start_epoch': 0,
        'routed_apg_end_epoch': -1,
        'routed_apg_warmup_epochs': 0,
        'routed_apg_min_weight': 0.0,
        'routed_apg_source': 'gt_count',
        'routed_apg_gate': 'detach',
        'apg_count_calibration': 'threshold',
        'apg_count_calibration_gate': 'hard',
        'apg_count_calibration_min': 0.05,
        'apg_count_calibration_max': 1.0,
        'count_head_loss_coef': 0.2,
        'count_head_loss_type': 'log_l1',
        'count_head_start_epoch': 0,
        'count_head_end_epoch': -1,
        'count_head_warmup_epochs': 0,
        'count_head_feature_grad_scale': 0.0,
        'count_head_feature_grad_start_epoch': 0,
        'count_head_feature_grad_warmup_epochs': 0,
        'allow_count_head_fresh_train': True,
        'density_map_loss_coef': 0.0,
        'eval_count_mode': 'threshold',
        'eval_score_calibration': 'none',
        'eval_score_calibration_strength': 1.0,
        'eval_score_calibration_start_epoch': 0,
        'eval_score_calibration_min_bias': -8.0,
        'eval_score_calibration_max_bias': 8.0,
        'eval_dense_start_epoch': 0,
        'eval_dense_residual_mode': 'none',
        'score_threshold': 0.5,
        'split_threshold': 0.5,
        'split_threshold_quantile': 0.5,
        'eval_nms_radius': 4.0,
        'eval_branch_gate': 'pred',
        'eval_soft_split_gate': 'none',
        'bad_count_start_epoch': 100,
        'bad_count_direction': 'all',
    },
}

# Encoder Context-Fusion PET (ECF-PET).
#
# This is the PET-safe version of multi-level feature fusion: 4x detail is
# injected only into the 8x encoder context through a zero-initialized residual
# adapter. The dense 4x proposal branch, quadtree split path, APG schedule, and
# evaluation protocol stay identical to vgg_apglc.
#
# Audit note: SHA scratch run did not improve over vgg_apglc (best observed
# 54.28 MAE at epoch 450, then drifted to 63+), so this is kept only as a
# negative multi-scale-fusion ablation.
MODEL_RECIPES['vgg_apglc_ecfpn'] = {
    **MODEL_RECIPES['vgg_apglc'],
    'encoder_context_fusion': 'detail_to_context',
    'encoder_context_fusion_activation': 'gelu',
    'encoder_context_fusion_gate': 'channel_spatial',
}

# APGCC-style shared IFI for PET.
#
# The older IFI-lite path used a separate auxiliary head, which improves
# feature supervision but does not directly train the sparse/dense heads used
# at inference. This recipe keeps the verified vgg_apglc scratch path and adds
# low-weight interpolated GT/local-negative supervision directly to both PET
# proposal heads. It is intentionally not a count-head, density-map, NMS, or
# calibration recipe.
#
# Audit note: SHA runs did not improve over vgg_apglc (best observed 53.3 MAE
# and later count drift), so this remains an experimental ablation rather than
# a recommended recipe.
MODEL_RECIPES['vgg_apglc_shared_ifi'] = {
    **MODEL_RECIPES['vgg_apglc'],
    'ifi_loss_coef': 0.02,
    'ifi_head_source': 'routed',
    'ifi_interpolation': 'bilinear',
    'ifi_point_coef': 0.2,
    'ifi_neg_k': 4,
    'ifi_neg_radius': 8.0,
    'ifi_neg_min_dist': 2.0,
    'ifi_start_epoch': 0,
    'ifi_end_epoch': 350,
}

# Full IFI-PET: APGCC-style implicit interpolation in the PET proposal path.
#
# This is structural, not just an auxiliary loss. PET's sparse/dense query
# features are extracted with local implicit interpolation, and the IFI
# auxiliary sampler uses the same four-neighbor implicit feature function.
MODEL_RECIPES['vgg_apglc_full_ifi'] = {
    **MODEL_RECIPES['vgg_apglc'],
    'query_feature_interpolation': 'implicit',
    'ifi_interpolation': 'implicit',
    'ifi_pos_dim': 32,
    'ifi_mlp_hidden_dim': 256,
    'ifi_activation': 'gelu',
    'ifi_loss_coef': 0.02,
    'ifi_head_source': 'routed',
    'ifi_point_coef': 0.2,
    'ifi_neg_k': 4,
    'ifi_neg_radius': 8.0,
    'ifi_neg_min_dist': 2.0,
    'ifi_start_epoch': 0,
    'ifi_end_epoch': 350,
}

# Controlled non-unified IFI ablation.
#
# This starts from the validated PET paper contract and changes only the query
# representation plus its APGCC-style local point supervision. Sparse and
# dense branches own separate IFI modules, and auxiliary samples use the same
# branch module and prediction head as inference. Do not inherit vgg_apglc
# here: combining its nearest-query APG objective with IFI supervision trains
# two different representations of the same auxiliary points.
MODEL_RECIPES['vgg_pet_branch_ifi'] = {
    **MODEL_RECIPES['vgg_pet_paper'],
    'query_feature_interpolation': 'implicit',
    'query_ifi_sharing': 'independent',
    'query_ifi_feature_source': 'branch',
    'ifi_interpolation': 'implicit',
    'ifi_feature_source': 'branch',
    'ifi_pos_dim': 32,
    'ifi_mlp_hidden_dim': 256,
    'ifi_activation': 'gelu',
    'ifi_loss_coef': 0.02,
    'ifi_head_source': 'routed',
    'ifi_point_coef': 0.2,
    'ifi_pos_k': 2,
    'ifi_pos_radius': 2.0,
    'ifi_random_sampling': True,
    'ifi_neg_k': 2,
    'ifi_neg_radius': 8.0,
    'ifi_neg_min_dist': 2.0,
    'ifi_start_epoch': 0,
    'ifi_end_epoch': -1,
}

# PET with paper-aligned APG sampling and structural multi-scale IFI.
#
# This is an integration of APGCC's two mechanisms into PET, not a claim that
# PET has become the published APGCC architecture. Unlike the legacy
# --apg_loss_coef path, every GT independently creates two positive and two
# negative auxiliary positions. The shared continuous representation reads
# PET's projected 4x/8x features and the normal Hungarian point-query losses
# remain active, matching APGCC's "Matcher + APG" optimization principle.
MODEL_RECIPES['vgg_apgcc_paper_ifi'] = {
    **MODEL_RECIPES['vgg_pet_paper'],
    'query_feature_interpolation': 'implicit',
    'query_ifi_sharing': 'shared',
    'query_ifi_feature_source': 'fpn4x8x',
    'query_ifi_residual': False,
    'ifi_interpolation': 'implicit',
    'ifi_feature_source': 'branch',
    'ifi_pos_dim': 32,
    'ifi_mlp_hidden_dim': 256,
    'ifi_activation': 'gelu',
    'ifi_loss_coef': 0.2,
    'ifi_head_source': 'routed',
    # PET predicts image-normalized offsets, unlike APGCC's pixel-scaled
    # regression. Keep PET's established point coefficient for usable offset
    # gradients while preserving APGCC's outer lambda_5=0.2.
    'ifi_point_coef': 5.0,
    'ifi_point_loss_type': 'mse',
    'ifi_balance_pos_neg': True,
    'ifi_pos_k': 2,
    'ifi_pos_radius': 2.0,
    'ifi_random_sampling': True,
    'ifi_neg_k': 2,
    'ifi_neg_radius': 8.0,
    'ifi_neg_min_dist': 2.0,
    'ifi_start_epoch': 0,
    'ifi_end_epoch': -1,
}

# Count-head adaptation that keeps the full APGCC/PET query architecture from
# stage 1. APG auxiliary samples are training-only, so stage 2 disables their
# loss but retains the same shared IFI module and routed PET prediction heads.
MODEL_RECIPES['vgg_apgcc_paper_ifi_counthead_stage2'] = {
    **MODEL_RECIPES['vgg_apglc_counthead_stage2_adapt'],
    'split_loss_variant': 'paper',
    'query_feature_interpolation': 'implicit',
    'query_ifi_sharing': 'shared',
    'query_ifi_feature_source': 'fpn4x8x',
    'query_ifi_residual': False,
    'ifi_interpolation': 'implicit',
    'ifi_feature_source': 'branch',
    'ifi_pos_dim': 32,
    'ifi_mlp_hidden_dim': 256,
    'ifi_activation': 'gelu',
    'ifi_loss_coef': 0.0,
    'ifi_head_source': 'routed',
    'ifi_point_coef': 5.0,
    'ifi_point_loss_type': 'mse',
    'ifi_balance_pos_neg': True,
    'ifi_pos_k': 2,
    'ifi_pos_radius': 2.0,
    'ifi_random_sampling': True,
    'ifi_neg_k': 2,
    'ifi_neg_radius': 8.0,
    'ifi_neg_min_dist': 2.0,
}

# PET-preserving APG with residual implicit feature interpolation.
#
# Replacing PET query features with a random implicit interpolator can regress
# counting before the new representation is learned. This variant keeps the
# native PET feature as an exact identity path and learns the shared 4x/8x IFI
# correction through a zero-initialized ReZero-style residual. The APG objective is
# otherwise the corrected per-point positive/negative formulation above.
MODEL_RECIPES['vgg_pet_apg_rifi'] = {
    **MODEL_RECIPES['vgg_apgcc_paper_ifi'],
    'lr_scheduler': 'step',
    'lr_drop': 700,
    'lr_gamma': 0.1,
    'query_ifi_residual': True,
    'query_ifi_residual_init': 0.0,
    # APGCC's 0.2 coefficient is normalized for a different proposal network.
    # PET has two full CE branches, so retain the empirically stable PET APG
    # scale while keeping the auxiliary active for the complete run.
    'ifi_loss_coef': 0.02,
}

MODEL_RECIPES['vgg_pet_apg_rifi_counthead_stage2'] = {
    **MODEL_RECIPES['vgg_apgcc_paper_ifi_counthead_stage2'],
    'query_ifi_residual': True,
    'query_ifi_residual_init': 0.0,
}

# Density-Routed IFI APG+LC.
#
# Preserve the verified sparse APG+LC representation and use residual
# multi-scale IFI only for PET's dense branch. Sparse nearest-query APG and
# dense arbitrary-point APG are mutually exclusive, avoiding the conflicting
# double guidance in vgg_apglc_branch_ifi. Both auxiliary objectives end at
# epoch 350; the detector then consolidates before the epoch-700 LR drop.
MODEL_RECIPES['vgg_apglc_density_routed_ifi'] = {
    **MODEL_RECIPES['vgg_apglc'],
    'lr_scheduler': 'step',
    'lr_drop': 700,
    'lr_gamma': 0.1,
    'apg_sparse_coef': 1.0,
    'apg_dense_coef': 0.0,
    'query_feature_interpolation': 'implicit',
    'query_ifi_sharing': 'shared',
    'query_ifi_feature_source': 'fpn4x8x',
    'query_ifi_branch_scope': 'dense',
    'query_ifi_residual': True,
    'query_ifi_residual_init': 1e-3,
    'ifi_interpolation': 'implicit',
    'ifi_feature_source': 'branch',
    'ifi_branch_scope': 'dense',
    'ifi_pos_dim': 32,
    'ifi_mlp_hidden_dim': 256,
    'ifi_activation': 'gelu',
    'ifi_loss_coef': 0.02,
    'ifi_head_source': 'routed',
    'ifi_point_coef': 5.0,
    'ifi_point_loss_type': 'mse',
    'ifi_balance_pos_neg': True,
    'ifi_pos_k': 2,
    'ifi_pos_radius': 2.0,
    'ifi_random_sampling': True,
    'ifi_neg_k': 2,
    'ifi_neg_radius': 8.0,
    'ifi_neg_min_dist': 2.0,
    'ifi_negative_policy': 'paper',
    'ifi_start_epoch': 0,
    'ifi_end_epoch': 350,
    'count_head_loss_coef': 0.0,
    'density_map_loss_coef': 0.0,
    'eval_count_mode': 'threshold',
    'eval_count_source': 'pet',
    'eval_score_calibration': 'none',
    'eval_nms_radius': 0.0,
    'eval_branch_gate': 'none',
    'eval_soft_split_gate': 'none',
}

# APG+LC + branch-local IFI.
#
# This is the missing combination that the repo did not expose cleanly before:
# keep the verified APG+LC training schedule, but replace PET's nearest-cell
# sparse/dense query features with branch-local implicit interpolation. Unlike
# unified/shared IFI, sparse and dense keep separate query interpolators and
# branch-local auxiliary supervision.
MODEL_RECIPES['vgg_apglc_branch_ifi'] = {
    **MODEL_RECIPES['vgg_apglc'],
    'query_feature_interpolation': 'implicit',
    'query_ifi_sharing': 'independent',
    'query_ifi_feature_source': 'branch',
    'ifi_interpolation': 'implicit',
    'ifi_feature_source': 'branch',
    'ifi_pos_dim': 32,
    'ifi_mlp_hidden_dim': 256,
    'ifi_activation': 'gelu',
    'ifi_loss_coef': 0.02,
    'ifi_head_source': 'routed',
    'ifi_point_coef': 0.2,
    'ifi_pos_k': 2,
    'ifi_pos_radius': 2.0,
    'ifi_random_sampling': True,
    'ifi_neg_k': 2,
    'ifi_neg_radius': 8.0,
    'ifi_neg_min_dist': 2.0,
    'ifi_start_epoch': 0,
    'ifi_end_epoch': 350,
}

MODEL_RECIPES['vgg_apglc_branch_ifi_counthead_stage2'] = {
    **MODEL_RECIPES['vgg_apglc_counthead_stage2_adapt'],
    'query_feature_interpolation': 'implicit',
    'query_ifi_sharing': 'independent',
    'query_ifi_feature_source': 'branch',
    'ifi_interpolation': 'implicit',
    'ifi_feature_source': 'branch',
    'ifi_pos_dim': 32,
    'ifi_mlp_hidden_dim': 256,
    'ifi_activation': 'gelu',
    'ifi_loss_coef': 0.0,
    'ifi_head_source': 'routed',
    'ifi_point_coef': 0.2,
    'ifi_pos_k': 2,
    'ifi_pos_radius': 2.0,
    'ifi_random_sampling': True,
    'ifi_neg_k': 2,
    'ifi_neg_radius': 8.0,
    'ifi_neg_min_dist': 2.0,
}

# Residual Multi-scale IFI PET (RMI-PET).
#
# PET's validated quadtree, matcher, and inference contract remain unchanged.
# One shared APGCC-style IFI reads projected 4x/8x features for both branches
# and auxiliary points. Its contribution is LayerScale-initialized as a small
# residual over each branch's native feature, avoiding the count-calibration
# regression observed when IFI replaced PET features outright.
MODEL_RECIPES['vgg_pet_rmi'] = {
    **MODEL_RECIPES['vgg_pet_paper'],
    'query_feature_interpolation': 'implicit',
    'query_ifi_sharing': 'shared',
    'query_ifi_feature_source': 'fpn4x8x',
    'query_ifi_residual': True,
    'query_ifi_residual_init': 1e-3,
    'ifi_interpolation': 'implicit',
    'ifi_feature_source': 'branch',
    'ifi_pos_dim': 32,
    'ifi_mlp_hidden_dim': 256,
    'ifi_activation': 'gelu',
    'ifi_loss_coef': 0.2,
    'ifi_head_source': 'routed',
    'ifi_point_coef': 0.2,
    'ifi_pos_k': 2,
    'ifi_pos_radius': 2.0,
    'ifi_random_sampling': True,
    'ifi_neg_k': 2,
    'ifi_neg_radius': 8.0,
    'ifi_neg_min_dist': 2.0,
    'ifi_start_epoch': 0,
    'ifi_end_epoch': -1,
}

MODEL_RECIPES['vgg_apglc_full_ifi_counthead_ft_legacy'] = {
    **MODEL_RECIPES['vgg_apglc_density_counthead_ft_legacy'],
    'query_feature_interpolation': 'implicit',
    'ifi_interpolation': 'implicit',
    'ifi_pos_dim': 32,
    'ifi_mlp_hidden_dim': 256,
    'ifi_activation': 'gelu',
    'ifi_loss_coef': 0.0,
    'ifi_head_source': 'routed',
    'ifi_point_coef': 0.2,
    'ifi_neg_k': 4,
    'ifi_neg_radius': 8.0,
    'ifi_neg_min_dist': 2.0,
}

# Quality-Calibrated PET (QC-PET).
#
# Keeps the verified APG+LC representation and inference path, but trains the
# person score to encode localization quality for matched point queries. This
# follows the detector lesson from GFL/VFNet: candidate ranking should reflect
# objectness and localization quality, not a binary label alone.
#
# Audit note: SHA sweep reached only 51.23 MAE and reduced localization F1
# relative to the verified APG+LC path, so this is kept as a reproducible
# negative ablation rather than a recommended recipe.
MODEL_RECIPES['vgg_apglc_quality'] = {
    **MODEL_RECIPES['vgg_apglc'],
    'quality_loss_coef': 0.05,
    'quality_loss_sigma': 16.0,
    'quality_loss_pos_floor': 0.50,
    'quality_loss_bg_weight': 0.05,
}

# Local Ordinal ZIP PET (LOZIP-PET): the verified APG+LC recipe plus a
# zero-initialized local-density representation module. Unlike the failed
# scalar count-feedback experiments, this module never shifts point logits or
# chooses the number of retained predictions.
#
# Audit note: this is not the full EBC-ZIP model from Ma et al.; it is only a
# local auxiliary on PET's 8x encoder features. SHA sweep reached 50.58 MAE,
# so keep it as a negative ablation rather than a recommended architecture.
MODEL_RECIPES['vgg_apglc_localzip'] = {
    **MODEL_RECIPES['vgg_apglc'],
    'local_density_mixer': 'zip_ordinal',
    'local_density_block_size': 16,
    'local_density_projection_dim': 64,
    'local_density_bin_centers': '1,2,3,4,5,6,7,8,9,10,11.38,13.38,16.26',
    'local_density_zero_prior': 0.9,
    'local_zip_loss_coef': 0.02,
    'local_zip_ce_coef': 1.0,
    'local_zip_count_coef': 0.1,
    'local_ordinal_loss_coef': 0.02,
    'local_ordinal_temperature': 0.1,
    'local_ordinal_edges': '1,2,4,8',
    'local_ordinal_max_per_level': 64,
}

# EBC-ZIP Count PET: PET/APG+LC remains the point-localization branch, while a
# separate blockwise Zero-Inflated Poisson head supplies the MAE/RMSE count.
# This is the paper-backed route from EBC-ZIP/ZIP, unlike the earlier scalar
# count-head experiments that globally shifted PET logits and caused count
# drift.
MODEL_RECIPES['vgg_apglc_ebczip'] = {
    **MODEL_RECIPES['vgg_apglc'],
    'zip_count_loss_coef': 1.0,
    'zip_count_block_size': 16,
    'zip_count_feature_source': 'fpn4x8x',
    'zip_count_bin_centers': '1,2,3,4,5,6,7,8,9,10,11.38,13.38,16.26',
    'zip_count_zero_prior': 0.9,
    'zip_count_ce_coef': 1.0,
    'zip_count_count_coef': 1.0,
    'zip_count_start_epoch': 0,
    'zip_count_end_epoch': -1,
    'zip_count_warmup_epochs': 20,
    # Keep the verified PET/APG+LC representation intact. The ZIP branch is a
    # count model trained on PET features; allowing its loss into the encoder
    # regressed the point branch to ~56 MAE in the first scratch run.
    'zip_count_feature_grad_scale': 0.0,
    'eval_count_source': 'zip',
}

# NWPU Tail-Robust APG+LC.
#
# NWPU differs from SHA in the exact way that broke the current run: it has
# high-resolution validation images with counts up to tens of thousands. A
# single resized full-image pass under-counts those extreme images, while
# tiling every large image over-counts normal scenes. This recipe keeps the
# verified PET/APG+LC detector unchanged, but changes the data/eval regime:
# - sample dense training crops and high-count images more often;
# - select checkpoints with adaptive tiled eval only when the first PET pass
#   already indicates an extreme count.
MODEL_RECIPES['vgg_apglc_nwpu_tail'] = {
    **MODEL_RECIPES['vgg_apglc'],
    'crop_attempts': 12,
    'min_crop_points': 1,
    'eval_max_size': 0,
    'nwpu_dense_crop_prob': 0.5,
    'nwpu_dense_crop_attempts': 32,
    'train_count_weight_power': 0.5,
    'train_count_weight_max': 8.0,
    'nwpu_sigma_mode': 'official',
    'localization_protocol': 'target_sigma',
    'eval_tile_size': 1536,
    'eval_tile_overlap': 128,
    'eval_tile_nms_radius': 8.0,
    'eval_tile_max_tiles': 16,
    'eval_tile_trigger_count': 1500.0,
    'eval_tile_trigger_area': 0,
}

# NWPU stage-2 count-head adaptation.
#
# This is the NWPU analogue of the SHA 50 -> 48 recovery path, but it keeps
# the safer stage-2 behavior: load a calibrated APG+LC detector, freeze the
# backbone, and lightly update PET heads with the scalar count auxiliary while
# preserving the same dense-crop/high-count sampling and tail-aware validation
# policy used by stage 1.
MODEL_RECIPES['vgg_apglc_counthead_stage2_nwpu'] = {
    **MODEL_RECIPES['vgg_apglc_counthead_stage2_adapt'],
    'crop_attempts': 12,
    'min_crop_points': 1,
    'eval_max_size': 0,
    'nwpu_dense_crop_prob': 0.5,
    'nwpu_dense_crop_attempts': 32,
    'train_count_weight_power': 0.5,
    'train_count_weight_max': 8.0,
    'nwpu_sigma_mode': 'official',
    'localization_protocol': 'target_sigma',
    'eval_tile_size': 1536,
    'eval_tile_overlap': 128,
    'eval_tile_nms_radius': 8.0,
    'eval_tile_max_tiles': 16,
    'eval_tile_trigger_count': 1500.0,
    'eval_tile_trigger_area': 0,
}

MODEL_RECIPES['vgg_apgcc_paper_ifi_nwpu'] = {
    **MODEL_RECIPES['vgg_apgcc_paper_ifi'],
    'crop_attempts': 12,
    'min_crop_points': 1,
    'eval_max_size': 0,
    'nwpu_dense_crop_prob': 0.5,
    'nwpu_dense_crop_attempts': 32,
    'train_count_weight_power': 0.5,
    'train_count_weight_max': 8.0,
    'nwpu_sigma_mode': 'official',
    'localization_protocol': 'target_sigma',
    'eval_tile_size': 1536,
    'eval_tile_overlap': 128,
    'eval_tile_nms_radius': 8.0,
    'eval_tile_max_tiles': 16,
    'eval_tile_trigger_count': 1500.0,
    'eval_tile_trigger_area': 0,
}

MODEL_RECIPES['vgg_apgcc_paper_ifi_counthead_stage2_nwpu'] = {
    **MODEL_RECIPES['vgg_apgcc_paper_ifi_counthead_stage2'],
    'crop_attempts': 12,
    'min_crop_points': 1,
    'eval_max_size': 0,
    'nwpu_dense_crop_prob': 0.5,
    'nwpu_dense_crop_attempts': 32,
    'train_count_weight_power': 0.5,
    'train_count_weight_max': 8.0,
    'nwpu_sigma_mode': 'official',
    'localization_protocol': 'target_sigma',
    'eval_tile_size': 1536,
    'eval_tile_overlap': 128,
    'eval_tile_nms_radius': 8.0,
    'eval_tile_max_tiles': 16,
    'eval_tile_trigger_count': 1500.0,
    'eval_tile_trigger_area': 0,
}

MODEL_RECIPES['vgg_pet_apg_rifi_nwpu'] = {
    **MODEL_RECIPES['vgg_apgcc_paper_ifi_nwpu'],
    'lr_scheduler': 'step',
    'lr_drop': 700,
    'lr_gamma': 0.1,
    'query_ifi_residual': True,
    'query_ifi_residual_init': 0.0,
    'ifi_loss_coef': 0.02,
}

MODEL_RECIPES['vgg_pet_apg_rifi_counthead_stage2_nwpu'] = {
    **MODEL_RECIPES['vgg_apgcc_paper_ifi_counthead_stage2_nwpu'],
    'query_ifi_residual': True,
    'query_ifi_residual_init': 0.0,
}

MODEL_RECIPES['vgg_apglc_density_routed_ifi_nwpu'] = {
    **MODEL_RECIPES['vgg_apglc_nwpu_tail'],
    **{
        key: value
        for key, value in MODEL_RECIPES['vgg_apglc_density_routed_ifi'].items()
        if key not in {
            'patch_size_choices',
            'crop_attempts',
            'min_crop_points',
            'nwpu_dense_crop_prob',
            'nwpu_dense_crop_attempts',
            'train_count_weight_power',
            'train_count_weight_max',
            'eval_max_size',
            'eval_tile_size',
            'eval_tile_overlap',
            'eval_tile_nms_radius',
            'eval_tile_max_tiles',
            'eval_tile_trigger_count',
            'eval_tile_trigger_area',
        }
    },
}

MODEL_RECIPES['vgg_apglc_full_ifi_nwpu'] = {
    **MODEL_RECIPES['vgg_apglc_nwpu_tail'],
    'query_feature_interpolation': 'implicit',
    'ifi_interpolation': 'implicit',
    'ifi_pos_dim': 32,
    'ifi_mlp_hidden_dim': 256,
    'ifi_activation': 'gelu',
    'ifi_loss_coef': 0.02,
    'ifi_head_source': 'routed',
    'ifi_point_coef': 0.2,
    'ifi_neg_k': 4,
    'ifi_neg_radius': 8.0,
    'ifi_neg_min_dist': 2.0,
    'ifi_start_epoch': 0,
    'ifi_end_epoch': 700,
}

MODEL_RECIPES['vgg_apglc_full_ifi_counthead_stage2_nwpu'] = {
    **MODEL_RECIPES['vgg_apglc_counthead_stage2_nwpu'],
    'query_feature_interpolation': 'implicit',
    'ifi_interpolation': 'implicit',
    'ifi_pos_dim': 32,
    'ifi_mlp_hidden_dim': 256,
    'ifi_activation': 'gelu',
    'ifi_loss_coef': 0.0,
    'ifi_head_source': 'routed',
    'ifi_point_coef': 0.2,
    'ifi_neg_k': 4,
    'ifi_neg_radius': 8.0,
    'ifi_neg_min_dist': 2.0,
}

UNIFIED_IFI_RECIPE_OVERRIDES = {
    'query_feature_interpolation': 'implicit',
    'query_ifi_sharing': 'shared',
    'query_ifi_feature_source': 'fpn4x8x',
    'ifi_interpolation': 'implicit',
    'ifi_feature_source': 'branch',
    'ifi_pos_dim': 32,
    'ifi_mlp_hidden_dim': 256,
    'ifi_activation': 'gelu',
    'ifi_loss_coef': 0.02,
    'ifi_head_source': 'routed',
    'ifi_point_coef': 0.2,
    'ifi_pos_k': 2,
    'ifi_pos_radius': 2.0,
    'ifi_random_sampling': True,
    'ifi_neg_k': 2,
    'ifi_neg_radius': 8.0,
    'ifi_neg_min_dist': 2.0,
    'ifi_start_epoch': 0,
    'ifi_end_epoch': 700,
    'branch_target_routing': 'gt_count',
    'split_loss_variant': 'paper_gt',
    'split_count_threshold': 2,
    'split_pos_weight': 1.0,
    'eval_branch_gate': 'none',
}

# Unified IFI-PET.
#
# Sparse, dense, and APG auxiliary points use one shared continuous feature
# representation and the same prediction heads. GT quadtree routing gives each
# target one branch owner during training, while inference retains PET's native
# window pruning rather than applying a second heuristic branch gate.
MODEL_RECIPES['vgg_apglc_unified_ifi'] = {
    **MODEL_RECIPES['vgg_apglc'],
    **UNIFIED_IFI_RECIPE_OVERRIDES,
    'scale_point_loss_coef': 0.0,
}

MODEL_RECIPES['vgg_apglc_unified_ifi_counthead_stage2'] = {
    **MODEL_RECIPES['vgg_apglc_counthead_stage2_adapt'],
    **UNIFIED_IFI_RECIPE_OVERRIDES,
    'ifi_loss_coef': 0.0,
    'scale_point_loss_coef': 0.0,
}

# NWPU specialization: retain the unified detector and add box-scale-aware
# localization plus tail-aware crop/evaluation settings.
MODEL_RECIPES['vgg_apglc_unified_ifi_nwpu'] = {
    **MODEL_RECIPES['vgg_apglc_nwpu_tail'],
    **UNIFIED_IFI_RECIPE_OVERRIDES,
    'scale_point_loss_coef': 0.05,
    'scale_point_sigma': 'small',
    'scale_point_sigma_min': 2.0,
    'scale_point_sigma_max': 128.0,
}

MODEL_RECIPES['vgg_apglc_unified_ifi_counthead_stage2_nwpu'] = {
    **MODEL_RECIPES['vgg_apglc_counthead_stage2_nwpu'],
    **UNIFIED_IFI_RECIPE_OVERRIDES,
    'ifi_loss_coef': 0.0,
    'scale_point_loss_coef': 0.05,
    'scale_point_sigma': 'small',
    'scale_point_sigma_min': 2.0,
    'scale_point_sigma_max': 128.0,
}

MODEL_RECIPES['vgg_apglc_branch_ifi_nwpu'] = {
    **MODEL_RECIPES['vgg_apglc_nwpu_tail'],
    'query_feature_interpolation': 'implicit',
    'query_ifi_sharing': 'independent',
    'query_ifi_feature_source': 'branch',
    'ifi_interpolation': 'implicit',
    'ifi_feature_source': 'branch',
    'ifi_pos_dim': 32,
    'ifi_mlp_hidden_dim': 256,
    'ifi_activation': 'gelu',
    'ifi_loss_coef': 0.02,
    'ifi_head_source': 'routed',
    'ifi_point_coef': 0.2,
    'ifi_pos_k': 2,
    'ifi_pos_radius': 2.0,
    'ifi_random_sampling': True,
    'ifi_neg_k': 2,
    'ifi_neg_radius': 8.0,
    'ifi_neg_min_dist': 2.0,
    'ifi_start_epoch': 0,
    'ifi_end_epoch': 700,
    'scale_point_loss_coef': 0.05,
    'scale_point_sigma': 'small',
    'scale_point_sigma_min': 2.0,
    'scale_point_sigma_max': 128.0,
}

MODEL_RECIPES['vgg_apglc_branch_ifi_counthead_stage2_nwpu'] = {
    **MODEL_RECIPES['vgg_apglc_counthead_stage2_nwpu'],
    'query_feature_interpolation': 'implicit',
    'query_ifi_sharing': 'independent',
    'query_ifi_feature_source': 'branch',
    'ifi_interpolation': 'implicit',
    'ifi_feature_source': 'branch',
    'ifi_pos_dim': 32,
    'ifi_mlp_hidden_dim': 256,
    'ifi_activation': 'gelu',
    'ifi_loss_coef': 0.0,
    'ifi_head_source': 'routed',
    'ifi_point_coef': 0.2,
    'ifi_pos_k': 2,
    'ifi_pos_radius': 2.0,
    'ifi_random_sampling': True,
    'ifi_neg_k': 2,
    'ifi_neg_radius': 8.0,
    'ifi_neg_min_dist': 2.0,
    'scale_point_loss_coef': 0.05,
    'scale_point_sigma': 'small',
    'scale_point_sigma_min': 2.0,
    'scale_point_sigma_max': 128.0,
}

EXPERIMENTAL_MODEL_RECIPES = {
    # RMI-PET is the current cross-dataset hypothesis. It is not promoted to a
    # production recipe until fixed-protocol SHA/SHB/QNRF/JHU/NWPU runs pass.
    'vgg_pet_rmi',
    # Paper-aligned sampling is implemented and contract-tested, but its PET
    # integration still requires fixed-protocol cross-dataset falsification.
    'vgg_apgcc_paper_ifi',
    'vgg_apgcc_paper_ifi_counthead_stage2',
    'vgg_apgcc_paper_ifi_nwpu',
    'vgg_apgcc_paper_ifi_counthead_stage2_nwpu',
    'vgg_pet_apg_rifi',
    'vgg_pet_apg_rifi_counthead_stage2',
    'vgg_pet_apg_rifi_nwpu',
    'vgg_pet_apg_rifi_counthead_stage2_nwpu',
    'vgg_apglc_density_routed_ifi',
    'vgg_apglc_density_routed_ifi_nwpu',
    # The remaining paths are kept for audit/reproduction only. Session runs
    # showed catastrophic drift or failed to improve on the PET/APG+LC baselines.
    'vgg_apglc_cbme_late_countreg',
    'vgg_apglc_foreground',
    'vgg_apglc_balanced_late_countreg',
    'vgg_apglc_budget_late_countreg',
    'vgg_apglc_count_feedback_late_countreg',
    'vgg_apglc_lowloss_count_feedback',
    'vgg_apglc_warmapg_late_countreg',
    'vgg_apglc_countcal',
    'vgg_apglc_ccpet',
    'vgg_apglc_bsf',
    'vgg_apglc_ecfpn',
    'vgg_apglc_shared_ifi',
    'vgg_apglc_quality',
    'vgg_apglc_localzip',
    'vgg_apglc_ebczip',
    'vgg_routed_apglc_countcal',
}

HEAVY_BACKBONE_PREFIXES = (
    'convnext_base',
    'convnextv2_base',
    'convnextv2_large',
    'convnextv2_huge',
    'swinv2',
    'maxvit',
    'resnet',
    'pvt_v2',
    'pvtv2',
)

MOBILE_BACKBONE_PREFIXES = (
    'convnext_tiny',
    'convnextv2_tiny',
    'convnextv2_small',
    'fastvit',
    'efficientvit',
    'efficientnet_',
    'efficientnetv2',
    'tf_efficientnetv2',
    'tf_efficientnet_',
    'mobilenetv4',
    'hgnet',
    'hgnetv2',
    'edgenext',
    'repvit',
)

BACKBONE_ABLATION_PRESETS = {
    'crowd_dense': [
        'convnextv2_base',
        'convnext_base',
        'swinv2_base',
        'maxvit_small',
        'pvtv2_b1',
    ],
    'latency': [
        'convnextv2_tiny',
        'fastvit_tiny',
        'efficientvit_tiny',
        'mobilenetv4_small',
        'repvit_tiny',
        'edgenext_tiny',
    ],
    'full': list(get_supported_timm_backbones()),
}

ARCHITECTURE_OVERRIDE_KEYS = {
    'backbone',
    'no_pretrained_backbone',
    'allow_random_backbone_fallback',
    'timm_adapter',
    'timm_output_norm',
    'scale_fusion',
    'scale_fusion_activation',
    'encoder_context_fusion',
    'encoder_context_fusion_activation',
    'encoder_context_fusion_gate',
    'position_embedding',
    'dec_layers',
    'dim_feedforward',
    'hidden_dim',
    'dropout',
    'nheads',
    'transformer_activation',
    'transformer_norm_style',
    'decoder_attention',
    'decoder_memory_halo',
    'decoder_global_context',
    'decoder_global_context_mode',
    'enc_win_sizes',
    'enc_shift_mode',
    'sparse_dec_win_size',
    'dense_dec_win_size',
    'query_feature_interpolation',
    'query_ifi_sharing',
    'query_ifi_feature_source',
    'query_ifi_branch_scope',
    'query_ifi_residual',
    'query_ifi_residual_init',
    'ifi_interpolation',
    'ifi_feature_source',
    'ifi_pos_dim',
    'ifi_mlp_hidden_dim',
    'ifi_activation',
    'ifi_branch_scope',
    'context_patch_size',
    'quad_context_mixer',
    'quad_context_levels',
    'quad_context_shift',
    'quad_context_mid_dim',
    'quad_context_activation',
    'perspective_mixer',
    'perspective_mixer_dilations',
    'perspective_mixer_mid_dim',
    'perspective_mixer_activation',
    'splitter_head',
    'splitter_hidden_dim',
    'splitter_activation',
    'fusion_mhf_mode',
    'fusion_mhf_heads',
    'fusion_mhf_position',
    'fusion_mhf_strength',
    'fusion_mhf_activation',
    'fusion_mhf_impl',
    'fusion_fpn_type',
    'fusion_mhf_reduction',
    'fusion_mhf_norm',
    'fusion_mhf_spatial_kernel',
    'fusion_mhf_output_activation',
    'foreground_loss_coef',
    'foreground_sigma',
    'foreground_neg_shrink',
    'foreground_init_prior',
    'zip_count_loss_coef',
    'zip_count_block_size',
    'zip_count_feature_source',
    'zip_count_bin_centers',
    'zip_count_zero_prior',
    'zip_count_ce_coef',
    'zip_count_count_coef',
    'zip_count_start_epoch',
    'zip_count_end_epoch',
    'zip_count_warmup_epochs',
    'zip_count_feature_grad_scale',
    'eval_count_blend_alpha',
    'vgg_fpn_main_lr',
}


def get_backbone_recipe(backbone_name):
    if backbone_name.startswith('vgg'):
        return BACKBONE_RECIPES['vgg']
    if any(backbone_name.startswith(prefix) for prefix in HEAVY_BACKBONE_PREFIXES):
        return BACKBONE_RECIPES['heavy']
    if any(backbone_name.startswith(prefix) for prefix in MOBILE_BACKBONE_PREFIXES):
        return BACKBONE_RECIPES['mobile']
    return None


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)

    # training Parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_backbone_adapter', default=-1.0, type=float,
                        help='learning rate for randomly initialized backbone adapters/FPN; negative uses --lr')
    parser.add_argument('--vgg_fpn_main_lr', action='store_true',
                        help='train the VGG FPN fusion block at --lr; disabled keeps original PET optimizer grouping')
    parser.add_argument('--freeze_backbone_epochs', default=0, type=int,
                        help='freeze pretrained backbone feature extractor for this many initial epochs')
    parser.add_argument('--freeze_bn', action='store_true',
                        help='keep BatchNorm running statistics fixed during training/fine-tuning')
    parser.add_argument('--amp', action='store_true',
                        help='train with CUDA automatic mixed precision to reduce activation memory')
    parser.add_argument('--amp_dtype', default='auto', choices=('auto', 'float16', 'bfloat16'),
                        help='CUDA autocast dtype. auto uses bfloat16 when supported, otherwise float16')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='gradient accumulation steps; effective batch size is batch_size * accum_iter')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1500, type=int)
    parser.add_argument('--lr_scheduler', default='step', type=str,
                        choices=('step', 'warmup_hold_cosine'),
                        help='learning-rate schedule to use')
    parser.add_argument('--lr_drop', default=-1, type=int,
                        help='epoch interval for StepLR decay; negative keeps original PET behavior and drops after --epochs')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
                        help='multiplicative LR decay factor for StepLR')
    parser.add_argument('--auto_backbone_recipe', action='store_true',
                        help='opt into backbone-specific lr/batch/warmup defaults')
    parser.add_argument('--model_recipe', default='none',
                        choices=('none',) + tuple(MODEL_RECIPES.keys()),
                        help='apply a vetted model/loss recipe before safety checks')
    parser.add_argument('--allow_experimental_model_recipe', action='store_true',
                        help='allow recipes kept only for ablation/reproduction after known bad SHA runs')
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help='number of warmup epochs')
    parser.add_argument('--hold_epochs', default=-1, type=int,
                        help='epochs to keep peak lr before cosine decay; -1 picks an automatic value')
    parser.add_argument('--min_lr', default=1e-7, type=float,
                        help='minimum learning rate reached by cosine annealing')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # model parameters
    # - backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--list_backbones', action='store_true',
                        help='print supported timm ablation backbones and exit')
    parser.add_argument('--no_pretrained_backbone', action='store_true',
                        help='initialize the backbone randomly instead of loading timm/ImageNet weights')
    parser.add_argument('--allow_random_backbone_fallback', action='store_true',
                        help='allow timm backbones to continue with random init if pretrained weights cannot load')
    parser.add_argument('--timm_adapter', default='lite_fpn', choices=('pet_fpn', 'lite_fpn', 'rcc_fpn', 'direct', 'fpn'),
                        help='adapter used to map timm features into PET 4x/8x features')
    parser.add_argument('--timm_output_norm', default='gn', choices=('gn', 'none'),
                        help='normalization after timm feature adapter; gn preserves old timm behavior, none is VGG-like')
    parser.add_argument('--scale_fusion', default='none', choices=('none', 'bidirectional'),
                        help='identity-initialized exchange between PET 4x detail and 8x context features')
    parser.add_argument('--scale_fusion_activation', default='gelu', choices=('relu', 'gelu'),
                        help='activation used by bidirectional scale fusion')
    parser.add_argument('--encoder_context_fusion', default='none', choices=('none', 'detail_to_context'),
                        help='zero-init 4x-detail to 8x-context fusion before PET encoder')
    parser.add_argument('--encoder_context_fusion_activation', default='gelu', choices=('relu', 'gelu'),
                        help='activation used by encoder context fusion')
    parser.add_argument('--encoder_context_fusion_gate', default='channel_spatial', choices=('none', 'channel_spatial'),
                        help='gating used by encoder context fusion residual')
    parser.add_argument('--fusion_mhf_mode', default='none', choices=('none', 'cem', 'cem_msem', 'full'),
                        help='VGG FPN high-level feature attention ablation inspired by VMambaCC MHF')
    parser.add_argument('--fusion_mhf_heads', default=1, type=int,
                        help='number of spatial heads for --fusion_mhf_mode cem_msem/full')
    parser.add_argument('--fusion_mhf_position', default='before', choices=('before', 'post'),
                        help='apply high-level feature multiplication before or after FPN fusion')
    parser.add_argument('--fusion_mhf_strength', default=1.0, type=float,
                        help='residual gate strength for VGG MHF-style feature fusion')
    parser.add_argument('--fusion_mhf_activation', default='gelu', choices=('relu', 'gelu'),
                        help='activation used by VGG MHF-style feature fusion')
    parser.add_argument('--fusion_mhf_impl', default='residual', choices=('residual', 'vmambacc'),
                        help='residual keeps PET-safe zero-init gates; vmambacc follows the VMambaCC MHF equations')
    parser.add_argument('--fusion_fpn_type', default='fpn', choices=('fpn', 'hs2fpn'),
                        help='fpn preserves original PET fusion; hs2fpn labels VMambaCC-style high-level guided fusion')
    parser.add_argument('--fusion_mhf_reduction', default=4, type=int,
                        help='channel bottleneck reduction for --fusion_mhf_impl vmambacc')
    parser.add_argument('--fusion_mhf_norm', default='none', choices=('none', 'bn', 'gn'),
                        help='optional norm inside VMambaCC-style CEM/HCEM conv stacks')
    parser.add_argument('--fusion_mhf_spatial_kernel', default=7, type=int,
                        help='spatial kernel size for VMambaCC-style MSEM')
    parser.add_argument('--fusion_mhf_output_activation', default='none', choices=('none', 'sigmoid'),
                        help='none follows the VMambaCC paper equation; sigmoid is a bounded ablation')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    # - transformer
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--transformer_activation', default='relu', choices=('relu', 'gelu'),
                        help='activation used in transformer feed-forward blocks')
    parser.add_argument('--transformer_norm_style', default='post', choices=('post', 'pre'),
                        help='post matches official PET; pre is more stable for deeper transformer stacks')
    parser.add_argument('--decoder_attention', default='softmax', choices=('softmax', 'linear'),
                        help='attention used inside decoder layers; softmax matches official PET')
    parser.add_argument('--decoder_memory_halo', default=0, type=int,
                        help='extra 8x encoder-feature tokens around each decoder cross-attention memory window')
    parser.add_argument('--decoder_global_context', action='store_true',
                        help='enable image-level global context in decoder memory windows')
    parser.add_argument('--decoder_global_context_mode', default='residual', choices=('residual', 'token'),
                        help='residual is identity-initialized and safer; token appends a competing cross-attention token')
    parser.add_argument('--enc_win_sizes', default='', type=str,
                        help='encoder window sizes as "w,h;w,h;..."; empty keeps paper PET defaults')
    parser.add_argument('--enc_shift_mode', default='none', choices=('none', 'swin'),
                        help='encoder window partition shift; swin alternates half-window shifted encoder layers')
    parser.add_argument('--sparse_dec_win_size', default='', type=str,
                        help='sparse decoder window size as "w,h"; empty keeps paper PET default')
    parser.add_argument('--dense_dec_win_size', default='', type=str,
                        help='dense decoder window size as "w,h"; empty keeps paper PET default')
    parser.add_argument('--context_patch_size', default='', type=str,
                        help='quadtree splitter context patch size as "w,h"; empty keeps paper PET default')
    parser.add_argument('--quad_context_mixer', default='none', choices=('none', 'lite'),
                        help='optional quadtree-aware context mixer after the encoder; none matches paper PET')
    parser.add_argument('--quad_context_levels', default=2, type=int,
                        help='number of quadtree parent pooling levels used by --quad_context_mixer lite')
    parser.add_argument('--quad_context_shift', default=1, type=int,
                        help='feature-token shift used for shifted parent context in --quad_context_mixer lite')
    parser.add_argument('--quad_context_mid_dim', default=128, type=int,
                        help='hidden channels for the quad context mixer gate')
    parser.add_argument('--quad_context_activation', default='gelu', choices=('relu', 'gelu'),
                        help='activation used by --quad_context_mixer lite')
    parser.add_argument('--perspective_mixer', default='none', choices=('none', 'drf'),
                        help='optional PANet-inspired dynamic receptive-field mixer after the encoder')
    parser.add_argument('--perspective_mixer_dilations', default='1,2,3', type=str,
                        help='comma-separated dilations used by --perspective_mixer drf')
    parser.add_argument('--perspective_mixer_mid_dim', default=64, type=int,
                        help='hidden channels for the dynamic receptive-field gate')
    parser.add_argument('--perspective_mixer_activation', default='gelu', choices=('relu', 'gelu'),
                        help='activation used by --perspective_mixer drf')
    parser.add_argument('--local_density_mixer', default='none', choices=('none', 'zip_ordinal'),
                        help='local ZIP count-distribution representation and zero-initialized PET feature residual')
    parser.add_argument('--local_density_block_size', default=16, type=int,
                        help='image-space block size for local density representation; must be divisible by encoder stride')
    parser.add_argument('--local_density_projection_dim', default=64, type=int,
                        help='embedding size for local ordinal density contrastive learning')
    parser.add_argument('--local_density_bin_centers',
                        default='1,2,3,4,5,6,7,8,9,10,11.38,13.38,16.26', type=str,
                        help='comma-separated positive block-count centers used by the ZIP rate head')
    parser.add_argument('--local_density_zero_prior', default=0.9, type=float,
                        help='initial structural-zero probability for local ZIP blocks')
    parser.add_argument('--local_zip_loss_coef', default=0.0, type=float,
                        help='weight for local zero-inflated Poisson NLL; 0 disables ZIP supervision')
    parser.add_argument('--local_zip_ce_coef', default=1.0, type=float,
                        help='positive count-bin CE multiplier inside local ZIP supervision')
    parser.add_argument('--local_zip_count_coef', default=0.1, type=float,
                        help='log image-count consistency multiplier inside local ZIP supervision')
    parser.add_argument('--local_ordinal_loss_coef', default=0.0, type=float,
                        help='weight for balanced ordinal local-density contrastive learning')
    parser.add_argument('--local_ordinal_temperature', default=0.1, type=float,
                        help='temperature for local ordinal contrastive similarity')
    parser.add_argument('--local_ordinal_edges', default='1,2,4,8', type=str,
                        help='positive-count boundaries defining ordinal local-density levels')
    parser.add_argument('--local_ordinal_max_per_level', default=64, type=int,
                        help='maximum sampled blocks per density level and training batch')
    parser.add_argument('--zip_count_loss_coef', default=0.0, type=float,
                        help='weight for blockwise EBC-ZIP count supervision; 0 disables the ZIP count branch')
    parser.add_argument('--zip_count_block_size', default=16, type=int,
                        help='image-space block size for the EBC-ZIP count branch; must be divisible by encoder stride')
    parser.add_argument('--zip_count_feature_source', default='encoder8x', choices=('encoder8x', 'fpn4x8x'),
                        help='features used by the EBC-ZIP branch: encoder8x or fused 4x-detail/8x-context')
    parser.add_argument('--zip_count_bin_centers',
                        default='1,2,3,4,5,6,7,8,9,10,11.38,13.38,16.26', type=str,
                        help='comma-separated positive block-count centers used by the ZIP count branch')
    parser.add_argument('--zip_count_zero_prior', default=0.9, type=float,
                        help='initial structural-zero probability for the ZIP count branch')
    parser.add_argument('--zip_count_ce_coef', default=1.0, type=float,
                        help='positive count-bin CE multiplier inside ZIP count supervision')
    parser.add_argument('--zip_count_count_coef', default=1.0, type=float,
                        help='global log-count consistency multiplier inside ZIP count supervision')
    parser.add_argument('--zip_count_start_epoch', default=0, type=int,
                        help='epoch when ZIP count supervision starts')
    parser.add_argument('--zip_count_end_epoch', default=-1, type=int,
                        help='epoch after which ZIP count supervision turns off; negative keeps it on')
    parser.add_argument('--zip_count_warmup_epochs', default=0, type=int,
                        help='linearly ramp ZIP count loss for this many epochs')
    parser.add_argument('--zip_count_feature_grad_scale', default=1.0, type=float,
                        help='scale gradients from ZIP count loss into PET encoder; 0 trains only the ZIP head')
    parser.add_argument('--splitter_head', default='pool', choices=('pool', 'conv'),
                        help='quadtree splitter head; pool matches official PET, conv adds local context')
    parser.add_argument('--splitter_hidden_dim', default=128, type=int,
                        help='hidden channels for --splitter_head conv')
    parser.add_argument('--splitter_activation', default='gelu', choices=('relu', 'gelu'),
                        help='activation used by --splitter_head conv')

    # loss parameters
    # - matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - loss coefficients
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)
    parser.add_argument('--point_loss_coef', default=5.0, type=float)
    parser.add_argument('--scale_point_loss_coef', default=0.0, type=float,
                        help='matched point loss normalized by per-head NWPU sigma; 0 disables it')
    parser.add_argument('--scale_point_sigma', default='small', choices=('small', 'large', 'geomean'),
                        help='head scale used by --scale_point_loss_coef')
    parser.add_argument('--scale_point_sigma_min', default=2.0, type=float,
                        help='minimum pixel scale used by scale-normalized point loss')
    parser.add_argument('--scale_point_sigma_max', default=128.0, type=float,
                        help='maximum pixel scale used by scale-normalized point loss')
    parser.add_argument('--quality_loss_coef', default=0.0, type=float,
                        help='quality-aware score calibration loss weight; 0 disables it')
    parser.add_argument('--quality_loss_sigma', default=16.0, type=float,
                        help='pixel sigma mapping matched localization error to score quality')
    parser.add_argument('--quality_loss_pos_floor', default=0.5, type=float,
                        help='minimum quality target assigned to matched positives')
    parser.add_argument('--quality_loss_bg_weight', default=0.1, type=float,
                        help='relative weight for unmatched-query quality negatives')
    parser.add_argument('--pq_sparse_coef', default=1.0, type=float,
                        help='branch multiplier for sparse base point-query CE/point losses')
    parser.add_argument('--pq_dense_coef', default=1.0, type=float,
                        help='branch multiplier for dense base point-query CE/point losses')
    parser.add_argument('--pq_dense_start_epoch', default=0, type=int,
                        help='epoch when dense base point-query loss starts')
    parser.add_argument('--pq_dense_warmup_epochs', default=0, type=int,
                        help='linearly ramp dense base point-query loss after --pq_dense_start_epoch')
    parser.add_argument('--branch_target_routing', default='none', choices=('none', 'gt_count'),
                        help='route base sparse/dense Hungarian targets by quadtree GT responsibility')
    parser.add_argument('--class_loss_type', default='ce', choices=('ce', 'focal'),
                        help='classification loss for person/background point-query logits')
    parser.add_argument('--focal_alpha', default=0.25, type=float,
                        help='person-class alpha for focal classification loss')
    parser.add_argument('--focal_gamma', default=2.0, type=float,
                        help='gamma for focal classification loss')
    parser.add_argument('--class_prior_prob', default=-1.0, type=float,
                        help='optional initial person prior for classification heads; <=0 keeps default init')
    parser.add_argument('--strict_model_checks', action='store_true',
                        help='run extra PET tensor finiteness checks during model forward')
    parser.add_argument('--count_loss_coef', default=0.0, type=float,
                        help='optional L1 loss on soft predicted count; 0 disables it')
    parser.add_argument('--count_loss_gate', default='detach', choices=('none', 'detach', 'soft', 'hard'),
                        help='routing gates used by count loss; detach calibrates scores without splitter gradients')
    parser.add_argument('--count_loss_type', default='log_l1', choices=('log_l1', 'l1', 'smooth_l1', 'over_log_l1'),
                        help='count-loss scale; log_l1 is safer early in training')
    parser.add_argument('--count_loss_budget_margin', default=1.0, type=float,
                        help='target-count multiplier for over_log_l1 count loss')
    parser.add_argument('--count_loss_start_epoch', default=-1, type=int,
                        help='epoch to enable count loss; negative uses warmup_epochs')
    parser.add_argument('--count_head_loss_coef', default=0.0, type=float,
                        help='separate global count-head loss weight; 0 disables it')
    parser.add_argument('--count_head_loss_type', default='log_l1', choices=('log_l1', 'l1', 'smooth_l1'),
                        help='loss scale for the separate count-head regressor')
    parser.add_argument('--count_head_start_epoch', default=0, type=int,
                        help='epoch when the separate count-head loss starts')
    parser.add_argument('--count_head_end_epoch', default=-1, type=int,
                        help='epoch after which the separate count-head loss turns off; negative keeps it on')
    parser.add_argument('--count_head_warmup_epochs', default=0, type=int,
                        help='linearly ramp the separate count-head loss after --count_head_start_epoch')
    parser.add_argument('--allow_count_head_fresh_train', action='store_true',
                        help='allow count-head auxiliary during fresh training; disabled by default after severe SHA over-counting')
    parser.add_argument('--allow_count_head_from_start', action='store_true',
                        help='deprecated: fresh count-head from epoch 0 is delayed unless --force_unsafe_count_head_from_start is also set')
    parser.add_argument('--force_unsafe_count_head_from_start', action='store_true',
                        help='dangerous ablation only: allow count-head auxiliary from epoch 0 during fresh training')
    parser.add_argument('--no_auto_freeze_bn_on_count_head_resume', action='store_true',
                        help='legacy reproduction only: do not auto-enable freeze_bn for count-head checkpoint fine-tuning')
    parser.add_argument('--safe_count_head_start_epoch', default=250, type=int,
                        help='auto-delay count-head auxiliary to this epoch for fresh training unless explicitly allowed')
    parser.add_argument('--count_head_init_count', default=40.0, type=float,
                        help='initial count prediction for a reference 256x256 crop in the separate count head')
    parser.add_argument('--count_head_init_cells', default=1024.0, type=float,
                        help='reference encoder-cell count for --count_head_init_count; 256/8 squared is 1024')
    parser.add_argument('--count_head_feature_grad_scale', default=1.0, type=float,
                        help='scale gradients from count/density auxiliaries into PET encoder; 0 trains only the head')
    parser.add_argument('--count_head_feature_grad_start_epoch', default=0, type=int,
                        help='epoch when count-head gradients may flow into PET features; before this, only the head is trained')
    parser.add_argument('--count_head_feature_grad_warmup_epochs', default=0, type=int,
                        help='linearly ramp count-head feature gradients after --count_head_feature_grad_start_epoch')
    parser.add_argument('--train_count_head_only', action='store_true',
                        help='freeze PET and train only the separate count head')
    parser.add_argument('--density_map_loss_coef', default=0.0, type=float,
                        help='spatial density-map auxiliary weight for the count head; 0 disables it')
    parser.add_argument('--allow_unstable_density_map_loss', action='store_true',
                        help='allow density-map auxiliary to train; disabled by default because it caused severe over-counting')
    parser.add_argument('--density_map_loss_type', default='log_l1', choices=('log_l1', 'l1', 'smooth_l1'),
                        help='loss scale for density-map auxiliary')
    parser.add_argument('--density_map_pos_weight', default=10.0, type=float,
                        help='extra weight on cells containing annotated points for density-map auxiliary')
    parser.add_argument('--density_map_grad_scale', default=1.0, type=float,
                        help='scale gradients from density-map auxiliary; scalar count-head loss is unaffected')
    parser.add_argument('--density_map_start_epoch', default=0, type=int,
                        help='epoch when density-map auxiliary starts')
    parser.add_argument('--density_map_end_epoch', default=-1, type=int,
                        help='epoch after which density-map auxiliary turns off; negative keeps it on')
    parser.add_argument('--foreground_loss_coef', default=0.0, type=float,
                        help='spatial foreground heatmap auxiliary weight; 0 disables it')
    parser.add_argument('--foreground_sigma', default=8.0, type=float,
                        help='pixel Gaussian sigma for the foreground heatmap target')
    parser.add_argument('--foreground_neg_shrink', default=16.0, type=float,
                        help='negative focal-loss shrink factor for foreground heatmap supervision')
    parser.add_argument('--foreground_init_prior', default=0.5, type=float,
                        help='initial foreground prior probability for the spatial foreground head')
    parser.add_argument('--region_count_loss_coef', default=0.0, type=float,
                        help='local region count calibration loss weight; 0 disables it')
    parser.add_argument('--region_count_grid', default=4, type=int,
                        help='number of y/x bins for local region count calibration')
    parser.add_argument('--region_count_gate', default='detach', choices=('none', 'detach', 'soft', 'hard'),
                        help='quadtree gate used by local region count calibration')
    parser.add_argument('--region_count_type', default='log_l1', choices=('log_l1', 'l1', 'smooth_l1'),
                        help='scale for local region count calibration')
    parser.add_argument('--region_count_start_epoch', default=-1, type=int,
                        help='epoch to enable local region count; negative uses warmup_epochs')
    parser.add_argument('--region_count_end_epoch', default=-1, type=int,
                        help='epoch after which local region count turns off; negative keeps it on')
    parser.add_argument('--branch_exclusion_loss_coef', default=0.0, type=float,
                        help='cross-branch sparse/dense mutual-exclusion loss weight; 0 disables it')
    parser.add_argument('--branch_exclusion_start_epoch', default=0, type=int,
                        help='epoch to enable cross-branch mutual exclusion')
    parser.add_argument('--branch_exclusion_end_epoch', default=-1, type=int,
                        help='epoch after which cross-branch mutual exclusion turns off; negative keeps it on')
    parser.add_argument('--bayesian_loss_coef', default=0.0, type=float,
                        help='optional point-level Bayesian expected-count loss weight; 0 disables it')
    parser.add_argument('--bayesian_sigma', default=8.0, type=float,
                        help='pixel Gaussian sigma for the Bayesian point-count auxiliary')
    parser.add_argument('--bayesian_bg_coef', default=0.05, type=float,
                        help='background suppression weight inside the Bayesian auxiliary')
    parser.add_argument('--bayesian_loss_gate', default='detach', choices=('none', 'detach', 'soft', 'hard'),
                        help='quadtree gate used by Bayesian auxiliary')
    parser.add_argument('--bayesian_start_epoch', default=-1, type=int,
                        help='epoch to enable Bayesian auxiliary; negative uses warmup_epochs')
    parser.add_argument('--bayesian_end_epoch', default=-1, type=int,
                        help='epoch after which Bayesian auxiliary turns off; negative keeps it on')
    parser.add_argument('--apg_loss_coef', default=0.0, type=float,
                        help='legacy nearest-grid-query guidance weight; this is not APGCC auxiliary-point sampling (use --ifi_loss_coef for full APG sampling)')
    parser.add_argument('--apg_pos_k', default=1, type=int,
                        help='nearest point queries per GT point supervised by APG')
    parser.add_argument('--apg_point_coef', default=5.0, type=float,
                        help='point-regression coefficient inside APG loss')
    parser.add_argument('--apg_bg_coef', default=0.0, type=float,
                        help='optional local background CE weight inside APG; 0 disables it')
    parser.add_argument('--apg_bg_k', default=0, type=int,
                        help='background point queries per GT used by APG local suppression')
    parser.add_argument('--apg_bg_min_dist', default=12.0, type=float,
                        help='minimum pixel distance from every GT for APG background samples')
    parser.add_argument('--apg_bg_offset_coef', default=0.0, type=float,
                        help='offset-to-zero coefficient for APG auxiliary negative points')
    parser.add_argument('--apg_local_neg_coef', default=0.0, type=float,
                        help='relative APGCC-style local negative CE weight inside APG; 0 disables it')
    parser.add_argument('--apg_local_neg_k', default=0, type=int,
                        help='local negative point queries per GT used by APGCC-style suppression')
    parser.add_argument('--apg_local_neg_min_dist', default=2.0, type=float,
                        help='minimum pixel distance from a GT for APGCC-style local negatives')
    parser.add_argument('--apg_local_neg_max_dist', default=8.0, type=float,
                        help='maximum pixel distance from a GT for APGCC-style local negatives; <=0 disables the upper bound')
    parser.add_argument('--apg_local_neg_offset_coef', default=1.0, type=float,
                        help='offset-to-zero coefficient for APGCC-style local negative points')
    parser.add_argument('--apg_start_epoch', default=0, type=int,
                        help='epoch when APG auxiliary supervision starts')
    parser.add_argument('--apg_warmup_epochs', default=0, type=int,
                        help='linearly ramp APG loss weight for this many epochs after --apg_start_epoch')
    parser.add_argument('--force_unsafe_apg_from_start', action='store_true',
                        help='dangerous ablation only: allow APG weight >0.1 from epoch 0 without warmup')
    parser.add_argument('--apg_sparse_coef', default=1.0, type=float,
                        help='branch multiplier for sparse APG inside --apg_loss_coef')
    parser.add_argument('--apg_dense_coef', default=1.0, type=float,
                        help='branch multiplier for dense APG inside --apg_loss_coef')
    parser.add_argument('--apg_dense_start_epoch', default=-1, type=int,
                        help='dense-branch APG start epoch; negative follows --apg_start_epoch')
    parser.add_argument('--apg_dense_warmup_epochs', default=-1, type=int,
                        help='dense-branch APG warmup epochs; negative follows --apg_warmup_epochs')
    parser.add_argument('--apg_end_epoch', default=-1, type=int,
                        help='epoch after which APG auxiliary supervision turns off; negative keeps it on')
    parser.add_argument('--apg_count_calibration', default='none', choices=('none', 'threshold', 'soft'),
                        help='modulate positive APG pressure by current PET count calibration')
    parser.add_argument('--apg_count_calibration_gate', default='detach', choices=('none', 'detach', 'soft', 'hard'),
                        help='split gate used when estimating PET count for APG count calibration')
    parser.add_argument('--apg_count_calibration_min', default=0.05, type=float,
                        help='minimum multiplier for positive APG terms under over-count')
    parser.add_argument('--apg_count_calibration_max', default=1.25, type=float,
                        help='maximum multiplier for positive APG terms under under-count')
    parser.add_argument('--apg_count_calibration_eps', default=1.0, type=float,
                        help='stability constant for APG count calibration ratio')
    parser.add_argument('--apg_contrastive_coef', default=0.0, type=float,
                        help='local APG contrastive margin loss weight inside APG; 0 disables it')
    parser.add_argument('--apg_neg_k', default=4, type=int,
                        help='nearby non-GT point queries per GT used as APG local negatives')
    parser.add_argument('--apg_margin', default=1.0, type=float,
                        help='person-logit margin between APG positive and local negative queries')
    parser.add_argument('--apg_consistency_coef', default=0.0, type=float,
                        help='neighboring APG proposal-point consistency weight; 0 disables it')
    parser.add_argument('--apg_consistency_k', default=4, type=int,
                        help='nearest queries per GT used by APG consistency')
    parser.add_argument('--apg_consistency_sigma', default=8.0, type=float,
                        help='pixel Gaussian sigma for APG consistency weights')
    parser.add_argument('--apg_soft_loss_coef', default=0.0, type=float,
                        help='Gaussian soft APG loss weight on PET inference logits; 0 disables it')
    parser.add_argument('--apg_soft_pos_k', default=4, type=int,
                        help='nearest point queries per GT used by Gaussian soft APG')
    parser.add_argument('--apg_soft_sigma', default=6.0, type=float,
                        help='pixel sigma for Gaussian soft APG classification targets')
    parser.add_argument('--apg_soft_point_coef', default=2.0, type=float,
                        help='point-regression coefficient inside Gaussian soft APG')
    parser.add_argument('--query_feature_interpolation', default='nearest', choices=('nearest', 'implicit'),
                        help='point-query feature extraction: PET nearest-cell lookup or APGCC-style implicit interpolation')
    parser.add_argument('--query_ifi_sharing', default='independent', choices=('independent', 'shared'),
                        help='share one implicit interpolator across sparse, dense, and auxiliary point paths')
    parser.add_argument('--query_ifi_feature_source', default='branch', choices=('branch', 'fpn4x8x'),
                        help='shared IFI input: native branch feature or identity-initialized 4x/8x fusion')
    parser.add_argument('--query_ifi_branch_scope', default='both', choices=('both', 'sparse', 'dense'),
                        help='PET branch whose normal grid queries use implicit interpolation')
    parser.add_argument('--query_ifi_residual', action='store_true',
                        help='learn shared IFI as a residual over PET native branch features')
    parser.add_argument('--query_ifi_residual_init', default=1e-3, type=float,
                        help='initial residual IFI contribution before tanh; must be non-negative')
    parser.add_argument('--ifi_interpolation', default='bilinear', choices=('bilinear', 'implicit'),
                        help='feature interpolation used by IFI auxiliary supervision')
    parser.add_argument('--ifi_feature_source', default='encoded', choices=('encoded', 'branch'),
                        help='IFI auxiliary source: encoded 8x context or the structural branch query features')
    parser.add_argument('--ifi_pos_dim', default=32, type=int,
                        help='relative positional encoding width for implicit IFI')
    parser.add_argument('--ifi_mlp_hidden_dim', default=256, type=int,
                        help='hidden width of the implicit IFI MLP; <=0 uses model hidden_dim')
    parser.add_argument('--ifi_activation', default='gelu', choices=('relu', 'gelu'),
                        help='activation used by implicit IFI MLP')
    parser.add_argument('--ifi_branch_scope', default='both', choices=('both', 'sparse', 'dense'),
                        help='PET branch supervised by arbitrary IFI/APG points')
    parser.add_argument('--ifi_loss_coef', default=0.0, type=float,
                        help='APGCC-style independent auxiliary-point guidance weight through interpolated features; 0 disables it')
    parser.add_argument('--ifi_head_source', default='separate', choices=('separate', 'sparse', 'dense', 'both', 'routed'),
                        help='prediction head used by IFI: separate auxiliary head, sparse PET head, dense PET head, both PET heads, or one routed PET head per point')
    parser.add_argument('--ifi_point_coef', default=1.0, type=float,
                        help='positive-to-GT and negative-to-zero offset coefficient inside full IFI/APG loss')
    parser.add_argument('--ifi_point_loss_type', default='smooth_l1', choices=('smooth_l1', 'mse'),
                        help='offset loss inside IFI/APG; the APGCC paper recipe uses mse')
    parser.add_argument('--ifi_balance_pos_neg', action='store_true',
                        help='average positive and negative APG groups separately, then sum them as in APGCC')
    parser.add_argument('--ifi_pos_k', default=1, type=int,
                        help='auxiliary positive points sampled per GT for IFI/APG guidance')
    parser.add_argument('--ifi_pos_radius', default=0.0, type=float,
                        help='maximum pixel displacement of IFI/APG positive points around each GT')
    parser.add_argument('--ifi_random_sampling', action='store_true',
                        help='randomize IFI/APG positive and negative points as in APGCC')
    parser.add_argument('--ifi_neg_k', default=4, type=int,
                        help='independent negative auxiliary points per GT for IFI/APG')
    parser.add_argument('--ifi_neg_radius', default=12.0, type=float,
                        help='maximum per-axis displacement for IFI/APG negative points')
    parser.add_argument('--ifi_neg_min_dist', default=4.0, type=float,
                        help='minimum per-axis random displacement and final distance from every GT for IFI/APG negatives')
    parser.add_argument('--ifi_negative_policy', default='filter', choices=('filter', 'paper'),
                        help='filter removes auxiliary negatives near any GT; paper keeps the exact sampled APG negatives')
    parser.add_argument('--ifi_start_epoch', default=0, type=int,
                        help='epoch when IFI-lite auxiliary supervision starts')
    parser.add_argument('--ifi_end_epoch', default=-1, type=int,
                        help='epoch after which IFI/APG supervision turns off; negative keeps it on')
    parser.add_argument('--qd_apg_loss_coef', default=0.0, type=float,
                        help='Quadtree-Dual APG loss weight; 0 disables it')
    parser.add_argument('--qd_apg_point_coef', default=5.0, type=float,
                        help='point-regression coefficient inside QD-APG loss')
    parser.add_argument('--qd_apg_suppress_coef', default=0.5, type=float,
                        help='background suppression coefficient for the non-routed PET branch')
    parser.add_argument('--qd_apg_start_epoch', default=0, type=int,
                        help='epoch when QD-APG auxiliary supervision starts')
    parser.add_argument('--qd_apg_end_epoch', default=-1, type=int,
                        help='epoch after which QD-APG turns off; negative keeps it on')
    parser.add_argument('--qd_apg_route_source', default='gt_count', choices=('gt_count', 'split_map'),
                        help='QD-APG branch teacher: GT local count target or live split map')
    parser.add_argument('--routed_apg_loss_coef', default=0.0, type=float,
                        help='split-responsibility APG loss weight for PET sparse/dense branches; 0 disables it')
    parser.add_argument('--routed_apg_point_coef', default=5.0, type=float,
                        help='point-regression coefficient inside split-responsibility APG')
    parser.add_argument('--routed_apg_pos_k', default=1, type=int,
                        help='nearest point queries per GT and branch used by split-responsibility APG')
    parser.add_argument('--routed_apg_bg_coef', default=0.0, type=float,
                        help='local background CE weight inside split-responsibility APG; 0 disables it')
    parser.add_argument('--routed_apg_bg_k', default=0, type=int,
                        help='background point queries per GT and branch used by split-responsibility APG')
    parser.add_argument('--routed_apg_bg_min_dist', default=12.0, type=float,
                        help='minimum pixel distance from every GT for routed APG background samples')
    parser.add_argument('--routed_apg_start_epoch', default=0, type=int,
                        help='epoch when split-responsibility APG starts')
    parser.add_argument('--routed_apg_end_epoch', default=-1, type=int,
                        help='epoch after which split-responsibility APG turns off; negative keeps it on')
    parser.add_argument('--routed_apg_warmup_epochs', default=0, type=int,
                        help='linearly ramp split-responsibility APG weight for this many epochs')
    parser.add_argument('--routed_apg_min_weight', default=0.1, type=float,
                        help='minimum branch responsibility so the non-primary branch is not starved')
    parser.add_argument('--routed_apg_source', default='gt_count', choices=('gt_count', 'split_map'),
                        help='responsibility teacher for split-responsibility APG')
    parser.add_argument('--routed_apg_gate', default='detach', choices=('detach', 'soft'),
                        help='whether split-map responsibilities backpropagate when --routed_apg_source split_map')
    parser.add_argument('--inheritance_loss_coef', default=0.0, type=float,
                        help='STEERER-style sparse/dense selective inheritance loss weight; 0 disables it')
    parser.add_argument('--inheritance_sparse_coef', default=1.0, type=float,
                        help='sparse parent occupancy term inside selective inheritance')
    parser.add_argument('--inheritance_dense_coef', default=1.0, type=float,
                        help='dense 2x2 child count term inside selective inheritance')
    parser.add_argument('--inheritance_consistency_coef', default=0.25, type=float,
                        help='simple-cell dense-child/sparse-parent consistency term')
    parser.add_argument('--inheritance_start_epoch', default=0, type=int,
                        help='epoch when selective inheritance starts')
    parser.add_argument('--inheritance_end_epoch', default=-1, type=int,
                        help='epoch after which selective inheritance turns off; negative keeps it on')
    parser.add_argument('--inheritance_gate', default='gt_count', choices=('gt_count', 'split_map'),
                        help='scale-selection teacher for selective inheritance')
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--pet_loss_variant', default='paper', choices=('paper', 'balanced'),
                        help='paper matches official PET; balanced enables experimental zero/negative-region losses')
    parser.add_argument('--split_loss_variant', default='auto', choices=('auto', 'none', 'paper', 'gt', 'paper_gt'),
                        help='split-map supervision: auto follows pet_loss_variant, none disables explicit split loss, paper uses PET min/max, gt uses per-cell GT BCE, paper_gt combines both')
    parser.add_argument('--negative_loss_coef', default=0.1, type=float,
                        help='extra scale for all-negative classification and split-map regions')
    parser.add_argument('--non_div_loss_coef', default=0.25, type=float,
                        help='auxiliary loss scale outside the current quadtree branch')
    parser.add_argument('--quadtree_loss_coef', default=0.1, type=float,
                        help='ground-truth split-map quality loss coefficient')
    parser.add_argument('--quadtree_prior_coef', default=0.025, type=float,
                        help='legacy image-level splitter prior coefficient')
    parser.add_argument('--split_count_threshold', default=2, type=int,
                        help='minimum people in a splitter cell before it should split')
    parser.add_argument('--split_pos_weight', default=1.0, type=float,
                        help='positive cell weight for quadtree quality loss')
    parser.add_argument('--split_threshold', default=0.5, type=float,
                        help='split-map threshold used to route sparse vs dense inference windows')
    parser.add_argument('--split_threshold_quantile', default=0.5, type=float,
                        help='legacy adaptive split-map quantile (unused when split_threshold is set)')
    parser.add_argument('--query_prune_threshold', default=0.5, type=float,
                        help='fixed PET decoder-window pruning threshold; independent from eval split threshold')
    parser.add_argument('--score_threshold', default=0.5, type=float,
                        help='point classification threshold; negative enables adaptive score thresholding')
    parser.add_argument('--eval_nms_radius', default=0.0, type=float,
                        help='optional eval-only point NMS radius in pixels; 0 disables duplicate suppression and matches PET')
    parser.add_argument('--eval_branch_gate', default='none', choices=('none', 'query', 'pred'),
                        help='eval-only split-aware sparse/dense ownership gate; none keeps PET concatenation')
    parser.add_argument('--eval_soft_split_gate', default='none', choices=('none', 'query', 'pred'),
                        help='eval-only soft split responsibility multiplied into person scores before thresholding')
    parser.add_argument('--eval_foreground_gate', default='none', choices=('none', 'query', 'pred'),
                        help='sample the foreground head at query/pred points during evaluation')
    parser.add_argument('--eval_foreground_gate_mode', default='suppress', choices=('suppress', 'logit_add'),
                        help='suppress only reduces point scores; logit_add is the older experimental additive prior')
    parser.add_argument('--eval_foreground_gate_strength', default=0.75, type=float,
                        help='foreground gate strength during evaluation')
    parser.add_argument('--eval_count_mode', default='threshold', choices=('threshold', 'count_head_topk'),
                        help='threshold keeps PET behavior; count_head_topk keeps top-K APG candidates using the separate count head')
    parser.add_argument('--eval_count_source', default='pet', choices=('pet', 'zip', 'zip_pet_blend'),
                        help='count used for MAE/RMSE: pet counts thresholded point predictions; zip sums the EBC-ZIP count branch; zip_pet_blend mixes both')
    parser.add_argument('--eval_count_blend_alpha', default=0.5, type=float,
                        help='ZIP weight for --eval_count_source zip_pet_blend; 0=PET count, 1=ZIP count')
    parser.add_argument('--eval_count_head_min_score', default=0.5, type=float,
                        help='minimum candidate score before count-head top-K selection')
    parser.add_argument('--eval_dense_start_epoch', default=0, type=int,
                        help='skip dense branch predictions during eval before this epoch')
    parser.add_argument('--eval_dense_residual_mode', default='none', choices=('none', 'count_head'),
                        help='before dense eval starts, optionally add top dense candidates only for count-head residual')
    parser.add_argument('--eval_dense_residual_start_epoch', default=0, type=int,
                        help='first eval epoch where residual dense top-up is allowed')
    parser.add_argument('--eval_dense_residual_min_score', default=0.0, type=float,
                        help='minimum dense score allowed for residual dense top-up')
    parser.add_argument('--eval_score_calibration', default='none', choices=('none', 'count_head_bias'),
                        help='eval-only score calibration; count_head_bias shifts person logits to match the scalar count head')
    parser.add_argument('--eval_score_calibration_strength', default=1.0, type=float,
                        help='fraction of the count-head logit bias applied during eval score calibration')
    parser.add_argument('--eval_score_calibration_start_epoch', default=0, type=int,
                        help='first eval epoch where score calibration is allowed')
    parser.add_argument('--eval_score_calibration_min_bias', default=-8.0, type=float,
                        help='minimum person-logit bias used by eval score calibration')
    parser.add_argument('--eval_score_calibration_max_bias', default=8.0, type=float,
                        help='maximum absolute person-logit bias used by eval score calibration')
    parser.add_argument('--eval_score_calibration_count_blend', default=1.0, type=float,
                        help='blend between PET expected count and count-head target for score calibration; 1 uses count head only')
    parser.add_argument('--eval_score_calibration_count_ratio_min', default=0.0, type=float,
                        help='minimum calibrated target as a ratio of PET expected count')
    parser.add_argument('--eval_score_calibration_count_ratio_max', default=1e6, type=float,
                        help='maximum calibrated target as a ratio of PET expected count')
    parser.add_argument('--no_eval_filter_invalid_points', action='store_true',
                        help='disable eval filtering of predicted points outside the real non-padded image area')
    parser.add_argument('--eval_debug_counting', action='store_true',
                        help='log sparse/dense query counts after each eval counting filter')
    parser.add_argument('--no_localization_metrics', action='store_true',
                        help='disable localization F1/precision/recall metrics during evaluation')
    parser.add_argument('--localization_large_threshold', default=8.0, type=float,
                        help='large pixel-distance threshold for localization F1/precision/recall')
    parser.add_argument('--localization_small_threshold', default=4.0, type=float,
                        help='small pixel-distance threshold for localization F1/precision/recall')
    parser.add_argument('--localization_protocol', default='fixed',
                        choices=('fixed', 'target_sigma', 'adaptive_nn'),
                        help='fixed uses pixel radii; target_sigma uses dataset per-GT sigma; adaptive_nn derives sigma from nearest GT distance')
    parser.add_argument('--localization_large_scale', default=1.0, type=float,
                        help='nearest-neighbor multiplier for adaptive large-threshold localization')
    parser.add_argument('--localization_small_scale', default=0.5, type=float,
                        help='nearest-neighbor multiplier for adaptive small-threshold localization')
    parser.add_argument('--eval_protocol', default='pet', choices=('pet', 'crowd_no_overlap'),
                        help='validation protocol used during training')
    parser.add_argument('--eval_tile_size', default=0, type=int,
                        help='positive value enables tiled eval for images larger than this size')
    parser.add_argument('--eval_tile_overlap', default=0, type=int,
                        help='pixel overlap between eval tiles; 0 gives non-overlap tiling')
    parser.add_argument('--eval_tile_nms_radius', default=0.0, type=float,
                        help='optional cross-tile NMS radius in pixels for overlapped tiled eval')
    parser.add_argument('--eval_tile_min_gt', default=0, type=int,
                        help='prohibited legacy oracle option; use a prediction/area trigger instead')
    parser.add_argument('--eval_tile_max_tiles', default=0, type=int,
                        help='skip tiled eval when the tile grid would exceed this many tiles; 0 disables limit')
    parser.add_argument('--eval_tile_trigger_count', default=0.0, type=float,
                        help='tile only when the normal full-image pass predicts at least this many people; 0 disables this trigger')
    parser.add_argument('--eval_tile_trigger_area', default=0, type=int,
                        help='tile only when valid image area is at least this many pixels; 0 disables this trigger')

    # dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="", type=str)
    parser.add_argument('--patch_size', default=256, type=int,
                        help='training crop size for crop-based crowd datasets')
    parser.add_argument('--patch_size_choices', default='', type=str,
                        help='comma-separated training crop sizes sampled per image; empty uses --patch_size')
    parser.add_argument('--crop_attempts', default=1, type=int,
                        help='number of random crop candidates tried per positive training image')
    parser.add_argument('--min_crop_points', default=0, type=int,
                        help='minimum people desired in a positive training crop')
    parser.add_argument('--eval_max_size', default=-1, type=int,
                        help='high-resolution dataset long-side cap applied before training crops and evaluation; -1 selects the published default (QNRF 1536, JHU/NWPU 2048), 0 disables resizing')
    parser.add_argument('--nwpu_eval_split', default='val', choices=('val', 'test', 'train'),
                        help='NWPU split used for validation/evaluation')
    parser.add_argument('--jhu_eval_split', default='val', choices=('val', 'test', 'train'),
                        help='JHU-Crowd++ split used for validation/evaluation')
    parser.add_argument('--ucfcc50_fold', default=0, type=int, choices=range(5),
                        help='held-out UCF-CC-50 fold index (0-4)')
    parser.add_argument('--ucfcc50_fold_seed', default=42, type=int,
                        help='seed used to create five UCF-CC-50 folds when no manifest is supplied')
    parser.add_argument('--ucfcc50_fold_manifest', default='', type=str,
                        help='JSON file containing the exact five UCF-CC-50 folds')
    parser.add_argument('--validation_protocol', default='auto',
                        choices=('auto', 'official_val', 'benchmark_test', 'train_holdout', 'final_test_once'),
                        help='checkpoint-selection protocol; auto uses train_holdout for SHA/SHB/QNRF and official_val for NWPU/JHU')
    parser.add_argument('--allow_benchmark_test_selection', action='store_true',
                        help='legacy-only override allowing repeated checkpoint selection on a benchmark test split')
    parser.add_argument('--train_holdout_fraction', default=0.1, type=float,
                        help='fraction of the training split reserved for checkpoint selection under train_holdout validation')
    parser.add_argument('--train_holdout_seed', default=42, type=int,
                        help='seed for deterministic train_holdout partitioning')
    parser.add_argument('--partial_annotation_ratio', default=1.0, type=float,
                        help='Shanghai train-only fixed annotated-region ratio in (0,1]; 1 keeps full supervision')
    parser.add_argument('--partial_annotation_seed', default=0, type=int,
                        help='seed for deterministic per-image partial annotation rectangles')
    parser.add_argument('--partial_annotation_height_ratio', default=0.5, type=float,
                        help='preferred height fraction of each fixed partial annotation rectangle')
    parser.add_argument('--annotation_override_dir', default='', type=str,
                        help='directory of complete GT_<image>.mat training annotations produced by refinement or partial-label completion')
    parser.add_argument('--nwpu_sigma_mode', default='official', choices=('area', 'diag', 'min_diag', 'official'),
                        help='fallback localization sigma derived from NWPU boxes when annotation sigma is absent')
    parser.add_argument('--nwpu_dense_crop_prob', default=0.0, type=float,
                        help='NWPU train only: probability of choosing the densest crop among random candidates')
    parser.add_argument('--nwpu_dense_crop_attempts', default=16, type=int,
                        help='NWPU train only: candidates for dense crop selection')
    parser.add_argument('--train_count_weight_power', default=0.0, type=float,
                        help='sample training images with weight (count+1)^power; 0 keeps uniform sampling')
    parser.add_argument('--train_count_weight_max', default=8.0, type=float,
                        help='maximum per-image sampling weight when --train_count_weight_power is enabled')

    # misc parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_model_only', action='store_true',
                        help='load only model weights from --resume and reset optimizer/scheduler/epoch counters')
    parser.add_argument('--resume_allow_arch_change', action='store_true',
                        help='with --resume_model_only, allow explicitly passed architecture flags to override checkpoint args')
    parser.add_argument('--allow_output_overwrite', action='store_true',
                        help='explicitly permit writing a resumed run into an unrelated non-empty output directory')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--eval_start_epoch', default=0, type=int,
                        help='skip validation before this epoch; useful for unstable from-scratch calibration warmup')
    parser.add_argument('--eval_model', default='auto', choices=('auto', 'raw', 'ema'),
                        help='model weights used for validation/checkpoint selection; auto uses EMA when enabled')
    parser.add_argument('--eval_before_train', action='store_true',
                        help='run validation once before the first training epoch')
    parser.add_argument('--no_abort_on_bad_count', action='store_true',
                        help='do not stop training when validation predicted count is catastrophically far from GT')
    parser.add_argument('--bad_count_ratio_max', default=2.0, type=float,
                        help='abort training if pred/gt or gt/pred exceeds this ratio and MAE is also high')
    parser.add_argument('--bad_count_mae_min', default=200.0, type=float,
                        help='minimum validation MAE needed before bad-count auto-abort can trigger')
    parser.add_argument('--bad_count_start_epoch', default=300, type=int,
                        help='first epoch where bad-count auto-abort is allowed')
    parser.add_argument('--bad_count_direction', default='all', choices=('all', 'over', 'under'),
                        help='which catastrophic count direction can trigger auto-abort')
    parser.add_argument('--syn_bn', default=0, type=int)
    parser.add_argument('--ema_decay', default=0.0, type=float,
                        help='exponential moving average decay for eval/checkpointing; 0 disables EMA')
    parser.add_argument('--deterministic', dest='deterministic', action='store_true', default=True,
                        help='enable deterministic CuDNN and seeded DataLoader workers')
    parser.add_argument('--no_deterministic', dest='deterministic', action='store_false',
                        help='disable deterministic CuDNN mode')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def get_explicit_arg_names(argv):
    names = set()
    for token in argv:
        if not token.startswith('--'):
            continue
        name = token[2:].split('=', 1)[0].replace('-', '_')
        if name.startswith('no_') and name[3:] == 'deterministic':
            names.add('deterministic')
        else:
            names.add(name)
    return names


def apply_backbone_recipe(args):
    """Apply backbone-specific fine-tuning defaults when the user left a generic setting in place."""
    recipe = get_backbone_recipe(args.backbone)
    if recipe is None:
        return

    explicit_args = set(getattr(args, '_explicit_args', set()))
    for key, tuned_value in recipe.items():
        if key in explicit_args:
            continue
        default_value = BASE_TRAINING_DEFAULTS.get(key)
        current_value = getattr(args, key, default_value)
        if default_value is not None and current_value == default_value:
            setattr(args, key, tuned_value)


def apply_model_recipe(args):
    recipe_name = getattr(args, 'model_recipe', 'none')
    if recipe_name == 'none':
        return
    if (
        recipe_name in EXPERIMENTAL_MODEL_RECIPES
        and not bool(getattr(args, 'allow_experimental_model_recipe', False))
    ):
        raise ValueError(
            f'model_recipe={recipe_name!r} is experimental and blocked by default '
            'because it is unvalidated or previously regressed counting in this '
            'repository. Use --allow_experimental_model_recipe only for a '
            'declared ablation.'
        )
    recipe = MODEL_RECIPES[recipe_name]
    explicit_args = set(getattr(args, '_explicit_args', set()))
    for key, value in recipe.items():
        if key in explicit_args:
            continue
        setattr(args, key, value)


def validate_partial_annotation_contract(args):
    ratio = float(getattr(args, 'partial_annotation_ratio', 1.0))
    if not 0.0 < ratio <= 1.0:
        raise ValueError('--partial_annotation_ratio must be in (0, 1]')
    if getattr(args, 'annotation_override_dir', '') and ratio < 1.0:
        raise ValueError(
            '--annotation_override_dir contains completed/refined full-image '
            'labels and must be trained with --partial_annotation_ratio 1.0'
        )
    if ratio >= 1.0:
        return
    if getattr(args, 'dataset_file', '') not in ('SHA', 'SHB'):
        raise ValueError(
            'fixed partial-region supervision is currently implemented only '
            'for ShanghaiTech SHA/SHB'
        )

    incompatible = {
        'count_loss_coef': getattr(args, 'count_loss_coef', 0.0),
        'region_count_loss_coef': getattr(args, 'region_count_loss_coef', 0.0),
        'count_head_loss_coef': getattr(args, 'count_head_loss_coef', 0.0),
        'density_map_loss_coef': getattr(args, 'density_map_loss_coef', 0.0),
        'zip_count_loss_coef': getattr(args, 'zip_count_loss_coef', 0.0),
        'local_zip_loss_coef': getattr(args, 'local_zip_loss_coef', 0.0),
        'foreground_loss_coef': getattr(args, 'foreground_loss_coef', 0.0),
        'apg_soft_loss_coef': getattr(args, 'apg_soft_loss_coef', 0.0),
        'qd_apg_loss_coef': getattr(args, 'qd_apg_loss_coef', 0.0),
        'routed_apg_loss_coef': getattr(args, 'routed_apg_loss_coef', 0.0),
        'inheritance_loss_coef': getattr(args, 'inheritance_loss_coef', 0.0),
    }
    enabled = [
        name for name, value in incompatible.items()
        if float(value or 0.0) > 0.0
    ]
    if enabled:
        raise ValueError(
            'partial-region training cannot use losses whose targets include '
            'the unannotated image area: ' + ', '.join(enabled)
        )


def is_safe_fresh_count_head(args):
    """Count head can start at epoch 0 only when it cannot corrupt PET logits."""
    bounded_count_bias = (
        getattr(args, 'eval_score_calibration', 'none') != 'count_head_bias'
        or (
            0.0 <= float(getattr(args, 'eval_score_calibration_strength', 1.0)) <= 0.5
            and -1.0 <= float(getattr(args, 'eval_score_calibration_min_bias', -8.0))
            and float(getattr(args, 'eval_score_calibration_max_bias', 8.0)) <= 1.0
            and 0.0 <= float(getattr(args, 'eval_score_calibration_count_blend', 1.0)) <= 0.5
            and 0.75 <= float(getattr(args, 'eval_score_calibration_count_ratio_min', 0.0))
            and float(getattr(args, 'eval_score_calibration_count_ratio_max', 1e6)) <= 1.25
        )
    )
    return (
        float(getattr(args, 'count_head_loss_coef', 0.0)) <= 0.25
        and float(getattr(args, 'count_head_feature_grad_scale', 1.0)) == 0.0
        and (int(getattr(args, 'count_head_warmup_epochs', 0)) >= 50 or bounded_count_bias)
        and float(getattr(args, 'density_map_loss_coef', 0.0)) == 0.0
        and getattr(args, 'eval_count_mode', 'threshold') == 'threshold'
    )


def sanitize_unstable_training_args(args):
    """Disable known-unstable experimental auxiliaries unless explicitly allowed."""
    explicit_args = set(getattr(args, '_explicit_args', set()))
    recipe_name = getattr(args, 'model_recipe', 'none')
    density_coef = float(getattr(args, 'density_map_loss_coef', 0.0))
    if density_coef > 0 and not bool(getattr(args, 'allow_unstable_density_map_loss', False)):
        print(
            'WARNING: --density_map_loss_coef was requested but is disabled by default. '
            'Recent SHA runs showed severe over-counting from this auxiliary. '
            'Use --allow_unstable_density_map_loss only for isolated debugging runs.'
        )
        args.density_map_loss_coef = 0.0
    count_coef = float(getattr(args, 'count_head_loss_coef', 0.0))
    count_start = int(getattr(args, 'count_head_start_epoch', 0))
    fresh_train = not bool(getattr(args, 'resume', ''))
    if recipe_name == 'vgg_apglc_density_counthead_ft_legacy' and fresh_train:
        raise ValueError(
            'model_recipe=vgg_apglc_density_counthead_ft_legacy is a recovery fine-tune recipe and requires '
            '--resume outputs/SHA/vgg16_bn_drop700_apg_lc_seed42/best_checkpoint.pth with --resume_model_only. '
            'Do not use it for fresh scratch training; it can reproduce the old count-head over-count failure.'
        )
    if count_coef > 0 and fresh_train and not bool(getattr(args, 'allow_count_head_fresh_train', False)):
        print(
            'WARNING: --count_head_loss_coef was requested for fresh training but is disabled by default. '
            'Recent SHA runs showed severe over-counting from count-head training before PET is calibrated. '
            'Use --allow_count_head_fresh_train only for isolated ablations.'
        )
        args.count_head_loss_coef = 0.0
    elif (
        count_coef > 0
        and count_start <= 0
        and fresh_train
        and not bool(getattr(args, 'force_unsafe_count_head_from_start', False))
    ):
        if is_safe_fresh_count_head(args):
            print(
                'Using safe detached count head: feature gradients are disabled, '
                'density-map supervision is off, and eval count calibration is bounded.'
            )
        else:
            delayed_start = max(1, int(getattr(args, 'safe_count_head_start_epoch', 250)))
            print(
                'WARNING: count-head auxiliary from epoch 0 is disabled for fresh training. '
                f'Setting count_head_start_epoch={delayed_start}. '
                'Use --force_unsafe_count_head_from_start only for isolated debugging runs.'
            )
            args.count_head_start_epoch = delayed_start
    if (
        count_coef > 0
        and bool(getattr(args, 'resume_model_only', False))
        and bool(getattr(args, 'resume', ''))
        and not bool(getattr(args, 'freeze_bn', False))
        and not bool(getattr(args, 'no_auto_freeze_bn_on_count_head_resume', False))
    ):
        print(
            'WARNING: enabling --freeze_bn for count-head fine-tuning from a checkpoint. '
            'Leaving BatchNorm in train mode can destroy PET score calibration in a few epochs.'
        )
        args.freeze_bn = True

    apg_coef_for_guard = float(getattr(args, 'apg_loss_coef', 0.0))
    if fresh_train and apg_coef_for_guard > 0.1 and int(getattr(args, 'apg_warmup_epochs', 0)) <= 0:
        if 'apg_loss_coef' in explicit_args and not bool(getattr(args, 'force_unsafe_apg_from_start', False)):
            raise ValueError(
                'Refusing fresh training with explicit --apg_loss_coef > 0.1 and no APG warmup. '
                'This repo repeatedly produced severe SHA over-counting in that regime; the '
                'verified vgg16_bn_drop700_apg_lc_seed42 checkpoint used apg_loss_coef=0.02. '
                'Use --apg_loss_coef 0.02, add --apg_warmup_epochs, or pass '
                '--force_unsafe_apg_from_start only for an isolated ablation.'
            )
        elif 'apg_loss_coef' in explicit_args:
            print(
                'WARNING: unsafe ablation accepted: explicit --apg_loss_coef > 0.1 with no APG warmup. '
                'Expect severe SHA over-counting unless this is a controlled diagnostic run.'
            )
        else:
            print(
                'WARNING: fresh APG+LC recipe requested apg_loss_coef > 0.1 with no warmup. '
                'The verified vgg16_bn_drop700_apg_lc_seed42 checkpoint used apg_loss_coef=0.02; '
                'setting apg_loss_coef=0.02. Pass --apg_loss_coef explicitly to override.'
            )
            args.apg_loss_coef = 0.02

    fresh_timm_train = fresh_train and is_timm_backbone(getattr(args, 'backbone', ''))
    if fresh_timm_train:
        class_prior_prob = float(getattr(args, 'class_prior_prob', -1.0))
        if class_prior_prob <= 0 and 'class_prior_prob' not in explicit_args:
            print(
                'WARNING: fresh timm/PET training without a foreground prior is prone to query explosion. '
                'Setting class_prior_prob=0.0023. Pass --class_prior_prob explicitly to override.'
            )
            args.class_prior_prob = 0.0023
        elif class_prior_prob > 0.02:
            print(
                'WARNING: class_prior_prob is high for PET full-image query counts on SHA. '
                'Values around 0.0023 are safer for fresh timm backbones.'
            )
        if (
            getattr(args, 'split_loss_variant', 'paper') in ('gt', 'paper_gt')
            and int(getattr(args, 'split_count_threshold', 2)) < 4
        ):
            if 'split_count_threshold' in explicit_args:
                print(
                    'WARNING: split_count_threshold<4 with GT split supervision can mark most SHA '
                    'split cells as dense and cause dense-branch over-counting.'
                )
            else:
                print(
                    'WARNING: fresh timm/PET GT split supervision with split_count_threshold=2 is '
                    'dense-biased on SHA. Setting split_count_threshold=5.'
                )
                args.split_count_threshold = 5
        if (
            float(getattr(args, 'score_threshold', 0.5)) < 0.3
            and 'score_threshold' in explicit_args
        ):
            print(
                'WARNING: score_threshold<0.3 is diagnostic-only for PET full-image evaluation and '
                'can turn mild dense-score drift into catastrophic over-counting.'
            )
        if getattr(args, 'class_loss_type', 'ce') == 'focal':
            focal_alpha = float(getattr(args, 'focal_alpha', 0.25))
            if focal_alpha > 0.5 and 'focal_alpha' not in explicit_args:
                print(
                    'WARNING: fresh timm/PET focal loss with focal_alpha>0.5 can over-amplify '
                    'foreground queries on SHA. Keeping the explicit focal mode, but setting '
                    'focal_alpha=0.25 unless you pass --focal_alpha explicitly.'
                )
                args.focal_alpha = 0.25
        apg_coef = float(getattr(args, 'apg_loss_coef', 0.0))
        apg_bg_coef = float(getattr(args, 'apg_bg_coef', 0.0))
        apg_bg_k = int(getattr(args, 'apg_bg_k', 0))
        if (
            apg_coef > 0
            and apg_bg_coef <= 0
            and apg_bg_k <= 0
            and 'apg_bg_coef' not in explicit_args
            and 'apg_bg_k' not in explicit_args
        ):
            print(
                'WARNING: positive-only APG on a fresh timm backbone caused severe SHA over-counting. '
                'Enabling APG local background negatives: apg_bg_coef=0.5, apg_bg_k=8. '
                'Pass --apg_bg_coef/--apg_bg_k explicitly to override.'
            )
            args.apg_bg_coef = 0.5
            args.apg_bg_k = 8
    return args


def should_abort_for_bad_count(args, epoch, test_stats):
    if bool(getattr(args, 'no_abort_on_bad_count', False)):
        return False, ''
    if epoch < int(getattr(args, 'bad_count_start_epoch', 300)):
        return False, ''
    pred_cnt = float(test_stats.get('pred_cnt', 0.0))
    gt_cnt = float(test_stats.get('gt_cnt', 0.0))
    mae = float(test_stats.get('mae', 0.0))
    if gt_cnt <= 0 or pred_cnt < 0:
        return False, ''
    ratio_limit = max(float(getattr(args, 'bad_count_ratio_max', 2.0)), 1.0)
    mae_limit = max(float(getattr(args, 'bad_count_mae_min', 200.0)), 0.0)
    over_ratio = pred_cnt / max(gt_cnt, 1e-6)
    under_ratio = gt_cnt / max(pred_cnt, 1e-6) if pred_cnt > 0 else float('inf')
    direction_mode = getattr(args, 'bad_count_direction', 'all')
    over_bad = mae >= mae_limit and over_ratio >= ratio_limit
    under_bad = mae >= mae_limit and under_ratio >= ratio_limit
    if direction_mode == 'over':
        should_abort = over_bad
        direction = 'over-count'
        bad_ratio = over_ratio
    elif direction_mode == 'under':
        should_abort = under_bad
        direction = 'under-count'
        bad_ratio = under_ratio
    else:
        should_abort = over_bad or under_bad
        direction = 'over-count' if over_ratio >= under_ratio else 'under-count'
        bad_ratio = max(over_ratio, under_ratio)
    if should_abort:
        message = (
            f'bad-count guard triggered: {direction} '
            f'pred_cnt={pred_cnt:.4f} gt_cnt={gt_cnt:.4f} '
            f'ratio={bad_ratio:.3f} mae={mae:.4f} '
            f'(limits: ratio>={ratio_limit:.3f}, mae>={mae_limit:.4f})'
        )
        return True, message
    return False, ''


def select_eval_model(model, model_without_ddp, model_ema, args):
    """Return the module used for validation and its label."""
    mode = getattr(args, 'eval_model', 'auto')
    if mode == 'raw':
        return model, 'raw'
    if mode == 'ema':
        if model_ema is None:
            print('WARNING: --eval_model ema requested but EMA is disabled; falling back to raw model.')
            return model, 'raw'
        return model_ema.module, 'ema'
    if model_ema is not None:
        return model_ema.module, 'ema'
    return model, 'raw'


def best_state_for_eval_model(model_without_ddp, model_ema, eval_model_name):
    if eval_model_name == 'ema' and model_ema is not None:
        return model_ema.state_dict(), True
    return model_without_ddp.state_dict(), model_ema is not None


def merge_checkpoint_args(args, checkpoint):
    checkpoint_args = checkpoint.get('args')
    if checkpoint_args is None:
        return args
    if isinstance(checkpoint_args, dict):
        checkpoint_args = argparse.Namespace(**checkpoint_args)

    merged = argparse.Namespace(**vars(checkpoint_args))
    for key, value in vars(args).items():
        if not hasattr(merged, key):
            setattr(merged, key, value)
    explicit_args = set(getattr(args, '_explicit_args', set()))
    explicit_only_runtime_keys = {
        'model_recipe', 'allow_experimental_model_recipe', 'auto_backbone_recipe',
        'eval_protocol', 'eval_max_size', 'nwpu_sigma_mode',
        'validation_protocol', 'train_holdout_fraction', 'train_holdout_seed',
        'allow_benchmark_test_selection',
        'allow_output_overwrite',
        'patch_size', 'patch_size_choices', 'crop_attempts', 'min_crop_points',
        'nwpu_dense_crop_prob', 'nwpu_dense_crop_attempts',
        'train_count_weight_power', 'train_count_weight_max',
        'no_localization_metrics', 'localization_large_threshold', 'localization_small_threshold',
        'localization_protocol', 'localization_large_scale', 'localization_small_scale',
        'eval_count_source', 'eval_count_blend_alpha',
        'eval_tile_size', 'eval_tile_overlap', 'eval_tile_nms_radius',
        'eval_tile_min_gt', 'eval_tile_max_tiles',
        'eval_tile_trigger_count', 'eval_tile_trigger_area',
        'score_threshold', 'split_threshold', 'split_threshold_quantile', 'query_prune_threshold',
        'eval_nms_radius', 'eval_branch_gate', 'eval_soft_split_gate',
        'eval_foreground_gate', 'eval_foreground_gate_mode', 'eval_foreground_gate_strength',
        'eval_count_mode', 'eval_count_head_min_score',
        'eval_score_calibration', 'eval_score_calibration_strength',
        'eval_score_calibration_start_epoch',
        'eval_score_calibration_min_bias', 'eval_score_calibration_max_bias',
        'eval_score_calibration_count_blend',
        'eval_score_calibration_count_ratio_min',
        'eval_score_calibration_count_ratio_max',
        'no_eval_filter_invalid_points', 'eval_debug_counting',
        'ucfcc50_fold', 'ucfcc50_fold_seed', 'ucfcc50_fold_manifest',
        'annotation_override_dir',
    }
    runtime_keys = {
        'resume', 'device', 'output_dir', 'seed', 'start_epoch',
        'resume_model_only', 'resume_allow_arch_change', 'num_workers', 'world_size', 'dist_url',
        'list_backbones', 'syn_bn', 'deterministic', 'freeze_bn', 'amp', 'amp_dtype',
        'strict_model_checks',
        # allow overriding schedule/eval settings at resume time
        'epochs', 'batch_size', 'accum_iter', 'eval_freq', 'eval_start_epoch', 'eval_model',
        'eval_before_train', 'data_path', 'nwpu_eval_split', 'jhu_eval_split',
        'bad_count_direction', 'bad_count_ratio_max', 'bad_count_mae_min', 'bad_count_start_epoch',
    }
    if getattr(args, 'resume_model_only', False):
        runtime_keys.update({
            'lr', 'lr_backbone', 'lr_backbone_adapter', 'weight_decay',
            'freeze_backbone_epochs', 'clip_max_norm',
            'lr_scheduler', 'lr_drop', 'lr_gamma', 'warmup_epochs', 'hold_epochs',
            'min_lr', 'ema_decay',
        })
        if 'eval_dense_start_epoch' in explicit_args:
            runtime_keys.add('eval_dense_start_epoch')
        for key in ('eval_dense_residual_mode', 'eval_dense_residual_start_epoch', 'eval_dense_residual_min_score'):
            if key in explicit_args:
                runtime_keys.add(key)
        aux_resume_keys = {
            'class_loss_type', 'focal_alpha', 'focal_gamma',
            'count_loss_coef', 'count_loss_gate', 'count_loss_type',
            'count_loss_budget_margin', 'count_loss_start_epoch',
            'quality_loss_coef', 'quality_loss_sigma', 'quality_loss_pos_floor',
            'quality_loss_bg_weight',
            'scale_point_loss_coef', 'scale_point_sigma',
            'scale_point_sigma_min', 'scale_point_sigma_max',
            'pq_sparse_coef', 'pq_dense_coef',
            'pq_dense_start_epoch', 'pq_dense_warmup_epochs',
            'branch_target_routing',
            'count_head_loss_coef', 'count_head_loss_type',
            'count_head_start_epoch', 'count_head_end_epoch', 'count_head_init_count',
            'count_head_warmup_epochs',
            'allow_count_head_fresh_train', 'allow_count_head_from_start',
            'force_unsafe_count_head_from_start', 'safe_count_head_start_epoch',
            'count_head_init_cells', 'count_head_feature_grad_scale',
            'count_head_feature_grad_start_epoch', 'count_head_feature_grad_warmup_epochs',
            'train_count_head_only',
            'density_map_loss_coef', 'allow_unstable_density_map_loss',
            'density_map_loss_type', 'density_map_pos_weight',
            'density_map_grad_scale',
            'density_map_start_epoch', 'density_map_end_epoch',
            'zip_count_loss_coef', 'zip_count_block_size', 'zip_count_feature_source', 'zip_count_bin_centers',
            'zip_count_zero_prior', 'zip_count_ce_coef', 'zip_count_count_coef',
            'zip_count_start_epoch', 'zip_count_end_epoch', 'zip_count_warmup_epochs',
            'zip_count_feature_grad_scale', 'eval_count_source', 'eval_count_blend_alpha',
            'foreground_loss_coef', 'foreground_sigma',
            'foreground_neg_shrink', 'foreground_init_prior',
            'eval_foreground_gate', 'eval_foreground_gate_mode', 'eval_foreground_gate_strength',
            'region_count_loss_coef', 'region_count_grid', 'region_count_gate',
            'region_count_type', 'region_count_start_epoch', 'region_count_end_epoch',
            'branch_exclusion_loss_coef', 'branch_exclusion_start_epoch', 'branch_exclusion_end_epoch',
            'bayesian_loss_coef', 'bayesian_sigma', 'bayesian_bg_coef',
            'bayesian_loss_gate', 'bayesian_start_epoch', 'bayesian_end_epoch',
            'apg_loss_coef', 'apg_pos_k', 'apg_point_coef',
            'apg_bg_coef', 'apg_bg_k', 'apg_bg_min_dist', 'apg_bg_offset_coef',
            'apg_local_neg_coef', 'apg_local_neg_k', 'apg_local_neg_min_dist',
            'apg_local_neg_max_dist', 'apg_local_neg_offset_coef',
            'apg_start_epoch', 'apg_warmup_epochs',
            'apg_sparse_coef', 'apg_dense_coef',
            'apg_dense_start_epoch', 'apg_dense_warmup_epochs',
            'apg_end_epoch',
            'apg_count_calibration', 'apg_count_calibration_gate',
            'apg_count_calibration_min', 'apg_count_calibration_max',
            'apg_count_calibration_eps',
            'apg_contrastive_coef', 'apg_neg_k', 'apg_margin',
            'apg_consistency_coef', 'apg_consistency_k', 'apg_consistency_sigma',
            'apg_soft_loss_coef', 'apg_soft_pos_k', 'apg_soft_sigma', 'apg_soft_point_coef',
            'query_feature_interpolation', 'query_ifi_sharing', 'query_ifi_feature_source',
            'query_ifi_branch_scope',
            'query_ifi_residual', 'query_ifi_residual_init',
            'ifi_interpolation', 'ifi_feature_source', 'ifi_pos_dim',
            'ifi_mlp_hidden_dim', 'ifi_activation', 'ifi_branch_scope',
            'ifi_loss_coef', 'ifi_head_source', 'ifi_point_coef',
            'ifi_point_loss_type', 'ifi_balance_pos_neg',
            'ifi_pos_k', 'ifi_pos_radius', 'ifi_random_sampling',
            'ifi_neg_k', 'ifi_neg_radius',
            'ifi_neg_min_dist', 'ifi_negative_policy',
            'ifi_start_epoch', 'ifi_end_epoch',
            'qd_apg_loss_coef', 'qd_apg_point_coef', 'qd_apg_suppress_coef',
            'qd_apg_start_epoch', 'qd_apg_end_epoch', 'qd_apg_route_source',
            'routed_apg_loss_coef', 'routed_apg_point_coef', 'routed_apg_pos_k',
            'routed_apg_bg_coef', 'routed_apg_bg_k', 'routed_apg_bg_min_dist',
            'routed_apg_start_epoch', 'routed_apg_end_epoch', 'routed_apg_warmup_epochs',
            'routed_apg_min_weight', 'routed_apg_source', 'routed_apg_gate',
            'inheritance_loss_coef', 'inheritance_sparse_coef', 'inheritance_dense_coef',
            'inheritance_consistency_coef', 'inheritance_start_epoch', 'inheritance_end_epoch',
            'inheritance_gate',
            'split_loss_variant',
        }
        runtime_keys.update(key for key in aux_resume_keys if key in explicit_args)
        if getattr(args, 'resume_allow_arch_change', False):
            runtime_keys.update(key for key in ARCHITECTURE_OVERRIDE_KEYS if key in explicit_args)
    runtime_keys.update(key for key in explicit_only_runtime_keys if key in explicit_args)
    for key in runtime_keys:
        if hasattr(args, key):
            setattr(merged, key, getattr(args, key))
    if hasattr(args, '_explicit_args'):
        setattr(merged, '_explicit_args', getattr(args, '_explicit_args'))
    return merged


def resolve_output_dir(args):
    output_arg = Path(args.output_dir)
    if output_arg.is_absolute():
        return output_arg
    parts = output_arg.parts
    if parts and parts[0] == 'outputs':
        return output_arg
    return Path("./outputs") / args.dataset_file / output_arg


def validate_training_output_dir(
    output_dir,
    checkpoint,
    resume_path='',
    allow_output_overwrite=False,
):
    """Prevent scratch or cross-run resumes from replacing earlier results."""
    if not output_dir.exists():
        return
    protected_names = (
        'checkpoint.pth',
        'best_checkpoint.pth',
        'best_complete_checkpoint.pth',
        'best_mae_checkpoint.pth',
        'best_mse_checkpoint.pth',
        'best_localization_checkpoint.pth',
        'best_eval_results.json',
        'best_mse_eval_results.json',
        'best_localization_eval_results.json',
        'latest_eval_results.json',
        'eval_history.jsonl',
    )
    existing = {output_dir / name for name in protected_names if (output_dir / name).exists()}
    existing.update(output_dir.glob('*.pth'))
    existing = sorted(existing)
    if not existing:
        return

    preview = '\n  - '.join(str(path) for path in existing[:8])
    if checkpoint is None:
        raise FileExistsError(
            'Refusing to start scratch training in an existing result directory. '
            'This prevents old checkpoints or metrics from being reused or overwritten. '
            'Choose a new --output_dir. '
            f'Found:\n  - {preview}'
        )
    if allow_output_overwrite:
        return

    resume_path = str(resume_path or '')
    same_run_resume = False
    if resume_path and not resume_path.startswith(('http://', 'https://')):
        try:
            same_run_resume = (
                Path(resume_path).expanduser().resolve().parent
                == output_dir.expanduser().resolve()
            )
        except OSError:
            same_run_resume = False
    if not same_run_resume:
        raise FileExistsError(
            'Refusing to write a resumed checkpoint into an unrelated existing '
            'result directory. Resume that directory from its own checkpoint, '
            'choose a new --output_dir, or pass --allow_output_overwrite after '
            'manually preserving the existing run. '
            f'Found:\n  - {preview}'
        )


def backup_existing_best_checkpoint(output_dir):
    """Keep the first existing best checkpoint before a training run overwrites it."""
    if not utils.is_main_process():
        return
    for name in ('best_checkpoint.pth', 'best_complete_checkpoint.pth'):
        best_path = output_dir / name
        backup_path = output_dir / name.replace('.pth', '.before_overwrite.pth')
        if not best_path.exists() or backup_path.exists():
            continue
        shutil.copy2(best_path, backup_path)
        print(f'backed up existing best checkpoint to: {backup_path}')


def ensure_mae_checkpoint_alias(output_dir):
    """Expose the legacy best checkpoint as initial explicit metric checkpoints."""
    legacy_path = output_dir / 'best_checkpoint.pth'
    complete_path = output_dir / 'best_complete_checkpoint.pth'
    mae_path = output_dir / 'best_mae_checkpoint.pth'
    mse_path = output_dir / 'best_mse_checkpoint.pth'
    localization_path = output_dir / 'best_localization_checkpoint.pth'
    legacy_eval_path = output_dir / 'best_eval_results.json'
    localization_eval_path = output_dir / 'best_localization_eval_results.json'
    if not utils.is_main_process() or not legacy_path.exists():
        return
    if not complete_path.exists():
        shutil.copy2(legacy_path, complete_path)
        print(f'created complete checkpoint alias: {complete_path}')
    if not mae_path.exists():
        shutil.copy2(legacy_path, mae_path)
        print(f'created MAE checkpoint alias: {mae_path}')
    if not mse_path.exists():
        # Historical checkpoints only retained the MSE paired with best MAE.
        # Use that available model as the initial MSE candidate; future
        # evaluations replace it using independently tracked MSE.
        shutil.copy2(legacy_path, mse_path)
        print(f'created initial MSE checkpoint alias: {mse_path}')
    if not localization_path.exists() and legacy_eval_path.exists():
        try:
            legacy_eval = json.loads(legacy_eval_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            legacy_eval = {}
        if 'loc_f1_large' in legacy_eval and 'loc_f1_small' in legacy_eval:
            shutil.copy2(legacy_path, localization_path)
            if not localization_eval_path.exists():
                shutil.copy2(legacy_eval_path, localization_eval_path)
            print(f'created initial localization checkpoint alias: {localization_path}')


def read_eval_record(path):
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def enrich_checkpoint_metadata(path, metadata):
    """Add metric metadata to a legacy checkpoint without changing its state."""
    if not utils.is_main_process() or not path.exists():
        return
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    missing = {key: value for key, value in metadata.items() if key not in checkpoint}
    if not missing:
        return
    checkpoint.update(missing)
    utils.save_on_master(checkpoint, path)
    print(f'embedded evaluation metadata in legacy checkpoint: {path}')


def checkpoint_args_snapshot(args):
    return argparse.Namespace(**{
        key: value for key, value in vars(args).items()
        if not key.startswith('_')
    })


def checkpoint_arg(checkpoint, key, default=None):
    checkpoint_args = checkpoint.get('args') if checkpoint is not None else None
    if checkpoint_args is None:
        return default
    if isinstance(checkpoint_args, dict):
        return checkpoint_args.get(key, default)
    return getattr(checkpoint_args, key, default)


def scalar_eval_metrics(test_stats, skip=()):
    skip = set(skip)
    metrics = {}
    for key, value in test_stats.items():
        if key in skip:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            metrics[key] = float(value)
    return metrics


def checkpoint_eval_metrics(test_stats):
    """Convert all scalar/string evaluation outputs to checkpoint-safe values."""
    metrics = {}
    for key, value in test_stats.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, bool, int, float)):
            metrics[key] = value
    return metrics


def should_skip_pretrained_backbone(args, checkpoint):
    if checkpoint is None:
        return bool(getattr(args, 'no_pretrained_backbone', False))
    explicit_args = set(getattr(args, '_explicit_args', set()))
    if 'no_pretrained_backbone' in explicit_args and bool(getattr(args, 'no_pretrained_backbone', False)):
        return True
    if getattr(args, 'resume_model_only', False):
        # Model-only fine-tuning can intentionally add or change heads/adapters.
        # Keep ImageNet/pretrained initialization for any missing compatible
        # modules, then let the checkpoint overwrite all matching weights.
        return False
    requested_backbone = getattr(args, 'backbone', None)
    checkpoint_backbone = checkpoint_arg(checkpoint, 'backbone', requested_backbone)
    changed_backbone = (
        getattr(args, 'resume_model_only', False)
        and getattr(args, 'resume_allow_arch_change', False)
        and 'backbone' in explicit_args
        and checkpoint_backbone != requested_backbone
    )
    if changed_backbone:
        return 'no_pretrained_backbone' in explicit_args and bool(getattr(args, 'no_pretrained_backbone', False))
    return True


def model_only_allowed_missing_prefixes(args):
    prefixes = []
    needs_count_head = (
        float(getattr(args, 'count_head_loss_coef', 0.0)) > 0
        or getattr(args, 'eval_count_mode', 'threshold') == 'count_head_topk'
        or getattr(args, 'eval_score_calibration', 'none') == 'count_head_bias'
        or getattr(args, 'eval_dense_residual_mode', 'none') == 'count_head'
    )
    if needs_count_head:
        prefixes.append('count_head.')
    needs_zip_count_head = (
        float(getattr(args, 'zip_count_loss_coef', 0.0)) > 0
        or getattr(args, 'eval_count_source', 'pet') in ('zip', 'zip_pet_blend')
    )
    if needs_zip_count_head:
        prefixes.append('zip_count_head.')
    needs_foreground_head = (
        float(getattr(args, 'foreground_loss_coef', 0.0)) > 0
        or getattr(args, 'eval_foreground_gate', 'none') != 'none'
    )
    if needs_foreground_head:
        prefixes.append('foreground_head.')
    if getattr(args, 'scale_fusion', 'none') != 'none':
        prefixes.append('scale_fusion.')
    if getattr(args, 'encoder_context_fusion', 'none') != 'none':
        prefixes.append('encoder_context_fusion.')
    if float(getattr(args, 'ifi_loss_coef', 0.0)) > 0 and getattr(args, 'ifi_head_source', 'separate') == 'separate':
        prefixes.extend(('ifi_cls_embed.', 'ifi_coord_embed.'))
    return tuple(prefixes)


def validate_model_only_incompatible(incompatible, allowed_missing_prefixes):
    missing = list(getattr(incompatible, 'missing_keys', []))
    unexpected = list(getattr(incompatible, 'unexpected_keys', []))
    bad_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if bad_missing or unexpected:
        details = []
        if bad_missing:
            details.append(f"unexpected missing keys: {bad_missing[:20]}")
        if unexpected:
            details.append(f"unexpected checkpoint keys: {unexpected[:20]}")
        allowed_text = ', '.join(allowed_missing_prefixes) or 'none'
        raise RuntimeError(
            'Unsafe non-strict model-only resume; '
            f'allowed missing prefixes: {allowed_text}; '
            + '; '.join(details)
        )


def build_optimizer_param_groups(model_without_ddp, args):
    """Keep pretrained backbone weights on low LR while training new adapters at main LR."""
    use_timm = is_timm_backbone(getattr(args, 'backbone', ''))
    timm_feature_prefix = 'backbone.backbone.backbone.'
    adapter_prefixes = [
        'backbone.backbone.fpn.',  # timm Joiner -> TimmBackbone -> BackboneFPN
        'backbone.backbone.lite_fpn.',  # timm Joiner -> TimmBackbone -> LiteFPNAdapter
        'backbone.backbone.rcc_fpn.',  # timm Joiner -> TimmBackbone -> RCCFPNAdapter
        'backbone.backbone.pet_fpn.',  # timm Joiner -> TimmBackbone -> PETFPNAdapter
        'backbone.backbone.direct_adapter.',  # timm Joiner -> TimmBackbone -> DirectFeatureAdapter
        'backbone.0.fpn.mhf_c4.',
        'backbone.0.fpn.mhf_c3.',
    ]
    if getattr(args, 'vgg_fpn_main_lr', False):
        adapter_prefixes.append('backbone.0.fpn.')  # VGG Joiner -> Backbone_VGG -> FeatsFusion
    adapter_lr = float(getattr(args, 'lr_backbone_adapter', -1.0))

    main_params, adapter_params, backbone_params = [], [], []
    for name, param in model_without_ddp.named_parameters():
        if not param.requires_grad:
            continue
        if use_timm and name.startswith(timm_feature_prefix):
            backbone_params.append(param)
        elif any(name.startswith(prefix) for prefix in adapter_prefixes):
            adapter_params.append(param)
        elif 'backbone' in name:
            backbone_params.append(param)
        else:
            main_params.append(param)

    param_groups = []
    group_summary = []
    if main_params:
        param_groups.append({'params': main_params})
        group_summary.append(('main', len(main_params), sum(p.numel() for p in main_params), args.lr))
    if adapter_params:
        adapter_group = {'params': adapter_params}
        effective_adapter_lr = args.lr if adapter_lr < 0 else adapter_lr
        if adapter_lr >= 0:
            adapter_group['lr'] = adapter_lr
        param_groups.append(adapter_group)
        group_summary.append(('backbone_adapter', len(adapter_params), sum(p.numel() for p in adapter_params), effective_adapter_lr))
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.lr_backbone})
        group_summary.append(('backbone', len(backbone_params), sum(p.numel() for p in backbone_params), args.lr_backbone))

    return param_groups, group_summary


def set_raw_backbone_trainability(model_without_ddp, args, trainable):
    use_timm = is_timm_backbone(getattr(args, 'backbone', ''))
    raw_backbone_prefixes = (
        ('backbone.backbone.backbone.',) if use_timm else ('backbone.0.body',)
    )
    changed = 0
    total = 0
    for name, param in model_without_ddp.named_parameters():
        if not any(name.startswith(prefix) for prefix in raw_backbone_prefixes):
            continue
        total += 1
        if param.requires_grad != trainable:
            param.requires_grad = trainable
            changed += 1
    return total, changed


def set_count_head_only_trainability(model_without_ddp):
    trainable, frozen = 0, 0
    for name, param in model_without_ddp.named_parameters():
        is_count_head = name.startswith('count_head.')
        param.requires_grad_(is_count_head)
        if is_count_head:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return trainable, frozen


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_reproducibility(seed, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ModelEma:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def set(self, model):
        self.module.load_state_dict(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(
                    model_value.to(device=ema_value.device, dtype=ema_value.dtype),
                    alpha=1.0 - self.decay,
                )
            else:
                ema_value.copy_(model_value.to(device=ema_value.device, dtype=ema_value.dtype))

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)


class IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(int(i) for i in indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def get_sample_counts(self):
        if not hasattr(self.dataset, 'get_sample_counts'):
            raise AttributeError('underlying dataset does not expose get_sample_counts')
        counts = self.dataset.get_sample_counts()
        return [counts[i] for i in self.indices]

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def resolve_validation_protocol(args):
    protocol = str(getattr(args, 'validation_protocol', 'auto'))
    nwpu_names = ('NWPU', 'NWPU_Crowd', 'NWPU-Crowd')
    jhu_names = ('JHU', 'JHU_Crowd', 'JHU-Crowd++')
    benchmark_only_names = (
        'SHA', 'SHB', 'QNRF',
        'UCFCC50', 'UCF_CC_50', 'UCF-CC-50',
    )
    if protocol == 'auto':
        protocol = (
            'official_val'
            if args.dataset_file in nwpu_names + jhu_names
            else 'train_holdout'
        )
    if protocol == 'official_val':
        if args.dataset_file not in nwpu_names + jhu_names:
            raise ValueError(
                'validation_protocol=official_val is only valid for NWPU or JHU'
            )
        if (
            args.dataset_file in nwpu_names
            and str(getattr(args, 'nwpu_eval_split', 'val')) != 'val'
        ):
            raise ValueError(
                'validation_protocol=official_val requires '
                '--nwpu_eval_split val'
            )
        if (
            args.dataset_file in jhu_names
            and str(getattr(args, 'jhu_eval_split', 'val')) != 'val'
        ):
            raise ValueError(
                'validation_protocol=official_val requires '
                '--jhu_eval_split val'
            )
    if (
        protocol == 'benchmark_test'
        and args.dataset_file in benchmark_only_names
        and not bool(getattr(args, 'allow_benchmark_test_selection', False))
    ):
        raise ValueError(
            'validation_protocol=benchmark_test repeatedly selects checkpoints '
            'on the benchmark test split. Use train_holdout for development or '
            'final_test_once for a fixed full-data refit. The legacy behavior '
            'requires --allow_benchmark_test_selection.'
        )
    if protocol == 'final_test_once':
        if args.dataset_file not in ('SHA', 'SHB', 'QNRF'):
            raise ValueError(
                'validation_protocol=final_test_once is only defined for '
                'SHA, SHB, and QNRF. Use official_val for NWPU/JHU and the '
                'five-fold runner for UCF-CC-50.'
            )
        if bool(getattr(args, 'eval_before_train', False)):
            raise ValueError(
                'final_test_once is incompatible with --eval_before_train'
            )
    return protocol


def build_train_holdout_indices(num_samples, holdout_fraction, seed):
    if num_samples < 2:
        raise ValueError('train-holdout validation requires at least 2 training samples')
    holdout_fraction = float(holdout_fraction)
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError('--train_holdout_fraction must be in (0, 1)')
    num_val = int(round(num_samples * holdout_fraction))
    num_val = max(1, min(num_samples - 1, num_val))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    permutation = torch.randperm(num_samples, generator=generator).tolist()
    val_indices = sorted(permutation[:num_val])
    train_indices = sorted(permutation[num_val:])
    return train_indices, val_indices


def main(args):
    utils.init_distributed_mode(args)
    if args.list_backbones:
        print('Supported timm ablation backbones:')
        for backbone in get_supported_timm_backbones():
            print(f'  {backbone}')
        return

    checkpoint = None
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        args = merge_checkpoint_args(args, checkpoint)
        args.no_pretrained_backbone = should_skip_pretrained_backbone(args, checkpoint)

    if getattr(args, 'auto_backbone_recipe', False):
        apply_backbone_recipe(args)
    apply_model_recipe(args)
    args = sanitize_unstable_training_args(args)
    validate_partial_annotation_contract(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    set_reproducibility(seed, deterministic=getattr(args, 'deterministic', True))
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(seed)

    # build model
    model, criterion = build_model(args)
    model.to(device)
    if args.syn_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if getattr(args, 'train_count_head_only', False):
        trainable_count, frozen_count = set_count_head_only_trainability(model_without_ddp)
        if utils.is_main_process():
            print(f'count-head-only training: trainable_params={trainable_count} frozen_params={frozen_count}')
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_ema = ModelEma(model_without_ddp, args.ema_decay) if args.ema_decay > 0 else None

    # build optimizer
    param_dicts, param_group_summary = build_optimizer_param_groups(model_without_ddp, args)
    if utils.is_main_process():
        for group_name, n_tensors, n_params, lr in param_group_summary:
            print(f'optimizer group {group_name}: tensors={n_tensors}, params={n_params}, lr={lr}')
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    amp_enabled = bool(getattr(args, 'amp', False) and device.type == 'cuda')
    amp_dtype = None
    scaler = None
    if amp_enabled:
        amp_dtype_name = getattr(args, 'amp_dtype', 'auto')
        if amp_dtype_name == 'auto':
            bf16_supported = bool(getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
            amp_dtype_name = 'bfloat16' if bf16_supported else 'float16'
        amp_dtype = torch.bfloat16 if amp_dtype_name == 'bfloat16' else torch.float16
        if amp_dtype == torch.float16:
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
                try:
                    scaler = torch.amp.GradScaler('cuda')
                except TypeError:
                    scaler = torch.amp.GradScaler(enabled=True)
            else:
                scaler = torch.cuda.amp.GradScaler()
        if utils.is_main_process():
            scaler_state = ' + GradScaler' if scaler is not None else ''
            print(f'AMP enabled: CUDA autocast dtype={amp_dtype_name}{scaler_state}')
    if args.lr_scheduler == 'step':
        lr_drop = args.epochs if args.lr_drop <= 0 else args.lr_drop
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop, gamma=args.lr_gamma)
    else:
        warmup_epochs = min(max(0, int(args.warmup_epochs)), max(args.epochs - 1, 0))
        hold_epochs = int(args.hold_epochs)
        if hold_epochs < 0:
            hold_epochs = max(0, args.epochs // 15)
        hold_epochs = min(max(0, hold_epochs), max(args.epochs - warmup_epochs - 1, 0))
        decay_epochs = max(1, args.epochs - warmup_epochs - hold_epochs)
        min_lr_ratio = float(args.min_lr) / max(float(args.lr), 1e-12)

        def warmup_hold_cosine_factor(epoch_idx):
            if args.epochs <= 1:
                return 1.0
            if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                warmup_progress = float(epoch_idx + 1) / float(warmup_epochs)
                return max(1e-3, warmup_progress)
            if epoch_idx < warmup_epochs + hold_epochs:
                return 1.0
            decay_progress = min(max(epoch_idx - warmup_epochs - hold_epochs, 0), decay_epochs) / float(decay_epochs)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return max(min_lr_ratio, cosine_factor)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_hold_cosine_factor)

    # build dataset
    validation_protocol = resolve_validation_protocol(args)
    dataset_train = build_dataset(image_set='train', args=args)
    if validation_protocol == 'train_holdout':
        dataset_train_eval = build_dataset(image_set='train_eval', args=args)
        train_indices, val_indices = build_train_holdout_indices(
            len(dataset_train_eval),
            getattr(args, 'train_holdout_fraction', 0.1),
            getattr(args, 'train_holdout_seed', args.seed),
        )
        dataset_train = IndexedSubset(dataset_train, train_indices)
        dataset_val = IndexedSubset(dataset_train_eval, val_indices)
    else:
        dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, seed=args.seed)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        if float(getattr(args, 'train_count_weight_power', 0.0)) > 0 and utils.is_main_process():
            print('WARNING: --train_count_weight_power is ignored with DistributedSampler')
    else:
        count_weight_power = float(getattr(args, 'train_count_weight_power', 0.0))
        if count_weight_power > 0 and hasattr(dataset_train, 'get_sample_counts'):
            counts = torch.as_tensor(dataset_train.get_sample_counts(), dtype=torch.float64)
            weights = torch.pow(counts + 1.0, count_weight_power)
            max_weight = float(getattr(args, 'train_count_weight_max', 0.0))
            if max_weight > 0:
                weights = weights.clamp(max=max_weight)
            weights = weights / weights.mean().clamp_min(1e-12)
            sampler_train = torch.utils.data.WeightedRandomSampler(
                weights,
                num_samples=len(weights),
                replacement=True,
                generator=data_loader_generator,
            )
            if utils.is_main_process():
                print(
                    'count-weighted sampler:',
                    f'power={count_weight_power}',
                    f'max_weight={max_weight}',
                    f'weight_range=[{float(weights.min()):.3f},{float(weights.max()):.3f}]',
                )
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train, generator=data_loader_generator)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                worker_init_fn=seed_worker, generator=data_loader_generator)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                worker_init_fn=seed_worker, generator=data_loader_generator)
    if utils.is_main_process():
        accum_iter = max(1, int(getattr(args, 'accum_iter', 1)))
        print(
            'batch config:',
            f'batch_size={args.batch_size}',
            f'accum_iter={accum_iter}',
            f'effective_batch_size={args.batch_size * accum_iter}',
            f'train_samples={len(dataset_train)}',
            f'val_samples={len(dataset_val)}',
            f'train_batches={len(batch_sampler_train)}',
        )
        print(
            'validation protocol:',
            validation_protocol,
            f'dataset={args.dataset_file}',
            f'nwpu_eval_split={getattr(args, "nwpu_eval_split", "")}',
            f'jhu_eval_split={getattr(args, "jhu_eval_split", "")}',
        )
        if validation_protocol == 'final_test_once':
            print(
                'final-test protocol: the benchmark test split will be '
                'evaluated only after the last training epoch'
            )

    # output directory and log
    output_dir = resolve_output_dir(args)
    validate_training_output_dir(
        output_dir,
        checkpoint,
        resume_path=args.resume,
        allow_output_overwrite=getattr(args, 'allow_output_overwrite', False),
    )
    run_log_name = output_dir / 'run_log.txt'
    if utils.is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        if checkpoint is not None:
            ensure_mae_checkpoint_alias(output_dir)
        print(f'outputs will be saved to: {output_dir.resolve()}')
        with open(run_log_name, "a", encoding="utf-8") as log_file:
            log_file.write('Run Log %s\n' % time.strftime("%c"))
            log_file.write("{}\n".format(args))
            log_file.write("parameters: {}\n".format(n_parameters))

    best_mae, best_mse, best_epoch = 1e8, 1e8, 0
    best_loc_f1_large = -1.0
    best_loc_f1_small = -1.0
    best_loc_mae = 1e8
    best_loc_mse = 1e8
    best_loc_epoch = 0
    lowest_mse = 1e8
    best_mse_mae = 1e8
    best_mse_epoch = 0
    latest_eval_metrics = {}
    best_mae_eval_metrics = {}
    best_mse_eval_metrics = {}
    best_localization_eval_metrics = {}
    if checkpoint is not None:
        model_key = 'model'
        if model_ema is not None and 'model_raw' in checkpoint and not args.resume_model_only:
            model_key = 'model_raw'
        allowed_missing_prefixes = model_only_allowed_missing_prefixes(args)
        auto_non_strict_model_only = (
            getattr(args, 'resume_model_only', False)
            and bool(allowed_missing_prefixes)
        )
        strict_load = not (
            getattr(args, 'resume_model_only', False)
            and (
                getattr(args, 'resume_allow_arch_change', False)
                or auto_non_strict_model_only
            )
        )
        resume_model_state = utils.upgrade_legacy_pet_state_dict(checkpoint[model_key])
        incompatible = model_without_ddp.load_state_dict(resume_model_state, strict=strict_load)
        if auto_non_strict_model_only and not getattr(args, 'resume_allow_arch_change', False):
            validate_model_only_incompatible(incompatible, allowed_missing_prefixes)
        if not strict_load and utils.is_main_process():
            missing = getattr(incompatible, 'missing_keys', [])
            unexpected = getattr(incompatible, 'unexpected_keys', [])
            print(
                'non-strict model-only resume:',
                f'missing_keys={len(missing)}',
                f'unexpected_keys={len(unexpected)}',
            )
            if missing:
                print('  missing:', missing[:20])
            if unexpected:
                print('  unexpected:', unexpected[:20])
        if model_ema is not None:
            if args.resume_model_only:
                model_ema.set(model_without_ddp)
            elif 'model_ema' in checkpoint:
                model_ema.load_state_dict(
                    utils.upgrade_legacy_pet_state_dict(checkpoint['model_ema'])
                )
            elif model_key == 'model_raw' and 'model' in checkpoint:
                model_ema.load_state_dict(
                    utils.upgrade_legacy_pet_state_dict(checkpoint['model'])
                )
            else:
                model_ema.set(model_without_ddp)
        if not args.resume_model_only and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            args.start_epoch = checkpoint['epoch'] + 1
            best_mae = checkpoint.get('best_mae', best_mae)
            best_mse = checkpoint.get('best_mse', best_mse)
            best_epoch = checkpoint.get('best_epoch', best_epoch)
            best_loc_f1_large = checkpoint.get('best_loc_f1_large', best_loc_f1_large)
            best_loc_f1_small = checkpoint.get('best_loc_f1_small', best_loc_f1_small)
            best_loc_mae = checkpoint.get('best_loc_mae', best_loc_mae)
            best_loc_mse = checkpoint.get('best_loc_mse', best_loc_mse)
            best_loc_epoch = checkpoint.get('best_loc_epoch', best_loc_epoch)
            lowest_mse = checkpoint.get('lowest_mse', checkpoint.get('best_mse', lowest_mse))
            best_mse_mae = checkpoint.get('best_mse_mae', checkpoint.get('best_mae', best_mse_mae))
            best_mse_epoch = checkpoint.get('best_mse_epoch', checkpoint.get('best_epoch', best_mse_epoch))
            latest_eval_metrics = dict(checkpoint.get('latest_eval_metrics', latest_eval_metrics))
            best_mae_eval_metrics = dict(checkpoint.get('best_mae_eval_metrics', best_mae_eval_metrics))
            best_mse_eval_metrics = dict(checkpoint.get('best_mse_eval_metrics', best_mse_eval_metrics))
            best_localization_eval_metrics = dict(
                checkpoint.get('best_localization_eval_metrics', best_localization_eval_metrics)
            )

    # Migrate complete evaluation metadata from runs created before metric
    # records were embedded in checkpoint payloads.
    saved_latest_record = read_eval_record(output_dir / 'latest_eval_results.json')
    saved_mae_record = read_eval_record(output_dir / 'best_eval_results.json')
    saved_mse_record = read_eval_record(output_dir / 'best_mse_eval_results.json')
    saved_loc_record = read_eval_record(output_dir / 'best_localization_eval_results.json')
    if not latest_eval_metrics and saved_latest_record:
        latest_eval_metrics = saved_latest_record
    if not best_mae_eval_metrics and saved_mae_record:
        best_mae_eval_metrics = saved_mae_record
    if not best_mse_eval_metrics:
        best_mse_eval_metrics = saved_mse_record or saved_mae_record
    if not best_localization_eval_metrics:
        best_localization_eval_metrics = saved_loc_record or saved_mae_record

    if best_mse_eval_metrics:
        saved_mse = float(best_mse_eval_metrics.get('test_mse', lowest_mse))
        if saved_mse <= lowest_mse:
            lowest_mse = saved_mse
            best_mse_mae = float(best_mse_eval_metrics.get('test_mae', best_mse_mae))
            best_mse_epoch = int(best_mse_eval_metrics.get('epoch', best_mse_epoch))
    if (
        'loc_f1_large' in best_localization_eval_metrics
        and 'loc_f1_small' in best_localization_eval_metrics
    ):
        saved_loc_rank = (
            float(best_localization_eval_metrics['loc_f1_large']),
            float(best_localization_eval_metrics['loc_f1_small']),
            -float(best_localization_eval_metrics.get('test_mae', 1e8)),
        )
        best_loc_rank = (best_loc_f1_large, best_loc_f1_small, -best_loc_mae)
        if saved_loc_rank > best_loc_rank:
            best_loc_f1_large = saved_loc_rank[0]
            best_loc_f1_small = saved_loc_rank[1]
            best_loc_mae = float(best_localization_eval_metrics.get('test_mae', best_loc_mae))
            best_loc_mse = float(best_localization_eval_metrics.get('test_mse', best_loc_mse))
            best_loc_epoch = int(best_localization_eval_metrics.get('epoch', best_loc_epoch))

    def checkpoint_payload(epoch, model_state=None, include_raw_model=False):
        payload = {
            'model': model_without_ddp.state_dict() if model_state is None else model_state,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': checkpoint_args_snapshot(args),
            'best_mae': best_mae,
            'best_mse': best_mse,
            'best_epoch': best_epoch,
            'best_loc_f1_large': best_loc_f1_large,
            'best_loc_f1_small': best_loc_f1_small,
            'best_loc_mae': best_loc_mae,
            'best_loc_mse': best_loc_mse,
            'best_loc_epoch': best_loc_epoch,
            'lowest_mse': lowest_mse,
            'best_mse_mae': best_mse_mae,
            'best_mse_epoch': best_mse_epoch,
            'latest_eval_metrics': dict(latest_eval_metrics),
            'best_mae_eval_metrics': dict(best_mae_eval_metrics),
            'best_mse_eval_metrics': dict(best_mse_eval_metrics),
            'best_localization_eval_metrics': dict(best_localization_eval_metrics),
            'checkpoint_eval_metrics': dict(latest_eval_metrics),
        }
        if model_ema is not None:
            payload['model_ema'] = model_ema.state_dict()
            payload['ema_decay'] = args.ema_decay
        if scaler is not None:
            payload['scaler'] = scaler.state_dict()
        if include_raw_model:
            payload['model_raw'] = model_without_ddp.state_dict()
        return payload

    legacy_metric_metadata = {
        'best_mae': best_mae,
        'best_mse': best_mse,
        'best_epoch': best_epoch,
        'lowest_mse': lowest_mse,
        'best_mse_mae': best_mse_mae,
        'best_mse_epoch': best_mse_epoch,
        'best_loc_f1_large': best_loc_f1_large,
        'best_loc_f1_small': best_loc_f1_small,
        'best_loc_mae': best_loc_mae,
        'best_loc_mse': best_loc_mse,
        'best_loc_epoch': best_loc_epoch,
        'latest_eval_metrics': dict(latest_eval_metrics),
        'best_mae_eval_metrics': dict(best_mae_eval_metrics),
        'best_mse_eval_metrics': dict(best_mse_eval_metrics),
        'best_localization_eval_metrics': dict(best_localization_eval_metrics),
    }
    for metric_path, metric_record in (
        (output_dir / 'best_checkpoint.pth', best_mae_eval_metrics),
        (output_dir / 'best_complete_checkpoint.pth', best_mae_eval_metrics),
        (output_dir / 'best_mae_checkpoint.pth', best_mae_eval_metrics),
        (output_dir / 'best_mse_checkpoint.pth', best_mse_eval_metrics),
        (output_dir / 'best_localization_checkpoint.pth', best_localization_eval_metrics),
    ):
        metric_metadata = dict(legacy_metric_metadata)
        metric_metadata['checkpoint_eval_metrics'] = dict(metric_record)
        enrich_checkpoint_metadata(metric_path, metric_metadata)

    if getattr(args, 'eval_before_train', False):
        t1 = time.time()
        pretrain_eval_epoch = (
            int(checkpoint.get('epoch', args.start_epoch))
            if checkpoint is not None
            else int(args.start_epoch)
        )
        eval_model, eval_model_name = select_eval_model(model, model_without_ddp, model_ema, args)
        if args.eval_protocol == 'crowd_no_overlap':
            test_stats = evaluate_crowd_no_overlap(
                eval_model,
                data_loader_val,
                device,
                epoch=pretrain_eval_epoch,
                vis_dir=None,
                localization_metrics=not args.no_localization_metrics,
                localization_large_threshold=args.localization_large_threshold,
                localization_small_threshold=args.localization_small_threshold,
                localization_protocol=args.localization_protocol,
                localization_large_scale=args.localization_large_scale,
                localization_small_scale=args.localization_small_scale,
            )
        else:
            test_stats = evaluate(
                eval_model,
                data_loader_val,
                device,
                pretrain_eval_epoch,
                None,
                localization_metrics=not args.no_localization_metrics,
                localization_large_threshold=args.localization_large_threshold,
                localization_small_threshold=args.localization_small_threshold,
                localization_protocol=args.localization_protocol,
                localization_large_scale=args.localization_large_scale,
                localization_small_scale=args.localization_small_scale,
                eval_tile_size=args.eval_tile_size,
                eval_tile_overlap=args.eval_tile_overlap,
                eval_tile_nms_radius=args.eval_tile_nms_radius,
                eval_tile_min_gt=args.eval_tile_min_gt,
                eval_tile_max_tiles=args.eval_tile_max_tiles,
                eval_tile_trigger_count=args.eval_tile_trigger_count,
                eval_tile_trigger_area=args.eval_tile_trigger_area,
            )
        t2 = time.time()
        print("\n==========================")
        print(
            "\npretrain_eval epoch:", pretrain_eval_epoch,
            "mae:", test_stats['mae'],
            "mse:", test_stats['mse'],
            "pred_cnt:", test_stats.get('pred_cnt', 0.0),
            "gt_cnt:", test_stats.get('gt_cnt', 0.0),
            format_localization_metrics(test_stats),
            "eval_model:", eval_model_name,
            "eval_time:", t2 - t1,
        )
        print("==========================\n")
        abort_bad_count, abort_reason = should_abort_for_bad_count(args, pretrain_eval_epoch, test_stats)
        if abort_bad_count:
            raise RuntimeError(f'eval_before_train failed sanity check: {abort_reason}')
        evaluated_epoch = pretrain_eval_epoch
        pretrain_mae = float(test_stats['mae'])
        pretrain_mse = float(test_stats['mse'])
        pretrain_mae_improved = pretrain_mae < best_mae
        if pretrain_mae_improved:
            best_mae = pretrain_mae
            best_mse = pretrain_mse
            best_epoch = evaluated_epoch
        pretrain_mse_improved = pretrain_mse < lowest_mse
        if pretrain_mse_improved:
            lowest_mse = pretrain_mse
            best_mse_mae = pretrain_mae
            best_mse_epoch = evaluated_epoch

        pretrain_loc_f1_large = test_stats.get('loc_f1_large')
        pretrain_loc_f1_small = test_stats.get('loc_f1_small')
        pretrain_loc_improved = False
        if pretrain_loc_f1_large is not None and pretrain_loc_f1_small is not None:
            pretrain_loc_f1_large = float(pretrain_loc_f1_large)
            pretrain_loc_f1_small = float(pretrain_loc_f1_small)
            pretrain_loc_rank = (
                pretrain_loc_f1_large,
                pretrain_loc_f1_small,
                -float(test_stats['mae']),
            )
            best_loc_rank = (best_loc_f1_large, best_loc_f1_small, -best_loc_mae)
            pretrain_loc_improved = pretrain_loc_rank > best_loc_rank
            if pretrain_loc_improved:
                best_loc_f1_large = pretrain_loc_f1_large
                best_loc_f1_small = pretrain_loc_f1_small
                best_loc_mae = pretrain_mae
                best_loc_mse = pretrain_mse
                best_loc_epoch = evaluated_epoch

        pretrain_eval_record = {
            'epoch': evaluated_epoch,
            'test_mae': pretrain_mae,
            'test_mse': pretrain_mse,
            'pred_cnt': float(test_stats.get('pred_cnt', 0.0)),
            'gt_cnt': float(test_stats.get('gt_cnt', 0.0)),
            'validation_protocol': validation_protocol,
            'best_epoch': int(best_epoch),
            'best_test_mae': float(best_mae),
            'best_test_mse': float(best_mse),
            'improved': bool(pretrain_mae_improved),
            'mse_improved': bool(pretrain_mse_improved),
            'lowest_mse': float(lowest_mse),
            'best_mse_mae': float(best_mse_mae),
            'best_mse_epoch': int(best_mse_epoch),
            'localization_improved': bool(pretrain_loc_improved),
            'best_loc_f1_large': float(best_loc_f1_large),
            'best_loc_f1_small': float(best_loc_f1_small),
            'best_loc_mae': float(best_loc_mae),
            'best_loc_mse': float(best_loc_mse),
            'best_loc_epoch': int(best_loc_epoch),
            'eval_model': eval_model_name,
            'eval_time': float(t2 - t1),
            'source': 'eval_before_train',
        }
        pretrain_eval_record.update(checkpoint_eval_metrics(test_stats))
        latest_eval_metrics = dict(pretrain_eval_record)
        if pretrain_mae_improved:
            best_mae_eval_metrics = dict(pretrain_eval_record)
        if pretrain_mse_improved:
            best_mse_eval_metrics = dict(pretrain_eval_record)
        if pretrain_loc_improved:
            best_localization_eval_metrics = dict(pretrain_eval_record)

        if utils.is_main_process():
            (output_dir / 'latest_eval_results.json').write_text(
                json.dumps(pretrain_eval_record, indent=2) + "\n",
                encoding="utf-8",
            )
            utils.save_on_master(checkpoint_payload(evaluated_epoch), output_dir / 'checkpoint.pth')
            evaluated_model_state, include_raw = best_state_for_eval_model(
                model_without_ddp,
                model_ema,
                eval_model_name,
            )
            if pretrain_mae_improved:
                (output_dir / 'best_eval_results.json').write_text(
                    json.dumps(pretrain_eval_record, indent=2) + "\n",
                    encoding="utf-8",
                )
                best_mae_payload = checkpoint_payload(
                    evaluated_epoch,
                    model_state=evaluated_model_state,
                    include_raw_model=include_raw,
                )
                utils.save_on_master(best_mae_payload, output_dir / 'best_checkpoint.pth')
                utils.save_on_master(best_mae_payload, output_dir / 'best_complete_checkpoint.pth')
                utils.save_on_master(best_mae_payload, output_dir / 'best_mae_checkpoint.pth')
            if pretrain_mse_improved:
                (output_dir / 'best_mse_eval_results.json').write_text(
                    json.dumps(pretrain_eval_record, indent=2) + "\n",
                    encoding="utf-8",
                )
                utils.save_on_master(
                    checkpoint_payload(
                        evaluated_epoch,
                        model_state=evaluated_model_state,
                        include_raw_model=include_raw,
                    ),
                    output_dir / 'best_mse_checkpoint.pth',
                )
            if pretrain_loc_improved:
                (output_dir / 'best_localization_eval_results.json').write_text(
                    json.dumps(pretrain_eval_record, indent=2) + "\n",
                    encoding="utf-8",
                )
                utils.save_on_master(
                    checkpoint_payload(
                        evaluated_epoch,
                        model_state=evaluated_model_state,
                        include_raw_model=include_raw,
                    ),
                    output_dir / 'best_localization_checkpoint.pth',
                )
                print(
                    'updated localization-best checkpoint:',
                    f'epoch={best_loc_epoch}',
                    f'sigma_l_f1={best_loc_f1_large:.4f}',
                    f'sigma_s_f1={best_loc_f1_small:.4f}',
                )

    # training
    print("Start training")
    start_time = time.time()
    bad_count_abort_reason = ''
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        freeze_epochs = int(getattr(args, 'freeze_backbone_epochs', 0))
        if freeze_epochs > 0:
            backbone_trainable = epoch >= freeze_epochs
            total_raw_backbone, changed_raw_backbone = set_raw_backbone_trainability(
                model_without_ddp,
                args,
                trainable=backbone_trainable,
            )
            if utils.is_main_process() and (epoch == args.start_epoch or changed_raw_backbone):
                state = 'trainable' if backbone_trainable else 'frozen'
                print(f'raw backbone is {state}: tensors={total_raw_backbone}, freeze_backbone_epochs={freeze_epochs}')
        
        t1 = time.time()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, model_ema=model_ema, model_without_ddp=model_without_ddp,
            freeze_bn=getattr(args, 'freeze_bn', False),
            amp_enabled=amp_enabled,
            scaler=scaler,
            amp_dtype=amp_dtype,
            accum_iter=getattr(args, 'accum_iter', 1))
        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        
        if utils.is_main_process():
            with open(run_log_name, "a", encoding="utf-8") as log_file:
                log_file.write('[ep %d][lr %.7f][%.2fs]\n' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))

        lr_scheduler.step()

        # save checkpoint
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master(checkpoint_payload(epoch), checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # write log
        if utils.is_main_process():
            with open(run_log_name, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        # evaluation
        scheduled_eval = (
            epoch % args.eval_freq == 0
            and epoch > 0
            and epoch >= int(getattr(args, 'eval_start_epoch', 0))
        )
        should_evaluate = (
            epoch == args.epochs - 1
            if validation_protocol == 'final_test_once'
            else scheduled_eval
        )
        if should_evaluate:
            t1 = time.time()
            eval_model, eval_model_name = select_eval_model(model, model_without_ddp, model_ema, args)
            if args.eval_protocol == 'crowd_no_overlap':
                test_stats = evaluate_crowd_no_overlap(
                    eval_model,
                    data_loader_val,
                    device,
                    epoch=epoch,
                    vis_dir=None,
                    localization_metrics=not args.no_localization_metrics,
                    localization_large_threshold=args.localization_large_threshold,
                    localization_small_threshold=args.localization_small_threshold,
                    localization_protocol=args.localization_protocol,
                    localization_large_scale=args.localization_large_scale,
                    localization_small_scale=args.localization_small_scale,
                )
            else:
                test_stats = evaluate(
                    eval_model,
                    data_loader_val,
                    device,
                    epoch,
                    None,
                    localization_metrics=not args.no_localization_metrics,
                    localization_large_threshold=args.localization_large_threshold,
                    localization_small_threshold=args.localization_small_threshold,
                    localization_protocol=args.localization_protocol,
                    localization_large_scale=args.localization_large_scale,
                    localization_small_scale=args.localization_small_scale,
                    eval_tile_size=args.eval_tile_size,
                    eval_tile_overlap=args.eval_tile_overlap,
                    eval_tile_nms_radius=args.eval_tile_nms_radius,
                    eval_tile_min_gt=args.eval_tile_min_gt,
                    eval_tile_max_tiles=args.eval_tile_max_tiles,
                    eval_tile_trigger_count=args.eval_tile_trigger_count,
                    eval_tile_trigger_area=args.eval_tile_trigger_area,
                )
            t2 = time.time()

            # output results
            mae, mse = test_stats['mae'], test_stats['mse']
            improved = mae < best_mae
            if improved:
                best_epoch = epoch
                best_mae = mae
                best_mse = mse
            mse_improved = mse < lowest_mse
            if mse_improved:
                lowest_mse = float(mse)
                best_mse_mae = float(mae)
                best_mse_epoch = epoch
            loc_f1_large = test_stats.get('loc_f1_large')
            loc_f1_small = test_stats.get('loc_f1_small')
            localization_improved = False
            if loc_f1_large is not None and loc_f1_small is not None:
                loc_f1_large = float(loc_f1_large)
                loc_f1_small = float(loc_f1_small)
                current_loc_rank = (loc_f1_large, loc_f1_small, -float(mae))
                best_loc_rank = (best_loc_f1_large, best_loc_f1_small, -best_loc_mae)
                localization_improved = current_loc_rank > best_loc_rank
                if localization_improved:
                    best_loc_f1_large = loc_f1_large
                    best_loc_f1_small = loc_f1_small
                    best_loc_mae = float(mae)
                    best_loc_mse = float(mse)
                    best_loc_epoch = epoch
            print("\n==========================")
            print(
                "\nepoch:", epoch,
                "mae:", mae,
                "mse:", mse,
                "pred_cnt:", test_stats.get('pred_cnt', 0.0),
                "gt_cnt:", test_stats.get('gt_cnt', 0.0),
                format_localization_metrics(test_stats),
                "\n\nbest mae:", best_mae,
                "best epoch:", best_epoch,
                "\nbest mse:", lowest_mse,
                "best mse epoch:", best_mse_epoch,
                "\nbest localization:",
                f"sigma_l_f1={best_loc_f1_large:.4f}",
                f"sigma_s_f1={best_loc_f1_small:.4f}",
                "best localization epoch:", best_loc_epoch,
            )
            print("==========================\n")
            if utils.is_main_process():
                eval_record = {
                    'epoch': epoch,
                    'test_mae': float(mae),
                    'test_mse': float(mse),
                    'pred_cnt': float(test_stats.get('pred_cnt', 0.0)),
                    'gt_cnt': float(test_stats.get('gt_cnt', 0.0)),
                    'best_epoch': int(best_epoch),
                    'best_test_mae': float(best_mae),
                    'best_test_mse': float(best_mse),
                    'improved': bool(improved),
                    'mse_improved': bool(mse_improved),
                    'lowest_mse': float(lowest_mse),
                    'best_mse_mae': float(best_mse_mae),
                    'best_mse_epoch': int(best_mse_epoch),
                    'localization_improved': bool(localization_improved),
                    'best_loc_f1_large': float(best_loc_f1_large),
                    'best_loc_f1_small': float(best_loc_f1_small),
                    'best_loc_mae': float(best_loc_mae),
                    'best_loc_mse': float(best_loc_mse),
                    'best_loc_epoch': int(best_loc_epoch),
                    'eval_time': float(t2 - t1),
                    'eval_model': eval_model_name,
                    'validation_protocol': validation_protocol,
                    'eval_count_mode': getattr(args, 'eval_count_mode', 'threshold'),
                    'eval_count_source': getattr(args, 'eval_count_source', 'pet'),
                    'eval_count_blend_alpha': float(getattr(args, 'eval_count_blend_alpha', 0.5)),
                    'eval_count_head_min_score': float(getattr(args, 'eval_count_head_min_score', 0.5)),
                    'eval_score_calibration': getattr(args, 'eval_score_calibration', 'none'),
                    'eval_score_calibration_strength': float(getattr(args, 'eval_score_calibration_strength', 1.0)),
                    'eval_score_calibration_start_epoch': int(getattr(args, 'eval_score_calibration_start_epoch', 0)),
                    'eval_score_calibration_min_bias': float(getattr(args, 'eval_score_calibration_min_bias', -8.0)),
                    'eval_score_calibration_max_bias': float(getattr(args, 'eval_score_calibration_max_bias', 8.0)),
                    'eval_score_calibration_count_blend': float(getattr(args, 'eval_score_calibration_count_blend', 1.0)),
                    'eval_score_calibration_count_ratio_min': float(getattr(args, 'eval_score_calibration_count_ratio_min', 0.0)),
                    'eval_score_calibration_count_ratio_max': float(getattr(args, 'eval_score_calibration_count_ratio_max', 1e6)),
                    'eval_filter_invalid_points': not bool(getattr(args, 'no_eval_filter_invalid_points', False)),
                    'eval_tile_size': int(getattr(args, 'eval_tile_size', 0)),
                    'eval_tile_overlap': int(getattr(args, 'eval_tile_overlap', 0)),
                    'eval_tile_nms_radius': float(getattr(args, 'eval_tile_nms_radius', 0.0)),
                    'eval_tile_min_gt': int(getattr(args, 'eval_tile_min_gt', 0)),
                    'eval_tile_max_tiles': int(getattr(args, 'eval_tile_max_tiles', 0)),
                    'eval_tile_trigger_count': float(getattr(args, 'eval_tile_trigger_count', 0.0)),
                    'eval_tile_trigger_area': int(getattr(args, 'eval_tile_trigger_area', 0)),
                    'localization_metrics': not bool(getattr(args, 'no_localization_metrics', False)),
                    'localization_large_threshold': float(getattr(args, 'localization_large_threshold', 8.0)),
                    'localization_small_threshold': float(getattr(args, 'localization_small_threshold', 4.0)),
                    'localization_protocol': getattr(args, 'localization_protocol', 'fixed'),
                    'localization_large_scale': float(getattr(args, 'localization_large_scale', 1.0)),
                    'localization_small_scale': float(getattr(args, 'localization_small_scale', 0.5)),
                    'loc_protocol': test_stats.get('loc_protocol', getattr(args, 'localization_protocol', 'fixed')),
                    'loc_protocol_large': test_stats.get('loc_protocol_large', ''),
                    'loc_protocol_small': test_stats.get('loc_protocol_small', ''),
                }
                eval_record.update(scalar_eval_metrics(test_stats, skip={'mae', 'mse', 'pred_cnt', 'gt_cnt'}))
                latest_eval_metrics = dict(eval_record)
                if improved:
                    best_mae_eval_metrics = dict(eval_record)
                if mse_improved:
                    best_mse_eval_metrics = dict(eval_record)
                if localization_improved:
                    best_localization_eval_metrics = dict(eval_record)
                with open(run_log_name, "a", encoding="utf-8") as log_file:
                    log_file.write("epoch:{}, mae:{}, mse:{}, time{}, \n\nbest mae:{}, best epoch: {}\n".format(
                                                epoch, mae, mse, t2 - t1, best_mae, best_epoch))
                    log_file.write(json.dumps(eval_record) + "\n")
                with open(output_dir / 'eval_history.jsonl', "a", encoding="utf-8") as eval_file:
                    eval_file.write(json.dumps(eval_record) + "\n")
                (output_dir / 'latest_eval_results.json').write_text(
                    json.dumps(eval_record, indent=2) + "\n",
                    encoding="utf-8",
                )
                # The epoch checkpoint is written before evaluation. Refresh it
                # so a resume retains this evaluation and every best-metric
                # record, even when MAE did not improve.
                utils.save_on_master(checkpoint_payload(epoch), output_dir / 'checkpoint.pth')

            if improved and utils.is_main_process():
                (output_dir / 'best_eval_results.json').write_text(
                    json.dumps(eval_record, indent=2) + "\n",
                    encoding="utf-8",
                )
                best_model_state, include_raw = best_state_for_eval_model(model_without_ddp, model_ema, eval_model_name)
                backup_existing_best_checkpoint(output_dir)
                best_mae_payload = checkpoint_payload(
                    epoch,
                    model_state=best_model_state,
                    include_raw_model=include_raw,
                )
                utils.save_on_master(best_mae_payload, output_dir / 'best_checkpoint.pth')
                utils.save_on_master(best_mae_payload, output_dir / 'best_complete_checkpoint.pth')
                utils.save_on_master(best_mae_payload, output_dir / 'best_mae_checkpoint.pth')

            if mse_improved and utils.is_main_process():
                (output_dir / 'best_mse_eval_results.json').write_text(
                    json.dumps(eval_record, indent=2) + "\n",
                    encoding="utf-8",
                )
                best_mse_model_state, include_raw = best_state_for_eval_model(
                    model_without_ddp,
                    model_ema,
                    eval_model_name,
                )
                utils.save_on_master(
                    checkpoint_payload(
                        epoch,
                        model_state=best_mse_model_state,
                        include_raw_model=include_raw,
                    ),
                    output_dir / 'best_mse_checkpoint.pth',
                )

            if localization_improved and utils.is_main_process():
                (output_dir / 'best_localization_eval_results.json').write_text(
                    json.dumps(eval_record, indent=2) + "\n",
                    encoding="utf-8",
                )
                best_loc_model_state, include_raw = best_state_for_eval_model(
                    model_without_ddp,
                    model_ema,
                    eval_model_name,
                )
                utils.save_on_master(
                    checkpoint_payload(
                        epoch,
                        model_state=best_loc_model_state,
                        include_raw_model=include_raw,
                    ),
                    output_dir / 'best_localization_checkpoint.pth',
                )

            abort_bad_count, abort_reason = should_abort_for_bad_count(args, epoch, test_stats)
            if abort_bad_count:
                bad_count_abort_reason = abort_reason
                print(f'\nWARNING: {abort_reason}')
                print('Stopping training early to avoid continuing a catastrophically miscalibrated run.\n')
                if utils.is_main_process():
                    abort_record = {
                        'epoch': epoch,
                        'aborted': True,
                        'reason': abort_reason,
                        'test_mae': float(mae),
                        'test_mse': float(mse),
                        'pred_cnt': float(test_stats.get('pred_cnt', 0.0)),
                        'gt_cnt': float(test_stats.get('gt_cnt', 0.0)),
                    }
                    with open(run_log_name, "a", encoding="utf-8") as log_file:
                        log_file.write(json.dumps(abort_record) + "\n")
                    (output_dir / 'abort_reason.json').write_text(
                        json.dumps(abort_record, indent=2) + "\n",
                        encoding="utf-8",
                    )
                break

    if utils.is_main_process():
        final_record = {
            'epoch': best_epoch,
            'test_mae': float(best_mae),
            'test_mse': float(best_mse),
            'best_epoch': best_epoch,
            'best_test_mae': float(best_mae),
            'best_test_mse': float(best_mse),
            'lowest_mse': float(lowest_mse),
            'best_mse_mae': float(best_mse_mae),
            'best_mse_epoch': int(best_mse_epoch),
            'best_loc_f1_large': float(best_loc_f1_large),
            'best_loc_f1_small': float(best_loc_f1_small),
            'best_loc_mae': float(best_loc_mae),
            'best_loc_mse': float(best_loc_mse),
            'best_loc_epoch': int(best_loc_epoch),
            'validation_protocol': validation_protocol,
            'final': True,
            'eval_model': getattr(args, 'eval_model', 'auto'),
        }
        if bad_count_abort_reason:
            final_record['aborted'] = True
            final_record['abort_reason'] = bad_count_abort_reason
        with open(run_log_name, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(final_record) + "\n")
        (output_dir / 'final_results.json').write_text(
            json.dumps(final_record, indent=2) + "\n",
            encoding="utf-8",
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args._explicit_args = get_explicit_arg_names(sys.argv[1:])
    main(args)
