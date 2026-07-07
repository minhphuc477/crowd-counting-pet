import argparse
import json
import random
import sys
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
from datasets import build_dataset
import util.misc as utils
from engine import (
    evaluate,
    evaluate_crowd_no_overlap,
    export_point_refinements,
    format_localization_metrics,
)
from models import build_model


ARCHITECTURE_OVERRIDE_KEYS = {
    'backbone',
    'no_pretrained_backbone',
    'allow_random_backbone_fallback',
    'timm_adapter',
    'timm_output_norm',
    'scale_fusion',
    'scale_fusion_activation',
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
    'ifi_interpolation',
    'ifi_pos_dim',
    'ifi_mlp_hidden_dim',
    'ifi_activation',
    'context_patch_size',
    'quad_context_mixer',
    'quad_context_levels',
    'quad_context_shift',
    'quad_context_mid_dim',
    'quad_context_activation',
    'splitter_head',
    'splitter_hidden_dim',
    'splitter_activation',
    'sparse_stride',
    'dense_stride',
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
    'eval_count_tail_threshold',
}


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)

    # model parameters
    # - backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--no_pretrained_backbone', action='store_true',
                        help='initialize the backbone randomly instead of loading timm/ImageNet weights')
    parser.add_argument('--allow_random_backbone_fallback', action='store_true',
                        help='allow timm backbones to continue with random init if pretrained weights cannot load')
    parser.add_argument('--timm_adapter', default='lite_fpn', choices=('pet_fpn', 'lite_fpn', 'rcc_fpn', 'direct', 'fpn'),
                        help='adapter used to map timm features into PET 4x/8x features')
    parser.add_argument('--timm_output_norm', default='gn', choices=('gn', 'none'),
                        help='normalization after timm feature adapter; gn preserves old timm behavior, none is VGG-like')
    parser.add_argument('--scale_fusion', default='none', choices=('none', 'bidirectional'))
    parser.add_argument('--scale_fusion_activation', default='gelu', choices=('relu', 'gelu'))
    parser.add_argument('--fusion_mhf_mode', default='none', choices=('none', 'cem', 'cem_msem', 'full'))
    parser.add_argument('--fusion_mhf_heads', default=1, type=int)
    parser.add_argument('--fusion_mhf_position', default='before', choices=('before', 'post'))
    parser.add_argument('--fusion_mhf_strength', default=1.0, type=float)
    parser.add_argument('--fusion_mhf_activation', default='gelu', choices=('relu', 'gelu'))
    parser.add_argument('--fusion_mhf_impl', default='residual', choices=('residual', 'vmambacc'))
    parser.add_argument('--fusion_fpn_type', default='fpn', choices=('fpn', 'hs2fpn'))
    parser.add_argument('--fusion_mhf_reduction', default=4, type=int)
    parser.add_argument('--fusion_mhf_norm', default='none', choices=('none', 'bn', 'gn'))
    parser.add_argument('--fusion_mhf_spatial_kernel', default=7, type=int)
    parser.add_argument('--fusion_mhf_output_activation', default='none', choices=('none', 'sigmoid'))
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
    parser.add_argument('--transformer_activation', default='relu', choices=('relu', 'gelu'))
    parser.add_argument('--transformer_norm_style', default='post', choices=('post', 'pre'))
    parser.add_argument('--decoder_attention', default='softmax', choices=('softmax', 'linear'),
                        help='attention used inside decoder layers; softmax matches official PET')
    parser.add_argument('--decoder_memory_halo', default=0, type=int,
                        help='extra 8x encoder-feature tokens around each decoder cross-attention memory window')
    parser.add_argument('--decoder_global_context', action='store_true')
    parser.add_argument('--decoder_global_context_mode', default='residual', choices=('residual', 'token'))
    parser.add_argument('--enc_win_sizes', default='', type=str,
                        help='encoder window sizes as "w,h;w,h;..."; empty keeps paper PET defaults')
    parser.add_argument('--enc_shift_mode', default='none', choices=('none', 'swin'))
    parser.add_argument('--sparse_dec_win_size', default='', type=str,
                        help='sparse decoder window size as "w,h"; empty keeps paper PET default')
    parser.add_argument('--dense_dec_win_size', default='', type=str,
                        help='dense decoder window size as "w,h"; empty keeps paper PET default')
    parser.add_argument('--context_patch_size', default='', type=str,
                        help='quadtree splitter context patch size as "w,h"; empty keeps paper PET default')
    parser.add_argument('--quad_context_mixer', default='none', choices=('none', 'lite'))
    parser.add_argument('--quad_context_levels', default=2, type=int)
    parser.add_argument('--quad_context_shift', default=1, type=int)
    parser.add_argument('--quad_context_mid_dim', default=128, type=int)
    parser.add_argument('--quad_context_activation', default='gelu', choices=('relu', 'gelu'))
    parser.add_argument('--perspective_mixer', default='none', choices=('none', 'drf'))
    parser.add_argument('--perspective_mixer_dilations', default='1,2,3', type=str)
    parser.add_argument('--perspective_mixer_mid_dim', default=64, type=int)
    parser.add_argument('--perspective_mixer_activation', default='gelu', choices=('relu', 'gelu'))
    parser.add_argument('--splitter_head', default='pool', choices=('pool', 'conv'))
    parser.add_argument('--splitter_hidden_dim', default=128, type=int)
    parser.add_argument('--splitter_activation', default='gelu', choices=('relu', 'gelu'))
    
    # loss parameters
    # - matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - loss coefficients
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)       # classification loss coefficient
    parser.add_argument('--point_loss_coef', default=5.0, type=float)    # regression loss coefficient
    parser.add_argument('--pq_sparse_coef', default=1.0, type=float)
    parser.add_argument('--pq_dense_coef', default=1.0, type=float)
    parser.add_argument('--pq_dense_start_epoch', default=0, type=int)
    parser.add_argument('--pq_dense_warmup_epochs', default=0, type=int)
    parser.add_argument('--branch_target_routing', default='none', choices=('none', 'gt_count'))
    parser.add_argument('--class_loss_type', default='ce', choices=('ce', 'focal'))
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--focal_gamma', default=2.0, type=float)
    parser.add_argument('--class_prior_prob', default=-1.0, type=float)
    parser.add_argument('--strict_model_checks', action='store_true')
    parser.add_argument('--count_loss_coef', default=0.0, type=float)
    parser.add_argument('--count_loss_gate', default='detach', choices=('none', 'detach', 'soft', 'hard'))
    parser.add_argument('--count_loss_type', default='log_l1', choices=('log_l1', 'l1', 'smooth_l1', 'over_log_l1'))
    parser.add_argument('--count_loss_budget_margin', default=1.0, type=float)
    parser.add_argument('--count_loss_start_epoch', default=-1, type=int)
    parser.add_argument('--count_head_loss_coef', default=0.0, type=float)
    parser.add_argument('--count_head_loss_type', default='log_l1', choices=('log_l1', 'l1', 'smooth_l1'))
    parser.add_argument('--count_head_start_epoch', default=0, type=int)
    parser.add_argument('--count_head_end_epoch', default=-1, type=int)
    parser.add_argument('--allow_count_head_fresh_train', action='store_true')
    parser.add_argument('--allow_count_head_from_start', action='store_true')
    parser.add_argument('--safe_count_head_start_epoch', default=250, type=int)
    parser.add_argument('--count_head_init_count', default=40.0, type=float)
    parser.add_argument('--count_head_init_cells', default=1024.0, type=float)
    parser.add_argument('--count_head_feature_grad_scale', default=1.0, type=float)
    parser.add_argument('--train_count_head_only', action='store_true')
    parser.add_argument('--density_map_loss_coef', default=0.0, type=float)
    parser.add_argument('--allow_unstable_density_map_loss', action='store_true')
    parser.add_argument('--density_map_loss_type', default='log_l1', choices=('log_l1', 'l1', 'smooth_l1'))
    parser.add_argument('--density_map_pos_weight', default=10.0, type=float)
    parser.add_argument('--density_map_grad_scale', default=1.0, type=float)
    parser.add_argument('--density_map_start_epoch', default=0, type=int)
    parser.add_argument('--density_map_end_epoch', default=-1, type=int)
    parser.add_argument('--measure_loss_coef', default=0.0, type=float)
    parser.add_argument('--measure_loss_distribution_coef', default=1.0, type=float)
    parser.add_argument('--measure_loss_count_coef', default=0.25, type=float)
    parser.add_argument('--measure_loss_transport_coef', default=0.0, type=float)
    parser.add_argument('--measure_loss_start_epoch', default=0, type=int)
    parser.add_argument('--measure_loss_end_epoch', default=-1, type=int)
    parser.add_argument('--measure_loss_warmup_epochs', default=0, type=int)
    parser.add_argument('--measure_loss_feature_grad_scale', default=1.0, type=float)
    parser.add_argument('--measure_loss_feature_grad_start_epoch', default=0, type=int)
    parser.add_argument('--measure_loss_feature_grad_warmup_epochs', default=0, type=int)
    parser.add_argument('--measure_loss_sinkhorn_iters', default=20, type=int)
    parser.add_argument('--measure_loss_sinkhorn_epsilon', default=0.05, type=float)
    parser.add_argument('--measure_loss_init_count', default=40.0, type=float)
    parser.add_argument('--measure_loss_init_cells', default=1024.0, type=float)
    parser.add_argument('--foreground_loss_coef', default=0.0, type=float)
    parser.add_argument('--foreground_sigma', default=8.0, type=float)
    parser.add_argument('--foreground_neg_shrink', default=16.0, type=float)
    parser.add_argument('--foreground_init_prior', default=0.5, type=float)
    parser.add_argument('--region_count_loss_coef', default=0.0, type=float)
    parser.add_argument('--region_count_grid', default=4, type=int)
    parser.add_argument('--region_count_gate', default='detach', choices=('none', 'detach', 'soft', 'hard'))
    parser.add_argument('--region_count_type', default='log_l1', choices=('log_l1', 'l1', 'smooth_l1'))
    parser.add_argument('--region_count_start_epoch', default=-1, type=int)
    parser.add_argument('--region_count_end_epoch', default=-1, type=int)
    parser.add_argument('--branch_exclusion_loss_coef', default=0.0, type=float)
    parser.add_argument('--branch_exclusion_start_epoch', default=0, type=int)
    parser.add_argument('--branch_exclusion_end_epoch', default=-1, type=int)
    parser.add_argument('--bayesian_loss_coef', default=0.0, type=float)
    parser.add_argument('--bayesian_sigma', default=8.0, type=float)
    parser.add_argument('--bayesian_bg_coef', default=0.05, type=float)
    parser.add_argument('--bayesian_loss_gate', default='detach', choices=('none', 'detach', 'soft', 'hard'))
    parser.add_argument('--bayesian_start_epoch', default=-1, type=int)
    parser.add_argument('--bayesian_end_epoch', default=-1, type=int)
    parser.add_argument('--apg_loss_coef', default=0.0, type=float)
    parser.add_argument('--apg_pos_k', default=1, type=int)
    parser.add_argument('--apg_point_coef', default=5.0, type=float)
    parser.add_argument('--apg_bg_coef', default=0.0, type=float)
    parser.add_argument('--apg_bg_k', default=0, type=int)
    parser.add_argument('--apg_bg_min_dist', default=12.0, type=float)
    parser.add_argument('--apg_bg_offset_coef', default=0.0, type=float)
    parser.add_argument('--apg_local_neg_coef', default=0.0, type=float)
    parser.add_argument('--apg_local_neg_k', default=0, type=int)
    parser.add_argument('--apg_local_neg_min_dist', default=2.0, type=float)
    parser.add_argument('--apg_local_neg_max_dist', default=8.0, type=float)
    parser.add_argument('--apg_local_neg_offset_coef', default=1.0, type=float)
    parser.add_argument('--apg_start_epoch', default=0, type=int)
    parser.add_argument('--apg_warmup_epochs', default=0, type=int)
    parser.add_argument('--apg_sparse_coef', default=1.0, type=float)
    parser.add_argument('--apg_dense_coef', default=1.0, type=float)
    parser.add_argument('--apg_dense_start_epoch', default=-1, type=int)
    parser.add_argument('--apg_dense_warmup_epochs', default=-1, type=int)
    parser.add_argument('--apg_end_epoch', default=-1, type=int)
    parser.add_argument('--apg_contrastive_coef', default=0.0, type=float)
    parser.add_argument('--apg_neg_k', default=4, type=int)
    parser.add_argument('--apg_margin', default=1.0, type=float)
    parser.add_argument('--apg_consistency_coef', default=0.0, type=float)
    parser.add_argument('--apg_consistency_k', default=4, type=int)
    parser.add_argument('--apg_consistency_sigma', default=8.0, type=float)
    parser.add_argument('--apg_soft_loss_coef', default=0.0, type=float)
    parser.add_argument('--apg_soft_pos_k', default=4, type=int)
    parser.add_argument('--apg_soft_sigma', default=6.0, type=float)
    parser.add_argument('--apg_soft_point_coef', default=2.0, type=float)
    parser.add_argument('--query_feature_interpolation', default='nearest', choices=('nearest', 'implicit'))
    parser.add_argument('--ifi_interpolation', default='bilinear', choices=('bilinear', 'implicit'))
    parser.add_argument('--ifi_pos_dim', default=32, type=int)
    parser.add_argument('--ifi_mlp_hidden_dim', default=256, type=int)
    parser.add_argument('--ifi_activation', default='gelu', choices=('relu', 'gelu'))
    parser.add_argument('--ifi_loss_coef', default=0.0, type=float)
    parser.add_argument('--ifi_head_source', default='separate', choices=('separate', 'sparse', 'dense', 'both', 'routed'))
    parser.add_argument('--ifi_point_coef', default=1.0, type=float)
    parser.add_argument('--ifi_neg_k', default=4, type=int)
    parser.add_argument('--ifi_neg_radius', default=12.0, type=float)
    parser.add_argument('--ifi_neg_min_dist', default=4.0, type=float)
    parser.add_argument('--ifi_start_epoch', default=0, type=int)
    parser.add_argument('--ifi_end_epoch', default=-1, type=int)
    parser.add_argument('--qd_apg_loss_coef', default=0.0, type=float)
    parser.add_argument('--qd_apg_point_coef', default=5.0, type=float)
    parser.add_argument('--qd_apg_suppress_coef', default=0.5, type=float)
    parser.add_argument('--qd_apg_start_epoch', default=0, type=int)
    parser.add_argument('--qd_apg_end_epoch', default=-1, type=int)
    parser.add_argument('--qd_apg_route_source', default='gt_count', choices=('gt_count', 'split_map'))
    parser.add_argument('--routed_apg_loss_coef', default=0.0, type=float)
    parser.add_argument('--routed_apg_point_coef', default=5.0, type=float)
    parser.add_argument('--routed_apg_pos_k', default=1, type=int)
    parser.add_argument('--routed_apg_bg_coef', default=0.0, type=float)
    parser.add_argument('--routed_apg_bg_k', default=0, type=int)
    parser.add_argument('--routed_apg_bg_min_dist', default=12.0, type=float)
    parser.add_argument('--routed_apg_start_epoch', default=0, type=int)
    parser.add_argument('--routed_apg_end_epoch', default=-1, type=int)
    parser.add_argument('--routed_apg_warmup_epochs', default=0, type=int)
    parser.add_argument('--routed_apg_min_weight', default=0.1, type=float)
    parser.add_argument('--routed_apg_source', default='gt_count', choices=('gt_count', 'split_map'))
    parser.add_argument('--routed_apg_gate', default='detach', choices=('detach', 'soft'))
    parser.add_argument('--inheritance_loss_coef', default=0.0, type=float)
    parser.add_argument('--inheritance_sparse_coef', default=1.0, type=float)
    parser.add_argument('--inheritance_dense_coef', default=1.0, type=float)
    parser.add_argument('--inheritance_consistency_coef', default=0.25, type=float)
    parser.add_argument('--inheritance_start_epoch', default=0, type=int)
    parser.add_argument('--inheritance_end_epoch', default=-1, type=int)
    parser.add_argument('--inheritance_gate', default='gt_count', choices=('gt_count', 'split_map'))
    parser.add_argument('--zip_count_loss_coef', default=0.0, type=float)
    parser.add_argument('--zip_count_block_size', default=16, type=int)
    parser.add_argument('--zip_count_feature_source', default='encoder8x', choices=('encoder8x', 'fpn4x8x'))
    parser.add_argument('--zip_count_bin_centers',
                        default='1,2,3,4,5,6,7,8,9,10,11.38,13.38,16.26', type=str)
    parser.add_argument('--zip_count_zero_prior', default=0.9, type=float)
    parser.add_argument('--zip_count_ce_coef', default=1.0, type=float)
    parser.add_argument('--zip_count_count_coef', default=1.0, type=float)
    parser.add_argument('--zip_count_start_epoch', default=0, type=int)
    parser.add_argument('--zip_count_end_epoch', default=-1, type=int)
    parser.add_argument('--zip_count_warmup_epochs', default=0, type=int)
    parser.add_argument('--zip_count_feature_grad_scale', default=1.0, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")   # cross-entropy weights
    parser.add_argument('--pet_loss_variant', default='paper', choices=('paper', 'balanced'))
    parser.add_argument('--split_loss_variant', default='auto', choices=('auto', 'none', 'paper', 'gt', 'paper_gt'))
    parser.add_argument('--negative_loss_coef', default=0.1, type=float)
    parser.add_argument('--non_div_loss_coef', default=0.25, type=float)
    parser.add_argument('--quadtree_loss_coef', default=0.1, type=float)
    parser.add_argument('--quadtree_prior_coef', default=0.025, type=float)
    parser.add_argument('--split_count_threshold', default=2, type=int)
    parser.add_argument('--split_pos_weight', default=1.0, type=float)
    parser.add_argument('--split_threshold', default=0.5, type=float)
    parser.add_argument('--split_threshold_quantile', default=0.5, type=float)
    parser.add_argument('--query_prune_threshold', default=0.5, type=float)
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--eval_nms_radius', default=0.0, type=float)
    parser.add_argument('--eval_branch_gate', default='none', choices=('none', 'query', 'pred'))
    parser.add_argument('--eval_soft_split_gate', default='none', choices=('none', 'query', 'pred'))
    parser.add_argument('--eval_foreground_gate', default='none', choices=('none', 'query', 'pred'))
    parser.add_argument('--eval_foreground_gate_mode', default='suppress', choices=('suppress', 'logit_add'))
    parser.add_argument('--eval_foreground_gate_strength', default=0.75, type=float)
    parser.add_argument('--eval_count_mode', default='threshold', choices=('threshold', 'count_head_topk'))
    parser.add_argument('--eval_count_source', default='pet', choices=('pet', 'zip', 'zip_pet_blend', 'zip_tail_blend'))
    parser.add_argument('--eval_count_blend_alpha', default=0.5, type=float)
    parser.add_argument('--eval_count_tail_threshold', default=1500.0, type=float)
    parser.add_argument('--eval_count_head_min_score', default=0.5, type=float)
    parser.add_argument('--eval_dense_start_epoch', default=0, type=int)
    parser.add_argument('--eval_dense_residual_mode', default='none', choices=('none', 'count_head'))
    parser.add_argument('--eval_dense_residual_start_epoch', default=0, type=int)
    parser.add_argument('--eval_dense_residual_min_score', default=0.0, type=float)
    parser.add_argument('--eval_score_calibration', default='none', choices=('none', 'count_head_bias'))
    parser.add_argument('--eval_score_calibration_strength', default=1.0, type=float)
    parser.add_argument('--eval_score_calibration_start_epoch', default=0, type=int)
    parser.add_argument('--eval_score_calibration_min_bias', default=-8.0, type=float)
    parser.add_argument('--eval_score_calibration_max_bias', default=8.0, type=float)
    parser.add_argument('--eval_score_calibration_count_blend', default=1.0, type=float)
    parser.add_argument('--eval_score_calibration_count_ratio_min', default=0.0, type=float)
    parser.add_argument('--eval_score_calibration_count_ratio_max', default=1e6, type=float)
    parser.add_argument('--no_eval_filter_invalid_points', action='store_true')
    parser.add_argument('--eval_debug_counting', action='store_true')
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

    # dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="", type=str)
    parser.add_argument('--patch_size', default=256, type=int,
                        help='training crop size for crop-based crowd datasets')
    parser.add_argument('--patch_size_choices', default='', type=str)
    parser.add_argument('--crop_attempts', default=1, type=int)
    parser.add_argument('--min_crop_points', default=0, type=int)
    parser.add_argument('--eval_max_size', default=-1, type=int,
                        help='high-resolution long-side cap; -1 uses PET defaults (QNRF 1536, JHU/NWPU 2048), 0 disables resizing')
    parser.add_argument('--nwpu_eval_split', default='val', choices=('val', 'test', 'train'),
                        help='NWPU split used when --dataset_file NWPU is evaluated')
    parser.add_argument('--jhu_eval_split', default='val', choices=('val', 'test', 'train'),
                        help='JHU-Crowd++ split used when --dataset_file JHU is evaluated')
    parser.add_argument('--ucfcc50_fold', default=0, type=int, choices=range(5),
                        help='held-out UCF-CC-50 fold index (0-4)')
    parser.add_argument('--ucfcc50_fold_seed', default=42, type=int,
                        help='seed used to create UCF-CC-50 folds without a manifest')
    parser.add_argument('--ucfcc50_fold_manifest', default='', type=str,
                        help='JSON file containing the exact five UCF-CC-50 folds')
    parser.add_argument('--nwpu_sigma_mode', default='official', choices=('area', 'diag', 'min_diag', 'official'),
                        help='fallback localization sigma derived from NWPU boxes when annotation sigma is absent')

    # misc parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_allow_arch_change', action='store_true',
                        help='allow explicitly requested architecture changes and load checkpoint non-strictly')
    parser.add_argument('--vis_dir', default="")
    parser.add_argument('--results_file', default='',
                        help='where to save eval metrics; empty writes eval_results.json next to checkpoint')
    parser.add_argument('--per_image_results_file', default='',
                        help='optional JSON path for per-image count/localization errors')
    parser.add_argument('--per_image_predictions_file', default='',
                        help='optional JSON path for final filtered point coordinates and scores')
    parser.add_argument('--refinement_predictions_file', default='',
                        help='run original annotations as custom PET queries and export refined points to JSON')
    parser.add_argument('--eval_image_set', default='val', choices=('val', 'train_eval', 'train_holdout'),
                        help='dataset split used by eval.py; train_holdout reproduces main.py validation_protocol=train_holdout')
    parser.add_argument('--train_holdout_fraction', default=0.1, type=float,
                        help='fraction of training split used by --eval_image_set train_holdout')
    parser.add_argument('--train_holdout_seed', default=42, type=int,
                        help='seed used by --eval_image_set train_holdout')
    parser.add_argument('--eval_tile_size', default=0, type=int,
                        help='positive value enables tiled full-resolution eval for images larger than this size')
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
    parser.add_argument('--override_score_threshold', default=None, type=float,
                        help='override the checkpoint score threshold at evaluation time')
    parser.add_argument('--override_split_threshold', default=None, type=float,
                        help='override the checkpoint split threshold at evaluation time')
    parser.add_argument('--override_split_threshold_quantile', default=None, type=float,
                        help='override the checkpoint split-threshold quantile at evaluation time')
    parser.add_argument('--override_query_prune_threshold', default=None, type=float,
                        help='override fixed PET decoder-window pruning threshold')
    parser.add_argument('--checkpoint_model_key', default='auto',
                        choices=('auto', 'model', 'model_ema', 'model_raw'),
                        help='checkpoint state to evaluate; auto prefers model_ema when present')
    parser.add_argument('--tta_flip', action='store_true',
                        help='average original and horizontal-flip predicted counts at evaluation time')
    parser.add_argument('--tta_scales', default='1.0',
                        help='comma-separated eval scales; dimensions are rounded to PET-compatible 256 multiples')
    parser.add_argument('--eval_protocol', default='pet', choices=('pet', 'crowd_no_overlap'),
                        help='pet uses the current metric dict; crowd_no_overlap uses P2PNet/APGCC-style MAE/RMSE')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--deterministic', dest='deterministic', action='store_true', default=True)
    parser.add_argument('--no_deterministic', dest='deterministic', action='store_false')
    parser.add_argument('--amp_dtype', default='auto', choices=('auto', 'float16', 'bfloat16'))

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


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
    always_runtime_keys = {
        'resume', 'device', 'vis_dir', 'results_file', 'data_path', 'dataset_file',
        'nwpu_eval_split', 'jhu_eval_split',
        'num_workers', 'seed',
        'per_image_results_file', 'per_image_predictions_file',
        'refinement_predictions_file',
        'eval_image_set',
        'override_score_threshold', 'override_split_threshold', 'override_split_threshold_quantile',
        'override_query_prune_threshold',
        'checkpoint_model_key', 'deterministic', 'amp_dtype', 'strict_model_checks',
    }
    explicit_only_runtime_keys = {
        'eval_max_size',
        'nwpu_sigma_mode',
        'eval_tile_size', 'eval_tile_overlap', 'eval_tile_nms_radius',
        'eval_tile_min_gt', 'eval_tile_max_tiles', 'eval_tile_trigger_count', 'eval_tile_trigger_area',
        'override_score_threshold', 'override_split_threshold', 'override_split_threshold_quantile',
        'override_query_prune_threshold',
        'tta_flip', 'tta_scales',
        'eval_nms_radius', 'eval_branch_gate', 'eval_soft_split_gate',
        'eval_foreground_gate', 'eval_foreground_gate_mode', 'eval_foreground_gate_strength',
        'eval_count_mode', 'eval_count_source', 'eval_count_blend_alpha',
        'eval_count_tail_threshold', 'eval_count_head_min_score',
        'eval_score_calibration', 'eval_score_calibration_strength',
        'eval_score_calibration_start_epoch',
        'eval_score_calibration_min_bias', 'eval_score_calibration_max_bias',
        'eval_score_calibration_count_blend',
        'eval_score_calibration_count_ratio_min',
        'eval_score_calibration_count_ratio_max',
        'no_eval_filter_invalid_points', 'eval_debug_counting',
        'no_localization_metrics', 'localization_large_threshold', 'localization_small_threshold',
        'localization_protocol', 'localization_large_scale', 'localization_small_scale',
        'eval_protocol', 'resume_allow_arch_change',
        'train_holdout_fraction', 'train_holdout_seed',
        'ucfcc50_fold', 'ucfcc50_fold_seed', 'ucfcc50_fold_manifest',
    }
    explicit_args = set(getattr(args, '_explicit_args', set()))
    if 'eval_dense_start_epoch' in explicit_args:
        explicit_only_runtime_keys.add('eval_dense_start_epoch')
    for key in ('eval_dense_residual_mode', 'eval_dense_residual_start_epoch', 'eval_dense_residual_min_score'):
        if key in explicit_args:
            explicit_only_runtime_keys.add(key)
    if getattr(args, 'resume_allow_arch_change', False):
        explicit_only_runtime_keys.update(key for key in ARCHITECTURE_OVERRIDE_KEYS if key in explicit_args)
    for key in always_runtime_keys:
        setattr(merged, key, getattr(args, key))
    for key in explicit_only_runtime_keys:
        if key in explicit_args:
            setattr(merged, key, getattr(args, key))
    if hasattr(args, '_explicit_args'):
        setattr(merged, '_explicit_args', getattr(args, '_explicit_args'))
    return merged


def checkpoint_arg(checkpoint, key, default=None):
    checkpoint_args = checkpoint.get('args') if checkpoint is not None else None
    if checkpoint_args is None:
        return default
    if isinstance(checkpoint_args, dict):
        return checkpoint_args.get(key, default)
    return getattr(checkpoint_args, key, default)


def should_skip_pretrained_backbone(args, checkpoint):
    if checkpoint is None:
        return bool(getattr(args, 'no_pretrained_backbone', False))
    explicit_args = set(getattr(args, '_explicit_args', set()))
    requested_backbone = getattr(args, 'backbone', None)
    checkpoint_backbone = checkpoint_arg(checkpoint, 'backbone', requested_backbone)
    changed_backbone = (
        getattr(args, 'resume_allow_arch_change', False)
        and 'backbone' in explicit_args
        and checkpoint_backbone != requested_backbone
    )
    if changed_backbone:
        return 'no_pretrained_backbone' in explicit_args and bool(getattr(args, 'no_pretrained_backbone', False))
    return True


def apply_eval_overrides(args):
    override_score_threshold = getattr(args, 'override_score_threshold', None)
    override_split_threshold = getattr(args, 'override_split_threshold', None)
    override_split_threshold_quantile = getattr(args, 'override_split_threshold_quantile', None)
    override_query_prune_threshold = getattr(args, 'override_query_prune_threshold', None)
    if override_score_threshold is not None:
        args.score_threshold = float(override_score_threshold)
    if override_split_threshold is not None:
        args.split_threshold = float(override_split_threshold)
    if override_split_threshold_quantile is not None:
        args.split_threshold_quantile = float(override_split_threshold_quantile)
    if override_query_prune_threshold is not None:
        args.query_prune_threshold = float(override_query_prune_threshold)
    return args


def parse_tta_scales(value):
    if isinstance(value, (list, tuple)):
        raw_values = value
    else:
        raw_values = str(value).replace(';', ',').split(',')
    scales = []
    for raw in raw_values:
        text = str(raw).strip()
        if not text:
            continue
        scale = float(text)
        if scale <= 0:
            raise ValueError('tta_scales must contain positive values')
        scales.append(scale)
    return tuple(dict.fromkeys(scales)) or (1.0,)


def scalar_eval_metrics(test_stats, skip=()):
    skip = set(skip)
    metrics = {}
    for key, value in test_stats.items():
        if key in skip:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            metrics[key] = float(value)
    return metrics


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


class IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(int(index) for index in indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def build_train_holdout_indices(num_samples, holdout_fraction, seed):
    if num_samples < 2:
        raise ValueError('train-holdout evaluation requires at least 2 training samples')
    holdout_fraction = float(holdout_fraction)
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError('--train_holdout_fraction must be in (0, 1)')
    num_val = int(round(num_samples * holdout_fraction))
    num_val = max(1, min(num_samples - 1, num_val))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    permutation = torch.randperm(num_samples, generator=generator).tolist()
    return sorted(permutation[num_val:]), sorted(permutation[:num_val])


def build_eval_dataset(args):
    eval_image_set = getattr(args, 'eval_image_set', 'val')
    if eval_image_set == 'train_holdout':
        dataset = build_dataset(image_set='train_eval', args=args)
        _, val_indices = build_train_holdout_indices(
            len(dataset),
            getattr(args, 'train_holdout_fraction', 0.1),
            getattr(args, 'train_holdout_seed', args.seed),
        )
        return IndexedSubset(dataset, val_indices), eval_image_set
    return build_dataset(image_set=eval_image_set, args=args), eval_image_set


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    set_reproducibility(seed, deterministic=getattr(args, 'deterministic', True))
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(seed)

    checkpoint = None
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        args = merge_checkpoint_args(args, checkpoint)
        args.no_pretrained_backbone = should_skip_pretrained_backbone(args, checkpoint)
    args = apply_eval_overrides(args)
    if (
        getattr(args, 'per_image_predictions_file', '')
        or getattr(args, 'refinement_predictions_file', '')
    ) and (
        bool(getattr(args, 'tta_flip', False))
        or any(
            abs(scale - 1.0) > 1e-6
            for scale in parse_tta_scales(getattr(args, 'tta_scales', '1.0'))
        )
    ):
        raise ValueError(
            'point-coordinate export does not support count-only TTA; use '
            '--tta_scales 1.0 without --tta_flip'
        )
    print(args)

    # build model
    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('params:', n_parameters/1e6)

    # build dataset
    dataset_val, eval_image_set = build_eval_dataset(args)
    if utils.is_main_process():
        print(f'eval image set: {eval_image_set}, samples={len(dataset_val)}')

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 worker_init_fn=seed_worker, generator=data_loader_generator)

    # load pretrained model
    cur_epoch = 0
    eval_model_key = 'model'
    if checkpoint is not None:
        model_state, eval_model_key = utils.get_checkpoint_model_state(
            checkpoint,
            getattr(args, 'checkpoint_model_key', 'auto'),
        )
        strict_load = not getattr(args, 'resume_allow_arch_change', False)
        incompatible = model_without_ddp.load_state_dict(model_state, strict=strict_load)
        if not strict_load and utils.is_main_process():
            missing = getattr(incompatible, 'missing_keys', [])
            unexpected = getattr(incompatible, 'unexpected_keys', [])
            print(
                'non-strict eval load:',
                f'missing_keys={len(missing)}',
                f'unexpected_keys={len(unexpected)}',
            )
            if missing:
                print('  missing:', missing[:20])
            if unexpected:
                print('  unexpected:', unexpected[:20])
        print(f'loaded checkpoint model state: {eval_model_key}')
        cur_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0

    if args.refinement_predictions_file:
        rows = export_point_refinements(
            model,
            data_loader_val,
            device,
            args.refinement_predictions_file,
            epoch=cur_epoch,
        )
        print(
            f'point refinements saved to: {args.refinement_predictions_file} '
            f'({len(rows)} images)'
        )
        return
    
    # evaluation
    vis_dir = None if args.vis_dir == "" else args.vis_dir
    tta_scales = parse_tta_scales(getattr(args, 'tta_scales', '1.0'))
    if args.eval_protocol == 'crowd_no_overlap':
        test_stats = evaluate_crowd_no_overlap(
            model,
            data_loader_val,
            device,
            epoch=cur_epoch,
            vis_dir=vis_dir,
            tta_flip=args.tta_flip,
            tta_scales=tta_scales,
            localization_metrics=not args.no_localization_metrics,
            localization_large_threshold=args.localization_large_threshold,
            localization_small_threshold=args.localization_small_threshold,
            localization_protocol=args.localization_protocol,
            localization_large_scale=args.localization_large_scale,
            localization_small_scale=args.localization_small_scale,
        )
        mae, mse = test_stats['mae'], test_stats['mse']
    else:
        test_stats = evaluate(
            model,
            data_loader_val,
            device,
            epoch=cur_epoch,
            vis_dir=vis_dir,
            tta_flip=args.tta_flip,
            tta_scales=tta_scales,
            localization_metrics=not args.no_localization_metrics,
            localization_large_threshold=args.localization_large_threshold,
            localization_small_threshold=args.localization_small_threshold,
            localization_protocol=args.localization_protocol,
            localization_large_scale=args.localization_large_scale,
            localization_small_scale=args.localization_small_scale,
            per_image_results_file=args.per_image_results_file,
            per_image_predictions_file=args.per_image_predictions_file,
            eval_tile_size=args.eval_tile_size,
            eval_tile_overlap=args.eval_tile_overlap,
            eval_tile_nms_radius=args.eval_tile_nms_radius,
            eval_tile_min_gt=args.eval_tile_min_gt,
            eval_tile_max_tiles=args.eval_tile_max_tiles,
            eval_tile_trigger_count=args.eval_tile_trigger_count,
            eval_tile_trigger_area=args.eval_tile_trigger_area,
        )
        mae, mse = test_stats['mae'], test_stats['mse']
    loc_text = format_localization_metrics(test_stats, prefix=', ')
    line = f'\nepoch: {cur_epoch}, mae: {mae}, mse: {mse}{loc_text}'
    print(line)
    if utils.is_main_process():
        if args.results_file:
            results_file = Path(args.results_file)
        elif args.resume and not args.resume.startswith('https'):
            results_file = Path(args.resume).resolve().parent / 'eval_results.json'
        else:
            results_file = Path('eval_results.json')
        results_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'epoch': int(cur_epoch),
            'eval_mae': float(mae),
            'eval_mse': float(mse),
            'pred_cnt': float(test_stats.get('pred_cnt', 0.0)),
            'gt_cnt': float(test_stats.get('gt_cnt', 0.0)),
            'checkpoint': args.resume,
            'eval_model': eval_model_key,
            'dataset_file': args.dataset_file,
            'eval_image_set': eval_image_set,
            'train_holdout_fraction': (
                float(getattr(args, 'train_holdout_fraction', 0.1))
                if eval_image_set == 'train_holdout'
                else None
            ),
            'train_holdout_seed': (
                int(getattr(args, 'train_holdout_seed', args.seed))
                if eval_image_set == 'train_holdout'
                else None
            ),
            'eval_protocol': args.eval_protocol,
            'tta_flip': bool(args.tta_flip),
            'tta_scales': list(tta_scales),
            'score_threshold': float(getattr(args, 'score_threshold', 0.5)),
            'split_threshold': float(getattr(args, 'split_threshold', 0.5)),
            'query_prune_threshold': float(getattr(args, 'query_prune_threshold', 0.5)),
            'eval_nms_radius': float(getattr(args, 'eval_nms_radius', 0.0)),
            'eval_branch_gate': getattr(args, 'eval_branch_gate', 'none'),
            'eval_soft_split_gate': getattr(args, 'eval_soft_split_gate', 'none'),
            'eval_count_mode': getattr(args, 'eval_count_mode', 'threshold'),
            'eval_count_source': getattr(args, 'eval_count_source', 'pet'),
            'eval_count_blend_alpha': float(getattr(args, 'eval_count_blend_alpha', 0.5)),
            'eval_count_tail_threshold': float(getattr(args, 'eval_count_tail_threshold', 1500.0)),
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
        payload.update(scalar_eval_metrics(test_stats, skip={'mae', 'mse', 'pred_cnt', 'gt_cnt'}))
        results_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f'eval results saved to: {results_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args._explicit_args = {
        token[2:].split('=', 1)[0].replace('-', '_')
        for token in sys.argv[1:]
        if token.startswith('--')
    }
    main(args)
