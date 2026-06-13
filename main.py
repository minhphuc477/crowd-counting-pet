import argparse
import copy
import datetime
import json
import math
import random
import sys
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets import build_dataset
from engine import evaluate, evaluate_crowd_no_overlap, train_one_epoch
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
    'context_patch_size',
    'quad_context_mixer',
    'quad_context_levels',
    'quad_context_shift',
    'quad_context_mid_dim',
    'quad_context_activation',
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
    parser.add_argument('--timm_adapter', default='lite_fpn', choices=('pet_fpn', 'lite_fpn', 'direct', 'fpn'),
                        help='adapter used to map timm features into PET 4x/8x features')
    parser.add_argument('--timm_output_norm', default='gn', choices=('gn', 'none'),
                        help='normalization after timm feature adapter; gn preserves old timm behavior, none is VGG-like')
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
    parser.add_argument('--count_loss_gate', default='detach', choices=('detach', 'soft', 'hard'),
                        help='routing gates used by count loss; detach calibrates scores without splitter gradients')
    parser.add_argument('--count_loss_type', default='log_l1', choices=('log_l1', 'l1', 'smooth_l1'),
                        help='count-loss scale; log_l1 is safer early in training')
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
    parser.add_argument('--allow_count_head_fresh_train', action='store_true',
                        help='allow count-head auxiliary during fresh training; disabled by default after severe SHA over-counting')
    parser.add_argument('--allow_count_head_from_start', action='store_true',
                        help='allow count-head auxiliary from epoch 0 when training from scratch; risky on SHA')
    parser.add_argument('--safe_count_head_start_epoch', default=250, type=int,
                        help='auto-delay count-head auxiliary to this epoch for fresh training unless explicitly allowed')
    parser.add_argument('--count_head_init_count', default=40.0, type=float,
                        help='initial count prediction for a reference 256x256 crop in the separate count head')
    parser.add_argument('--count_head_init_cells', default=1024.0, type=float,
                        help='reference encoder-cell count for --count_head_init_count; 256/8 squared is 1024')
    parser.add_argument('--count_head_feature_grad_scale', default=1.0, type=float,
                        help='scale gradients from count/density auxiliaries into PET encoder; 0 trains only the head')
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
                        help='Auxiliary Point Guidance loss weight; 0 disables it')
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
    parser.add_argument('--apg_start_epoch', default=0, type=int,
                        help='epoch when APG auxiliary supervision starts')
    parser.add_argument('--apg_warmup_epochs', default=0, type=int,
                        help='linearly ramp APG loss weight for this many epochs after --apg_start_epoch')
    parser.add_argument('--apg_end_epoch', default=-1, type=int,
                        help='epoch after which APG auxiliary supervision turns off; negative keeps it on')
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
    parser.add_argument('--ifi_loss_coef', default=0.0, type=float,
                        help='Interpolated Feature Guidance auxiliary loss weight; 0 disables it')
    parser.add_argument('--ifi_point_coef', default=1.0, type=float,
                        help='zero-offset coefficient inside IFI-lite APG loss')
    parser.add_argument('--ifi_neg_k', default=4, type=int,
                        help='local negative interpolated points per GT for IFI-lite')
    parser.add_argument('--ifi_neg_radius', default=12.0, type=float,
                        help='pixel radius for IFI-lite local negative ring')
    parser.add_argument('--ifi_neg_min_dist', default=4.0, type=float,
                        help='discard IFI-lite negatives closer than this many pixels to any GT')
    parser.add_argument('--ifi_start_epoch', default=0, type=int,
                        help='epoch when IFI-lite auxiliary supervision starts')
    parser.add_argument('--ifi_end_epoch', default=-1, type=int,
                        help='epoch after which IFI-lite turns off; negative keeps it on')
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
    parser.add_argument('--split_loss_variant', default='auto', choices=('auto', 'paper', 'gt', 'paper_gt'),
                        help='split-map supervision: auto follows pet_loss_variant, paper uses PET min/max, gt uses per-cell GT BCE, paper_gt combines both')
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
    parser.add_argument('--split_threshold_quantile', default=0.55, type=float,
                        help='legacy adaptive split-map quantile (unused when split_threshold is set)')
    parser.add_argument('--score_threshold', default=0.5, type=float,
                        help='point classification threshold; negative enables adaptive score thresholding')
    parser.add_argument('--eval_nms_radius', default=0.0, type=float,
                        help='optional eval-only point NMS radius in pixels; 0 disables duplicate suppression')
    parser.add_argument('--eval_branch_gate', default='none', choices=('none', 'query', 'pred'),
                        help='eval-only split-aware sparse/dense ownership gate; none keeps PET concatenation')
    parser.add_argument('--eval_soft_split_gate', default='none', choices=('none', 'query', 'pred'),
                        help='eval-only soft split responsibility multiplied into person scores before thresholding')
    parser.add_argument('--eval_count_mode', default='threshold', choices=('threshold', 'count_head_topk'),
                        help='threshold keeps PET behavior; count_head_topk keeps top-K APG candidates using the separate count head')
    parser.add_argument('--eval_count_head_min_score', default=0.0, type=float,
                        help='minimum candidate score before count-head top-K selection')
    parser.add_argument('--eval_protocol', default='pet', choices=('pet', 'crowd_no_overlap'),
                        help='validation protocol used during training')

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
    parser.add_argument('--eval_max_size', default=1536, type=int,
                        help='QNRF/UCF validation long-side cap; non-positive disables resizing')

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
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--eval_start_epoch', default=0, type=int,
                        help='skip validation before this epoch; useful for unstable from-scratch calibration warmup')
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


def sanitize_unstable_training_args(args):
    """Disable known-unstable experimental auxiliaries unless explicitly allowed."""
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
        and not bool(getattr(args, 'allow_count_head_from_start', False))
    ):
        delayed_start = max(1, int(getattr(args, 'safe_count_head_start_epoch', 250)))
        print(
            'WARNING: count-head auxiliary from epoch 0 is disabled for fresh training. '
            f'Setting count_head_start_epoch={delayed_start}. '
            'Use --allow_count_head_from_start only for isolated debugging runs.'
        )
        args.count_head_start_epoch = delayed_start
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
    bad_ratio = max(over_ratio, under_ratio)
    if mae >= mae_limit and bad_ratio >= ratio_limit:
        direction = 'over-count' if over_ratio >= under_ratio else 'under-count'
        message = (
            f'bad-count guard triggered: {direction} '
            f'pred_cnt={pred_cnt:.4f} gt_cnt={gt_cnt:.4f} '
            f'ratio={bad_ratio:.3f} mae={mae:.4f} '
            f'(limits: ratio>={ratio_limit:.3f}, mae>={mae_limit:.4f})'
        )
        return True, message
    return False, ''


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
    runtime_keys = {
        'resume', 'device', 'output_dir', 'seed', 'start_epoch',
        'resume_model_only', 'resume_allow_arch_change', 'num_workers', 'world_size', 'dist_url',
        'list_backbones', 'syn_bn', 'deterministic', 'freeze_bn', 'amp', 'amp_dtype',
        'strict_model_checks',
        # allow overriding schedule/eval settings at resume time
        'epochs', 'batch_size', 'accum_iter', 'eval_freq', 'eval_start_epoch',
        'eval_before_train', 'eval_protocol', 'data_path', 'eval_max_size',
        'patch_size', 'patch_size_choices', 'crop_attempts', 'min_crop_points',
    }
    if getattr(args, 'resume_model_only', False):
        runtime_keys.update({
            'lr', 'lr_backbone', 'lr_backbone_adapter', 'weight_decay',
            'freeze_backbone_epochs', 'clip_max_norm',
            'lr_scheduler', 'lr_drop', 'lr_gamma', 'warmup_epochs', 'hold_epochs',
            'min_lr', 'ema_decay',
            'score_threshold', 'split_threshold', 'split_threshold_quantile',
            'eval_nms_radius', 'eval_branch_gate', 'eval_soft_split_gate',
            'eval_count_mode', 'eval_count_head_min_score',
        })
        explicit_args = set(getattr(args, '_explicit_args', set()))
        aux_resume_keys = {
            'class_loss_type', 'focal_alpha', 'focal_gamma',
            'count_head_loss_coef', 'count_head_loss_type',
            'count_head_start_epoch', 'count_head_end_epoch', 'count_head_init_count',
            'allow_count_head_fresh_train', 'allow_count_head_from_start', 'safe_count_head_start_epoch',
            'count_head_init_cells', 'count_head_feature_grad_scale', 'train_count_head_only',
            'density_map_loss_coef', 'allow_unstable_density_map_loss',
            'density_map_loss_type', 'density_map_pos_weight',
            'density_map_grad_scale',
            'density_map_start_epoch', 'density_map_end_epoch',
            'region_count_loss_coef', 'region_count_grid', 'region_count_gate',
            'region_count_type', 'region_count_start_epoch', 'region_count_end_epoch',
            'bayesian_loss_coef', 'bayesian_sigma', 'bayesian_bg_coef',
            'bayesian_loss_gate', 'bayesian_start_epoch', 'bayesian_end_epoch',
            'apg_loss_coef', 'apg_pos_k', 'apg_point_coef',
            'apg_bg_coef', 'apg_bg_k', 'apg_bg_min_dist',
            'apg_start_epoch', 'apg_warmup_epochs', 'apg_end_epoch',
            'apg_contrastive_coef', 'apg_neg_k', 'apg_margin',
            'apg_consistency_coef', 'apg_consistency_k', 'apg_consistency_sigma',
            'apg_soft_loss_coef', 'apg_soft_pos_k', 'apg_soft_sigma', 'apg_soft_point_coef',
            'ifi_loss_coef', 'ifi_point_coef', 'ifi_neg_k', 'ifi_neg_radius',
            'ifi_neg_min_dist', 'ifi_start_epoch', 'ifi_end_epoch',
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
    for key in runtime_keys:
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


def should_skip_pretrained_backbone(args, checkpoint):
    if checkpoint is None:
        return bool(getattr(args, 'no_pretrained_backbone', False))
    explicit_args = set(getattr(args, '_explicit_args', set()))
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


def build_optimizer_param_groups(model_without_ddp, args):
    """Keep pretrained backbone weights on low LR while training new adapters at main LR."""
    use_timm = is_timm_backbone(getattr(args, 'backbone', ''))
    timm_feature_prefix = 'backbone.backbone.backbone.'
    adapter_prefixes = [
        'backbone.backbone.fpn.',  # timm Joiner -> TimmBackbone -> BackboneFPN
        'backbone.backbone.lite_fpn.',  # timm Joiner -> TimmBackbone -> LiteFPNAdapter
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
    args = sanitize_unstable_training_args(args)
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
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, seed=args.seed)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
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
            f'train_batches={len(batch_sampler_train)}',
        )

    # output directory and log
    output_dir = resolve_output_dir(args)
    run_log_name = output_dir / 'run_log.txt'
    if utils.is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        print(f'outputs will be saved to: {output_dir.resolve()}')
        with open(run_log_name, "a", encoding="utf-8") as log_file:
            log_file.write('Run Log %s\n' % time.strftime("%c"))
            log_file.write("{}\n".format(args))
            log_file.write("parameters: {}\n".format(n_parameters))

    best_mae, best_mse, best_epoch = 1e8, 1e8, 0
    if checkpoint is not None:
        model_key = 'model'
        if model_ema is not None and 'model_raw' in checkpoint and not args.resume_model_only:
            model_key = 'model_raw'
        strict_load = not (
            getattr(args, 'resume_model_only', False)
            and getattr(args, 'resume_allow_arch_change', False)
        )
        incompatible = model_without_ddp.load_state_dict(checkpoint[model_key], strict=strict_load)
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
                model_ema.load_state_dict(checkpoint['model_ema'])
            elif model_key == 'model_raw' and 'model' in checkpoint:
                model_ema.load_state_dict(checkpoint['model'])
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
        }
        if model_ema is not None:
            payload['model_ema'] = model_ema.state_dict()
            payload['ema_decay'] = args.ema_decay
        if scaler is not None:
            payload['scaler'] = scaler.state_dict()
        if include_raw_model:
            payload['model_raw'] = model_without_ddp.state_dict()
        return payload

    if getattr(args, 'eval_before_train', False):
        t1 = time.time()
        eval_model = model_ema.module if model_ema is not None else model
        if args.eval_protocol == 'crowd_no_overlap':
            test_stats = evaluate_crowd_no_overlap(eval_model, data_loader_val, device, vis_dir=None)
        else:
            test_stats = evaluate(eval_model, data_loader_val, device, args.start_epoch, None)
        t2 = time.time()
        print("\n==========================")
        print(
            "\npretrain_eval epoch:", args.start_epoch,
            "mae:", test_stats['mae'],
            "mse:", test_stats['mse'],
            "pred_cnt:", test_stats.get('pred_cnt', 0.0),
            "gt_cnt:", test_stats.get('gt_cnt', 0.0),
            "eval_time:", t2 - t1,
        )
        print("==========================\n")

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
        if epoch % args.eval_freq == 0 and epoch > 0 and epoch >= int(getattr(args, 'eval_start_epoch', 0)):
            t1 = time.time()
            eval_model = model_ema.module if model_ema is not None else model
            eval_model_name = 'ema' if model_ema is not None else 'raw'
            if args.eval_protocol == 'crowd_no_overlap':
                test_stats = evaluate_crowd_no_overlap(eval_model, data_loader_val, device, vis_dir=None)
            else:
                test_stats = evaluate(eval_model, data_loader_val, device, epoch, None)
            t2 = time.time()

            # output results
            mae, mse = test_stats['mae'], test_stats['mse']
            improved = mae < best_mae
            if improved:
                best_epoch = epoch
                best_mae = mae
                best_mse = mse
            print("\n==========================")
            print(
                "\nepoch:", epoch,
                "mae:", mae,
                "mse:", mse,
                "pred_cnt:", test_stats.get('pred_cnt', 0.0),
                "gt_cnt:", test_stats.get('gt_cnt', 0.0),
                "\n\nbest mae:", best_mae,
                "best epoch:", best_epoch,
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
                    'eval_time': float(t2 - t1),
                    'eval_model': eval_model_name,
                    'eval_count_mode': getattr(args, 'eval_count_mode', 'threshold'),
                    'eval_count_head_min_score': float(getattr(args, 'eval_count_head_min_score', 0.0)),
                }
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

            if improved and utils.is_main_process():
                utils.save_on_master(checkpoint_payload(epoch), output_dir / 'checkpoint.pth')
                (output_dir / 'best_eval_results.json').write_text(
                    json.dumps(eval_record, indent=2) + "\n",
                    encoding="utf-8",
                )
                best_model_state = model_ema.state_dict() if model_ema is not None else model_without_ddp.state_dict()
                utils.save_on_master(
                    checkpoint_payload(
                        epoch,
                        model_state=best_model_state,
                        include_raw_model=model_ema is not None,
                    ),
                    output_dir / 'best_checkpoint.pth',
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
            'final': True,
            'eval_model': 'ema' if model_ema is not None else 'raw',
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
