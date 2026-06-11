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
from engine import evaluate, evaluate_crowd_no_overlap
from models import build_model


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
    parser.add_argument('--timm_adapter', default='lite_fpn', choices=('pet_fpn', 'lite_fpn', 'direct', 'fpn'),
                        help='adapter used to map timm features into PET 4x/8x features')
    parser.add_argument('--timm_output_norm', default='gn', choices=('gn', 'none'),
                        help='normalization after timm feature adapter; gn preserves old timm behavior, none is VGG-like')
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
    parser.add_argument('--class_loss_type', default='ce', choices=('ce', 'focal'))
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--focal_gamma', default=2.0, type=float)
    parser.add_argument('--class_prior_prob', default=-1.0, type=float)
    parser.add_argument('--strict_model_checks', action='store_true')
    parser.add_argument('--count_loss_coef', default=0.0, type=float)
    parser.add_argument('--count_loss_gate', default='detach', choices=('detach', 'soft', 'hard'))
    parser.add_argument('--count_loss_type', default='log_l1', choices=('log_l1', 'l1', 'smooth_l1'))
    parser.add_argument('--count_loss_start_epoch', default=-1, type=int)
    parser.add_argument('--region_count_loss_coef', default=0.0, type=float)
    parser.add_argument('--region_count_grid', default=4, type=int)
    parser.add_argument('--region_count_gate', default='detach', choices=('none', 'detach', 'soft', 'hard'))
    parser.add_argument('--region_count_type', default='log_l1', choices=('log_l1', 'l1', 'smooth_l1'))
    parser.add_argument('--region_count_start_epoch', default=-1, type=int)
    parser.add_argument('--region_count_end_epoch', default=-1, type=int)
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
    parser.add_argument('--apg_start_epoch', default=0, type=int)
    parser.add_argument('--apg_warmup_epochs', default=0, type=int)
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
    parser.add_argument('--ifi_loss_coef', default=0.0, type=float)
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
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")   # cross-entropy weights
    parser.add_argument('--pet_loss_variant', default='paper', choices=('paper', 'balanced'))
    parser.add_argument('--split_loss_variant', default='auto', choices=('auto', 'paper', 'gt', 'paper_gt'))
    parser.add_argument('--negative_loss_coef', default=0.1, type=float)
    parser.add_argument('--non_div_loss_coef', default=0.25, type=float)
    parser.add_argument('--quadtree_loss_coef', default=0.1, type=float)
    parser.add_argument('--quadtree_prior_coef', default=0.025, type=float)
    parser.add_argument('--split_count_threshold', default=2, type=int)
    parser.add_argument('--split_pos_weight', default=1.0, type=float)
    parser.add_argument('--split_threshold', default=0.5, type=float)
    parser.add_argument('--split_threshold_quantile', default=0.55, type=float)
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--eval_nms_radius', default=0.0, type=float)
    parser.add_argument('--eval_branch_gate', default='none', choices=('none', 'query', 'pred'))
    parser.add_argument('--eval_soft_split_gate', default='none', choices=('none', 'query', 'pred'))

    # dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="", type=str)
    parser.add_argument('--patch_size', default=256, type=int,
                        help='training crop size for crop-based crowd datasets')
    parser.add_argument('--patch_size_choices', default='', type=str)
    parser.add_argument('--crop_attempts', default=1, type=int)
    parser.add_argument('--min_crop_points', default=0, type=int)
    parser.add_argument('--eval_max_size', default=1536, type=int,
                        help='QNRF/UCF validation long-side cap; non-positive disables resizing')

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
    parser.add_argument('--override_score_threshold', default=None, type=float,
                        help='override the checkpoint score threshold at evaluation time')
    parser.add_argument('--override_split_threshold', default=None, type=float,
                        help='override the checkpoint split threshold at evaluation time')
    parser.add_argument('--override_split_threshold_quantile', default=None, type=float,
                        help='override the checkpoint split-threshold quantile at evaluation time')
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
    runtime_keys = {
        'resume', 'device', 'vis_dir', 'results_file', 'data_path', 'dataset_file',
        'eval_max_size', 'num_workers', 'seed',
        'override_score_threshold', 'override_split_threshold', 'override_split_threshold_quantile',
        'checkpoint_model_key', 'deterministic', 'tta_flip', 'tta_scales',
        'eval_nms_radius', 'eval_branch_gate', 'eval_soft_split_gate',
        'eval_protocol', 'resume_allow_arch_change',
        'amp_dtype', 'strict_model_checks',
    }
    explicit_args = set(getattr(args, '_explicit_args', set()))
    if getattr(args, 'resume_allow_arch_change', False):
        runtime_keys.update(key for key in ARCHITECTURE_OVERRIDE_KEYS if key in explicit_args)
    for key in runtime_keys:
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
    if override_score_threshold is not None:
        args.score_threshold = float(override_score_threshold)
    if override_split_threshold is not None:
        args.split_threshold = float(override_split_threshold)
    if override_split_threshold_quantile is not None:
        args.split_threshold_quantile = float(override_split_threshold_quantile)
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
    val_image_set = 'val'
    dataset_val = build_dataset(image_set=val_image_set, args=args)

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
    
    # evaluation
    vis_dir = None if args.vis_dir == "" else args.vis_dir
    tta_scales = parse_tta_scales(getattr(args, 'tta_scales', '1.0'))
    if args.eval_protocol == 'crowd_no_overlap':
        test_stats = evaluate_crowd_no_overlap(
            model,
            data_loader_val,
            device,
            vis_dir=vis_dir,
            tta_flip=args.tta_flip,
            tta_scales=tta_scales,
        )
        mae, mse = test_stats['mae'], test_stats['mse']
    else:
        test_stats = evaluate(model, data_loader_val, device, vis_dir=vis_dir, tta_flip=args.tta_flip, tta_scales=tta_scales)
        mae, mse = test_stats['mae'], test_stats['mse']
    line = f'\nepoch: {cur_epoch}, mae: {mae}, mse: {mse}' 
    print(line)
    if utils.is_main_process():
        if args.results_file:
            results_file = Path(args.results_file)
        elif args.resume and not args.resume.startswith('https'):
            results_file = Path(args.resume).resolve().parent / 'eval_results.json'
        else:
            results_file = Path('eval_results.json')
        results_file.parent.mkdir(parents=True, exist_ok=True)
        results_file.write_text(json.dumps({
            'epoch': int(cur_epoch),
            'eval_mae': float(mae),
            'eval_mse': float(mse),
            'pred_cnt': float(test_stats.get('pred_cnt', 0.0)),
            'gt_cnt': float(test_stats.get('gt_cnt', 0.0)),
            'checkpoint': args.resume,
            'eval_model': eval_model_key,
            'eval_protocol': args.eval_protocol,
            'tta_flip': bool(args.tta_flip),
            'tta_scales': list(tta_scales),
            'eval_nms_radius': float(getattr(args, 'eval_nms_radius', 0.0)),
            'eval_branch_gate': getattr(args, 'eval_branch_gate', 'none'),
            'eval_soft_split_gate': getattr(args, 'eval_soft_split_gate', 'none'),
        }, indent=2) + "\n", encoding="utf-8")
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
