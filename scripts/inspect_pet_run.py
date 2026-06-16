#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


KEY_ARGS = (
    'dataset_file',
    'data_path',
    'backbone',
    'no_pretrained_backbone',
    'resume',
    'resume_model_only',
    'lr',
    'lr_backbone',
    'lr_backbone_adapter',
    'lr_scheduler',
    'lr_drop',
    'batch_size',
    'accum_iter',
    'amp',
    'freeze_bn',
    'freeze_backbone_epochs',
    'fusion_mhf_mode',
    'fusion_mhf_heads',
    'fusion_mhf_position',
    'fusion_mhf_strength',
    'fusion_mhf_impl',
    'fusion_fpn_type',
    'fusion_mhf_reduction',
    'fusion_mhf_norm',
    'fusion_mhf_spatial_kernel',
    'fusion_mhf_output_activation',
    'score_threshold',
    'eval_nms_radius',
    'eval_branch_gate',
    'eval_soft_split_gate',
    'split_threshold',
    'patch_size',
    'patch_size_choices',
    'crop_attempts',
    'min_crop_points',
    'splitter_head',
    'class_prior_prob',
    'strict_model_checks',
    'class_loss_type',
    'focal_alpha',
    'focal_gamma',
    'count_loss_coef',
    'count_loss_gate',
    'count_loss_type',
    'count_loss_budget_margin',
    'count_loss_start_epoch',
    'region_count_loss_coef',
    'region_count_grid',
    'region_count_gate',
    'region_count_type',
    'region_count_start_epoch',
    'region_count_end_epoch',
    'bayesian_loss_coef',
    'bayesian_sigma',
    'bayesian_bg_coef',
    'bayesian_loss_gate',
    'bayesian_start_epoch',
    'bayesian_end_epoch',
    'apg_loss_coef',
    'apg_pos_k',
    'apg_point_coef',
    'apg_bg_coef',
    'apg_bg_k',
    'apg_bg_min_dist',
    'apg_bg_offset_coef',
    'apg_start_epoch',
    'apg_warmup_epochs',
    'apg_end_epoch',
    'apg_contrastive_coef',
    'apg_neg_k',
    'apg_margin',
    'apg_consistency_coef',
    'apg_consistency_k',
    'apg_consistency_sigma',
    'apg_soft_loss_coef',
    'apg_soft_pos_k',
    'apg_soft_sigma',
    'apg_soft_point_coef',
    'ifi_loss_coef',
    'ifi_point_coef',
    'ifi_neg_k',
    'ifi_neg_radius',
    'ifi_neg_min_dist',
    'ifi_start_epoch',
    'ifi_end_epoch',
    'qd_apg_loss_coef',
    'qd_apg_point_coef',
    'qd_apg_suppress_coef',
    'qd_apg_start_epoch',
    'qd_apg_end_epoch',
    'qd_apg_route_source',
    'pet_loss_variant',
    'split_loss_variant',
    'dec_layers',
    'enc_shift_mode',
    'decoder_attention',
    'decoder_memory_halo',
    'decoder_global_context',
    'decoder_global_context_mode',
    'quad_context_mixer',
    'quad_context_levels',
    'quad_context_shift',
    'seed',
)


def get_arg(args, key, default=None):
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        for name in ('best_checkpoint.pth', 'checkpoint.pth'):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f'No best_checkpoint.pth or checkpoint.pth found in {path}')
    return path


def load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return None


def print_eval_record(label, record):
    if not record:
        return
    pred = record.get('pred_cnt')
    gt = record.get('gt_cnt')
    ratio = None
    if pred is not None and gt:
        ratio = float(pred) / max(float(gt), 1e-12)
    parts = [
        f"epoch={record.get('epoch')}",
        f"mae={record.get('test_mae')}",
        f"mse={record.get('test_mse')}",
        f"pred_cnt={pred}",
        f"gt_cnt={gt}",
    ]
    if ratio is not None:
        parts.append(f"pred/gt={ratio:.3f}")
    print(f'{label}: ' + ', '.join(parts))


def main():
    parser = argparse.ArgumentParser(description='Inspect PET checkpoint args and count-calibration symptoms.')
    parser.add_argument('path', help='run directory or checkpoint path')
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint(args.path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    ckpt_args = checkpoint.get('args')

    print(f'checkpoint: {checkpoint_path}')
    print(f"checkpoint_epoch: {checkpoint.get('epoch')}")
    print(f"checkpoint_best_mae: {checkpoint.get('best_mae')}")
    print(f"checkpoint_best_epoch: {checkpoint.get('best_epoch')}")
    print('args:')
    for key in KEY_ARGS:
        print(f'  {key}: {get_arg(ckpt_args, key)}')

    run_dir = checkpoint_path.parent
    print_eval_record('best_eval_results', load_json(run_dir / 'best_eval_results.json'))
    print_eval_record('latest_eval_results', load_json(run_dir / 'latest_eval_results.json'))

    backbone = get_arg(ckpt_args, 'backbone', '')
    if str(backbone).startswith('vgg') and get_arg(ckpt_args, 'no_pretrained_backbone', False):
        print('warning: this checkpoint was trained with --no_pretrained_backbone; VGG random init usually fails badly.')

    latest = load_json(run_dir / 'latest_eval_results.json')
    if latest and latest.get('pred_cnt') is not None and latest.get('gt_cnt'):
        pred_ratio = float(latest['pred_cnt']) / max(float(latest['gt_cnt']), 1e-12)
        if pred_ratio < 0.5:
            print('warning: latest eval is severe undercounting; try lower score-threshold sweep, but stop this run if best MAE is still >100 on SHA.')
        elif pred_ratio > 1.5:
            print('warning: latest eval is severe overcounting; try higher score-threshold sweep.')


if __name__ == '__main__':
    main()
