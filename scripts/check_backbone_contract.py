#!/usr/bin/env python3
"""Check whether a backbone satisfies PET's tensor contract.

The check does not use training data or pretrained weights. It builds PET with
the requested backbone, runs one synthetic image through inference, and reports
the output tensor shapes. This catches most replacement-backbone integration
errors before starting a long training run.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import build_model
from models.backbones import get_supported_timm_backbones
from util.misc import nested_tensor_from_tensor_list


def make_args(
    backbone,
    device,
    timm_adapter,
    decoder_attention,
    decoder_memory_halo,
    decoder_global_context,
    quad_context_mixer,
    fusion_mhf_mode,
    fusion_mhf_heads,
    fusion_mhf_position,
    fusion_mhf_impl,
    fusion_fpn_type,
    fusion_mhf_reduction,
    fusion_mhf_norm,
    fusion_mhf_spatial_kernel,
    fusion_mhf_output_activation,
):
    return SimpleNamespace(
        device=device,
        backbone=backbone,
        timm_adapter=timm_adapter,
        fusion_mhf_mode=fusion_mhf_mode,
        fusion_mhf_heads=fusion_mhf_heads,
        fusion_mhf_position=fusion_mhf_position,
        fusion_mhf_strength=1.0,
        fusion_mhf_activation='gelu',
        fusion_mhf_impl=fusion_mhf_impl,
        fusion_fpn_type=fusion_fpn_type,
        fusion_mhf_reduction=fusion_mhf_reduction,
        fusion_mhf_norm=fusion_mhf_norm,
        fusion_mhf_spatial_kernel=fusion_mhf_spatial_kernel,
        fusion_mhf_output_activation=fusion_mhf_output_activation,
        no_pretrained_backbone=True,
        position_embedding='sine',
        dec_layers=2,
        dim_feedforward=512,
        hidden_dim=256,
        dropout=0.0,
        nheads=8,
        transformer_activation='relu',
        transformer_norm_style='post',
        decoder_attention=decoder_attention,
        decoder_memory_halo=decoder_memory_halo,
        decoder_global_context=decoder_global_context,
        enc_win_sizes='',
        enc_shift_mode='none',
        sparse_dec_win_size='',
        dense_dec_win_size='',
        context_patch_size='',
        quad_context_mixer=quad_context_mixer,
        quad_context_levels=2,
        quad_context_shift=1,
        quad_context_mid_dim=128,
        quad_context_activation='gelu',
        splitter_head='pool',
        splitter_hidden_dim=128,
        splitter_activation='gelu',
        set_cost_class=1.0,
        set_cost_point=0.05,
        ce_loss_coef=1.0,
        point_loss_coef=5.0,
        count_loss_coef=0.0,
        count_loss_gate='detach',
        count_loss_type='log_l1',
        count_loss_start_epoch=-1,
        eos_coef=0.5,
        pet_loss_variant='paper',
        negative_loss_coef=0.1,
        non_div_loss_coef=0.25,
        quadtree_loss_coef=0.1,
        quadtree_prior_coef=0.025,
        split_count_threshold=2,
        split_pos_weight=1.0,
        split_threshold=0.5,
        split_threshold_quantile=0.55,
        score_threshold=0.5,
    )


def check_backbone(
    backbone,
    image_size,
    device,
    timm_adapter,
    decoder_attention,
    decoder_memory_halo,
    decoder_global_context,
    quad_context_mixer,
    fusion_mhf_mode,
    fusion_mhf_heads,
    fusion_mhf_position,
    fusion_mhf_impl,
    fusion_fpn_type,
    fusion_mhf_reduction,
    fusion_mhf_norm,
    fusion_mhf_spatial_kernel,
    fusion_mhf_output_activation,
):
    args = make_args(
        backbone,
        device,
        timm_adapter,
        decoder_attention,
        decoder_memory_halo,
        decoder_global_context,
        quad_context_mixer,
        fusion_mhf_mode,
        fusion_mhf_heads,
        fusion_mhf_position,
        fusion_mhf_impl,
        fusion_fpn_type,
        fusion_mhf_reduction,
        fusion_mhf_norm,
        fusion_mhf_spatial_kernel,
        fusion_mhf_output_activation,
    )
    model, _ = build_model(args)
    model.to(device).eval()

    image = torch.randn(3, image_size[0], image_size[1], device=device)
    samples = nested_tensor_from_tensor_list([image])
    samples = samples.to(device)

    with torch.no_grad():
        features, _ = model.backbone(samples)
        outputs = model(samples, test=True)

    print(f'{backbone}: OK')
    for name in ('4x', '8x'):
        tensor = features[name].tensors
        mask = features[name].mask
        print(f'  feature {name}: tensor={tuple(tensor.shape)} mask={tuple(mask.shape)}')
    print(f"  pred_logits={tuple(outputs['pred_logits'].shape)} pred_points={tuple(outputs['pred_points'].shape)}")


def parse_args():
    parser = argparse.ArgumentParser(description='Validate PET backbone compatibility')
    parser.add_argument('--backbone', default='convnextv2_base')
    parser.add_argument('--timm_adapter', default='lite_fpn', choices=('lite_fpn', 'direct', 'fpn'))
    parser.add_argument('--all', action='store_true', help='Check vgg16_bn and every supported timm backbone')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--decoder_attention', default='softmax', choices=('softmax', 'linear'))
    parser.add_argument('--decoder_memory_halo', default=0, type=int)
    parser.add_argument('--decoder_global_context', action='store_true')
    parser.add_argument('--quad_context_mixer', default='none', choices=('none', 'lite'))
    parser.add_argument('--fusion_mhf_mode', default='none', choices=('none', 'cem', 'cem_msem', 'full'))
    parser.add_argument('--fusion_mhf_heads', default=1, type=int)
    parser.add_argument('--fusion_mhf_position', default='before', choices=('before', 'post'))
    parser.add_argument('--fusion_mhf_impl', default='residual', choices=('residual', 'vmambacc'))
    parser.add_argument('--fusion_fpn_type', default='fpn', choices=('fpn', 'hs2fpn'))
    parser.add_argument('--fusion_mhf_reduction', default=4, type=int)
    parser.add_argument('--fusion_mhf_norm', default='none', choices=('none', 'bn', 'gn'))
    parser.add_argument('--fusion_mhf_spatial_kernel', default=7, type=int)
    parser.add_argument('--fusion_mhf_output_activation', default='none', choices=('none', 'sigmoid'))
    return parser.parse_args()


def main():
    args = parse_args()
    backbones = ['vgg16_bn'] + list(get_supported_timm_backbones()) if args.all else [args.backbone]
    failures = []
    for backbone in backbones:
        try:
            check_backbone(
                backbone,
                (args.height, args.width),
                args.device,
                args.timm_adapter,
                args.decoder_attention,
                args.decoder_memory_halo,
                args.decoder_global_context,
                args.quad_context_mixer,
                args.fusion_mhf_mode,
                args.fusion_mhf_heads,
                args.fusion_mhf_position,
                args.fusion_mhf_impl,
                args.fusion_fpn_type,
                args.fusion_mhf_reduction,
                args.fusion_mhf_norm,
                args.fusion_mhf_spatial_kernel,
                args.fusion_mhf_output_activation,
            )
        except Exception as exc:
            failures.append((backbone, exc))
            print(f'{backbone}: FAIL - {exc}')

    if failures:
        print('\nFailures:')
        for backbone, exc in failures:
            print(f'  {backbone}: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
