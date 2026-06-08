import torch
import os
from models import build_model
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
old_mha_forward = nn.MultiheadAttention.forward
def new_mha_forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, average_attn_weights=True, is_causal=False):
    return old_mha_forward(self, query, key, value, key_padding_mask=key_padding_mask, need_weights=False, attn_mask=attn_mask, average_attn_weights=average_attn_weights, is_causal=is_causal)
nn.MultiheadAttention.forward = new_mha_forward


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)
    parser.add_argument('--backbone', default='convnextv2_base', type=str)
    parser.add_argument('--no_pretrained_backbone', action='store_true', default=True)
    parser.add_argument('--allow_random_backbone_fallback', action='store_true')
    parser.add_argument('--timm_adapter', default='lite_fpn')
    parser.add_argument('--fusion_mhf_mode', default='none')
    parser.add_argument('--fusion_mhf_heads', default=1, type=int)
    parser.add_argument('--fusion_mhf_position', default='before')
    parser.add_argument('--fusion_mhf_strength', default=1.0, type=float)
    parser.add_argument('--fusion_mhf_activation', default='gelu')
    parser.add_argument('--fusion_mhf_impl', default='residual')
    parser.add_argument('--fusion_fpn_type', default='fpn')
    parser.add_argument('--fusion_mhf_reduction', default=4, type=int)
    parser.add_argument('--fusion_mhf_norm', default='none')
    parser.add_argument('--fusion_mhf_spatial_kernel', default=7, type=int)
    parser.add_argument('--fusion_mhf_output_activation', default='none')
    parser.add_argument('--position_embedding', default='sine')
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dim_feedforward', default=512, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--transformer_activation', default='relu')
    parser.add_argument('--transformer_norm_style', default='post')
    parser.add_argument('--decoder_attention', default='softmax')
    parser.add_argument('--decoder_memory_halo', default=0, type=int)
    parser.add_argument('--decoder_global_context', action='store_true')
    parser.add_argument('--decoder_global_context_mode', default='residual')
    parser.add_argument('--enc_win_sizes', default='')
    parser.add_argument('--enc_shift_mode', default='none')
    parser.add_argument('--sparse_dec_win_size', default='')
    parser.add_argument('--dense_dec_win_size', default='')
    parser.add_argument('--context_patch_size', default='')
    parser.add_argument('--quad_context_mixer', default='none')
    parser.add_argument('--quad_context_levels', default=2, type=int)
    parser.add_argument('--quad_context_shift', default=1, type=int)
    parser.add_argument('--quad_context_mid_dim', default=128, type=int)
    parser.add_argument('--quad_context_activation', default='gelu')
    parser.add_argument('--splitter_head', default='pool')
    parser.add_argument('--splitter_hidden_dim', default=128, type=int)
    parser.add_argument('--splitter_activation', default='gelu')
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_point', default=0.05, type=float)
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)
    parser.add_argument('--point_loss_coef', default=5.0, type=float)
    parser.add_argument('--count_loss_coef', default=0.0, type=float)
    parser.add_argument('--count_loss_gate', default='detach')
    parser.add_argument('--count_loss_type', default='log_l1')
    parser.add_argument('--count_loss_start_epoch', default=-1, type=int)
    parser.add_argument('--region_count_loss_coef', default=0.0, type=float)
    parser.add_argument('--region_count_grid', default=4, type=int)
    parser.add_argument('--region_count_gate', default='detach')
    parser.add_argument('--region_count_type', default='log_l1')
    parser.add_argument('--region_count_start_epoch', default=-1, type=int)
    parser.add_argument('--region_count_end_epoch', default=-1, type=int)
    parser.add_argument('--bayesian_loss_coef', default=0.0, type=float)
    parser.add_argument('--bayesian_sigma', default=8.0, type=float)
    parser.add_argument('--bayesian_bg_coef', default=0.05, type=float)
    parser.add_argument('--bayesian_loss_gate', default='detach')
    parser.add_argument('--bayesian_start_epoch', default=-1, type=int)
    parser.add_argument('--bayesian_end_epoch', default=-1, type=int)
    parser.add_argument('--apg_loss_coef', default=1.0, type=float)
    parser.add_argument('--apg_pos_k', default=1, type=int)
    parser.add_argument('--apg_point_coef', default=5.0, type=float)
    parser.add_argument('--apg_start_epoch', default=0, type=int)
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
    parser.add_argument('--qd_apg_route_source', default='gt_count')
    parser.add_argument('--eos_coef', default=0.5, type=float)
    parser.add_argument('--pet_loss_variant', default='paper')
    parser.add_argument('--split_loss_variant', default='gt')
    parser.add_argument('--negative_loss_coef', default=0.1, type=float)
    parser.add_argument('--non_div_loss_coef', default=0.25, type=float)
    parser.add_argument('--quadtree_loss_coef', default=0.5, type=float)
    parser.add_argument('--quadtree_prior_coef', default=0.025, type=float)
    parser.add_argument('--split_count_threshold', default=2, type=int)
    parser.add_argument('--split_pos_weight', default=2.0, type=float)
    parser.add_argument('--split_threshold', default=0.5, type=float)
    parser.add_argument('--split_threshold_quantile', default=0.55, type=float)
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--eval_nms_radius', default=0.0, type=float)
    parser.add_argument('--eval_branch_gate', default='none')
    parser.add_argument('--eval_soft_split_gate', default='none')
    parser.add_argument('--eval_protocol', default='pet')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dataset_file', default='SHA')
    return parser

args = get_args_parser().parse_args([])
model, criterion = build_model(args)
model.cuda()
criterion.cuda()
model.train()

from util.misc import NestedTensor
samples = NestedTensor(torch.randn(2, 3, 256, 256).cuda(), torch.zeros(2, 256, 256, dtype=torch.bool).cuda())
targets = [{'points': torch.rand(5, 2).cuda() * 256, 'labels': torch.ones(5).long().cuda(), 'density': torch.ones(1).cuda()} for _ in range(2)]

import torch.cuda.amp as amp
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Create a scenario where padding mask is active
samples.mask[1, :, 128:] = True

outputs = model(samples, train=True, epoch=0, criterion=criterion, targets=targets)
losses = outputs['losses']

print("Losses before backward:")
print(losses)
for k, v in outputs['loss_dict'].items():
    print(k, v)

losses.backward()

nan_grads = 0
total_grads = 0
for name, p in model.named_parameters():
    if p.grad is not None:
        total_grads += 1
        if not torch.isfinite(p.grad).all():
            print(f"NaN grad in {name}")
            nan_grads += 1

print(f"Total gradients: {total_grads}, NaN gradients: {nan_grads}")
if nan_grads > 0:
    print("WARNING: Model still produces NaN gradients!")
else:
    print("SUCCESS: Gradients are finite.")
