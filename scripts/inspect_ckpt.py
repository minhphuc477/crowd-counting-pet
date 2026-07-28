"""Inspect checkpoint for available architecture components."""
import torch, sys, os

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else \
    'outputs/NWPU/vgg16_bn_apglc_nwpu_tail_rifi_seed42/best_mae_checkpoint.pth'

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state = ckpt.get('model', ckpt)

# Key component groups
groups = {
    'count_head': [],
    'zip_count': [],
    'ifi': [],
    'foreground': [],
    'scale_point': [],
    'apg': [],
    'split': [],
}
for k in state.keys():
    for grp in groups:
        if grp in k.lower():
            groups[grp].append(k)
            break

print(f'Total keys: {len(state)}')
print()
for grp, keys in groups.items():
    has = len(keys) > 0
    print(f'{"✓" if has else "✗"} {grp}: {len(keys)} params')
    if has:
        for k in keys[:3]:
            print(f'    {k}: {tuple(state[k].shape) if hasattr(state[k], "shape") else state[k]}')
        if len(keys) > 3:
            print(f'    ... and {len(keys)-3} more')

# Checkpoint args
args = ckpt.get('args')
if args:
    print()
    print('=== Checkpoint training args (eval-relevant) ===')
    for attr in [
        'eval_count_mode', 'eval_score_calibration', 'eval_branch_gate',
        'eval_foreground_gate', 'count_head_loss_coef', 'zip_count_loss_coef',
        'eval_tile_trigger_count', 'eval_tile_trigger_area', 'eval_tile_nms_radius',
        'score_threshold', 'split_threshold', 'apglc',
    ]:
        val = getattr(args, attr, 'NOT_SET')
        print(f'  {attr}: {val}')

print()
print('=== Epoch ===', ckpt.get('epoch', 'unknown'))
