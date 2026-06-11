"""
Continuous Feature Interpolation (CFI) for PET multi-scale features.

Blends coarse (8x) and fine (4x) pyramid maps at several learned continuous
scale ratios, then feeds the aggregated context back into both branches with
zero-initialized residuals so training starts from the original PET features.
"""
import torch
import torch.nn.functional as F
from torch import nn


def _parse_positive_int(value, default, name):
    value = int(value) if value is not None else int(default)
    if value <= 0:
        raise ValueError(f'{name} must be positive, got {value}')
    return value


class ContinuousFeatureInterpolation(nn.Module):
    """Cross-scale continuous feature interpolation between 4x and 8x maps."""

    def __init__(self, channels, num_scales=3, mid_dim=64, mode='full'):
        super().__init__()
        if mode == 'none':
            self.enabled = False
            return
        if mode not in ('lite', 'full'):
            raise ValueError(f'Unsupported cfi_mode: {mode}. Use "none", "lite", or "full".')

        self.enabled = True
        self.mode = mode
        num_scales = _parse_positive_int(num_scales, 3, 'cfi_num_scales')
        mid_dim = max(16, min(int(mid_dim), channels))

        init_logits = torch.linspace(-0.5, 0.5, num_scales)
        self.scale_logits = nn.Parameter(init_logits)

        self.coarse_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.fine_proj = nn.Conv2d(channels, channels, 1, bias=False)

        if mode == 'lite':
            self.interp_refine = nn.Sequential(
                nn.Conv2d(channels, mid_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_dim, channels, 1, bias=False),
            )
            self.fuse_4x = None
            self.fuse_8x = None
        else:
            self.interp_refine = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels, mid_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(mid_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_dim, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_scales)
            ])
            self.fuse_4x = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.fuse_8x = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )

        self.out_4x = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_8x = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.out_4x.weight)
        nn.init.zeros_(self.out_8x.weight)

    def forward(self, feat_4x, feat_8x):
        if not self.enabled:
            return feat_4x, feat_8x

        fine = self.fine_proj(feat_4x)
        coarse = self.coarse_proj(feat_8x)
        coarse_up = F.interpolate(coarse, size=fine.shape[-2:], mode='bilinear', align_corners=False)

        if self.mode == 'lite':
            refined_maps = []
            for logits in self.scale_logits:
                alpha = torch.sigmoid(logits)
                refined_maps.append((1.0 - alpha) * coarse_up + alpha * fine)
            interp_agg = self.interp_refine(torch.stack(refined_maps, dim=0).mean(dim=0))
            delta_4x = self.out_4x(interp_agg)
            delta_8x = self.out_8x(F.adaptive_avg_pool2d(interp_agg, coarse.shape[-2:]))
        else:
            refined_maps = []
            for logits, refine in zip(self.scale_logits, self.interp_refine):
                alpha = torch.sigmoid(logits)
                blended = (1.0 - alpha) * coarse_up + alpha * fine
                refined_maps.append(refine(blended))
            interp_agg = torch.stack(refined_maps, dim=0).mean(dim=0)
            delta_4x = self.out_4x(self.fuse_4x(torch.cat([fine, interp_agg], dim=1)))
            pooled = F.adaptive_avg_pool2d(interp_agg, coarse.shape[-2:])
            delta_8x = self.out_8x(self.fuse_8x(torch.cat([coarse, pooled], dim=1)))

        return feat_4x + delta_4x, feat_8x + delta_8x


def build_cfi_module(args, channels):
    mode = getattr(args, 'cfi_mode', 'none')
    if mode == 'none':
        return ContinuousFeatureInterpolation(channels, mode='none')
    return ContinuousFeatureInterpolation(
        channels,
        num_scales=getattr(args, 'cfi_num_scales', 3),
        mid_dim=getattr(args, 'cfi_mid_dim', 64),
        mode=mode,
    )
