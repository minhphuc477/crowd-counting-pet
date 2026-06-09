"""
Multi-Scale Guidance for Implicit Feature Interpolation (MSG-IFI).

Extends APGCC-style IFI by fusing implicit interpolations from PET's fine (4x)
and coarse (8x / context-encoded) feature maps with learned scale guidance.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


def _positional_encoding(values, num_freqs=4):
    """NeRF-style encoding for sub-pixel offsets / distances."""
    freqs = 2.0 ** torch.arange(num_freqs, device=values.device, dtype=values.dtype)
    scaled = values.unsqueeze(-1) * freqs.view(1, 1, -1) * math.pi
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


def _bilinear_sample(feat, batch_idx, points_abs, img_h, img_w):
    """Sample BxCxHxW feature map at image-space points (y, x). Returns NxC."""
    if points_abs.numel() == 0:
        return feat.new_zeros((0, feat.shape[1]))
    grid_x = (points_abs[:, 1] + 0.5) / max(float(img_w), 1.0) * 2.0 - 1.0
    grid_y = (points_abs[:, 0] + 0.5) / max(float(img_h), 1.0) * 2.0 - 1.0
    sample_grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        feat[batch_idx:batch_idx + 1],
        sample_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=False,
    )
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1)


class NeighborImplicitBlock(nn.Module):
    """Four-neighbor implicit interpolation on one feature pyramid level."""

    def __init__(self, channels, mid_dim=128, pe_freq=4):
        super().__init__()
        self.pe_freq = int(pe_freq)
        pe_dim = 4 * self.pe_freq
        in_dim = channels + pe_dim
        hidden = max(32, min(int(mid_dim), channels * 2))
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.out_proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, feat, batch_idx, points_abs, img_h, img_w):
        if points_abs.numel() == 0:
            return feat.new_zeros((0, feat.shape[1]))

        _, _, fh, fw = feat.shape
        device, dtype = feat.device, feat.dtype
        n_pts = points_abs.shape[0]

        fy = (points_abs[:, 0] + 0.5) / max(float(img_h), 1.0) * fh - 0.5
        fx = (points_abs[:, 1] + 0.5) / max(float(img_w), 1.0) * fw - 0.5

        x0 = fx.floor().clamp(0, max(fw - 2, 0)).long()
        y0 = fy.floor().clamp(0, max(fh - 2, 0)).long()
        x1 = (x0 + 1).clamp(max=fw - 1)
        y1 = (y0 + 1).clamp(max=fh - 1)

        corners = [(y0, x0), (y0, x1), (y1, x0), (y1, x1)]
        corner_feats = []
        for cy, cx in corners:
            corner_feats.append(feat[batch_idx, :, cy, cx].permute(1, 0))

        dx = fx - x0.to(dtype)
        dy = fy - y0.to(dtype)
        w00 = (1.0 - dx) * (1.0 - dy)
        w01 = dx * (1.0 - dy)
        w10 = (1.0 - dx) * dy
        w11 = dx * dy
        weights = torch.stack([w00, w01, w10, w11], dim=1)

        interp_terms = []
        deltas = [
            torch.stack([-dx, -dy], dim=-1),
            torch.stack([1.0 - dx, -dy], dim=-1),
            torch.stack([-dx, 1.0 - dy], dim=-1),
            torch.stack([1.0 - dx, 1.0 - dy], dim=-1),
        ]
        for corner_feat, delta, weight in zip(corner_feats, deltas, weights.unbind(dim=1)):
            pe = _positional_encoding(delta, self.pe_freq).reshape(n_pts, -1)
            term = self.neighbor_mlp(torch.cat([corner_feat, pe], dim=-1))
            interp_terms.append(term * weight.unsqueeze(-1))

        fused = torch.stack(interp_terms, dim=0).sum(dim=0)
        baseline = _bilinear_sample(feat, batch_idx, points_abs, img_h, img_w)
        return baseline + self.out_proj(fused)


class MultiScaleGuidedInterpolator(nn.Module):
    """Fuse 4x and 8x/context implicit interpolations with learned scale guidance."""

    def __init__(self, channels, mid_dim=128, pe_freq=4, use_encode_src=True, mode='msg'):
        super().__init__()
        if mode == 'none':
            self.enabled = False
            return
        if mode not in ('lite', 'msg'):
            raise ValueError(f'Unsupported ifi_mode: {mode}. Use "lite", "msg", or "none".')

        self.enabled = True
        self.mode = mode
        self.use_encode_src = bool(use_encode_src)

        if mode == 'lite':
            self.interp_4x = None
            self.interp_8x = None
            self.scale_gate = None
            self.refine = None
            return

        self.interp_4x = NeighborImplicitBlock(channels, mid_dim=mid_dim, pe_freq=pe_freq)
        self.interp_8x = NeighborImplicitBlock(channels, mid_dim=mid_dim, pe_freq=pe_freq)
        gate_in = channels * (3 if use_encode_src else 2)
        hidden = max(32, min(int(mid_dim), channels))
        self.scale_gate = nn.Sequential(
            nn.Linear(gate_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3 if use_encode_src else 2),
        )
        self.refine = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, feat_4x, feat_8x, encode_src, batch_idx, points_abs, img_h, img_w):
        if not self.enabled or points_abs.numel() == 0:
            ref = encode_src if encode_src is not None else feat_8x
            return ref.new_zeros((0, ref.shape[1]))

        if self.mode == 'lite':
            src = encode_src if encode_src is not None else feat_8x
            return _bilinear_sample(src, batch_idx, points_abs, img_h, img_w)

        f4 = self.interp_4x(feat_4x, batch_idx, points_abs, img_h, img_w)
        coarse = encode_src if (self.use_encode_src and encode_src is not None) else feat_8x
        f8 = self.interp_8x(coarse, batch_idx, points_abs, img_h, img_w)

        gate_inputs = [f4, f8]
        if self.use_encode_src and encode_src is not None:
            gate_inputs.append(_bilinear_sample(encode_src, batch_idx, points_abs, img_h, img_w))

        weights = torch.softmax(self.scale_gate(torch.cat(gate_inputs, dim=-1)), dim=-1)
        fused = sum(w.unsqueeze(-1) * f for w, f in zip(weights.unbind(dim=-1), gate_inputs))
        return fused + self.refine(fused)


def build_ifi_interpolator(args, channels):
    mode = getattr(args, 'ifi_mode', 'lite')
    if float(getattr(args, 'ifi_loss_coef', 0.0)) <= 0.0:
        return MultiScaleGuidedInterpolator(channels, mode='none')
    if mode == 'none':
        mode = 'lite'
    return MultiScaleGuidedInterpolator(
        channels,
        mid_dim=getattr(args, 'ifi_mid_dim', 128),
        pe_freq=getattr(args, 'ifi_pe_freq', 4),
        use_encode_src=not getattr(args, 'no_ifi_encode_src', False),
        mode=mode,
    )
