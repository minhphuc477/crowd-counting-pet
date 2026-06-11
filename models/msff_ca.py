"""
Multi-Scale Feature Fusion and Convolutional Attention (MSFF-CA).

Adapted from crowd-counting methods that combine densely connected dilated
convolutions (DCAM / MSCA) with channel-spatial convolutional attention (SAM/CAM)
to capture scale diversity and suppress background clutter.
"""
import torch
import torch.nn.functional as F
from torch import nn


def _parse_dilation_rates(value, default=(2, 4, 6)):
    if value is None or value == '':
        return tuple(int(v) for v in default)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(',') if part.strip()]
    else:
        parts = list(value)
    rates = tuple(int(part) for part in parts)
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError(f'msff_ca_dilations must be positive integers, got {value!r}')
    return rates


class DenseContextAwareModule(nn.Module):
    """Dense context-aware module with densely connected dilated convolutions."""

    def __init__(self, channels, dilation_rates=(2, 4, 6)):
        super().__init__()
        self.dilation_rates = tuple(int(rate) for rate in dilation_rates)
        self.branches = nn.ModuleList()
        in_channels = channels
        for rate in self.dilation_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels += channels
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        fused_channels = channels * (len(self.dilation_rates) + 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        branch_outputs = []
        dense_input = x
        for branch in self.branches:
            branch_out = branch(dense_input)
            branch_outputs.append(branch_out)
            dense_input = torch.cat([dense_input, branch_out], dim=1)

        global_context = self.global_context(x)
        global_context = F.interpolate(global_context, size=x.shape[-2:], mode='nearest')
        fused = torch.cat(branch_outputs + [global_context], dim=1)
        return self.fuse(fused)


class SemanticAttentionHead(nn.Module):
    """Lightweight foreground attention head (MSCANet-style SAM)."""

    def __init__(self, channels, mid_channels=64):
        super().__init__()
        mid = min(int(mid_channels), channels)
        mid2 = max(mid // 2, 16)
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid2, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_point_attention_targets(targets, feat_h, feat_w, img_h, img_w, sigma, device, dtype):
    """Soft foreground maps from GT point annotations."""
    bs = len(targets)
    out = torch.zeros(bs, 1, feat_h, feat_w, device=device, dtype=dtype)
    if sigma <= 0:
        raise ValueError(f'msff_ca_attn_sigma must be positive, got {sigma}')
    sigma2 = float(sigma) ** 2

    yy = (torch.arange(feat_h, device=device, dtype=dtype) + 0.5) * (float(img_h) / feat_h)
    xx = (torch.arange(feat_w, device=device, dtype=dtype) + 0.5) * (float(img_w) / feat_w)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
    grid_y = grid_y.reshape(1, 1, feat_h, feat_w)
    grid_x = grid_x.reshape(1, 1, feat_h, feat_w)

    for batch_idx, target in enumerate(targets):
        points = target['points'].to(device=device, dtype=dtype)
        if points.numel() == 0:
            continue
        py = points[:, 0].view(-1, 1, 1, 1)
        px = points[:, 1].view(-1, 1, 1, 1)
        dist2 = (grid_y - py) ** 2 + (grid_x - px) ** 2
        heat = torch.exp(-dist2 / (2.0 * sigma2))
        out[batch_idx, 0] = heat.max(dim=0)[0]
    return out.clamp(0.0, 1.0)


class ConvolutionalAttention(nn.Module):
    """Channel + spatial convolutional attention (CBAM-style)."""

    def __init__(self, channels, reduction=4, spatial_kernel=7):
        super().__init__()
        hidden = max(1, channels // int(reduction))
        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        padding = int(spatial_kernel) // 2
        self.spatial_gate = nn.Conv2d(2, 1, int(spatial_kernel), padding=padding, bias=False)

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        channel_weight = torch.sigmoid(self.channel_gate(avg_pool) + self.channel_gate(max_pool))
        x = x * channel_weight

        spatial_stats = torch.cat(
            [x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)],
            dim=1,
        )
        spatial_weight = torch.sigmoid(self.spatial_gate(spatial_stats))
        return x * spatial_weight


class MSFFCAEnhancer(nn.Module):
    """Residual multi-scale fusion + convolutional attention block for PET features."""

    def __init__(
        self,
        channels,
        mode='full',
        stacks=1,
        dilation_rates=(2, 4, 6),
        reduction=4,
        spatial_kernel=7,
        with_attn_head=False,
        with_foreground_head=False,
        attn_mid_channels=64,
    ):
        super().__init__()
        if mode == 'none':
            self.enabled = False
            return
        if mode not in ('msca', 'attn', 'full'):
            raise ValueError(f'Unsupported msff_ca_mode: {mode}. Use "none", "msca", "attn", or "full".')

        self.enabled = True
        self.mode = mode
        stacks = max(1, int(stacks))
        self.use_msca = mode in ('msca', 'full')
        self.use_attn = mode in ('attn', 'full')

        if self.use_msca:
            self.msca = nn.Sequential(
                *[
                    DenseContextAwareModule(channels, dilation_rates=dilation_rates)
                    for _ in range(stacks)
                ]
            )
        else:
            self.msca = nn.Identity()

        self.attn = (
            ConvolutionalAttention(channels, reduction=reduction, spatial_kernel=spatial_kernel)
            if self.use_attn
            else nn.Identity()
        )
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        self.attn_head = (
            SemanticAttentionHead(channels, mid_channels=attn_mid_channels)
            if (with_attn_head or with_foreground_head)
            else None
        )

    def _foreground_map(self, x, out):
        if self.attn_head is not None:
            return torch.sigmoid(self.attn_head(out))
        if self.use_attn:
            spatial_stats = torch.cat(
                [out.max(dim=1, keepdim=True)[0], out.mean(dim=1, keepdim=True)],
                dim=1,
            )
            return torch.sigmoid(self.attn.spatial_gate(spatial_stats))
        return torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)

    def forward(self, x, return_attn=False, return_foreground=False):
        if not self.enabled:
            if return_attn or return_foreground:
                return x, None, None
            return x
        out = x
        if self.use_msca:
            out = self.msca(out)
        attn_logits = self.attn_head(out) if self.attn_head is not None and return_attn else None
        if self.use_attn:
            out = self.attn(out)
        enhanced = x + self.out_proj(out)
        foreground = self._foreground_map(x, out) if return_foreground else None
        if return_attn or return_foreground:
            return enhanced, attn_logits, foreground
        return enhanced


def build_msff_ca_enhancer(args, channels):
    mode = getattr(args, 'msff_ca_mode', 'none')
    if mode == 'none':
        return nn.Identity()
    msff_calib = getattr(args, 'msff_calib_mode', 'none') == 'full'
    with_attn_head = float(getattr(args, 'msff_ca_attn_loss_coef', 0.0)) > 0.0
    with_foreground_head = msff_calib or bool(getattr(args, 'msff_foreground_gate', False))
    return MSFFCAEnhancer(
        channels,
        mode=mode,
        stacks=getattr(args, 'msff_ca_stacks', 1),
        dilation_rates=_parse_dilation_rates(
            getattr(args, 'msff_ca_dilations', ''),
            default=(2, 4, 6),
        ),
        reduction=getattr(args, 'msff_ca_reduction', 4),
        spatial_kernel=getattr(args, 'msff_ca_spatial_kernel', 7),
        with_attn_head=with_attn_head,
        with_foreground_head=with_foreground_head,
        attn_mid_channels=getattr(args, 'msff_ca_attn_mid_dim', 64),
    )
