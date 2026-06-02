"""
Backbone modules
"""
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from util.misc import NestedTensor
from .vgg import *
from ..position_encoding import build_position_encoding


def _make_activation(name):
    if name == 'gelu':
        return nn.GELU()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    raise ValueError(f'Unsupported fusion MHF activation: {name}. Use "gelu" or "relu".')


def _make_norm(num_channels, norm):
    if norm == 'none':
        return nn.Identity()
    if norm == 'bn':
        return nn.BatchNorm2d(num_channels)
    if norm == 'gn':
        groups = min(32, num_channels)
        while num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f'Unsupported fusion MHF norm: {norm}. Use "none", "bn", or "gn".')


class ConvStack(nn.Module):
    def __init__(self, channels, activation='gelu', reduction=4, norm='none'):
        super().__init__()
        hidden = max(1, channels // int(reduction))
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=norm == 'none'),
            _make_norm(hidden, norm),
            _make_activation(activation),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x):
        return self.net(x)


class VMambaCCCEM(nn.Module):
    """VMambaCC Channel Enhancement Module.

    Implements the paper equation W = sigmoid(Convs(maxpool(Fh)) +
    Convs(avgpool(Fh))), Fout = W * Fh.
    """
    def __init__(self, channels, activation='gelu', reduction=4, norm='none'):
        super().__init__()
        self.shared_convs = ConvStack(channels, activation=activation, reduction=reduction, norm=norm)

    def forward(self, x):
        max_pool = F.adaptive_max_pool2d(x, 1)
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        weight = torch.sigmoid(self.shared_convs(max_pool) + self.shared_convs(avg_pool))
        return x * weight


class VMambaCCMSEM(nn.Module):
    """VMambaCC Multi-head Spatial Enhancement Module."""
    def __init__(self, channels, heads=4, kernel_size=7):
        super().__init__()
        heads = max(1, int(heads))
        if channels % heads != 0:
            raise ValueError(f'fusion_mhf_heads={heads} must divide hidden channels={channels}')
        self.heads = heads
        padding = int(kernel_size) // 2
        self.spatial_convs = nn.ModuleList(
            nn.Conv2d(2, 1, int(kernel_size), padding=padding)
            for _ in range(heads)
        )

    def forward(self, x):
        groups = x.chunk(self.heads, dim=1)
        enhanced_groups = []
        for group, conv in zip(groups, self.spatial_convs):
            pooled = torch.cat(
                [group.max(dim=1, keepdim=True)[0], group.mean(dim=1, keepdim=True)],
                dim=1,
            )
            spatial_weight = torch.sigmoid(conv(pooled))
            enhanced_groups.append(group * spatial_weight)
        return torch.cat(enhanced_groups, dim=1)


class VMambaCCHCEM(nn.Module):
    """VMambaCC high-level channel enhancement module."""
    def __init__(self, channels, activation='gelu', reduction=4, norm='none'):
        super().__init__()
        self.shared_convs = ConvStack(channels, activation=activation, reduction=reduction, norm=norm)
        self.out_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        max_pool = F.adaptive_max_pool2d(x, 1)
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        return self.out_conv(self.shared_convs(max_pool) + self.shared_convs(avg_pool))


class VMambaCCMHF(nn.Module):
    """MHF attention as described by VMambaCC.

    The block consumes high-level features and produces an upsampled attention
    feature that is multiplied into a lower-level feature map.
    """
    def __init__(
        self,
        channels,
        mode='none',
        heads=4,
        activation='gelu',
        reduction=4,
        norm='none',
        spatial_kernel=7,
        output_activation='none',
    ):
        super().__init__()
        self.mode = mode
        self.output_activation = output_activation
        self.use_cem = mode in ('cem', 'cem_msem', 'full')
        self.use_msem = mode in ('cem_msem', 'full')
        self.use_hcem = mode == 'full'

        if mode == 'none':
            self.enabled = False
            return
        if mode not in ('cem', 'cem_msem', 'full'):
            raise ValueError(f'Unsupported fusion_mhf_mode: {mode}. Use "none", "cem", "cem_msem", or "full".')
        if output_activation not in ('none', 'sigmoid'):
            raise ValueError('fusion_mhf_output_activation must be "none" or "sigmoid"')

        self.enabled = True
        self.cem = VMambaCCCEM(channels, activation=activation, reduction=reduction, norm=norm)
        self.msem = VMambaCCMSEM(channels, heads=heads, kernel_size=spatial_kernel) if self.use_msem else nn.Identity()
        self.hcem = VMambaCCHCEM(channels, activation=activation, reduction=reduction, norm=norm) if self.use_hcem else nn.Identity()

    def forward(self, high_feature, target_size):
        if not self.enabled:
            return None
        x = high_feature
        if self.use_cem:
            x = self.cem(x)
        x = self.msem(x)
        x = self.hcem(x)
        if self.output_activation == 'sigmoid':
            x = torch.sigmoid(x)
        return F.interpolate(x, size=target_size, mode='nearest')


class _ResidualChannelGate(nn.Module):
    def __init__(self, channels, activation='gelu', reduction=4):
        super().__init__()
        hidden = max(1, channels // int(reduction))
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            _make_activation(activation),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.shared[-1].weight)
        nn.init.zeros_(self.shared[-1].bias)

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        gate = torch.sigmoid(self.shared(avg) + self.shared(max_pool))
        return 1.0 + 2.0 * (gate - 0.5)


class _ResidualSpatialGate(nn.Module):
    def __init__(self, channels, heads=1):
        super().__init__()
        heads = max(1, int(heads))
        if channels % heads != 0:
            raise ValueError(f'fusion_mhf_heads={heads} must divide hidden channels={channels}')
        self.heads = heads
        self.convs = nn.ModuleList(nn.Conv2d(2, 1, 7, padding=3) for _ in range(heads))
        for conv in self.convs:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, x):
        groups = x.chunk(self.heads, dim=1)
        out = []
        for group, conv in zip(groups, self.convs):
            pooled = torch.cat(
                [group.max(dim=1, keepdim=True)[0], group.mean(dim=1, keepdim=True)],
                dim=1,
            )
            gate = torch.sigmoid(conv(pooled))
            out.append(group * (1.0 + 2.0 * (gate - 0.5)))
        return torch.cat(out, dim=1)


class MHFEnhance(nn.Module):
    """High-level feature enhancement for VGG FPN.

    ``residual`` keeps the PET-safe zero-init approximation. ``vmambacc``
    follows the CEM/MSEM/HCEM equations from the VMambaCC paper.
    """
    def __init__(
        self,
        channels,
        mode='none',
        heads=1,
        strength=1.0,
        activation='gelu',
        impl='residual',
        reduction=4,
        norm='none',
        spatial_kernel=7,
        output_activation='none',
    ):
        super().__init__()
        if impl == 'vmambacc':
            self.impl = impl
            self.paper_mhf = VMambaCCMHF(
                channels,
                mode=mode,
                heads=heads,
                activation=activation,
                reduction=reduction,
                norm=norm,
                spatial_kernel=spatial_kernel,
                output_activation=output_activation,
            )
            return
        if impl != 'residual':
            raise ValueError(f'Unsupported fusion_mhf_impl: {impl}. Use "residual" or "vmambacc".')
        self.impl = impl
        self.paper_mhf = None
        self.mode = mode
        self.strength = float(strength)
        self.use_cem = mode in ('cem', 'cem_msem', 'full')
        self.use_msem = mode in ('cem_msem', 'full')
        self.use_hcem = mode == 'full'

        if mode == 'none':
            self.out_proj = None
            return
        if mode not in ('cem', 'cem_msem', 'full'):
            raise ValueError(f'Unsupported fusion_mhf_mode: {mode}. Use "none", "cem", "cem_msem", or "full".')

        self.cem = _ResidualChannelGate(channels, activation=activation) if self.use_cem else nn.Identity()
        self.msem = _ResidualSpatialGate(channels, heads=heads) if self.use_msem else nn.Identity()
        self.hcem = _ResidualChannelGate(channels, activation=activation) if self.use_hcem else nn.Identity()
        self.out_proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, high_feature, target_size):
        if self.impl == 'vmambacc':
            return self.paper_mhf(high_feature, target_size)
        if self.out_proj is None:
            return None
        x = high_feature
        if self.use_cem:
            x = x * self.cem(x)
        if self.use_msem:
            x = self.msem(x)
        if self.use_hcem:
            x = x * self.hcem(x)
        logits = F.interpolate(self.out_proj(x), size=target_size, mode='nearest')
        return 1.0 + self.strength * torch.tanh(logits)


class FeatsFusion(nn.Module):
    def __init__(
        self,
        C3_size,
        C4_size,
        C5_size,
        hidden_size=256,
        out_size=256,
        out_kernel=3,
        mhf_mode='none',
        mhf_heads=1,
        mhf_position='before',
        mhf_strength=1.0,
        mhf_activation='gelu',
        mhf_impl='residual',
        mhf_reduction=4,
        mhf_norm='none',
        mhf_spatial_kernel=7,
        mhf_output_activation='none',
        fusion_type='fpn',
    ):
        super(FeatsFusion, self).__init__()
        self.mhf_mode = mhf_mode
        self.mhf_position = mhf_position
        self.fusion_type = fusion_type
        if mhf_position not in ('before', 'post'):
            raise ValueError(f'Unsupported fusion_mhf_position: {mhf_position}. Use "before" or "post".')
        if fusion_type not in ('fpn', 'hs2fpn', 'lite_fpn'):
            raise ValueError(f'Unsupported fusion_fpn_type: {fusion_type}. Use "fpn", "hs2fpn", or "lite_fpn".')
        self.P5_1 = nn.Conv2d(C5_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)

        self.P4_1 = nn.Conv2d(C4_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)

        self.P3_1 = nn.Conv2d(C3_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)
        rng_state = torch.random.get_rng_state()
        self.mhf_c4 = MHFEnhance(
            hidden_size,
            mode=mhf_mode,
            heads=mhf_heads,
            strength=mhf_strength,
            activation=mhf_activation,
            impl=mhf_impl,
            reduction=mhf_reduction,
            norm=mhf_norm,
            spatial_kernel=mhf_spatial_kernel,
            output_activation=mhf_output_activation,
        )
        self.mhf_c3 = MHFEnhance(
            hidden_size,
            mode=mhf_mode,
            heads=mhf_heads,
            strength=mhf_strength,
            activation=mhf_activation,
            impl=mhf_impl,
            reduction=mhf_reduction,
            norm=mhf_norm,
            spatial_kernel=mhf_spatial_kernel,
            output_activation=mhf_output_activation,
        )
        torch.random.set_rng_state(rng_state)

    def _apply_high_level_gate(self, low_feature, high_feature, attention):
        gate = attention(high_feature, low_feature.shape[-2:])
        if gate is None:
            return low_feature
        return low_feature * gate

    def _fuse_low_high(self, low_feature, high_feature, attention):
        return low_feature + high_feature

    def forward(self, inputs):
        C3, C4, C5 = inputs
        C3_shape, C4_shape, C5_shape = C3.shape[-2:], C4.shape[-2:], C5.shape[-2:]

        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, C4_shape)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        if self.mhf_position == 'before':
            P4_x = self._apply_high_level_gate(P4_x, P5_upsampled_x, self.mhf_c4)
        P4_x = self._fuse_low_high(P4_x, P5_upsampled_x, self.mhf_c4)
        if self.mhf_position == 'post':
            P4_x = self._apply_high_level_gate(P4_x, P5_upsampled_x, self.mhf_c4)
        P4_upsampled_x = F.interpolate(P4_x, C3_shape)
        if self.fusion_type != 'lite_fpn':
            P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        if self.mhf_position == 'before':
            P3_x = self._apply_high_level_gate(P3_x, P4_upsampled_x, self.mhf_c3)
        P3_x = self._fuse_low_high(P3_x, P4_upsampled_x, self.mhf_c3)
        if self.mhf_position == 'post':
            P3_x = self._apply_high_level_gate(P3_x, P4_upsampled_x, self.mhf_c3)
        if self.fusion_type != 'lite_fpn':
            P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]
    

class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool, args=None):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers
        self.fpn = FeatsFusion(
            256,
            512,
            512,
            hidden_size=num_channels,
            out_size=num_channels,
            out_kernel=3,
            mhf_mode=getattr(args, 'fusion_mhf_mode', 'none'),
            mhf_heads=getattr(args, 'fusion_mhf_heads', 1),
            mhf_position=getattr(args, 'fusion_mhf_position', 'before'),
            mhf_strength=getattr(args, 'fusion_mhf_strength', 1.0),
            mhf_activation=getattr(args, 'fusion_mhf_activation', 'gelu'),
            mhf_impl=getattr(args, 'fusion_mhf_impl', 'residual'),
            mhf_reduction=getattr(args, 'fusion_mhf_reduction', 4),
            mhf_norm=getattr(args, 'fusion_mhf_norm', 'none'),
            mhf_spatial_kernel=getattr(args, 'fusion_mhf_spatial_kernel', 7),
            mhf_output_activation=getattr(args, 'fusion_mhf_output_activation', 'none'),
            fusion_type=getattr(args, 'fusion_fpn_type', 'fpn'),
        )

    def forward(self, tensor_list: NestedTensor):
        feats = []
        if self.return_interm_layers:
            xs = tensor_list.tensors
            for idx, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                feats.append(xs)
                        
            # feature fusion
            features_fpn = self.fpn([feats[1], feats[2], feats[3]])
            features_fpn_4x = features_fpn[0]
            features_fpn_8x = features_fpn[1]

            # get tensor mask
            m = tensor_list.mask
            assert m is not None
            mask_4x = F.interpolate(m[None].float(), size=features_fpn_4x.shape[-2:]).to(torch.bool)[0]
            mask_8x = F.interpolate(m[None].float(), size=features_fpn_8x.shape[-2:]).to(torch.bool)[0]

            out: Dict[str, NestedTensor] = {}
            out['4x'] = NestedTensor(features_fpn_4x, mask_4x)
            out['8x'] = NestedTensor(features_fpn_8x, mask_8x)
        else:
            xs = self.body(tensor_list)
            out.append(xs)

        return out


class Backbone_VGG(BackboneBase_VGG):
    """
    VGG backbone
    """
    def __init__(self, name: str, return_interm_layers: bool, pretrained: bool = True, args=None):
        if name == 'vgg16_bn':
            backbone = vgg16_bn(pretrained=pretrained)
        else:
            raise ValueError(f'Unsupported VGG backbone: {name}')
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers, args=args)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: Dict[str, NestedTensor] = {}
        pos = {}
        for name, x in xs.items():            
            out[name] = x
            # position encoding
            pos[name] = self[1](x).to(x.tensors.dtype)
        return out, pos


def build_backbone_vgg(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone_VGG(
        args.backbone,
        True,
        pretrained=not getattr(args, 'no_pretrained_backbone', False),
        args=args,
    )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


if __name__ == '__main__':
    Backbone_VGG('vgg16_bn', True)
