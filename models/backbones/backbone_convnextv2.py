"""
Backbone modules for modern timm backbones used by PET.
"""
from typing import Dict
import importlib
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from util.misc import NestedTensor
from ..position_encoding import build_position_encoding


CONVNEXTV2_VARIANTS = (
    'convnextv2_atto',
    'convnextv2_femto',
    'convnextv2_pico',
    'convnextv2_nano',
    'convnextv2_tiny',
    'convnextv2_small',
    'convnextv2_base',
    'convnextv2_large',
)

SWIN_VARIANTS = (
    'swin_tiny_patch4_window7_224',
    'swin_small_patch4_window7_224',
    'swinv2_tiny_window8_256',
    'swinv2_small_window8_256',
)

MODERN_TIMM_BACKBONES = CONVNEXTV2_VARIANTS + SWIN_VARIANTS


def is_modern_timm_backbone(name: str) -> bool:
    return name in {'auto', 'auto_swin'} or name.startswith('convnextv2_') or name.startswith('swin')


def resolve_convnextv2_backbone_name(name: str) -> str:
    """
    Resolve the legacy ConvNeXt V2 auto path without changing existing semantics.
    """
    if name != 'auto':
        if name not in CONVNEXTV2_VARIANTS:
            raise ValueError(f'Unsupported ConvNeXt V2 backbone: {name}')
        return name

    if not torch.cuda.is_available():
        return 'convnextv2_nano'

    total_memory_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
    if total_memory_gb >= 24:
        return 'convnextv2_large'
    if total_memory_gb >= 14:
        return 'convnextv2_base'
    if total_memory_gb >= 8:
        return 'convnextv2_small'
    return 'convnextv2_nano'


def resolve_timm_backbone_name(name: str) -> str:
    """
    Resolve auto selectors and validate supported timm backbones.
    """
    if name == 'auto':
        return resolve_convnextv2_backbone_name(name)
    if name == 'auto_swin':
        if not torch.cuda.is_available():
            return 'swinv2_tiny_window8_256'
        total_memory_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
        if total_memory_gb >= 16:
            return 'swinv2_small_window8_256'
        return 'swinv2_tiny_window8_256'
    if name not in MODERN_TIMM_BACKBONES:
        raise ValueError(f'Unsupported timm backbone: {name}')
    return name


def get_convnextv2_training_defaults(backbone_name: str):
    """
    Backward-compatible alias for the original ConvNeXt V2 helper.
    """
    return get_timm_training_defaults(backbone_name)


def get_timm_training_defaults(backbone_name: str):
    """
    Conservative batch-size and learning-rate defaults for modern PET backbones.
    """
    defaults = {
        'convnextv2_atto': (8, 1.0e-4, 1.0e-5),
        'convnextv2_femto': (8, 1.0e-4, 1.0e-5),
        'convnextv2_pico': (8, 1.0e-4, 1.0e-5),
        'convnextv2_nano': (8, 1.0e-4, 1.0e-5),
        'convnextv2_tiny': (6, 7.5e-5, 7.5e-6),
        'convnextv2_small': (4, 5.0e-5, 5.0e-6),
        'convnextv2_base': (2, 2.5e-5, 2.5e-6),
        'convnextv2_large': (1, 1.25e-5, 1.25e-6),
        'swin_tiny_patch4_window7_224': (4, 5.0e-5, 5.0e-6),
        'swin_small_patch4_window7_224': (2, 2.5e-5, 2.5e-6),
        'swinv2_tiny_window8_256': (4, 5.0e-5, 5.0e-6),
        'swinv2_small_window8_256': (2, 2.5e-5, 2.5e-6),
    }
    if backbone_name not in defaults:
        raise ValueError(f'Unsupported backbone defaults for {backbone_name}')
    return defaults[backbone_name]


def _requires_relaxed_img_size(backbone_name: str) -> bool:
    return backbone_name.startswith('swin')


class ModernBackboneFPN(nn.Module):
    def __init__(self, stage_channels, hidden_size=256, out_size=256, out_kernel=3):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(ch, hidden_size, kernel_size=1, stride=1, padding=0)
            for ch in stage_channels
        )
        self.output_convs = nn.ModuleList(
            nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel // 2)
            for _ in stage_channels
        )

    def forward(self, inputs):
        c2, c3, c4, c5 = inputs

        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')

        p2 = self.output_convs[0](p2)
        p3 = self.output_convs[1](p3)
        p4 = self.output_convs[2](p4)
        p5 = self.output_convs[3](p5)
        return [p2, p3, p4, p5]


class BackboneBase_Timm(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, return_interm_layers: bool):
        super().__init__()
        self.backbone = backbone
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

        self.stage_channels = [info['num_chs'] for info in backbone.feature_info.get_dicts()]
        if len(self.stage_channels) != 4:
            raise ValueError(
                f'The selected timm backbone must expose 4 stages, but got {len(self.stage_channels)} stages.'
            )
        self.fpn = ModernBackboneFPN(self.stage_channels, hidden_size=num_channels, out_size=num_channels, out_kernel=3)

    @staticmethod
    def _to_nchw(feature: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if feature.dim() != 4:
            raise ValueError(f'Expected a 4D feature tensor, but got shape {tuple(feature.shape)}')
        if feature.shape[1] == expected_channels:
            return feature
        if feature.shape[-1] == expected_channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f'Could not infer channel dimension for feature shape {tuple(feature.shape)} '
            f'with expected channels {expected_channels}.'
        )

    def forward(self, tensor_list: NestedTensor):
        feats = self.backbone(tensor_list.tensors)
        feats = [self._to_nchw(feat, ch) for feat, ch in zip(feats, self.stage_channels)]

        features_fpn = self.fpn(feats)
        features_fpn_4x = features_fpn[0]
        features_fpn_8x = features_fpn[1]

        m = tensor_list.mask
        assert m is not None
        mask_4x = F.interpolate(m[None].float(), size=features_fpn_4x.shape[-2:]).to(torch.bool)[0]
        mask_8x = F.interpolate(m[None].float(), size=features_fpn_8x.shape[-2:]).to(torch.bool)[0]

        out: Dict[str, NestedTensor] = {}
        out['4x'] = NestedTensor(features_fpn_4x, mask_4x)
        out['8x'] = NestedTensor(features_fpn_8x, mask_8x)
        return out


class Backbone_Timm(BackboneBase_Timm):
    """
    Generic timm backbone adapter for PET.
    """

    def __init__(self, name: str, return_interm_layers: bool, pretrained: bool = True, img_size: int = 256):
        backbone_name = resolve_timm_backbone_name(name)
        timm = importlib.import_module('timm')

        create_kwargs = {
            'pretrained': pretrained,
            'features_only': True,
            'out_indices': (0, 1, 2, 3),
        }
        if _requires_relaxed_img_size(backbone_name):
            create_kwargs.update({
                'img_size': img_size,
                'strict_img_size': False,
            })

        try:
            backbone = timm.create_model(backbone_name, **create_kwargs)
        except (RuntimeError, OSError) as exc:
            if not pretrained:
                raise
            warnings.warn(
                f'Could not load pretrained weights for {backbone_name}: {exc}. Falling back to random init.',
                RuntimeWarning,
            )
            create_kwargs['pretrained'] = False
            backbone = timm.create_model(backbone_name, **create_kwargs)

        num_channels = 256
        super().__init__(backbone, num_channels, return_interm_layers)


class Joiner(nn.Module):
    def __init__(self, backbone, position_embedding):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone(tensor_list)
        out: Dict[str, NestedTensor] = {}
        pos = {}
        for name, x in xs.items():
            out[name] = x
            pos[name] = self.position_embedding(x).to(x.tensors.dtype)
        return out, pos


def build_backbone_timm(args):
    position_embedding = build_position_encoding(args)
    backbone_name = resolve_timm_backbone_name(args.backbone)
    args.backbone = backbone_name
    backbone = Backbone_Timm(
        backbone_name,
        True,
        pretrained=True,
        img_size=max(256, int(getattr(args, 'patch_size', 256))),
    )
    model = Joiner(backbone, position_embedding)
    return model


def build_backbone_convnextv2(args):
    return build_backbone_timm(args)


if __name__ == '__main__':
    Backbone_Timm('convnextv2_nano', True)
